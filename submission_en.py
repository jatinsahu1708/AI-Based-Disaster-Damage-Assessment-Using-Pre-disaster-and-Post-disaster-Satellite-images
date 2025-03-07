import os
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import zipfile
from model.dual_encoder_segformer import DualEncoderSegformer
from scipy.stats import mode

# -------------- Configuration ----------------

# Device setup: Use GPU if available.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# List of best model checkpoints for all 5 folds (change these paths to your actual checkpoint paths)

MODEL_PATHS = ["/home/jothi/jatin/kfold/models/best_segformer_fold_1_epoch_20_iou_0.588196.pth",
"/home/jothi/jatin/kfold/models/best_segformer_fold_2_epoch_12_iou_0.597912.pth",
"/home/jothi/jatin/kfold/models/best_segformer_fold_3_epoch_30_iou_0.577298.pth",
"/home/jothi/jatin/kfold/models/best_segformer_fold_4_epoch_30_iou_0.633900.pth",
"/home/jothi/jatin/kfold/models/best_segformer_fold_5_epoch_26_iou_0.561013.pth"]

NUM_CLASSES = 4
RESIZE_SIZE = (1024, 1024)
PRETRAINED_MODEL = "nvidia/mit-b4"

# Voting method: set to "hard" or "soft"
VOTING = "hard"

# Paths for input folders and output zip file.
PRE_EVENT_FOLDER = "/home/jothi/jatin/damage_assessment/dfc25_track2_trainval/val/pre-event/"
POST_EVENT_FOLDER = "/home/jothi/jatin/damage_assessment/dfc25_track2_trainval/val/post-event/"
OUTPUT_ZIP = "ensemble_predictions_hard.zip"

# -------------- Data Transform ----------------

albumentations_transform = A.Compose([
    A.Resize(height=RESIZE_SIZE[0], width=RESIZE_SIZE[1]),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0),
    ToTensorV2()
])

def preprocess_image(image_path, is_grayscale=False):
    image = np.array(Image.open(image_path))
    if is_grayscale and len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 1:
        image = np.concatenate([image] * 3, axis=-1)
    transformed = albumentations_transform(image=image)
    return transformed['image'].unsqueeze(0).to(DEVICE)

# -------------- Load Ensemble Models ----------------

# Add the problematic global to the allowed globals once before loading any checkpoints.
torch.serialization.add_safe_globals(["numpy._core.multiarray.scalar"])

ensemble_models = []
for path in MODEL_PATHS:
    model = DualEncoderSegformer(pretrained_rgb=PRETRAINED_MODEL, num_labels=NUM_CLASSES).to(DEVICE)
    # Load model checkpoint with map_location to DEVICE and disable weights_only
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    # Check if the checkpoint is a dictionary with 'model_state_dict'
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    ensemble_models.append(model)
print("Loaded all ensemble models.")


# -------------- Ensemble Prediction Functions ----------------

def ensemble_predict_soft(pre_image, post_image):
    """Soft voting: average the softmax probabilities and then take argmax."""
    predictions = []
    with torch.no_grad():
        for model in ensemble_models:
            output = model(pre_image, post_image)  # shape: (1, num_classes, H, W)
            probs = torch.softmax(output, dim=1)
            predictions.append(probs)
        avg_probs = torch.mean(torch.stack(predictions), dim=0)
        final_pred = torch.argmax(avg_probs, dim=1).squeeze(0)
    return final_pred.cpu().numpy()

def ensemble_predict_hard(pre_image, post_image):
    """Hard voting: take the mode (majority vote) of the predicted classes."""
    predictions = []
    with torch.no_grad():
        for model in ensemble_models:
            output = model(pre_image, post_image)  # shape: (1, num_classes, H, W)
            pred = torch.argmax(output, dim=1).squeeze(0)
            predictions.append(pred.cpu().numpy())
    predictions = np.stack(predictions, axis=0)  # shape: (5, H, W)
    # Compute the mode along the first axis for each pixel.
    mode_result = mode(predictions, axis=0)
    final_pred = mode_result.mode.squeeze(0)  # shape: (H, W)
    return final_pred

def ensemble_predict(pre_image, post_image, voting="soft"):
    if voting == "soft":
        return ensemble_predict_soft(pre_image, post_image)
    elif voting == "hard":
        return ensemble_predict_hard(pre_image, post_image)
    else:
        raise ValueError("Voting must be either 'soft' or 'hard'.")

# -------------- Evaluation and Saving Predictions ----------------

def evaluate_and_save(pre_event_folder, post_event_folder, output_zip, voting="soft"):
    pre_images = sorted(os.listdir(pre_event_folder))
    post_images = sorted(os.listdir(post_event_folder))
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for pre_img, post_img in zip(pre_images, post_images):
            pre_path = os.path.join(pre_event_folder, pre_img)
            post_path = os.path.join(post_event_folder, post_img)
            original_size = Image.open(pre_path).size
            
            pre_image = preprocess_image(pre_path)
            # For post-event image, we assume it might be grayscale.
            post_image = preprocess_image(post_path, is_grayscale=True)
            
            # Get ensemble prediction using the selected voting method.
            final_mask = ensemble_predict(pre_image, post_image, voting=voting)
            
            # Resize mask to original image size.
            mask_img = Image.fromarray(final_mask.astype(np.uint8)).resize(original_size, resample=Image.NEAREST)
            # Modify output filename as needed.
            output_filename = pre_img.replace("_pre_disaster", "").replace(".tif", "_building_damage.png")
            mask_img.save(output_filename, format="PNG")
            zipf.write(output_filename)
            os.remove(output_filename)
            print(f"Saved and added to zip: {output_filename}")

if __name__ == "__main__":
    evaluate_and_save(PRE_EVENT_FOLDER, POST_EVENT_FOLDER, OUTPUT_ZIP, voting=VOTING)
    print(f"All ensemble predictions saved in {OUTPUT_ZIP}")
