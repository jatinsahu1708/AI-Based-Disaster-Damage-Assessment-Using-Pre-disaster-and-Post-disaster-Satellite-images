import os
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import zipfile
from model.shared_encodder_segformer import SharedEncoderSegformer

# Define device, hyperparameters, and paths
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/home/jothi/jatin/sharedsegformer/best_sharedsegformer_epoch_30_iou_0.603978.pth"
NUM_CLASSES = 4
PRETRAINED_MODEL = "nvidia/mit-b4"
RESIZE_SIZE = (1024, 1024)  # Ensure this is defined

# Instantiate the model using the same architecture as during training
model = SharedEncoderSegformer(pretrained_rgb=PRETRAINED_MODEL, num_labels=NUM_CLASSES).to(DEVICE)

# Load the checkpoint dictionary

with torch.serialization.safe_globals(["numpy._core.multiarray.scalar"]):
    checkpoint = torch.load(MODEL_PATH, weights_only=False)

# Load the model state dictionary from the checkpoint
model.load_state_dict(checkpoint['model_state_dict'])
# Set the model to evaluation mode
model.eval()

print(f"Model loaded from epoch {checkpoint['epoch']} with best Mean IoU: {checkpoint['best_mean_iou']:.6f}")

# Define the Albumentations transformation
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
        # If already (H, W, 1), concatenate to form 3 channels
        image = np.concatenate([image] * 3, axis=-1)
    transformed = albumentations_transform(image=image)
    return transformed['image'].unsqueeze(0).to(DEVICE)

def evaluate_and_save(pre_event_folder, post_event_folder, output_zip):
    pre_images = sorted(os.listdir(pre_event_folder))
    post_images = sorted(os.listdir(post_event_folder))
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for pre_img, post_img in zip(pre_images, post_images):
            pre_path = os.path.join(pre_event_folder, pre_img)
            post_path = os.path.join(post_event_folder, post_img)
            original_size = Image.open(pre_path).size
            pre_image = preprocess_image(pre_path)
            post_image = preprocess_image(post_path, is_grayscale=True)
            
            with torch.no_grad():
                output = model(pre_image, post_image)
                mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            
            mask_img = Image.fromarray(mask).resize(original_size, resample=Image.NEAREST)
            output_filename = pre_img.replace("_pre_disaster", "").replace(".tif", "_building_damage.png")
            mask_img.save(output_filename, format="PNG")
            zipf.write(output_filename)
            os.remove(output_filename)
            print(f"Saved and added to zip: {output_filename}")

if __name__ == "__main__":
    pre_event_folder = "/home/jothi/jatin/damage_assessment/dfc25_track2_trainval/val/pre-event/"
    post_event_folder = "/home/jothi/jatin/damage_assessment/dfc25_track2_trainval/val/post-event/"
    output_zip = "predictions_shared.zip"
    
    evaluate_and_save(pre_event_folder, post_event_folder, output_zip)
    print(f"All predictions saved in {output_zip}")
