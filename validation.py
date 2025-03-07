import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from model.dual_encoder_segformer import DualEncoderSegformer

# Load model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/home/jothi/jatin/segformer_epoch_100.pth"  # Change based on your saved model
NUM_CLASSES = 4
RESIZE_SIZE = (256, 256)
PRETRAINED_MODEL = "nvidia/mit-b0"

# Load the trained model
model = DualEncoderSegformer(pretrained_rgb=PRETRAINED_MODEL, num_labels=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Albumentations Transform (Same as dataset.py)
albumentations_transform = A.Compose([
    A.Resize(height=RESIZE_SIZE[0], width=RESIZE_SIZE[1]),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def preprocess_image(image_path, is_grayscale=False):
    """ Load and preprocess the image using Albumentations """
    image = np.array(Image.open(image_path))
    
    # Ensure grayscale image has shape [H, W, 1] like in dataset.py
    if is_grayscale and len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)

    # Apply Albumentations transform
    transformed = albumentations_transform(image=image)
    
    return transformed['image'].unsqueeze(0).to(DEVICE)  # Add batch dimension

def evaluate(pre_event_path, post_event_path, save_path="/home/jothi/jatin/output_mask_la-palma_volcano_00000465.tif"):
    """ Evaluate the model on a single image pair """
    pre_image = preprocess_image(pre_event_path)
    post_image = preprocess_image(post_event_path, is_grayscale=True)  # Ensure grayscale post-event image
    
    with torch.no_grad():
        output = model(pre_image, post_image)
        mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    
    # Save the mask
    mask_colored = (mask * (255 / NUM_CLASSES)).astype(np.uint8)  # Scale mask values
    mask_img = Image.fromarray(mask_colored)
    mask_img.save(save_path)
    print(f"Segmentation mask saved to {save_path}")

# Example Usage
if __name__ == "__main__":
    pre_event_image_path = "/home/jothi/jatin/damage_assessment/dfc25_track2_trainval/train/pre-event/la_palma-volcano_00000465_pre_disaster.tif"  # Update with actual path
    post_event_image_path = "/home/jothi/jatin/damage_assessment/dfc25_track2_trainval/train/post-event/la_palma-volcano_00000465_post_disaster.tif"  # Update with actual path
    evaluate(pre_event_image_path, post_event_image_path)
