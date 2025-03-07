import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
import albumentations as A
from tqdm import tqdm
import matplotlib.pyplot as plt
from model.dual_encoder_segformer import DualEncoderSegformer
from utils.dataset import BRIGHTDataset
from utils.losses import CombinedLoss
from evaluate import evaluate


# Hyperparameters
BATCH_SIZE = 2
LR = 1e-4
EPOCHS = 100  # Total epochs to train
RESIZE_SIZE = (1024, 1024)
PRETRAINED_MODEL = "nvidia/mit-b4"

# Paths (Update as needed)
PRE_EVENT_DIR = "/home/jothi/kushal/data/cracks/images/"
POST_EVENT_DIR = "/home/jothi/kushal/data/cracks/images/"
MASK_DIR = "/home/jothi/kushal/data/cracks/masks/"
LOSS_PLOT_PATH = "/home/jothi/jatin/kushalloss_plot.png"

# Dataset and Transforms
train_transform = A.Compose([
    A.Resize(height=RESIZE_SIZE[0], width=RESIZE_SIZE[1]),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0),
], additional_targets={'image1': 'image'})

# Load the full dataset
full_dataset = BRIGHTDataset(
    pre_event_dir=PRE_EVENT_DIR,
    post_event_dir=POST_EVENT_DIR,
    mask_dir=MASK_DIR,
    transform=train_transform,
    
    resize_size=RESIZE_SIZE
)

# Split dataset: 80% train, 20% validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Class frequencies and weights
freqs = torch.tensor([0.876, 0.1076, 0.0083, 0.00806], dtype=torch.float32)#
class_weights = 1.0 / (freqs + 1e-6)
class_weights /= class_weights.max()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DualEncoderSegformer(pretrained_rgb=PRETRAINED_MODEL, num_labels=4).to(device)
criterion = CombinedLoss(class_weights=class_weights.to(device))
optimizer = AdamW(model.parameters(), lr=LR)

best_mean_iou = 0.0
train_losses = []
val_losses = []
train_focal_losses = []
train_dice_losses = []

for epoch in range(1, EPOCHS + 1):
    
    torch.cuda.empty_cache()
    model.train()
    
    total_loss = 0.0
    total_focal_loss = 0.0
    total_dice_loss = 0.0
    num_batches = 0
    
    for pre_img, post_img, mask in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        pre_img, post_img, mask = pre_img.to(device), post_img.to(device), mask.to(device).long()
        optimizer.zero_grad()
        
        outputs = model(pre_img, post_img)
        loss, focal_loss, dice_loss = criterion(outputs, mask)
        
        total_loss += loss.item()
        total_focal_loss += focal_loss.item()
        total_dice_loss += dice_loss.item()
        num_batches += 1
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    epoch_train_loss = total_loss / num_batches
    epoch_focal_loss = total_focal_loss / num_batches
    epoch_dice_loss = total_dice_loss / num_batches
    
    train_losses.append(epoch_train_loss)
    train_focal_losses.append(epoch_focal_loss)
    train_dice_losses.append(epoch_dice_loss)
    
    print(f"Epoch {epoch}: Train Loss: {epoch_train_loss:.6f} (Focal: {epoch_focal_loss:.6f}, Dice: {epoch_dice_loss:.6f})")
    
    if epoch % 5 == 0:
       # Calculate IoU metrics on training set
       viou, vmean_iou = evaluate(model, train_loader, device, num_classes=4)
       print(f"Train IoU per class: {viou}")
       print(f"Mean train IoU: {vmean_iou:.4f}")    
    
    model.eval()
    val_total_loss = 0.0
    val_batches = 0
    with torch.no_grad():
        for pre_img, post_img, mask in val_loader:
            pre_img, post_img, mask = pre_img.to(device), post_img.to(device), mask.to(device).long()
            outputs = model(pre_img, post_img)
            loss, _, _ = criterion(outputs, mask)
            val_total_loss += loss.item()
            val_batches += 1
    epoch_val_loss = val_total_loss / val_batches
    val_losses.append(epoch_val_loss)
    print(f"Epoch {epoch}: Validation Loss: {epoch_val_loss:.6f}")
    
    iou, mean_iou = evaluate(model, val_loader, device, num_classes=4)
    print(f"Validation IoU per class: {iou}")
    print(f"Mean IoU: {mean_iou:.4f}")
    
    # Save best model based on validation mean IoU (higher is better)
    if mean_iou > best_mean_iou:
        best_mean_iou = mean_iou
        best_model_path = f"/home/jothi/jatin/kushalbest_segformer_model_epoch_{epoch}_iou_{best_mean_iou:.6f}.pth"
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved at epoch {epoch} with mean_iou: {best_mean_iou:.6f}")
    if epoch % 5 == 0:  # or any frequency you prefer
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epoch+1), train_losses, label='Train Loss', marker='o')
        plt.plot(range(1, epoch+1), val_losses, label='Validation Loss', marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss")
        plt.grid(True)
        plt.savefig(LOSS_PLOT_PATH)
        print(f"Intermediate loss plot saved at {LOSS_PLOT_PATH}")
        plt.close()  # Close the figure to free memory

    # Save a checkpoint every 25 epochs
    if epoch % 25 == 0:
        checkpoint_path = f"/home/jothi/jatin/kushalsegformer_epoch_{epoch}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at epoch {epoch}: {checkpoint_path}")

# Plot and save train and validation loss only




        
# import torch
# from torch.utils.data import DataLoader,random_split
# from torch.optim import AdamW
# import albumentations as A
# from tqdm import tqdm
# from model.dual_encoder_segformer import DualEncoderSegformer
# from utils.dataset import BRIGHTDataset
# from utils.losses import CombinedLoss
# from evaluate import evaluate


# # Hyperparameters
# BATCH_SIZE = 4
# LR = 1e-4
# EPOCHS = 20
# RESIZE_SIZE = (1024,1024)
# PRETRAINED_MODEL = "nvidia/mit-b4"

# # Paths (Update these!)
# PRE_EVENT_DIR = "/home/jothi/jatin/damage_assessment/dfc25_track2_trainval/train/pre-event"
# POST_EVENT_DIR = "/home/jothi/jatin/damage_assessment/dfc25_track2_trainval/train/post-event"
# MASK_DIR = "/home/jothi/jatin/damage_assessment/dfc25_track2_trainval/train/masks"

# # Dataset and Transforms
# train_transform = A.Compose([
#     A.Resize(height=RESIZE_SIZE[0], width=RESIZE_SIZE[1]),
#     A.HorizontalFlip(p=0.5),
#     A.VerticalFlip(p=0.5),
#     A.RandomRotate90(p=0.5),
#     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
# ], additional_targets={'image1': 'image'})

# # Load the full dataset
# full_dataset = BRIGHTDataset(
#     pre_event_dir=PRE_EVENT_DIR,
#     post_event_dir=POST_EVENT_DIR,
#     mask_dir=MASK_DIR,
#     transform=train_transform,
#     resize_size=RESIZE_SIZE
# )

# # Split the dataset into 80% training and 20% validation
# train_size = int(0.8 * len(full_dataset))
# val_size = len(full_dataset) - train_size
# train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# # Class frequencies (percentages to proportions)
# freqs = torch.tensor([0.876, 0.1076, 0.0083, 0.00806], dtype=torch.float32)
# # Compute inverse frequencies and normalize
# class_weights = 1.0 / freqs
# class_weights /= class_weights.sum()
# print("Static class weights shape:", class_weights.shape)  # Should be torch.Size([4])

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = DualEncoderSegformer(pretrained_rgb=PRETRAINED_MODEL, num_labels=4).to(device)
# criterion = CombinedLoss(class_weights=class_weights.to(device))
# optimizer = AdamW(model.parameters(), lr=LR)

#   # Assuming you save the above function in evaluate.py

# # Training loop
# for epoch in range(EPOCHS):
#     model.train()
#     epoch_loss = 0.0

#     for pre_img, post_img, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
#         pre_img = pre_img.to(device)
#         post_img = post_img.to(device)
#         mask = mask.to(device)
#         mask = mask.long()  # Ensure the target is a long tensor

#         optimizer.zero_grad()
#         outputs = model(pre_img, post_img)
#         loss = criterion(outputs, mask)
#         loss.backward()
#         optimizer.step()

#         epoch_loss += loss.item()

#     print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader):.4f}")
    
#     train_iou, train_mean_iou = evaluate(model, train_loader, device, num_classes=4)
#     print(f"Training IoU per class: {train_iou}")
#     print(f"Mean Training IoU: {train_mean_iou:.4f}")

#     # Validation loop and IoU calculation
#     iou, mean_iou = evaluate(model, val_loader, device, num_classes=4)
#     print(f"Validation IoU per class: {iou}")
#     print(f"Mean IoU: {mean_iou:.4f}")

#     # Save model after each epoch
    
#     if (epoch + 1) % 10 == 0:
#         torch.save(model.state_dict(), f"/home/jothi/jatin/segformer_1batchsize_b4_focl+dicl_caf_epoch_{epoch+1}.pth")
#         print(f"Model saved as segformer_epoch_{epoch+1}.pth")



