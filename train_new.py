
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import albumentations as A
from tqdm import tqdm
import matplotlib.pyplot as plt
from model.dual_encoder_segformer import DualEncoderSegformer
from utils.dataset import BRIGHTDataset
from utils.wce_lov_loss import LovaszWCE
from evaluate import evaluate
from sklearn.model_selection import KFold

# Hyperparameters
BATCH_SIZE = 2
LR = 1e-4
EPOCHS = 30  # Total epochs to train
RESIZE_SIZE = (1024, 1024)
PRETRAINED_MODEL = "nvidia/mit-b4"
K_FOLDS = 5

# Paths
PRE_EVENT_DIR = "/home/jothi/jatin/damage_assessment/dfc25_track2_trainval/train/pre-event"
POST_EVENT_DIR = "/home/jothi/jatin/damage_assessment/dfc25_track2_trainval/train/post-event"
MASK_DIR = "/home/jothi/jatin/damage_assessment/dfc25_track2_trainval/train/masks"
BASE_LOSS_PLOT_PATH = "/home/jothi/jatin/kfold/loss_plot"  # Base name for loss plots

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

# Load full dataset
full_dataset = BRIGHTDataset(
    pre_event_dir=PRE_EVENT_DIR,
    post_event_dir=POST_EVENT_DIR,
    mask_dir=MASK_DIR,
    transform=train_transform,
    resize_size=RESIZE_SIZE
)

# Set device to GPU 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Running folds 3-4 on {device}")

# K-Fold Cross Validation with fixed random state
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
    # Only process folds 3 and 4
    if fold==0 or fold >= 3:
        continue

    print(f"Fold {fold+1}/{K_FOLDS}")
    
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model on GPU 1
    model = DualEncoderSegformer(pretrained_rgb=PRETRAINED_MODEL, num_labels=4).to(device)
    class_frequencies = torch.tensor([0.876, 0.1076, 0.0083, 0.00806], dtype=torch.float32)
    class_weights = 1.0 / class_frequencies
    class_weights /= class_weights.max()
    criterion = LovaszWCE(class_weights=class_weights.to(device))
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    best_mean_iou = 0.0
    train_losses, val_losses = [], []
    
    for epoch in range(1, EPOCHS + 1):
        torch.cuda.empty_cache()
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for pre_img, post_img, mask in tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch}/{EPOCHS}"):
            pre_img, post_img, mask = pre_img.to(device), post_img.to(device), mask.to(device).long()
            optimizer.zero_grad()
            
            outputs = model(pre_img, post_img)
            loss, _, _ = criterion(outputs, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        epoch_train_loss = total_loss / num_batches
        train_losses.append(epoch_train_loss)
        print(f"Fold {fold+1} Epoch {epoch}: Train Loss: {epoch_train_loss:.6f}")
        
        # Validation
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
        print(f"Fold {fold+1} Epoch {epoch}: Validation Loss: {epoch_val_loss:.6f}")
        
        # IoU Evaluation on validation set
        iou, mean_iou = evaluate(model, val_loader, device, num_classes=4)
        print(f"Fold {fold+1} Epoch {epoch}: Mean IoU: {mean_iou:.4f}")
        
        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            best_model_path = f"/home/jothi/jatin/kfold/models/best_segformer_fold_{fold+1}_epoch_{epoch}_iou_{best_mean_iou:.6f}.pth"
            checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_mean_iou': best_mean_iou,
    }
            torch.save(model.state_dict(), best_model_path)
            print(f"Fold {fold+1}: New best model saved at epoch {epoch} with mean IoU: {best_mean_iou:.6f}")
        
        # Save loss plot every 5 epochs with fold info in filename
        if epoch % 5 == 0:
            loss_plot_path = f"{BASE_LOSS_PLOT_PATH}_fold_{fold+1}.png"
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, epoch+1), train_losses, label='Train Loss', marker='o')
            plt.plot(range(1, epoch+1), val_losses, label='Validation Loss', marker='o')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.title(f"Training and Validation Loss (Fold {fold+1})")
            plt.grid(True)
            plt.savefig(loss_plot_path)
            print(f"Fold {fold+1}: Loss plot saved at {loss_plot_path}")
            plt.close()
        
        # Save a checkpoint every 25 epochs with fold info in filename
        if epoch % 25 == 0 or epoch==30:
            checkpoint_path = f"/home/jothi/jatin/kfold/models/segformer_fold_{fold+1}_epoch_{epoch}.pth"
            checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_mean_iou': best_mean_iou,
    }
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Fold {fold+1}: Checkpoint saved at epoch {epoch}: {checkpoint_path}")


#import torch
# import os
# from torch.utils.data import DataLoader, random_split
# from torch.optim import AdamW
# import albumentations as A
# from tqdm import tqdm
# from model.dual_encoder_segformer import DualEncoderSegformer
# from utils.dataset import BRIGHTDataset
# from utils.wce_lov_loss import LovaszWCE
# from evaluate import evaluate
# from torch.optim.lr_scheduler import CosineAnnealingLR

# # Hyperparameters
# BATCH_SIZE = 2
# LR = 1e-4
# EPOCHS = 100  # Train from epoch 21 to 100
# RESIZE_SIZE = (1024, 1024)
# PRETRAINED_MODEL = "nvidia/mit-b4"

# # Paths (Update these!)
# PRE_EVENT_DIR = "/home/jothi/jatin/damage_assessment/dfc25_track2_trainval/train/pre-event"
# POST_EVENT_DIR = "/home/jothi/jatin/damage_assessment/dfc25_track2_trainval/train/post-event"
# MASK_DIR = "/home/jothi/jatin/damage_assessment/dfc25_track2_trainval/train/masks"
# MODEL_SAVE_DIR = "/home/jothi/jatin/"  # Base directory for saving models

# # Dataset and Transforms
# train_transform = A.Compose([
#     A.Resize(height=RESIZE_SIZE[0], width=RESIZE_SIZE[1]),
#     A.HorizontalFlip(p=0.5),
#     A.VerticalFlip(p=0.5),
#     A.RandomRotate90(p=0.5),
#     A.Normalize(mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225],
#                 max_pixel_value=255.0),
# ], additional_targets={'image1': 'image'})

# # Load the full dataset
# full_dataset = BRIGHTDataset(
#     pre_event_dir=PRE_EVENT_DIR,
#     post_event_dir=POST_EVENT_DIR,
#     mask_dir=MASK_DIR,
#     transform=train_transform,
#     resize_size=RESIZE_SIZE
# )

# # Split dataset: 80% training, 20% validation
# train_size = int(0.8 * len(full_dataset))
# val_size = len(full_dataset) - train_size
# train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# # Class frequencies (percentages to proportions)
# freqs = torch.tensor([0.876, 0.1076, 0.0083, 0.00806], dtype=torch.float32)
# class_weights = 1.0 / (freqs + 1e-6)  # Avoid division by zero
# class_weights /= class_weights.max()  # Normalize properly
# print("Class Weights:", class_weights)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = DualEncoderSegformer(pretrained_rgb=PRETRAINED_MODEL, num_labels=4).to(device)
# criterion = LovaszWCE(class_weights=class_weights.to(device))
# optimizer = AdamW(model.parameters(), lr=LR)

# # Load previous checkpoint
# #checkpoint_path = "/home/jothi/jatin/best_segformer_model_lov+wce_epoch_28.pth"
# #model.load_state_dict(torch.load(checkpoint_path))
# #print("Loaded model from previous checkpoint.")

# best_mean_iou = 0.0

# # Define scheduler
# scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# # Training loop
# for epoch in range(1, EPOCHS + 1):
#     torch.cuda.empty_cache()  # Clear GPU cache before each epoch
#     model.train()

#     total_lovasz_loss, total_wce_loss, total_loss = 0.0, 0.0, 0.0
#     num_batches = len(train_loader)

#     for pre_img, post_img, mask in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
#         pre_img, post_img, mask = pre_img.to(device), post_img.to(device), mask.to(device).long()
#         optimizer.zero_grad()

#         # Forward pass (FP32)
#         outputs = model(pre_img, post_img)
#         loss, wce_loss, lovasz_loss = criterion(outputs, mask)

#         # Backward pass
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()

#         # Track losses
#         total_lovasz_loss += lovasz_loss.item()
#         total_wce_loss += wce_loss.item()
#         total_loss += loss.item()

#     # Update scheduler
#     scheduler.step()

#     # Print per-epoch loss statistics
#     print(f"Epoch {epoch}: Avg Total Loss: {total_loss/num_batches:.4f}, "
#           f"Avg WCE Loss: {total_wce_loss/num_batches:.4f}, "
#           f"Avg Lovasz Loss: {total_lovasz_loss/num_batches:.4f}")

#     # Validation
#     torch.cuda.empty_cache()  # Clear GPU before validation
#     with torch.no_grad():
#         model.eval()


#         # Every 5 epochs, calculate and display validation loss
#         if epoch % 5 == 0:
#             # Calculate IoU metrics
#             viou, vmean_iou = evaluate(model, train_loader, device, num_classes=4)
#             print(f"Train IoU per class: {viou}")
#             print(f"Mean train IoU: {vmean_iou:.4f}")
#             total_val_loss = 0.0
#             num_val_batches = 0
#             for pre_img, post_img, mask in val_loader:
#                 pre_img, post_img, mask = pre_img.to(device), post_img.to(device), mask.to(device).long()
#                 outputs = model(pre_img, post_img)
#                 loss, _, _ = criterion(outputs, mask)
#                 total_val_loss += loss.item()
#                 num_val_batches += 1
#             avg_val_loss = total_val_loss / num_val_batches
#             print(f"Epoch {epoch}: Avg Validation Loss: {avg_val_loss:.4f}")
#         # Calculate IoU metrics
#         iou, mean_iou = evaluate(model, val_loader, device, num_classes=4)
#         print(f"Validation IoU per class: {iou}")
#         print(f"Mean IoU: {mean_iou:.4f}")            
#         model.train()

#     # Save best model
#     best_model_path = os.path.join(MODEL_SAVE_DIR, f"best_segformer_model_lov+wce_channel-att_epoch_{epoch}_miou_{mean_iou}.pth")
#     if mean_iou > best_mean_iou:
#         best_mean_iou = mean_iou
#         torch.save(model.state_dict(), best_model_path)
#         print(f"New best model saved at epoch {epoch} with Mean IoU: {best_mean_iou:.4f}")

#     # Periodic checkpoint saving
#     if epoch % 25 == 0:
#         checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"segformer_lov+wce_channel-att_epoch_{epoch}.pth")
#         torch.save(model.state_dict(), checkpoint_path)
#import torch
# import os
# from torch.utils.data import DataLoader, random_split
# from torch.optim import AdamW
# import albumentations as A
# from tqdm import tqdm
# from model.dual_encoder_segformer import DualEncoderSegformer
# from utils.dataset import BRIGHTDataset
# from utils.wce_lov_loss import LovaszWCE
# from evaluate import evaluate
# from torch.optim.lr_scheduler import CosineAnnealingLR

# # Hyperparameters
# BATCH_SIZE = 2
# LR = 1e-4
# EPOCHS = 100  # We'll train from epoch 21 to 100
# RESIZE_SIZE = (1024, 1024)
# PRETRAINED_MODEL = "nvidia/mit-b4"

# # Paths (Update these!)
# PRE_EVENT_DIR = "/home/jothi/jatin/damage_assessment/dfc25_track2_trainval/train/pre-event"
# POST_EVENT_DIR = "/home/jothi/jatin/damage_assessment/dfc25_track2_trainval/train/post-event"
# MASK_DIR = "/home/jothi/jatin/damage_assessment/dfc25_track2_trainval/train/masks"
# MODEL_SAVE_DIR = "/home/jothi/jatin/"  # Set a base directory for saving models

# # Dataset and Transforms
# train_transform = A.Compose([
#     A.Resize(height=RESIZE_SIZE[0], width=RESIZE_SIZE[1]),
#     A.HorizontalFlip(p=0.5),
#     A.VerticalFlip(p=0.5),
#     A.RandomRotate90(p=0.5),
#     A.Normalize(mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225],
#                 max_pixel_value=255.0),
# ], additional_targets={'image1': 'image'})

# # Load the full dataset
# full_dataset = BRIGHTDataset(
#     pre_event_dir=PRE_EVENT_DIR,
#     post_event_dir=POST_EVENT_DIR,
#     mask_dir=MASK_DIR,
#     transform=train_transform,
#     resize_size=RESIZE_SIZE
# )

# # Split dataset: 80% training, 20% validation
# train_size = int(0.8 * len(full_dataset))
# val_size = len(full_dataset) - train_size
# train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# # Class frequencies (percentages to proportions)
# freqs = torch.tensor([0.876, 0.1076, 0.0083, 0.00806], dtype=torch.float32)
# # Compute inverse frequencies and normalize
# class_weights = 1.0 / (freqs + 1e-6)  # Avoid division by zero
# class_weights /= class_weights.max()  # Normalize properly
# print("Class Weights:", class_weights)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = DualEncoderSegformer(pretrained_rgb=PRETRAINED_MODEL, num_labels=4).to(device)
# criterion = LovaszWCE(class_weights=class_weights.to(device))
# optimizer = AdamW(model.parameters(), lr=LR)

# # Mixed Precision Training
# scaler = torch.amp.GradScaler()

# # Load previous checkpoint
# checkpoint_path = "/home/jothi/jatin/best_segformer_model_lov+wce_epoch.pth"
# model.load_state_dict(torch.load(checkpoint_path))
# print("Loaded model from previous checkpoint.")

# best_mean_iou = 0.0

# # Define scheduler
# scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# # Training loop
# for epoch in range(13, EPOCHS + 1):
#     torch.cuda.empty_cache()  # Clear GPU cache before each epoch
#     model.train()

#     total_lovasz_loss, total_wce_loss, total_loss = 0.0, 0.0, 0.0
#     num_batches = len(train_loader)

#     for pre_img, post_img, mask in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
#         pre_img, post_img, mask = pre_img.to(device), post_img.to(device), mask.to(device).long()
#         optimizer.zero_grad()

#         # Mixed Precision Training
#         with torch.amp.autocast(device_type="cuda",dtype=torch.float16):
#             outputs = model(pre_img, post_img)
#             loss, wce_loss, lovasz_loss = criterion(outputs, mask)

#         scaler.scale(loss).backward()
#         scaler.unscale_(optimizer)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         scaler.step(optimizer)
#         scaler.update()

#         # Track losses
#         total_lovasz_loss += lovasz_loss.item()
#         total_wce_loss += wce_loss.item()
#         total_loss += loss.item()

#     # Update scheduler
#     scheduler.step()

#     # Print per-epoch loss statistics
#     print(f"Epoch {epoch}: Avg Total Loss: {total_loss/num_batches:.4f}, "
#           f"Avg WCE Loss: {total_wce_loss/num_batches:.4f}, "
#           f"Avg Lovasz Loss: {total_lovasz_loss/num_batches:.4f}")

#     # Validation
#     torch.cuda.empty_cache()  # Clear GPU before validation
#     with torch.no_grad():
#         model.eval()
#         iou, mean_iou = evaluate(model, val_loader, device, num_classes=4)
#         print(f"Validation IoU per class: {iou}")
#         print(f"Mean IoU: {mean_iou:.4f}")
#         model.train()

#     # Save best model
#     best_model_path = os.path.join(MODEL_SAVE_DIR, f"best_segformer_model_lov+wce_epoch_{epoch}.pth")
#     if mean_iou > best_mean_iou:
#         best_mean_iou = mean_iou
#         torch.save(model.state_dict(), best_model_path)
#         print(f"New best model saved at epoch {epoch} with Mean IoU: {best_mean_iou:.4f}")

#     # Periodic checkpoint saving
#     if epoch % 25 == 0:
#         checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"segformer_lov+wce_epoch_{epoch}.pth")
#         torch.save(model.state_dict(), checkpoint_path)
