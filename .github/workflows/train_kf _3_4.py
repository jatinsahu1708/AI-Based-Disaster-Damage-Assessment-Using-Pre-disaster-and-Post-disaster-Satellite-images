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

# Set device to GPU 1 (or CPU if not available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Running folds 3-4 on {device}")

# K-Fold Cross Validation with fixed random state
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
    # Only process folds 3 and 4
    if fold <=3:
        continue

    print(f"Running Fold {fold+1}/{K_FOLDS}")
    
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
            # Save full checkpoint dictionary with extra keys
            torch.save(checkpoint, best_model_path)
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
        if epoch % 25 == 0 or epoch == 30:
            checkpoint_path = f"/home/jothi/jatin/kfold/models/segformer_fold_{fold+1}_epoch_{epoch}.pth"
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mean_iou': best_mean_iou,
            }
            # Save full checkpoint dictionary
            torch.save(checkpoint, checkpoint_path)
            print(f"Fold {fold+1}: Checkpoint saved at epoch {epoch}: {checkpoint_path}")
