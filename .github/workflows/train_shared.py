import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import albumentations as A
from tqdm import tqdm
import matplotlib.pyplot as plt
from model.shared_encodder_segformer import SharedEncoderSegformer
from utils.dataset import BRIGHTDataset
from utils.wce_lov_loss import LovaszWCE
from evaluate import evaluate

# Hyperparameters
BATCH_SIZE = 2
LR = 1e-4
EPOCHS = 30  # Total epochs to train
RESIZE_SIZE = (1024, 1024)
PRETRAINED_MODEL = "nvidia/mit-b4"
TRAIN_SPLIT = 0.8  # 80% for training, 20% for validation

# Paths
PRE_EVENT_DIR = "/home/jothi/jatin/damage_assessment/dfc25_track2_trainval/train/pre-event"
POST_EVENT_DIR = "/home/jothi/jatin/damage_assessment/dfc25_track2_trainval/train/post-event"
MASK_DIR = "/home/jothi/jatin/damage_assessment/dfc25_track2_trainval/train/masks"
BASE_LOSS_PLOT_PATH = "/home/jothi/jatin/sharedsegformer/loss_plot.png"  # Loss plot path

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

# Split dataset into training and validation sets
dataset_length = len(full_dataset)
train_length = int(dataset_length * TRAIN_SPLIT)
val_length = dataset_length - train_length
train_subset, val_subset = random_split(full_dataset, [train_length, val_length])

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model, loss, optimizer, and scheduler
model = SharedEncoderSegformer(pretrained_rgb=PRETRAINED_MODEL, num_labels=4).to(device)
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
    
    # Initialize accumulators for individual loss components (training)
    total_loss_epoch = 0.0
    wce_loss_epoch = 0.0
    lovasz_loss_epoch = 0.0
    num_batches = 0

    for pre_img, post_img, mask in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        pre_img, post_img, mask = pre_img.to(device), post_img.to(device), mask.to(device).long()
        optimizer.zero_grad()
        
        outputs = model(pre_img, post_img)
        # Expecting criterion to return a tuple: (total_loss, wce_loss, lovasz_loss)
        total_loss_batch, wce_loss, lovasz_loss = criterion(outputs, mask)
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss_epoch += total_loss_batch.item()
        wce_loss_epoch += wce_loss.item()
        lovasz_loss_epoch += lovasz_loss.item()
        num_batches += 1

    scheduler.step()
    
    # Compute average losses for the epoch
    avg_total_loss = total_loss_epoch / num_batches
    avg_wce_loss = wce_loss_epoch / num_batches
    avg_lovasz_loss = lovasz_loss_epoch / num_batches
    train_losses.append(avg_total_loss)
    print(f"Epoch {epoch}: Train [Total: {avg_total_loss:.6f}, WCE: {avg_wce_loss:.6f}, Lovasz: {avg_lovasz_loss:.6f}]")
    # Validation phase with individual loss tracking
    model.eval()
    val_total_loss_epoch = 0.0
    val_wce_loss_epoch = 0.0
    val_lovasz_loss_epoch = 0.0
    val_batches = 0
    with torch.no_grad():
        for pre_img, post_img, mask in val_loader:
            pre_img, post_img, mask = pre_img.to(device), post_img.to(device), mask.to(device).long()
            outputs = model(pre_img, post_img)
            total_loss_batch, wce_loss, lovasz_loss = criterion(outputs, mask)
            val_total_loss_epoch += total_loss_batch.item()
            val_wce_loss_epoch += wce_loss.item()
            val_lovasz_loss_epoch += lovasz_loss.item()
            val_batches += 1

    avg_val_total_loss = val_total_loss_epoch / val_batches
    avg_val_wce_loss = val_wce_loss_epoch / val_batches
    avg_val_lovasz_loss = val_lovasz_loss_epoch / val_batches
    val_losses.append(avg_val_total_loss)
    print(f"Val [Total: {avg_val_total_loss:.6f}, WCE: {avg_val_wce_loss:.6f}, Lovasz: {avg_val_lovasz_loss:.6f}]")

    # Evaluate IoU on validation set
    iou, mean_iou = evaluate(model, val_loader, device, num_classes=4)
    
    # Print all losses in a single line
    print(f"Mean IoU: {mean_iou:.4f}")

    # Save best model checkpoint if current mean IoU is the best
    if mean_iou > best_mean_iou:
        best_mean_iou = mean_iou
        best_model_path = f"/home/jothi/jatin/sharedsegformer/best_sharedsegformer_epoch_{epoch}_iou_{best_mean_iou:.6f}.pth"
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_mean_iou': best_mean_iou,
        }
        torch.save(checkpoint, best_model_path)
        print(f"New best model saved at epoch {epoch} with Mean IoU: {best_mean_iou:.6f}")

    # Save loss plot every 5 epochs
    if epoch % 5 == 0:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epoch+1), train_losses, label='Train Total Loss', marker='o')
        plt.plot(range(1, epoch+1), val_losses, label='Validation Total Loss', marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Total Loss")
        plt.grid(True)
        plt.savefig(BASE_LOSS_PLOT_PATH)
        print(f"Loss plot saved at {BASE_LOSS_PLOT_PATH}")
        plt.close()

    # Save a checkpoint every 25 epochs or at the final epoch
    if epoch % 25 == 0 or epoch == EPOCHS:
        checkpoint_path = f"/home/jothi/jatin/sharedsegformer/sharedsegformer_epoch_{epoch}.pth"
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_mean_iou': best_mean_iou,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch}: {checkpoint_path}")
