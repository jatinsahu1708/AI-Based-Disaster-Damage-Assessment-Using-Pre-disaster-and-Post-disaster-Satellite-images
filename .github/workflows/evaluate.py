import torch
import numpy as np

def evaluate(model, val_loader, device, num_classes=4):
    """
    Evaluate the model using IoU (Intersection over Union) for each class.

    Args:
        model: The trained model.
        val_loader: DataLoader for the validation dataset.
        device: The device (cuda or cpu).
        num_classes: Number of classes for the segmentation task.

    Returns:
        iou: A dictionary containing IoU values for each class.
        mean_iou: The mean IoU across all classes.
    """
    model.eval()  # Set model to evaluation mode
    iou = {i: 0 for i in range(num_classes)}
    total_pixels = {i: 0 for i in range(num_classes)}  # Count of pixels per class
    intersection = {i: 0 for i in range(num_classes)}  # Intersection of predictions and ground truth
    union = {i: 0 for i in range(num_classes)}  # Union of predictions and ground truth

    with torch.no_grad():
        for pre_img, post_img, mask in val_loader:
            pre_img = pre_img.to(device)
            post_img = post_img.to(device)
            mask = mask.to(device)
            
            # Get predictions from the model
            outputs = model(pre_img, post_img)
            preds = torch.argmax(outputs, dim=1)  # Get predicted class labels
            
            # Calculate intersection and union for each class
            for i in range(num_classes):
                # True Positives (TP): pixels where both predicted and ground truth are class i
                tp = torch.sum((preds == i) & (mask == i)).item()
                # False Positives (FP): pixels where predicted is class i but ground truth is not
                fp = torch.sum((preds == i) & (mask != i)).item()
                # False Negatives (FN): pixels where ground truth is class i but predicted is not
                fn = torch.sum((preds != i) & (mask == i)).item()

                intersection[i] += tp
                union[i] += tp + fp + fn
                total_pixels[i] += torch.sum(mask == i).item()

        # Compute IoU for each class
        for i in range(num_classes):
            iou[i] = intersection[i] / (union[i] + 1e-6)  # Adding a small value to avoid division by zero

        mean_iou = np.mean(list(iou.values()))

    return iou, mean_iou
