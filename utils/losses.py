#import torch
#import torch.nn as nn
#import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, class_weights, alpha=50.0, gamma=2.0, eps=1e-6):
        """
        Combined Loss = alpha * Focal Loss + Dice Loss.
        This implementation is vectorized for efficiency.
        
        Args:
            class_weights (Tensor): A tensor of shape (num_classes,) containing class weights.
            alpha (float): Weight for the focal loss component.
            gamma (float): Focusing parameter for focal loss.
            eps (float): A small constant for numerical stability.
        """
        super().__init__()
        self.class_weights = class_weights.to(torch.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.num_classes = class_weights.shape[0]

    def forward(self, logits, targets):
        # Normalize logits for stability and clamp extreme values.
        logits = logits - logits.max(dim=1, keepdim=True)[0]
        logits = torch.clamp(logits, min=-10, max=10)

        # Compute probabilities and one-hot encode targets.
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # ----------------------------
        # Vectorized Focal Loss
        # ----------------------------
        ce_loss = -torch.log(torch.clamp(probs, self.eps, 1.0 - self.eps))
        focal_mod = (1 - probs) ** self.gamma
        focal_loss = targets_one_hot * focal_mod * ce_loss  # (B, C, H, W)
        # Average over batch and spatial dims (per class)
        focal_loss = focal_loss.mean(dim=(0, 2, 3))  # shape: (C,)
        focal_loss_total = (self.class_weights * focal_loss).sum()
        
        # ----------------------------
        # Vectorized Dice Loss
        # ----------------------------
        intersection = (probs * targets_one_hot).sum(dim=(0, 2, 3))  # shape: (C,)
        p_sum = probs.sum(dim=(0, 2, 3))
        t_sum = targets_one_hot.sum(dim=(0, 2, 3))
        dice_score = (2.0 * intersection + self.eps) / (p_sum + t_sum + self.eps)
        dice_loss = 1.0 - dice_score  # shape: (C,)
        dice_loss_total = (self.class_weights * dice_loss).sum()

        # Total combined loss
        total_loss = self.alpha * focal_loss_total + dice_loss_total

        return total_loss, focal_loss_total, dice_loss_total

class CombinedLoss3(nn.Module):
    def __init__(self, class_weights, alpha=0.5, gamma=2.0, eps=1e-6):
        """
        Args:
            class_weights (Tensor): A tensor of shape [num_classes] containing the weight for each class.
            alpha (float): Weighting factor for the Dice loss component.
            gamma (float): Focusing parameter for Focal Loss.
            eps (float): A small constant to avoid division by zero.
        """
        super().__init__()
        self.class_weights = class_weights  # should be a tensor of shape [num_classes]
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.num_classes = class_weights.shape[0]

    def forward(self, logits, targets):
        """
        Args:
            logits: Tensor of shape (N, C, H, W), raw predictions.
            targets: Tensor of shape (N, H, W) with class labels (0 to num_classes-1).
        Returns:
            loss: Scalar loss value.
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)  # shape: (N, C, H, W)
        # One-hot encode targets: shape becomes (N, H, W, C), then permute to (N, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        focal_loss_total = 0.0
        dice_loss_total = 0.0

        # Loop over each class to compute per-class losses
        for c in range(self.num_classes):
            # Get probability map and target mask for class c
            p_c = probs[:, c, :, :]  # shape (N, H, W)
            t_c = targets_one_hot[:, c, :, :]  # shape (N, H, W)

            # Focal Loss for class c:
            # Standard CE: -log(p) for pixels where t=1; we mask out others.
            ce_loss = -torch.log(p_c + self.eps)
            # Focal modulating factor: (1 - p)^gamma
            focal_mod = (1 - p_c) ** self.gamma
            focal_loss_c = (t_c * focal_mod * ce_loss).mean()

            # Dice Loss for class c:
            # Intersection and sums over pixels (batch + spatial dims)
            intersection = (p_c * t_c).sum()
            p_sum = p_c.sum()
            t_sum = t_c.sum()
            dice_score = (2.0 * intersection + self.eps) / (p_sum + t_sum + self.eps)
            dice_loss_c = 1.0 - dice_score

            # Weight the per-class losses with w_i
            weight = self.class_weights[c]
            focal_loss_total += weight * focal_loss_c
            dice_loss_total += weight * dice_loss_c

        # Combine losses with alpha scaling Dice Loss
        loss = focal_loss_total + self.alpha * dice_loss_total
        return loss





import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss2(nn.Module):
    def __init__(self, class_weights, alpha=100, gamma=2.0, eps=1e-6):
        super().__init__()
        self.class_weights = class_weights.to(torch.float32)  
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.num_classes = class_weights.shape[0]

    def forward(self, logits, targets):
        # Normalize logits for stability
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("NaN or Inf detected in logits!")

        logits = logits - logits.max(dim=1, keepdim=True)[0]  # Normalize logits before softmax
        logits = torch.clamp(logits, min=-10, max=10)  # Prevent extreme values

        probs = F.softmax(logits, dim=1)  

        # Debugging: Check for NaN
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            print("NaN detected in softmax probabilities!")

        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Debugging: Check for NaN in targets
        if torch.isnan(targets_one_hot).any():
            print("NaN detected in one-hot encoded targets!")

        focal_loss_total = 0.0
        dice_loss_total = 0.0

        for c in range(self.num_classes):
            p_c = torch.clamp(probs[:, c, :, :], min=self.eps, max=1.0 - self.eps)  
            t_c = targets_one_hot[:, c, :, :]

            # Focal Loss (Stable)
            ce_loss = -torch.log(torch.clamp(p_c, min=self.eps, max=1.0 - self.eps))  
            focal_mod = (1 - p_c).pow(self.gamma)  
            focal_loss_c = (t_c * focal_mod * ce_loss).mean()

            # Dice Loss (Stable)
            intersection = (p_c * t_c).sum()
            p_sum = p_c.sum()
            t_sum = t_c.sum()
            if (p_sum + t_sum).item() == 0:
                dice_score = 1.0  
            else:
                dice_score = (2.0 * intersection + self.eps) / (p_sum + t_sum + self.eps)

            dice_loss_c = 1.0 - dice_score

            weight = self.class_weights[c]
            focal_loss_total += weight * focal_loss_c
            dice_loss_total += weight * dice_loss_c

        total_loss = self.alpha * focal_loss_total + dice_loss_total

        # Debugging: Check for NaN
        if torch.isnan(total_loss):
            print(f"NaN detected! Focal: {focal_loss_total}, Dice: {dice_loss_total}")

        return total_loss, focal_loss_total, dice_loss_total


# import torch.nn as nn
# import torch.nn.functional as F

# class CombinedLoss(nn.Module):
#     def __init__(self, class_weights=None, lambda_dice=0.7):
#         """
#         Args:
#             class_weights (Tensor): Precomputed static weights for all classes.
#             lambda_dice (float): Weighting factor for the Dice loss component.
#         """
#         super().__init__()
#         self.lambda_dice = lambda_dice
#         # Initialize CrossEntropyLoss with the static class weights.
#         self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1, reduction='mean')

#     def dice_loss(self, preds, targets):
#         smooth = 1.0
#         preds = F.softmax(preds, dim=1)  # Convert logits to probabilities
#         targets_one_hot = F.one_hot(targets, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float()
        
#         # Calculate intersection and union for Dice coefficient
#         intersection = (preds * targets_one_hot).sum(dim=(2, 3))
#         union = preds.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
#         dice = (2.0 * intersection + smooth) / (union + smooth)
#         return 1.0 - dice.mean()

#     def forward(self, preds, targets):
#         ce = self.ce_loss(preds, targets)
#         dice = self.dice_loss(preds, targets)
#         return ce + self.lambda_dice * dice


