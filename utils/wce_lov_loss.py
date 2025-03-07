import torch
import torch.nn as nn
import torch.nn.functional as F
def lovasz_softmax1(probas, labels):
    """
    Vectorized Lovasz-Softmax loss.
    Args:
        probas: [B, C, H, W] logits (after softmax)
        labels: [B, H, W] ground truth labels
    Returns:
        Lovasz-Softmax loss.
    """
    B, C, H, W = probas.shape
    probas = probas.permute(0, 2, 3, 1).reshape(-1, C)  # Flatten to [N, C]
    labels = labels.view(-1)  # Flatten to [N]
    
    # Compute per-class foreground masks
    fg = torch.zeros_like(probas)  # [N, C]
    fg.scatter_(1, labels.unsqueeze(1), 1)  # One-hot encoding of labels

    # Compute absolute errors for all classes
    errors = (fg - probas).abs()

    # Sort errors in descending order per class
    sorted_errors, indices = torch.sort(errors, dim=0, descending=True)
    sorted_fg = torch.gather(fg, 0, indices)  # Sort foregrounds accordingly

    # Compute intersection and union
    intersection = sorted_fg.cumsum(0)
    union = intersection + (1 - sorted_fg).cumsum(0)
    
    # Compute Lovasz gradient
    jaccard = 1.0 - intersection / (union + 1e-6)
    jaccard[1:] = jaccard[1:].clone() - jaccard[:-1]
  # Difference gives gradient
    
    # Compute Lovasz loss
    loss = (sorted_errors * jaccard).sum(0).mean()  # Average over all classes
    return loss

# Weighted Cross-Entropy Loss
class WeightedCrossEntropyLoss1(nn.Module):
    def __init__(self, class_weights):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def forward(self, logits, labels):
        return F.cross_entropy(logits, labels, weight=self.class_weights.to(logits.device))

# Combined Lovasz-WCE Loss
class LovaszWCE_new(nn.Module):
    def __init__(self, class_weights, alpha=0.75):
        super(LovaszWCE_new, self).__init__()
        self.wce = WeightedCrossEntropyLoss(class_weights)
        self.alpha = alpha

    def forward(self, logits, labels):
        probas = F.softmax(logits, dim=1)
        lovasz_loss = lovasz_softmax(probas, labels)
        wce_loss = self.wce(logits, labels)
        combined_loss = self.alpha * lovasz_loss + wce_loss
        return combined_loss, wce_loss, lovasz_loss

#####################
# Lovasz Softmax Loss
def lovasz_softmax(probas, labels, classes="present"):
    """
    Lovasz-Softmax loss.
    Args:
        probas: [B, C, H, W] logits (after softmax)
        labels: [B, H, W] ground truth labels
        classes: 'present' to include only classes present in the labels.
    Returns:
        Lovasz-Softmax loss.
    """
    def lovasz_grad(gt_sorted):
        """Calculate the gradient of the Lovasz extension."""
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / (union+1e-6)
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard

    B, C, H, W = probas.shape
    probas = probas.permute(0, 2, 3, 1).reshape(-1, C)  # Flatten
    labels = labels.view(-1)

    loss = 0.0
    classes_to_sum = list(range(C)) if classes == "all" else torch.unique(labels)
    for c in classes_to_sum:
        fg = (labels == c).float()  # Binary ground truth for class c
        if fg.sum() == 0:
            continue
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        loss += torch.dot(errors_sorted, lovasz_grad(fg_sorted))
    return loss / len(classes_to_sum)

# Weighted Cross-Entropy Loss
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights):
        """
        Weighted Cross-Entropy Loss.
        Args:
            class_weights: List of class weights for WCE.
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def forward(self, logits, labels):
        """
        Args:
            logits: [B, C, H, W] output logits (not probabilities)
            labels: [B, H, W] ground truth labels
        """
        loss = F.cross_entropy(logits, labels, weight=self.class_weights.to(logits.device))
        return loss

# Combined Lovasz-WCE Loss
class LovaszWCE(nn.Module):
    def __init__(self, class_weights, alpha=0.75):
        """
        Combined Lovasz-Weighted Cross-Entropy Loss.
        Args:
            class_weights: List of class weights for WCE.
            alpha: Weight for the Lovasz loss (0 <= alpha <= 1).
        """
        super(LovaszWCE, self).__init__()
        self.wce = WeightedCrossEntropyLoss(class_weights)
        self.alpha = alpha

    def forward(self, logits, labels):
        """
        Args:
            logits: [B, C, H, W] output logits (not probabilities)
            labels: [B, H, W] ground truth labels
        """
        probas = F.softmax(logits, dim=1)  # Convert logits to probabilities
        lovasz_loss = lovasz_softmax(probas, labels)
        wce_loss = self.wce(logits, labels)
        comined_oss= self.alpha * lovasz_loss +  wce_loss
        return comined_oss,wce_loss,lovasz_loss














































def lovasz_softmax_flat(probas, labels, classes='present'):
    
    if probas.numel() == 0:
        return probas * 0.0
    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float()  # foreground mask for class c
        if classes == 'present' and fg.sum() == 0:
            continue
        # Errors: 1 - probability if foreground, probability if background
        errors = (1.0 - probas[:, c]) * fg + probas[:, c] * (1 - fg)
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        grad = torch.cumsum(torch.ones_like(fg_sorted) / fg_sorted.numel(), dim=0)
        losses.append(torch.dot(errors_sorted, grad))
    if losses:
        return torch.mean(torch.stack(losses))
    else:
        return torch.tensor(0.0, device=probas.device)

class CombinedLoss1(nn.Module):
    def __init__(self, class_weights, lambda_val=0.5, eps=1e-6):

        super().__init__()
        self.class_weights = class_weights.to(torch.float32)
        self.lambda_val = lambda_val
        self.eps = eps

    def forward(self, logits, targets):
        # Compute Weighted Cross-Entropy Loss (WCE)
        wce_loss = F.cross_entropy(logits, targets, weight=self.class_weights, reduction='mean')
        
        # Compute Lovász-Softmax Loss
        probas = F.softmax(logits, dim=1)  # (N, C, H, W)
        B, C, H, W = probas.shape
        probas_flat = probas.view(B, C, -1).permute(0, 2, 1).contiguous().view(-1, C)
        targets_flat = targets.view(-1)  # Corrected reshaping

        lovasz_loss = lovasz_softmax_flat(probas_flat, targets_flat, classes='present')

        
        total_loss = self.lambda_val * wce_loss + (1 - self.lambda_val) * lovasz_loss
        return total_loss, wce_loss, lovasz_loss