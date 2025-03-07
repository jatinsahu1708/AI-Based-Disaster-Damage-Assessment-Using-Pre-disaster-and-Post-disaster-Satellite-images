import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import albumentations as A

class BRIGHTDataset(Dataset):
    def __init__(self, pre_event_dir, post_event_dir, mask_dir, transform=None, resize_size=(1024, 1024)):
        # Save directories and transform parameters
        self.pre_event_dir = pre_event_dir
        self.post_event_dir = post_event_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.resize_size = resize_size

        # Load sorted filenames to ensure correspondence
        self.pre_images = sorted(os.listdir(pre_event_dir))
        self.post_images = sorted(os.listdir(post_event_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.pre_images)

    def __getitem__(self, idx):
        # Load pre-disaster RGB image
        pre_img = np.array(Image.open(os.path.join(self.pre_event_dir, self.pre_images[idx])))
        
        # Load post-disaster SAR image (grayscale), then add a channel dimension
        post_img = np.array(Image.open(os.path.join(self.post_event_dir, self.post_images[idx])))
        #post_img = np.expand_dims(post_img, axis=-1)
        if post_img.ndim == 2:
            # If the image is grayscale (H, W), stack it to get (H, W, 3)
            post_img = np.stack([post_img] * 3, axis=-1)
        elif post_img.shape[-1] == 1:
            # If already (H, W, 1), concatenate to form 3 channels
            post_img = np.concatenate([post_img] * 3, axis=-1)
        
        # Load segmentation mask
        mask = np.array(Image.open(os.path.join(self.mask_dir, self.masks[idx])))

        # Resize images and mask using albumentations
        resize_transform = A.Compose(
                  [A.Resize(height=self.resize_size[0], width=self.resize_size[1])],
                  additional_targets={'image1': 'image'}  # This tells Albumentations to treat the key "image1" as an image
  )
        resized = resize_transform(image=pre_img, image1=post_img, mask=mask)

        pre_img = resized['image']
        post_img = resized['image1']
        mask = resized['mask']

        if self.transform:
            # Apply additional augmentation transforms
            transformed = self.transform(image=pre_img, image1=post_img, mask=mask)
            pre_img = transformed['image']
            post_img = transformed['image1']
            mask = transformed['mask']

        # Convert images to PyTorch tensors; rearrange dimensions to [C, H, W]
        pre_img = torch.from_numpy(pre_img).permute(2, 0, 1).float()
        post_img = torch.from_numpy(post_img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()

        return pre_img, post_img, mask
