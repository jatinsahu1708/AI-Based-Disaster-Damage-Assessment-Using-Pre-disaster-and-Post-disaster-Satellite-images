import torch
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from torch import nn
from .fusion import ChannelAttentionFusion  # Assuming this is implemented elsewhere
import torch.nn.functional as F

class SharedEncoderSegformer(nn.Module):
    def __init__(self, pretrained_rgb="nvidia/mit-b4", num_labels=4):
        super().__init__()
        
        # Load the configuration for the pretrained model
        config = SegformerConfig.from_pretrained(pretrained_rgb)
        config.num_labels = num_labels  # Set the number of output classes
        
        # Shared encoder for both pre- and post-disaster images
        self.encoder = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_rgb,
            config=config
        )
        
        # Define the embedding dimensions manually
        embed_dims = [64, 128, 320, 512]
        
        # Channel attention fusion modules for each scale
        self.fusions = nn.ModuleList([
            ChannelAttentionFusion(channels=dim) 
            for dim in embed_dims
        ])
        
        # Use the decoder head from the pretrained model
        self.decode_head = self.encoder.decode_head

    def forward(self, pre_img, post_img):
        # Extract multi-scale features from the shared encoder for both inputs
        pre_features = self.encoder(pre_img, output_hidden_states=True).hidden_states
        post_features = self.encoder(post_img, output_hidden_states=True).hidden_states
        
        # Fuse features at each corresponding stage
        fused_features = []
        for pre_feat, post_feat, fusion in zip(pre_features, post_features, self.fusions):
            fused = fusion(pre_feat, post_feat)
            fused_features.append(fused)
        
        # Decode the fused features using the decoder head
        logits = self.decode_head(fused_features)
        
        # Ensure the output size is correctly upsampled to match the input image size
        logits = F.interpolate(logits, size=pre_img.shape[2:], mode='bilinear', align_corners=False)
        
        return logits
