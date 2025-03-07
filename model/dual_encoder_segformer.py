import torch
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from torch import nn
from .fusion import ChannelAttentionFusion,DifferenceFusion  # Assuming this is implemented elsewhere
import torch.nn.functional as F

class DualEncoderSegformer(nn.Module):
    def __init__(self, pretrained_rgb="nvidia/mit-b4", num_labels=4):
        super().__init__()
        
        # Load the configuration for the pretrained model
        config = SegformerConfig.from_pretrained(pretrained_rgb)
        
        # Ensure the number of classes is set correctly
        config.num_labels = num_labels  # Set the number of output classes
        
        # Pre-disaster encoder (RGB, 3 channels)
        self.pre_encoder = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_rgb,
            config=config
        )
        
        # Post-disaster encoder (SAR, 1 channel)
        # Modify the config to handle 1 input channel
        sar_config = SegformerConfig.from_pretrained(pretrained_rgb)
        sar_config.num_channels = 3  # Set input channels to 1 for SAR
        sar_config.num_labels = num_labels  # Ensure the same number of output classes
        self.post_encoder = SegformerForSemanticSegmentation(sar_config)
        
        # Initialize the first layer weights for SAR encoder by copying and averaging the RGB weights
        #with torch.no_grad():
            # Access the first patch embedding conv layer in the RGB encoder
            #rgb_first_layer = self.pre_encoder.segformer.encoder.patch_embeddings[0].proj
            # Access the corresponding layer in the SAR encoder
            #sar_first_layer = self.post_encoder.segformer.encoder.patch_embeddings[0].proj
            
            # Average the RGB weights over the channel dimension (from 3 to 1)
            #sar_first_layer.weight.data = rgb_first_layer.weight.data.mean(dim=1, keepdim=True)
            # Copy the bias if present
            #if rgb_first_layer.bias is not None:
                #sar_first_layer.bias.data = rgb_first_layer.bias.data
        
        # Define the embedding dimensions manually
        embed_dims = [64, 128, 320, 512]
        
        # Channel attention fusion modules for each scale
        self.fusions = nn.ModuleList([
            ChannelAttentionFusion(channels=dim) 
            #DifferenceFusion()
            for dim in embed_dims
        ])
        
        # Use the decoder head from the pretrained RGB model
        self.decode_head = self.pre_encoder.decode_head

    def forward(self, pre_img, post_img):
        # Extract multi-scale features from both encoders with hidden_states output
        pre_features = self.pre_encoder(pre_img, output_hidden_states=True).hidden_states
        post_features = self.post_encoder(post_img, output_hidden_states=True).hidden_states
        
        # Fuse features at each corresponding stage
        fused_features = []
        for pre_feat, post_feat, fusion in zip(pre_features, post_features, self.fusions):
            fused = fusion(pre_feat, post_feat)
            fused_features.append(fused)
        
        # Decode the fused features using the decoder head
        logits = self.decode_head(fused_features)
        
        # Ensure the output size is correctly upsampled to match the input image size
        logits = F.interpolate(logits, size=pre_img.shape[2:], mode='bilinear', align_corners=False)
        
        # Print the shape of logits to verify correct dimensions
        print("Logits shape:", logits.shape)
        
        return logits
