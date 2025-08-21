import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .stems import ConvStem
class RangeViTSegmentationModel(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()

        input_height = 48
        input_width = 480
        hidden_dim = 256
        embed_dim = 384  # ViT embed dimension, can get it by self.backbone.embed_dim
        base_channels = 32

        # Stem block: preprocess input before ViT encoder
        self.patch_embed = ConvStem(
            in_channels=in_channels,
            base_channels=base_channels,
            embed_dim=embed_dim,
            flatten=True,
            hidden_dim=hidden_dim)
        
        # Create ViT model with weights trained on ImageNet21k
        self.backbone = timm.create_model(
            'vit_small_patch16_384',
            pretrained=True,
            in_chans=in_channels,
            num_classes=0,
            global_pool='',
            features_only=False,
            drop_path_rate=0.1,  # Add DropPath regularization (0.1 is typical)
            drop_rate=0.1,
            attn_drop_rate=0.1
        )

        # Replace built-in patch embedding with identity since we already have a custom one
        self.backbone.patch_embed = nn.Identity()
        
        # Decoder: 
        # Use two Conv2d blocks
        # Use PixelShuffle as upsampling layer:
        #       Input Shape: (Batch, Channels * 4, Height, Width)
        #       Output Shape: (Batch, Channels, Height * 2, Width * 2)
        n_foo = hidden_dim * 2 * 8 # 4096
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, n_foo, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(n_foo),
            nn.ConvTranspose2d(n_foo, embed_dim, kernel_size=(2, 20), stride=(2, 20)),  
            nn.GELU(),
            nn.BatchNorm2d(embed_dim),
        )

        # Classification head: turn [32, 640, 48, 480] to [32, 20, 48, 480]
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(embed_dim + hidden_dim, n_classes, kernel_size=1),
            nn.Upsample(size=(input_height, input_width), mode='bilinear', align_corners=False)
        )

        self.original_size = None  # Store original size for resizing back

    def forward(self, x): 
        """
        Forward pass for the model.
        input x: Tensor of shape [Batch, Channels, Height, Width]: e.g [32, 9, 48, 480]
        """
        self.original_size = x.shape[2:] # original Height, Width: e.g [48, 480]
        # Resize input to what ViT expects
        # Note that skip has shape [Batch, hidden_dim, original Height, original Width]: e.g [32, 256, 48, 480]
        x, skip = self.patch_embed(x)
        feats = self.backbone(x)   # Feats shape: [Batch, Tokens + CLS, EmbedDim]: e.g[32, 577, 384]
        # ViT returns tokens, we need to reshape to 2D feature map
        B = x.shape[0]
        h = w = int(384 / 16)  # 16 is patch size of vit_small_patch16_384
        C = feats.shape[-1]
        # Remove CLS token and reshape to [B, C, h, w]: e.g [32, 384, 24, 24]
        feats = feats[:, 1:, :].reshape(B, h, w, C).permute(0, 3, 1, 2)
        # decoder, the new decoder will have the size of [Batch, embed_dim, original Height, original Width]
        feats = self.decoder(feats)
        # Concatenate skip connectionto decoder
        feats = torch.cat([feats, skip], dim=1)
        # Apply segmentation head, it design to output [Batch, num_classes, original Height, original Width]
        # result no need to interpolate: e.g [32, 20, 48, 480] 
        logits = self.segmentation_head(feats)
        return logits