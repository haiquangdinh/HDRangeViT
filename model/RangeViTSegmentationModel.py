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
        embed_dim = 384  # ViT embed dimension
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

        # Replace built-in patch embedding with identity
        self.backbone.patch_embed = nn.Identity()
        
        # Decoder: two upsampling blocks for refinement
        self.decoder = nn.Sequential(
            nn.Conv2d(self.backbone.embed_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(hidden_dim, n_classes * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2)
        )

        self.original_size = None  # Store original size for resizing back

    def forward(self, x):
        self.original_size = x.shape[2:]
        # Resize input to what ViT expects
        x, skip = self.patch_embed(x)
        feats = self.backbone(x)
        # ViT returns tokens, we need to reshape to 2D feature map
        B = x.shape[0]
        h = w = int(384 / 16)  # 16 is patch size of vit_small_patch16_384
        C = feats.shape[-1]
        # Remove CLS token and reshape to [B, C, h, w]
        feats = feats[:, 1:, :].reshape(B, h, w, C).permute(0, 3, 1, 2)
        
        # Apply segmentation head
        logits = self.decoder(feats)
        # Resize back to original dimensions
        logits = F.interpolate(logits, size=self.original_size, mode='bilinear', align_corners=False)
       
        return logits