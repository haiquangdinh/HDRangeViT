import torch.nn as nn
import torch.nn.functional as F
import timm

class RangeViTSegmentationModel(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()

        self.input_height = 48
        self.input_width = 480

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
        
        # Decoder: two upsampling blocks for refinement
        conv11dim = 256 # out-of-memory: self.backbone.embed_dim*self.input_height*self.input_width
        self.decoder = nn.Sequential(
            nn.Conv2d(self.backbone.embed_dim, conv11dim, kernel_size=1),
            nn.BatchNorm2d(conv11dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.ConvTranspose2d(conv11dim, conv11dim, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(conv11dim, n_classes * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2)
        )

        self.original_size = None  # Store original size for resizing back

    def forward(self, x):
        # Store original size for later upsampling
        self.original_size = x.shape[2:]
        # Resize input to 384x384 (what ViT expects)
        x_resized = F.interpolate(x, size=(384, 384), mode='bilinear', align_corners=False)
        # Extract features from backbone
        feats = self.backbone(x_resized)
        # Reshape features for segmentation head 
        # ViT returns tokens, we need to reshape to 2D feature map
        B = x_resized.shape[0]
        h = w = int(384 / 16)  # 16 is patch size of vit_small_patch16_384
        C = feats.shape[-1]
        # Remove CLS token and reshape to [B, C, h, w]
        feats = feats[:, 1:, :].reshape(B, h, w, C).permute(0, 3, 1, 2)
        
        # Apply segmentation head
        logits = self.decoder(feats)
        # Resize back to original dimensions
        logits = F.interpolate(logits, size=self.original_size, mode='bilinear', align_corners=False)
       
        return logits