import torch.nn as nn
import torch.nn.functional as F
import timm

class RangeViTSegmentationModel(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()

        self.input_height = 48
        self.input_width = 480

        # Create ViT model without features_only to see what we actually get
        self.backbone = timm.create_model(
            'vit_small_patch16_384',       
            pretrained=True,
            in_chans=in_channels,
            num_classes=0,  # Set num_classes to 0 to avoid classification head; this has no effect on number of classes in seg_head (still 20)
            global_pool='', # disables CLS token pooling
            features_only=False  # Don't use features_only
        )
        
        # Get the actual feature dimension from the model
        feat_dim = self.backbone.embed_dim #384  # This should be 384 for vit_small
        hidden_dim = 256
        # Print for debugging
        print(f"ViT feature dimension: {feat_dim}")
        print(f"number of classes: {n_classes}")
        # Create segmentation head with the correct input dimension
        self.seg_head = nn.Sequential(
            nn.Conv2d(feat_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, n_classes, kernel_size=1)
        )
        # Decoder: two upsampling blocks for refinement
        self.decoder = nn.Sequential(
            nn.Conv2d(self.backbone.embed_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, n_classes * 4, kernel_size=3, padding=1),
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