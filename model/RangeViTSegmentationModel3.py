import torch
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
        

        # Transformer-based decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.backbone.embed_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.decoder_proj = nn.Linear(self.backbone.embed_dim, n_classes)

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

        # Prepare for transformer decoder: flatten spatial dims to sequence
        B, C, H, W = feats.shape
        src = feats.flatten(2).permute(2, 0, 1)  # [H*W, B, C]

        # Use learned queries (can also use src as tgt for simplicity)
        tgt = torch.zeros_like(src)
        decoded = self.transformer_decoder(tgt, src)  # [H*W, B, C]

        # Project to class logits
        logits = self.decoder_proj(decoded)  # [H*W, B, n_classes]
        logits = logits.permute(1, 2, 0).reshape(B, self.decoder_proj.out_features, H, W)
        logits = F.interpolate(logits, size=self.original_size, mode='bilinear', align_corners=False)
        return logits