### Model
import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import PatchEmbed
import torch.nn.functional as F

class RangeViTSegmentationModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.input_height = 48
        self.input_width = 460
        self.patch_height = 2
        self.patch_width = 8

        self.backbone = timm.create_model(
            'vit_small_patch16_384',
            pretrained=True,
            in_chans=in_channels,
            num_classes=0,
            global_pool='',
            features_only=False
        )

        # Override patch embedding
        self.backbone.patch_embed = PatchEmbed(
            img_size=(self.input_height, self.input_width),
            patch_size=(self.patch_height, self.patch_width),
            in_chans=in_channels,
            embed_dim=self.backbone.embed_dim
        )

        self.grid_h, self.grid_w = self.backbone.patch_embed.grid_size  # (32, 256)
        self.num_patches = self.grid_h * self.grid_w
        print(f"Grid size: {self.grid_h} x {self.grid_w}, Patches: {self.num_patches}")

        expected_tokens = 1 + self.num_patches
        if self.backbone.pos_embed.shape[1] != expected_tokens:
            self.update_pos_embed()

        self.seg_head = nn.Sequential(
            nn.Conv2d(self.backbone.embed_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

        self.original_size = None

    def update_pos_embed(self):
        old_pos_embed = self.backbone.pos_embed
        cls_token = old_pos_embed[:, :1, :]
        patch_pos = old_pos_embed[:, 1:, :]

        # Original pretrained ViT size was 24x24 : (384x384)/(16x16)
        patch_pos = patch_pos.reshape(1, 24, 24, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(self.grid_h, self.grid_w), mode='bilinear', align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, self.num_patches, -1)
        new_pos_embed = torch.cat([cls_token, patch_pos], dim=1)
        self.backbone.pos_embed = nn.Parameter(new_pos_embed)

    def forward(self, x):
        B = x.shape[0]
        self.original_size = x.shape[2:]  # Expect (64, 2048)

        # DO NOT resize
        feats = self.backbone(x)  # [B, 8193, C]
        C = feats.shape[-1]
        feats = feats[:, 1:, :].reshape(B, self.grid_h, self.grid_w, C).permute(0, 3, 1, 2)

        logits = self.seg_head(feats)  # [B, num_classes, 32, 256]
        logits = F.interpolate(logits, size=self.original_size, mode='bilinear', align_corners=False)
        return logits