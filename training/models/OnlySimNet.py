import torch
import torch.nn as nn

class OnlySimNet(nn.Module):
    def __init__(self, num_modalities=4, feat_channels=32, out_channels=3):
        super().__init__()
        self.num_modalities = num_modalities
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(1, feat_channels, 3, padding=1),
                nn.InstanceNorm3d(feat_channels),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(feat_channels, feat_channels, 3, padding=1),
                nn.InstanceNorm3d(feat_channels),
                nn.LeakyReLU(inplace=True)
            ) for _ in range(num_modalities)
        ])
        self.decoder = nn.Sequential(
            nn.Conv3d(feat_channels, 32, 3, padding=1),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, out_channels, 1)
        )

    def forward(self, x, mask):
        # x: [B, M, H, W, D]
        B, M, H, W, D = x.shape
        feats = []
        for i in range(M):
            feats.append(self.encoders[i](x[:, i:i+1]))  # [B,C,H,W,D]
        feats = torch.stack(feats, dim=1)                # [B,M,C,H,W,D]
        # 简单“可用模态平均”
        mask_ = mask.view(B, M, 1, 1, 1, 1).float()
        fused = (feats * mask_).sum(1) / (mask_.sum(1) + 1e-8)
        out = self.decoder(fused)
        return out
