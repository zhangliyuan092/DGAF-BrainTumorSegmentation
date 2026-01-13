import torch
import torch.nn as nn

class DIMNet_NoQAM(nn.Module):
    def __init__(self, num_modalities=4, feat_channels=32, out_channels=3):
        super().__init__()
        self.num_modalities = num_modalities
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(1, feat_channels, 3, padding=1),
                nn.InstanceNorm3d(feat_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(feat_channels, feat_channels, 3, padding=1),
                nn.InstanceNorm3d(feat_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(num_modalities)
        ])
        # 无 QAM、无 Gate
        self.decoder = nn.Sequential(
            nn.Conv3d(feat_channels, 32, 3, padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, out_channels, 1)
        )

    def forward(self, x, mask=None):
        # x: [B, M, H, W, D]
        B, M, H, W, D = x.shape
        if mask is None:
            mask = torch.ones((B, M), dtype=x.dtype, device=x.device)
        feats = []
        for i in range(M):
            feat = self.encoders[i](x[:, i:i+1])  # [B,C,H,W,D]
            feats.append(feat)
        feats = torch.stack(feats, dim=1)  # [B,M,C,H,W,D]
        mask_ = mask[..., None, None, None, None]  # [B,M,1,1,1,1]
        denom = mask_.sum(dim=1, keepdim=True).clamp_min(1.0)
        fused = (feats * mask_).sum(dim=1) / denom.squeeze(1)  # [B,C,H,W,D]
        out = self.decoder(fused)
        return out, None
