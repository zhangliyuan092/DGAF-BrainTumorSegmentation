import torch
import torch.nn as nn

class DIMNet_NoGate(nn.Module):
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
        # 质量评分（保留 QAM）
        self.quality_mlps = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(feat_channels, 1),
                nn.Sigmoid()
            ) for _ in range(num_modalities)
        ])
        # 无 Gate：用 mask * q 做静态加权
        self.decoder = nn.Sequential(
            nn.Conv3d(feat_channels, 32, 3, padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, out_channels, 1)
        )

    def forward(self, x, mask=None):
        B, M, H, W, D = x.shape
        if mask is None:
            mask = torch.ones((B, M), dtype=x.dtype, device=x.device)
        feats, qualities = [], []
        for i in range(M):
            f = self.encoders[i](x[:, i:i+1])  # [B,C,H,W,D]
            feats.append(f)
            q = self.quality_mlps[i](f)       # [B,1]
            qualities.append(q)
        feats = torch.stack(feats, dim=1)        # [B,M,C,H,W,D]
        q = torch.cat(qualities, dim=1)          # [B,M]
        w = (q * mask).clamp_min(1e-6)           # [B,M]
        w = w / (w.sum(dim=1, keepdim=True))     # 归一化
        w = w[..., None, None, None, None]       # [B,M,1,1,1,1]
        fused = (feats * w).sum(dim=1)           # [B,C,H,W,D]
        out = self.decoder(fused)
        return out, w.squeeze(-1).squeeze(-1).squeeze(-1).squeeze(-1)
