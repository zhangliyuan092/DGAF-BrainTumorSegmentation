import torch
import torch.nn as nn

class OnlyQNet(nn.Module):
    def __init__(self, num_modalities=4, feat_channels=32, out_channels=3):
        super().__init__()
        self.num_modalities = num_modalities
        self.feat_channels = feat_channels
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
        # Q-Net: GAP -> FC -> Sigmoid
        self.qnets = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(feat_channels, 1),
                nn.Sigmoid()
            ) for _ in range(num_modalities)
        ])
        self.decoder = nn.Sequential(
            nn.Conv3d(feat_channels, 32, 3, padding=1),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, out_channels, 1)
        )

    def forward(self, x, mask):
        # x: [B,M,H,W,D]
        B, M, H, W, D = x.shape
        feats, qs = [], []
        for i in range(M):
            f = self.encoders[i](x[:, i:i+1])     # [B,C,H,W,D]
            feats.append(f)
            q = self.qnets[i](f)                  # [B,1]
            qs.append(q)
        feats = torch.stack(feats, dim=1)         # [B,M,C,H,W,D]
        q = torch.cat(qs, dim=1)                  # [B,M]
        # 权重：Softmax(q * mask)
        logits = q + (mask * 0 + 1e-8)            # 保持同维
        logits = logits + torch.log(mask + 1e-8)  # 去掉缺失模态
        alpha = torch.softmax(logits, dim=1).view(B, M, 1, 1, 1, 1)
        fused = (feats * alpha).sum(1)
        out = self.decoder(fused)
        return out
