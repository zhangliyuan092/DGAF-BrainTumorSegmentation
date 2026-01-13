import torch
import torch.nn as nn

class OnlyGateNet(nn.Module):
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
        # Gate controller: concat -> 1x1x1 conv -> softmax
        self.gate = nn.Sequential(
            nn.Conv3d(num_modalities * feat_channels, 64, 1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(64, num_modalities, 1, bias=True)  # produce M logits map
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(feat_channels, 32, 3, padding=1),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, out_channels, 1)
        )

    def forward(self, x, mask):
        B, M, H, W, D = x.shape
        feats = [self.encoders[i](x[:, i:i+1]) for i in range(M)]  # list of [B,C,H,W,D]
        Fcat = torch.cat(feats, dim=1)                             # [B,M*C,H,W,D]
        logits = self.gate(Fcat)                                   # [B,M,H,W,D]
        # 局部 softmax + 掩模（缺失模态不分配权重）
        logits = logits + torch.log(mask.view(B, M, 1, 1, 1) + 1e-8)
        alpha = torch.softmax(logits, dim=1).unsqueeze(2)          # [B,M,1,H,W,D]
        feats_stacked = torch.stack(feats, dim=1)                  # [B,M,C,H,W,D]
        fused = (feats_stacked * alpha).sum(1)                     # [B,C,H,W,D]
        out = self.decoder(fused)
        return out
