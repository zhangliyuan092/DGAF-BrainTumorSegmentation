import torch
import torch.nn as nn
import torch.nn.functional as F

class DGAFModule(nn.Module):
    """
    动态门控自适应特征融合模块
    - 融合模态特征、质量分数和mask信息
    - 可直接插入UNet或自定义主干
    """
    def __init__(self, num_modalities, feat_channels):
        super(DGAFModule, self).__init__()
        self.num_modalities = num_modalities
        self.feat_channels = feat_channels

        # 质量-门控MLP (每模态)
        self.quality_gate = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_channels + 2, 32),  # +2: mask和质量分数
                nn.ReLU(),
                nn.Linear(32, 8),
                nn.ReLU(),
                nn.Linear(8, 1)
            ) for _ in range(num_modalities)
        ])

        # 最终融合权重softmax
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feats, qualities, mask):
        """
        feats: [B, M, C, H, W, D]        # M模态
        qualities: [B, M]                # 每模态质量分数
        mask: [B, M]                     # 每模态mask（1=可用, 0=缺失）
        """
        B, M, C, H, W, D = feats.shape
        weights = []
        for i in range(M):
            feat_mean = feats[:, i].mean(dim=[2, 3, 4])  # [B, H, W, D]->[B]
            feat_in = torch.cat([feat_mean, qualities[:, i:i+1], mask[:, i:i+1]], dim=1)  # [B, C+2]
            w = self.quality_gate[i](feat_in)  # [B, 1]
            w = w * mask[:, i:i+1]             # 缺失模态权重强制为0
            weights.append(w)
        weights = torch.cat(weights, dim=1)  # [B, M]

        # 防止全缺失导致除0
        weights_sum = weights.sum(dim=1, keepdim=True) + 1e-6
        weights = weights / weights_sum       # 有效模态归一化

        # 加权融合
        weights_ = weights.view(B, M, 1, 1, 1, 1)
        fused = (feats * weights_).sum(dim=1)    # [B, C, H, W, D]
        return fused, weights
