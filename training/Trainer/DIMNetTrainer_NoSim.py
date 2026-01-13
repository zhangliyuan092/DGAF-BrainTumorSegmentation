import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.models.DIMNet import DIMNet

class DIMNetTrainer_NoSim(nnUNetTrainer):
    """
    Ablation: NoSim
    - 训练时不做随机模态缺失模拟（仍然支持真实缺失；验证/推理也按真实输入mask处理）
    - 其余配置（网络、损失、优化器等）全部沿用 nnUNetTrainer 默认逻辑
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.network = DIMNet(num_modalities=4, feat_channels=32, out_channels=3)

    @torch.no_grad()
    def _build_mask_from_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        根据输入数据是否为全0生成 (B, M) 的可用模态掩模。
        data: (B, M, C=1, H, W, D) 或 (B, M, H, W, D) 取决于你的DIMNet前向写法
        这里按你之前的写法处理：x shape 是 [B, M, H, W, D]
        如果你的 dataloader 给到的是 [B, M, 1, H, W, D]，请改成 sum(dim=(2,3,4,5))
        """
        # 兼容两种维度
        if data.dim() == 6:  # [B,M,1,H,W,D]
            red_dims = (2, 3, 4, 5)
        else:                # [B,M,H,W,D]
            red_dims = (2, 3, 4)
        return (data.sum(dim=red_dims) != 0).float()

    def run_iteration(self, data, target):
        """
        不做 simulate_missing_modalities。其它与 nnUNetTrainer 保持一致。
        """
        x = data  

        # 构造可用模态mask（训练时也允许真实缺失）
        mask = self._build_mask_from_data(x)

        # 前向
        output, _ = self.network(x, mask)

        # 计算损失（沿用父类 self.compute_loss）
        loss = self.compute_loss(output, target)
        loss.backward()
        return loss.detach().cpu().numpy()
