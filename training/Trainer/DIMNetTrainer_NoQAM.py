import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.models.DIMNet_NoQAM import DIMNet_NoQAM
from nnunetv2.preprocessing.missing_modality_simulator import simulate_missing_modalities

class DIMNetTrainer_NoQAM(nnUNetTrainer):
    """
    去掉质量评分模块；保留训练阶段随机遮蔽（与 DGAF 一致）
    """
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.network = DIMNet_NoQAM(num_modalities=4, feat_channels=32, out_channels=3)

    def run_iteration(self, data, target):
        # 与基线一致：训练时随机遮蔽
        if self.training:
            data, mask = simulate_missing_modalities(data, mode='random', missing_prob=0.25)
        else:
            mask = (data.sum(dim=(2,3,4,5)) != 0).float()
        output, _ = self.network(data, mask)
        loss = self.compute_loss(output, target)
        loss.backward()
        return loss.detach().cpu().numpy()
