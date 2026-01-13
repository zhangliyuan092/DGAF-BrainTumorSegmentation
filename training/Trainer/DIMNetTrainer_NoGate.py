import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.models.DIMNet_NoGate import DIMNet_NoGate
from nnunetv2.preprocessing.missing_modality_simulator import simulate_missing_modalities

class DIMNetTrainer_NoGate(nnUNetTrainer):
    """
    禁用 Gate，仅用 mask * QAM 静态加权融合；训练时仍随机遮蔽
    """
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.network = DIMNet_NoGate(num_modalities=4, feat_channels=32, out_channels=3)

    def run_iteration(self, data, target):
        if self.training:
            data, mask = simulate_missing_modalities(data, mode='random', missing_prob=0.25)
        else:
            mask = (data.sum(dim=(2,3,4,5)) != 0).float()
        output, _ = self.network(data, mask)
        loss = self.compute_loss(output, target)
        loss.backward()
        return loss.detach().cpu().numpy()
