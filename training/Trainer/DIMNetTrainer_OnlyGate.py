import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.models.OnlyGateNet import OnlyGateNet

class DIMNetTrainer_OnlyGate(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.network = OnlyGateNet(num_modalities=4, feat_channels=32, out_channels=3)

    def run_iteration(self, data, target):
        # 不做随机模拟，隔离门控贡献
        mask = (data.sum(dim=(2,3,4,5)) != 0).float()
        output = self.network(data, mask)
        loss = self.compute_loss(output, target)
        loss.backward()
        return loss.detach().cpu().numpy()
