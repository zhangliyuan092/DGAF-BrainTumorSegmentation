import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.models.OnlyQNet import OnlyQNet

class DIMNetTrainer_OnlyQ(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.network = OnlyQNet(num_modalities=4, feat_channels=32, out_channels=3)

    def run_iteration(self, data, target):
        # 不做随机模拟，纯粹考察 Q-Net 的权重作用
        mask = (data.sum(dim=(2,3,4,5)) != 0).float()
        output = self.network(data, mask)
        loss = self.compute_loss(output, target)
        loss.backward()
        return loss.detach().cpu().numpy()
