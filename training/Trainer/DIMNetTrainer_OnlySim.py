import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.models.OnlySimNet import OnlySimNet
from nnunetv2.preprocessing.missing_modality_simulator import simulate_missing_modalities

class DIMNetTrainer_OnlySim(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.network = OnlySimNet(num_modalities=4, feat_channels=32, out_channels=3)

    def run_iteration(self, data, target):
        if self.training:
            data, mask = simulate_missing_modalities(data, mode='random', missing_prob=0.25)
        else:
            mask = (data.sum(dim=(2,3,4,5)) != 0).float()
        output = self.network(data, mask)
        loss = self.compute_loss(output, target)
        loss.backward()
        return loss.detach().cpu().numpy()
