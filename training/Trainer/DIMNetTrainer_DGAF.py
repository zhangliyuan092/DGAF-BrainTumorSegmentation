from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.models.DIMNet import DIMNet
from nnunetv2.preprocessing.missing_modality_simulator import simulate_missing_modalities
import torch

class DIMNetTrainer_DGAF(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device('cuda')
    ):
        super().__init__(
            plans,
            configuration,
            fold,
            dataset_json,
            device
        )
        self.network = DIMNet(num_modalities=4, feat_channels=32, out_channels=3)

        import inspect
        print("======== nnUNetTrainer loaded from:", inspect.getfile(nnUNetTrainer))

    def run_iteration(self, data, target):
        if self.training:
            data, mask = simulate_missing_modalities(data, mode='random', missing_prob=0.25)
        else:
            mask = (data.sum(dim=(2,3,4,5)) != 0).float()
        output, weights = self.network(data, mask)
        loss = self.compute_loss(output, target)
        loss.backward()
        return loss.detach().cpu().numpy()
