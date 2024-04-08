import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class AnalysisMetric:
    def __init__(self) -> None:
        self.min_ade = []
        self.min_fde = []
        self.b_min_fde = []
        self.seq_idx = []

    @torch.no_grad()
    def log(self, out, target, seq_idx):
        pred = out["traj"]
        pi = F.softmax(out["prob"], -1)  # [N, F]

        endpoint_error = torch.norm(
            pred[..., -1, :2] - target.unsqueeze(1)[..., -1, :2], p=2, dim=-1
        )  # [N, F]
        # minFDE
        min_fde = torch.min(endpoint_error, dim=-1).values  # [N]

        # b-minFDE
        b_min_fde = torch.min((1 - pi) ** 2 + endpoint_error, dim=-1).values  # [N]

        # minADE
        best_mode = torch.argmin(endpoint_error, dim=-1)  # [N
        best_pred = pred[torch.arange(pred.shape[0]), best_mode]  # [N, 30, 2]
        min_ade = torch.norm(best_pred - target, p=2, dim=-1).mean(-1)  # [N]

        self.min_fde.append(min_fde)
        self.b_min_fde.append(b_min_fde)
        self.min_ade.append(min_ade)
        self.seq_idx.append(seq_idx)

    def save(self, path):
        torch.save(
            {
                "min_ade": torch.cat(self.min_ade, dim=0),
                "min_fde": torch.cat(self.min_fde, dim=0),
                "b_min_fde": torch.cat(self.b_min_fde, dim=0),
                "seq_idx": torch.cat(self.seq_idx, dim=0),
            },
            path,
        )
