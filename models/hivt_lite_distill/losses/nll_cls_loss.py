import torch
import torch.nn as nn
import torch.nn.functional as F


class NllClsLoss(nn.Module):
    def __init__(self, eps: float = 1e-6, reduction: str = "mean") -> None:
        super(NllClsLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(
        self, preds: torch.Tensor, pis: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        preds: [F, N, 4]
        pis: [F, N]
        """
        loc, scale = preds.chunk(2, dim=-1)  # [F, N, 2]
        log_pi = F.log_softmax(pis, dim=0).unsqueeze(-1)  # [F, N, 1]

        scale = scale.clone()
        with torch.no_grad():
            scale.clamp_(min=self.eps)
        nll = -log_pi + torch.log(2 * scale) + torch.abs(target - loc) / scale

        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        elif self.reduction == "none":
            return nll
        else:
            raise ValueError(
                "{} is not a valid value for reduction".format(self.reduction)
            )
