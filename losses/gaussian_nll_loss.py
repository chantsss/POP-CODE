
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianNLLLoss(nn.Module):

    def __init__(self,
                 full: bool = False,
                 eps: float = 1e-6,
                 reduction: str = 'mean') -> None:
        super(GaussianNLLLoss, self).__init__()
        self.full = full
        self.eps = eps
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        mean, var = pred.chunk(2, dim=-1)
        return F.gaussian_nll_loss(input=mean, target=target, var=var, full=self.full, eps=self.eps,
                                   reduction=self.reduction)
