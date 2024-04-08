import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MultipathLoss(nn.Module):
    def __init__(self, use_rho: bool = False) -> None:
        super().__init__()
        self.use_rho = use_rho

    def forward(
        self, gmm: torch.Tensor, log_pi: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        gmm: [N, 4/5]
        pi: [N]
        """
        if not self.use_rho:
            return self.gaussian_nll_no_rho(gmm, log_pi, target)
        else:
            raise NotImplementedError

    def gaussian_nll_no_rho(
        self, gmm: torch.Tensor, log_pi: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        gmm: [N, 4]
        target: [N, 2]
        pi: [N]
        """
        mu = gmm[..., :2]
        log_sigma = gmm[..., 2:4]
        sigma = torch.exp(log_sigma)

        diff = target - mu
        reg_nll = (
            torch.sum(log_sigma, dim=-1)
            + 0.5 * torch.sum((diff / sigma) ** 2, dim=-1)
            + 2 * np.log(np.pi)
        )
        cls_nll = -log_pi

        loss = (reg_nll + cls_nll).mean()

        return loss, reg_nll.mean(), cls_nll.mean()


class MultipathLaplaceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self, pred: torch.Tensor, log_pi: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        pred: [N, 4/5]
        log_pi: [N]
        """

        loc, scale = pred.chunk(2, dim=-1)
        scale = scale.clone()
        with torch.no_grad():
            scale.clamp_(min=self.eps)
        reg_nll = torch.sum(
            torch.log(2 * scale) + torch.abs(target - loc) / scale, dim=-1
        )
        cls_nll = -log_pi
        loss = (reg_nll + cls_nll).mean()

        return loss, reg_nll.mean(), cls_nll.mean()
