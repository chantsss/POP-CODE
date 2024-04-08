import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianNLLLoss(nn.Module):
    def __init__(self) -> None:
        super(GaussianNLLLoss, self).__init__()

    def forward(
        self, gaussian: torch.Tensor, pi: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        # return self.implementation1(gaussian, pi, target)
        return self.implementation2(gaussian, pi, target)

    def implementation1(self, gaussian, pi, target):
        mu = gaussian[..., :2]  # [F, n, 2]
        log_sigma = gaussian[..., 2:4]  # [F, n, 2
        sigma = torch.exp(log_sigma)  # [F, n, 2]
        rho = torch.tanh(gaussian[..., 4])  # [F, n]
        one_minus_rho2 = torch.max(1 - rho ** 2, torch.tensor(1e-5, device=rho.device))

        # log_pi = F.log_softmax(pi, dim=0)
        diff = target[None, ..., :2] - mu  # [F, n, 2]

        nll = (
            # -log_pi
            torch.sum(log_sigma, dim=-1)
            + 0.5 * torch.log(one_minus_rho2)
            + 0.5
            * (
                torch.sum((diff / sigma) ** 2, dim=-1)
                - 2 * rho * torch.prod(diff / sigma, dim=-1)
            )
            / one_minus_rho2
        )

        return nll.mean()

    def implementation2(self, gaussian, pi, target):
        # multi-path-plus-plus implementation
        mu = gaussian[..., :2]  # [n, 2]
        a = gaussian[..., 2]
        b = gaussian[..., 3]
        c = gaussian[..., 4]
        sigma = torch.sqrt(
            torch.stack(
                [torch.exp(a) * torch.cosh(b), torch.exp(-a) * torch.cosh(b)], dim=1
            )
            * torch.exp(c).view(-1, 1)
        )
        rho = torch.tanh(b)
        one_minus_rho2 = torch.max(1 - rho ** 2, torch.tensor(1e-5, device=rho.device))

        diff = target[None, ..., :2] - mu  # [F, n, 2]

        nll = (
            # -log_pi
            torch.sum(torch.log(sigma), dim=-1)
            + 0.5 * torch.log(one_minus_rho2)
            + 0.5
            * (
                torch.sum((diff / sigma) ** 2, dim=-1)
                - 2 * rho * torch.prod(diff / sigma, dim=-1)
            )
            / one_minus_rho2
        )

        return nll.mean()
