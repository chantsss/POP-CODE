
import torch
import torch.nn as nnfrom torchvision.ops import sigmoid_focal_loss


class FocalLoss(nn.Module):

    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        return sigmoid_focal_loss(pred, target, self.alpha, self.gamma, self.reduction)
