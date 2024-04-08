from typing import Optional

import torchfrom torchmetrics import Metric
from metrics.qc_metrics.utils import topkfrom metrics.qc_metrics.utils import valid_filterfrom utils import wrap_angle


class minAHE(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(minAHE, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,
               keep_invalid_final_step: bool = True,
               min_criterion: str = 'FDE') -> None:
        pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
        pred_topk, _ = topk(self.max_guesses, pred, prob)
        if min_criterion == 'FDE':
            inds_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=self.device)).argmax(dim=-1)
            inds_best = torch.norm(
                pred_topk[torch.arange(pred.size(0)), :, inds_last, :-1] -
                target[torch.arange(pred.size(0)), inds_last, :-1].unsqueeze(-2), p=2, dim=-1).argmin(dim=-1)
        elif min_criterion == 'ADE':
            inds_best = (torch.norm(pred_topk[..., :-1] - target[..., :-1].unsqueeze(1), p=2, dim=-1) *
                         valid_mask.unsqueeze(1)).sum(dim=-1).argmin(dim=-1)
        else:
            raise ValueError('{} is not a valid criterion'.format(min_criterion))
        self.sum += ((wrap_angle(pred_topk[torch.arange(pred.size(0)), inds_best, :, -1] - target[..., -1]).abs() *
                      valid_mask).sum(dim=-1) / valid_mask.sum(dim=-1)).sum()
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
