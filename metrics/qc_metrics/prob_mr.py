from typing import Optional

import torchfrom torchmetrics import Metric
from metrics.qc_metrics.utils import topkfrom metrics.qc_metrics.utils import valid_filter


class ProbMR(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(ProbMR, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,
               keep_invalid_final_step: bool = True,
               miss_criterion: str = 'FDE',
               miss_threshold: float = 2.0) -> None:
        pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
        pred_topk, prob_topk = topk(self.max_guesses, pred, prob)
        if miss_criterion == 'FDE':
            inds_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=self.device)).argmax(dim=-1)
            min_fde, inds_best = torch.norm(pred_topk[torch.arange(pred.size(0)), :, inds_last] -
                                            target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2),
                                            p=2, dim=-1).min(dim=-1)
            self.sum += torch.where(min_fde > miss_threshold,
                                    1.0,
                                    1.0 - prob_topk[torch.arange(pred.size(0)), inds_best]).sum()
        elif miss_criterion == 'MAXDE':
            min_maxde, inds_best = ((torch.norm(pred_topk - target.unsqueeze(1), p=2, dim=-1) *
                                     valid_mask.unsqueeze(1)).max(dim=-1)[0]).min(dim=-1)
            self.sum += torch.where(min_maxde > miss_threshold,
                                    1.0,
                                    1.0 - prob_topk[torch.arange(pred.size(0)), inds_best]).sum()
        else:
            raise ValueError('{} is not a valid criterion'.format(miss_criterion))
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
