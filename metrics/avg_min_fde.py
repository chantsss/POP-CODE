from typing import Any, Callable, Optional, Dict

import torchfrom torchmetrics import Metricfrom torchmetrics.classification.accuracy import Accuracy


class avgMinFDE(Metric):
    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super(avgMinFDE, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.name = "avgminFDE"
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        y_hat: torch.Tensor,
        pi: torch.Tensor,
        y: torch.Tensor,
        scored_mask: torch.Tensor,
    ) -> None:
        """scored agents are assumed to have full observations
        y_hat: [B, K, N, T, 2]
        pi: [B, K]
        y: [B, N, T, 2]
        scored_mask: [B, N]
        """
        with torch.no_grad():
            # num_valid_agents = scored_mask.sum(-1)
            l2_norm = (torch.norm(y_hat[:, :, :, :2] - y[..., :, :2], \
                    p=2, dim=-1) * scored_mask).sum( dim=-1) 
            best_mode = l2_norm.argmin(dim=0)
            num_nodes = best_mode.size(0)
            avg_min_fde = l2_norm[best_mode, torch.arange(num_nodes)]

            self.sum += avg_min_fde.sum()
            self.count += num_nodes

    # def update(
    #     self,
    #     y_hat: torch.Tensor,
    #     pi: torch.Tensor,
    #     y: torch.Tensor,
    #     scored_mask: torch.Tensor,
    # ) -> None:
    #     """scored agents are assumed to have full observations
    #     y_hat: [B, K, N, T, 2]
    #     pi: [B, K]
    #     y: [B, N, T, 2]
    #     scored_mask: [B, N]
    #     """
    #     with torch.no_grad():
    #         bs, K, N, T, _ = y_hat.shape
    #         valid_mask = scored_mask.unsqueeze(1).float()  # [B, 1, N]
    #         num_valid_agents = valid_mask.sum(-1)
    #         avg_fde = (
    #             torch.norm(y_hat[..., -1, :2] - y.unsqueeze(1)[..., -1, :2], dim=-1)
    #             * valid_mask
    #         ).sum(-1) / num_valid_agents
    #         avg_min_fde = torch.min(avg_fde, dim=-1)[0]

    #         self.sum += avg_min_fde.sum()
    #         self.count += bs

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
