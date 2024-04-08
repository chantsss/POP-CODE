from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as Ffrom torchmetrics import Metric


class berierFDE(Metric):
    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super(berierFDE, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.name = "b-FDE"
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> None:
        with torch.no_grad():
            pred = outputs["traj"]
            prob = F.softmax(outputs["prob"], -1)
            self.sum += torch.min(
                (1 - prob) ** 2
                + torch.norm(
                    pred[..., -1, :2] - target.unsqueeze(1)[..., -1, :2], p=2, dim=-1
                ),
                dim=-1,
            ).values.sum()
            self.count += pred.shape[0]

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
