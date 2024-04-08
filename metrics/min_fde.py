from typing import Any, Callable, Optional, Dict

import torchfrom torchmetrics import Metric


class minFDE1(Metric):
    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super(minFDE1, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.name = "minFDE1"
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> None:
        with torch.no_grad():
            pred = outputs["traj"]
            pi = outputs["prob"]
            best_mode = torch.argmax(pi, dim=-1)
            best_pred = pred[torch.arange(pred.shape[0]), best_mode]
            self.sum += torch.norm(
                best_pred[:, -1, :2] - target[:, -1, :2], p=2, dim=-1
            ).sum()
            self.count += pred.shape[0]

    def compute(self) -> torch.Tensor:
        return self.sum / self.count


class minFDE(Metric):
    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super(minFDE, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.name = "minFDE"
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> None:
        with torch.no_grad():
            pred = outputs["traj"]
            self.sum += torch.min(
                torch.norm(
                    pred[..., -1, :2] - target.unsqueeze(1)[..., -1, :2], p=2, dim=-1
                ),
                dim=-1,
            ).values.sum()
            self.count += pred.shape[0]

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
