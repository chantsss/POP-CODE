from typing import Any, Callable, Optional, Dict

import torchfrom torchmetrics import Metric


class MR1(Metric):
    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
        self,
        miss_threshold: float = 2.0,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super(MR1, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.name = "MR1"
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.miss_threshold = miss_threshold

    def update(self, outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> None:
        with torch.no_grad():
            pred = outputs["traj"]
            pi = outputs["prob"]
            best_mode = torch.argmax(pi, dim=-1)
            best_pred = pred[torch.arange(pred.shape[0]), best_mode]  # [N, 60, 2]
            missed_pred = (
                torch.norm(
                    best_pred[..., -1, :2] - target[..., -1, :2],
                    p=2,
                    dim=-1,
                )
                > self.miss_threshold
            )
            self.sum += missed_pred.sum()
            self.count += pred.shape[0]

    def compute(self) -> torch.Tensor:
        return self.sum / self.count


class MR(Metric):
    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
        self,
        miss_threshold: float = 2.0,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super(MR, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.name = "MR"
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.miss_threshold = miss_threshold

    def update(self, outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> None:
        with torch.no_grad():
            pred = outputs["traj"]
            missed_pred = (
                torch.norm(
                    pred[..., -1, :2] - target.unsqueeze(1)[..., -1, :2], p=2, dim=-1
                )
                > self.miss_threshold
            )
            self.sum += missed_pred.all(-1).sum()
            self.count += pred.shape[0]

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
