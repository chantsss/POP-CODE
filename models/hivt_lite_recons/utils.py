from typing import Optional, Tuple

import torch


class DistanceDropEdge(object):
    def __init__(self, max_distance: Optional[float] = None) -> None:
        self.max_distance = max_distance

    def __call__(
        self, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.max_distance is None:
            return edge_index, edge_attr
        row, col = edge_index
        mask = torch.norm(edge_attr, p=2, dim=-1) < self.max_distance
        edge_index = torch.stack([row[mask], col[mask]], dim=0)
        edge_attr = edge_attr[mask]
        return edge_index, edge_attr
