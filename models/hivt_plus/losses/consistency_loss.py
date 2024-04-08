
#from itertools import permutations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyLoss(nn.Module):
    def __init__(self, num_modes: int, history_steps: int=20 ,future_steps: int=30) -> None:
        super(ConsistencyLoss, self).__init__()
        self.num_modes = num_modes
        all_permutations = torch.Tensor(list(permutations(range(num_modes)))).long()
        self.num_permutations = len(all_permutations)
        all_permutations_flat = all_permutations.flatten()
        self.register_buffer("all_permutations", all_permutations_flat)
        self.future_steps = future_steps
        self.history_steps = history_steps

    def forward(
        self,
        pred_past: torch.Tensor,
        pred_now: torch.Tensor,
        pad_loc: torch.Tensor,
        pad_loc_mask: torch.Tensor,
        pad_loc_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        pred_past: [F, N, 30, 2]
        pred_now: [F, N, 30, 2]
        target: [N, 30, 2]
        pad_loc: [F, N, 2]
        pad_loc_target: [N, 2]
        """
        valid_mask = ~pad_loc_mask
        loc_past, _ = pred_past[:, valid_mask].chunk(2, dim=-1)
        loc_now, _ = pred_now[:, valid_mask].chunk(2, dim=-1)
        valid_pad_loc = pad_loc[:, valid_mask]
        valid_pad_loc_target = (
            pad_loc_target[valid_mask].unsqueeze(0).repeat(self.num_modes, 1, 1)
        )

        pad_loc_reg_loss = F.smooth_l1_loss(valid_pad_loc, valid_pad_loc_target)

        loc_past_all = loc_past + valid_pad_loc.unsqueeze(-2).detach()
        loc_now_all = loc_now + valid_pad_loc_target.unsqueeze(-2)

        consistency_loss = self.global_minimum_consistency_loss(
            loc_past_all=loc_past_all, loc_now_all=loc_now_all
        )

        return pad_loc_reg_loss, consistency_loss

    def global_minimum_consistency_loss(
        self, loc_past_all: torch.Tensor, loc_now_all: torch.Tensor
    ):
        """Find optimal match from all permutations
        loc_past: [F, N 30, 2]
        loc_now: [F, N, 30, 2]
        """
        loc_past = loc_past_all.transpose(1, 0)  # [N, F, 30, 2]
        loc_now = loc_now_all.transpose(1, 0)  # [N, F, 30, 2]

        batch = loc_past.shape[0]

        loc_now_permutes = loc_now[:, self.all_permutations].view(
            batch, self.num_permutations, self.num_modes, self.future_steps, 2
        )  # [N, P, F, 30, 2]
        match_error = torch.norm(
            loc_past.unsqueeze(1)[..., -1, :2] - loc_now_permutes[..., -1, :2], dim=-1
        ).mean(-1)
        optimal_match = torch.argmin(match_error, dim=-1)

        consistency_loss = F.smooth_l1_loss(
            loc_past, loc_now_permutes[torch.arange(batch), optimal_match]
        )
        return consistency_loss
