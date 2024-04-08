
#
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as Ffrom utils.utils_hivt import TemporalDatafrom metrics import MR, berierFDE, minADE, minFDEfrom models.hivt_plus import GlobalInteractor, LocalEncoder, MLPDecoderfrom models.hivt_plus.losses import (
    ConsistencyLoss,
    LaplaceNLLLoss,
    SoftTargetCrossEntropyLoss,
)from ..hivt import HiVTfrom submission.submission_base import SubmissionBasefrom torchmetrics import MetricCollection


class HiVTPlus(pl.LightningModule):
    def __init__(
        self,
        historical_steps: int,
        future_steps: int,
        num_modes: int,
        rotate: bool,
        node_dim: int,
        edge_dim: int,
        dim: int,
        num_heads: int,
        dropout: float,
        num_temporal_layers: int,
        num_global_layers: int,
        local_radius: float,
        parallel: bool,
        lr: float,
        weight_decay: float,
        T_max: int,
        submission_handler=None,
    ) -> None:
        super(HiVTPlus, self).__init__()
        self.historical_steps = historical_steps
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.rotate = rotate
        self.parallel = parallel
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max

        self.local_encoder = LocalEncoder(
            historical_steps=historical_steps,
            node_dim=node_dim,
            edge_dim=edge_dim,
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            num_temporal_layers=num_temporal_layers,
            local_radius=local_radius,
            parallel=parallel,
        )
        self.global_interactor = GlobalInteractor(
            historical_steps=historical_steps,
            embed_dim=dim,
            edge_dim=edge_dim,
            num_modes=num_modes,
            num_heads=num_heads,
            num_layers=num_global_layers,
            dropout=dropout,
            rotate=rotate,
        )
        self.decoder = MLPDecoder(
            local_channels=dim,
            global_channels=dim,
            future_steps=future_steps,
            num_modes=num_modes,
            uncertain=True,
        )
        self.reg_loss = LaplaceNLLLoss(reduction="mean")
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction="mean")
        self.consistency_loss = ConsistencyLoss(num_modes=num_modes, history_steps=historical_steps,future_steps=future_steps)

        metrics = MetricCollection([minADE(), minFDE(), MR(), berierFDE()])
        self.val_metrics = metrics.clone(prefix="val_")

        if submission_handler:
            self.submission_handler: SubmissionBase = submission_handler

    def forward(self, data: TemporalData):
        if self.rotate:
            rotate_mat = torch.empty(data.num_nodes, 2, 2, device=self.device)
            sin_vals = torch.sin(data["rotate_angles"])
            cos_vals = torch.cos(data["rotate_angles"])
            rotate_mat[:, 0, 0] = cos_vals
            rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals
            rotate_mat[:, 1, 1] = cos_vals
            if data.y is not None:
                data.y = torch.bmm(data.y, rotate_mat)
            data["rotate_mat"] = rotate_mat
        else:
            data["rotate_mat"] = None

        local_embed0, local_embed1 = self.local_encoder(data=data)

        global_embed0 = self.global_interactor(
            data=data, local_embed=local_embed0, offset=1
        )
        global_embed1 = self.global_interactor(
            data=data, local_embed=local_embed1, offset=0
        )

        y_hat0, pi0, pad_loc = self.decoder(
            local_embed=local_embed0, global_embed=global_embed0, add_pad_loc=True
        )
        y_hat1, pi1 = self.decoder(local_embed=local_embed1, global_embed=global_embed1)

        return {
            "0": (y_hat0, pi0, pad_loc),
            "1": (y_hat1, pi1),
        }

    def training_step(self, data, batch_idx):
        out = self(data)

        (y_hat, pi) = out["1"]
        reg_mask = ~data["padding_mask"][:, self.historical_steps :]  # [N, 30]
        valid_steps = reg_mask.sum(dim=-1)  # [N]
        cls_mask = valid_steps > 0  # [N]
        l2_norm = (torch.norm(y_hat[:, :, :, :2] - data.y, p=2, dim=-1) * reg_mask).sum(
            dim=-1
        )  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]  # [N, 30, 2]
        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        soft_target = (
            F.softmax(-l2_norm[:, cls_mask] / valid_steps[cls_mask], dim=0).t().detach()
        )
        cls_loss = self.cls_loss(pi[cls_mask], soft_target)

        (y_hat0, pi0, pad_loc) = out["0"]
        pad_loc_mask = data["padding_mask"][:, self.historical_steps-2] | data["padding_mask"][:, self.historical_steps-1]
        pad_loc_target = data.x[:, self.historical_steps-1]  # [N]
        pad_loc_loss, consistency_loss = self.consistency_loss(
            y_hat0, y_hat, pad_loc, pad_loc_mask, pad_loc_target
        )

        loss = reg_loss + cls_loss + pad_loc_loss + consistency_loss

        self.log(
            "train/reg_loss",
            reg_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )
        self.log("train/cls_loss", cls_loss, on_step=True, on_epoch=True, batch_size=1)
        self.log(
            "train/pad_loss", pad_loc_loss, on_step=True, on_epoch=True, batch_size=1
        )
        self.log(
            "train/consistency_loss",
            consistency_loss,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )
        return loss

    def validation_step(self, data, batch_idx):
        out = self(data)

        (y_hat, pi) = out["1"]
        reg_mask = ~data["padding_mask"][:, self.historical_steps :]
        l2_norm = (torch.norm(y_hat[:, :, :, :2] - data.y, p=2, dim=-1) * reg_mask).sum(
            dim=-1
        )  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])

        # (y_hat0, pi0, pad_loc) = out["0"]
        # pad_loc_mask = data["padding_mask"][:, 18] | data["padding_mask"][:, 19]
        # pad_loc_target = data.x[:, 19]  # [N]
        # pad_loc_loss, consistency_loss = self.consistency_loss(
        #     y_hat0, y_hat, pad_loc, pad_loc_mask, pad_loc_target
        # )

        self.log(
            "val/reg_loss",
            reg_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        # self.log("val/pad_loss", pad_loc_loss, on_step=True, on_epoch=True)
        # self.log("val/consistency_loss", consistency_loss, on_step=True, on_epoch=True)

        y_hat_agent = y_hat[:, data["agent_index"], :, :2]
        pi_hat_agent = pi[data["agent_index"]]
        y_agent = data.y[data["agent_index"]]

        outputs = {"traj": y_hat_agent.transpose(1, 0), "prob": pi_hat_agent}

        metrics = self.val_metrics(outputs, y_agent)
        self.log_dict(
            metrics, prog_bar=True, on_step=False, on_epoch=True, batch_size=1
        )

    def test_step(self, data, batch_idx) -> None:
        traj, prob = self(data)["0"]

        agent_traj = traj[:, data["agent_index"], :, :2].transpose(1, 0)
        agent_pi = prob[data["agent_index"]]
        rot_mat = data["rotate_mat"][data["agent_index"]].transpose(-1, -2)
        position = (
            data["positions"][data["agent_index"], self.historical_steps-1, :2].unsqueeze(1).unsqueeze(1)
        )
        agent_traj = torch.matmul(agent_traj, rot_mat.unsqueeze(1)) + position

        self.submission_handler.format_data(data, agent_traj, agent_pi)

    def on_test_end(self) -> None:
        self.submission_handler.generate_submission_file()

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
            nn.GRUCell,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.T_max, eta_min=0.0
        )
        return [optimizer], [scheduler]
