
#
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as Ffrom torchmetrics import MetricCollection
from utils.utils_hivt import TemporalDatafrom metrics import MR, berierFDE, minADE, minFDEfrom metrics.analysis_metric import AnalysisMetricfrom submission.submission_base import SubmissionBase
from .decoder import GaussianDecoder, MLPDecoderfrom .global_interactor import GlobalInteractorfrom .local_encoder import LocalEncoderfrom .losses.gaussian_nll_loss import GaussianNLLLossfrom .losses.laplace_nll_loss import LaplaceNLLLossfrom .losses.multipath_loss import MultipathLaplaceLoss, MultipathLossfrom .losses.nll_cls_loss import NllClsLossfrom .losses.soft_target_cross_entropy_loss import SoftTargetCrossEntropyLoss
import random
import copyfrom models.common.utils import drop_his

class HiVTLiteMTask(pl.LightningModule):
    def __init__(
        self,
        historical_steps: int = 50,
        future_steps: int = 60,
        num_modes: int = 6,
        rotate: bool = True,
        node_dim: int = 2,
        edge_dim: int = 2,
        dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        nat_backbone: bool = False,
        pos_embed: bool = False,
        nll_cls: bool = False,
        num_temporal_layers: int = 4,
        num_global_layers: int = 3,

#recons
        future_steps_recons: int = 50,
        num_modes_recons: int = 6,
        dim_recons: int = 128,

        mtask: bool = True,
        gaussian: bool = False,
        use_multipath_loss: bool = False,
        local_radius: float = 50,
        parallel: bool = False,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        T_max: int = 64,
        reduce_his_length: bool = False,
        random_his_length: bool = False,
        random_interpolate_zeros: bool = False,
        valid_observation_length: int = 20,
        drop_all_agent: bool = True,
        submission_handler=None,
    ) -> None:
        super(HiVTLiteMTask, self).__init__()
        self.save_hyperparameters(ignore=["submission_handler"])

        self.historical_steps = historical_steps
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.rotate = rotate
        self.parallel = parallel
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.nll_cls = nll_cls
        self.gaussian = gaussian
        self.use_multipath_loss = use_multipath_loss
        self.validate_mode = False
        self.drop_all_agent = drop_all_agent
        self.mtask = mtask

        self.local_encoder = LocalEncoder(
            historical_steps=historical_steps,
            node_dim=node_dim,
            edge_dim=edge_dim,
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            local_radius=local_radius,
            parallel=parallel,
            nat_backbone=nat_backbone,
            pos_embed=pos_embed,
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
        if self.gaussian:
            self.decoder = GaussianDecoder(
                local_channels=dim,
                global_channels=dim,
                future_steps=future_steps,
                num_modes=num_modes,
                use_rho=False,
            )
            if self.mtask:
                self.decoder_recons = GaussianDecoder(
                    local_channels=dim_recons,
                    global_channels=dim_recons,
                    future_steps=future_steps_recons,
                    num_modes=num_modes_recons,
                    use_rho=False,
                )
            self.multipath_loss = MultipathLoss(use_rho=False)
        else:
            self.decoder = MLPDecoder(
                local_channels=dim,
                global_channels=dim,
                future_steps=future_steps,
                num_modes=num_modes,
                uncertain=True,
            )
            if self.mtask:
                self.decoder_recons = MLPDecoder(
                    local_channels=dim_recons,
                    global_channels=dim_recons,
                    future_steps=future_steps_recons,
                    num_modes=num_modes_recons,
                    uncertain=True,
                )

            if self.nll_cls:
                self.reg_loss = LaplaceNLLLoss(reduction="mean")
                self.cls_loss = NllClsLoss(reduction="mean")
            elif self.use_multipath_loss:
                self.multipath_loss = MultipathLaplaceLoss()
            else:
                self.reg_loss = LaplaceNLLLoss(reduction="mean")
                self.cls_loss = SoftTargetCrossEntropyLoss(reduction="mean")

        metrics = MetricCollection([minADE(), minFDE(), MR(), berierFDE()])
        self.val_metrics = metrics.clone(prefix="val_")

        if submission_handler:
            self.submission_handler: SubmissionBase = submission_handler

        self.reduce_his_length = reduce_his_length
        self.random_his_length = random_his_length
        self.random_interpolate_zeros = random_interpolate_zeros
        self.valid_observation_length = valid_observation_length

    def forward(self, data: TemporalData):

        if self.reduce_his_length:
            data = drop_his(self.valid_observation_length, self.historical_steps, \
                                                self.reduce_his_length, self.random_his_length, \
                                                self.random_interpolate_zeros, data, drop_all=self.drop_all_agent, drop_for_recons=True)
        if self.rotate:
            rotate_mat = torch.empty(data.num_nodes, 2, 2, device=self.device)
            sin_vals = torch.sin(data["rotate_angles"])
            cos_vals = torch.cos(data["rotate_angles"])
            rotate_mat[:, 0, 0] = cos_vals
            rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals
            rotate_mat[:, 1, 1] = cos_vals
            if data.y is not None:  # [N, 30, 2]
                data.y = torch.bmm(data.y, rotate_mat)
            data["rotate_mat"] = rotate_mat
        else:
            data["rotate_mat"] = None

        if 'x_type' not in data:
            data["x_type"] = torch.zeros(data.num_nodes, dtype=torch.long)
        
        if 'x_category' not in data:
            data["x_category"] = torch.ones(data.num_nodes, dtype=torch.long) * 2

        local_embed = self.local_encoder(data=data)
        global_embed = self.global_interactor(data=data, local_embed=local_embed)
        y_hat, pi = self.decoder(local_embed=local_embed, global_embed=global_embed)
        if self.mtask:
            y_hat_recons, pi_recons = self.decoder_recons(local_embed=local_embed, global_embed=global_embed)
        # for x in [y_hat, pi]:
        #     print('sum is', x.sum())
            return y_hat, pi, y_hat_recons, pi_recons
        else:
            return y_hat, pi, None, None

    def training_step(self, data, batch_idx):
        # laplace nll loss
        y_hat, pi, y_hat_recons, pi_recons = self(data)
        reg_mask = ~data["padding_mask"][:, self.historical_steps :]
        scored_mask = data["x_category"] >= 2
        reg_mask[~scored_mask, :] = False

        valid_steps = reg_mask.sum(dim=-1)
        cls_mask = valid_steps > 0
        l2_norm = (torch.norm(y_hat[:, :, :, :2] - data.y, p=2, dim=-1) * reg_mask).sum(
            dim=-1
        )  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]

        if self.mtask:
            #recons
            reg_mask_recons = ~data["his_pred_padding_mask"][:, :self.historical_steps]
            scored_mask_recons = data["x_category"] >= 2
            reg_mask_recons[~scored_mask_recons, :] = False

            valid_steps_recons = reg_mask_recons.sum(dim=-1)
            cls_mask_recons = valid_steps_recons > 0
            l2_norm_recons = (torch.norm(y_hat_recons[:, :, :, :2] - data.x, p=2, dim=-1) * reg_mask_recons).sum(
                dim=-1
            )  # [F, N]
            best_mode_recons = l2_norm_recons.argmin(dim=0)
            y_hat_best_recons = y_hat_recons[best_mode_recons, torch.arange(data.num_nodes)]

        # pred loss
        soft_target = (
            F.softmax(-l2_norm[:, cls_mask] / valid_steps[cls_mask], dim=0)
            .t()
            .detach()
        )
        cls_loss = self.cls_loss(pi[cls_mask], soft_target)
        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])

        loss = reg_loss + cls_loss 

        reg_loss_recons = 0 
        cls_loss_recons = 0
        if self.mtask:
            # recons loss
            soft_target_recons = (
                F.softmax(-l2_norm_recons[:, cls_mask_recons] / valid_steps_recons[cls_mask_recons], dim=0)
                .t()
                .detach()
            )
            cls_loss_recons = self.cls_loss(pi_recons[cls_mask_recons], soft_target_recons)
            reg_loss_recons = self.reg_loss(y_hat_best_recons[reg_mask_recons], data.x[reg_mask_recons])
            loss = loss + reg_loss_recons + cls_loss_recons

        self.log(
            "train/reg_loss",
            reg_loss.item(),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=1,
            sync_dist=True
        )
        self.log(
            "train/cls_loss", cls_loss.item(), on_step=True, on_epoch=True, batch_size=1,sync_dist=True
        )

        if self.mtask:
            self.log(
                "train/reg_recons_loss",
                reg_loss_recons.item(),
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                batch_size=1,
                sync_dist=True
            )
            self.log(
                "train/cls_recons_loss", cls_loss_recons.item(), on_step=True, on_epoch=True, batch_size=1,sync_dist=True
            )


        return loss

    def on_validation_start(self) -> None:
        if self.validate_mode:
            self.analysis_metrics = AnalysisMetric()
            self.analysis_metrics_recons = AnalysisMetric('_recons')

    def validation_step(self, data, batch_idx):
        y_hat, pi, y_hat_recons, pi_recons = self(data)
        reg_mask = ~data["padding_mask"][:, self.historical_steps :]
        scored_mask = data["x_category"] >= 2
        reg_mask[~scored_mask, :] = False

        l2_norm = (torch.norm(y_hat[:, :, :, :2] - data.y, p=2, dim=-1) * reg_mask).sum(
            dim=-1
        )  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]

        #recons
        if self.mtask:
            reg_mask_recons = ~data["his_pred_padding_mask"][:, :self.historical_steps]
            scored_mask_recons = data["x_category"] >= 2
            reg_mask_recons[~scored_mask_recons, :] = False

            valid_steps_recons = reg_mask_recons.sum(dim=-1)
            cls_mask_recons = valid_steps_recons > 0
            l2_norm_recons = (torch.norm(y_hat_recons[:, :, :, :2] - data.x, p=2, dim=-1) * reg_mask_recons).sum(
                dim=-1
            )  # [F, N]
            best_mode_recons = l2_norm_recons.argmin(dim=0)
            y_hat_best_recons = y_hat_recons[best_mode_recons, torch.arange(data.num_nodes)]
            reg_loss_recons = self.reg_loss(y_hat_best_recons[reg_mask_recons], data.x[reg_mask_recons])
            self.log(
            "val/reg_recons_loss",
            reg_loss_recons.item(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True)



        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        self.log(
            "val/reg_loss",
            reg_loss.item(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True
        )

        y_hat_agent = y_hat[:, data["agent_index"], :, :2]
        pi_hat_agent = pi[data["agent_index"]]
        y_agent = data.y[data["agent_index"]]

        outputs = {"traj": y_hat_agent.transpose(1, 0), "prob": pi_hat_agent}

        metrics = self.val_metrics(outputs, y_agent)

        for k, v in metrics.items():
            metrics[k] = v.item()

        if self.mtask:
            #recons
            y_hat_agent_recons = y_hat_recons[:, data["agent_index"], :, :2]
            pi_hat_agent_recons = pi_recons[data["agent_index"]]
            y_agent_recons = data.x[data["agent_index"]]

            outputs_recons = {"traj": y_hat_agent_recons.transpose(1, 0), "prob": pi_hat_agent_recons}

            metrics_recons = self.val_metrics(outputs_recons, y_agent_recons)
            metrics_recons_ = {}
            for k, v in metrics_recons.items():
                metrics_recons_[k+'_recons'] = v.item()
            self.log_dict(
            metrics_recons_, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

        # print(metrics)
        self.log_dict(
            metrics, prog_bar=True, on_step=False, on_epoch=True, batch_size=1
        )
        if self.validate_mode:
            self.analysis_metrics.log(outputs, y_agent, data["sequence_id"])
            if self.mtask:
                self.analysis_metrics.log(outputs_recons, y_agent_recons, data["sequence_id"])

    def on_validation_end(self) -> None:
        if self.validate_mode:
            print("save analysis metrics")
            self.analysis_metrics.save("./analysis_metrics.pt")
            self.analysis_metrics_recons.save("./analysis_metrics_recons.pt")

    @torch.no_grad()
    def test_step(self, data, batch_idx) -> None:
        traj, prob, _, _ = self(data)

        agent_traj = (
            traj[:, data["agent_index"], :, :2]
            .view(self.num_modes, -1, self.future_steps, 2)
            .transpose(1, 0)
        )  # [N, F, 30, 2]
        agent_pi = prob[data["agent_index"]].view(-1, self.num_modes)  # [N, F]
        rot_mat = (
            data["rotate_mat"][data["agent_index"]].transpose(-1, -2).view(-1, 2, 2)
        )  # [N, 2, 2]
        position = data["positions"][data["agent_index"], self.historical_steps-1, :2].view(-1, 1, 1, 2)
        agent_traj = torch.matmul(agent_traj, rot_mat.unsqueeze(1)) + position

        if hasattr(self, "submission_handler"):
            self.submission_handler.format_data(data, agent_traj, agent_pi)
        else:
            return agent_traj, agent_pi

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
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.GroupNorm,
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
