
#
import random
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from losses import LaplaceNLLLossfrom losses import SoftTargetCrossEntropyLossfrom metrics import ADEfrom metrics import FDEfrom metrics import MRHivtfrom models.hivt import GlobalInteractorfrom models.hivt  import LocalEncoderfrom models.hivt  import MLPDecoderfrom utils.utils_hivt import TemporalDatafrom submission.submission_av2 import SubmissionAv2from models.common.utils import drop_his

class HiVT(pl.LightningModule):

    def __init__(self,
                 historical_steps: int,
                 future_steps: int,
                 num_modes: int,
                 rotate: bool,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float,
                 num_temporal_layers: int,
                 num_global_layers: int,
                 local_radius: float,
                 parallel: bool,
                 lr: float,
                 weight_decay: float,
                 T_max: int,
                 reduce_his_length: False,
                 random_his_length: False,
                 random_interpolate_zeros: False,
                 valid_observation_length: 20,
                 submission_handler=SubmissionAv2(submit=False),
                 **kwargs) -> None:
        super(HiVT, self).__init__()
        self.save_hyperparameters(ignore=["submission_handler"])
        self.historical_steps = historical_steps
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.rotate = rotate
        self.parallel = parallel
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max

        self.local_encoder = LocalEncoder(historical_steps=historical_steps,
                                          node_dim=node_dim,
                                          edge_dim=edge_dim,
                                          embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          num_temporal_layers=num_temporal_layers,
                                          local_radius=local_radius,
                                          parallel=parallel)
        self.global_interactor = GlobalInteractor(historical_steps=historical_steps,
                                                  embed_dim=embed_dim,
                                                  edge_dim=edge_dim,
                                                  num_modes=num_modes,
                                                  num_heads=num_heads,
                                                  num_layers=num_global_layers,
                                                  dropout=dropout,
                                                  rotate=rotate)
        self.decoder = MLPDecoder(local_channels=embed_dim,
                                  global_channels=embed_dim,
                                  future_steps=future_steps,
                                  num_modes=num_modes,
                                  uncertain=True)
        self.reg_loss = LaplaceNLLLoss(reduction='mean')
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction='mean')

        self.minADE = ADE()
        self.minFDE = FDE()
        self.minMR = MRHivt()

        # self.reduce_his_length = False
        # self.random_his_length = False
        # self.random_interpolate_zeros = False
        # self.valid_observation_length = valid_observation_length
        self.reduce_his_length = reduce_his_length
        self.random_his_length = random_his_length
        self.random_interpolate_zeros = random_interpolate_zeros
        self.valid_observation_length = valid_observation_length
        self.submission_handler = submission_handler

    # def drop_his(self, data: TemporalData):

    #     if self.reduce_his_length: 
    #         valid_observation_length = self.valid_observation_length
    #         if self.random_his_length:
    #             valid_observation_length = torch.randint(low=1, high=self.historical_steps+1, size=(1,)).item()
    #         if self.random_interpolate_zeros:
    #             # Make sure agent is visible at least at current frame
    #             indices = torch.arange(1, self.historical_steps)
    #             shuffle = torch.randperm(self.historical_steps - 1)
    #             set_zeros = indices[shuffle][:self.historical_steps - valid_observation_length]
    #             # set_zeros.sort()

    #         if len(set_zeros) != 0:
    #             batch_size = len(data)
    #             # print('Start forgetting history in training, remain ', valid_observation_length,' steps.......')
    #             dropped_x = copy.copy(data['x'])
    #             dropped_padding_mask = copy.copy(data['padding_mask'])
    #             agent_index = data['agent_index']
    #             device = dropped_x.device

    #             for bz_idx in range(batch_size):
    #                 agent_idx = agent_index[bz_idx]
    #                 dropped_x[agent_idx][set_zeros] = torch.zeros(self.historical_steps-valid_observation_length, 2, device=device)
    #                 dropped_padding_mask[agent_idx][set_zeros] = torch.ones(self.historical_steps-valid_observation_length, device=device).bool()

    #         else:
    #             dropped_x = data['x']
    #             dropped_padding_mask = data['padding_mask']

    #     return dropped_x, dropped_padding_mask


    def forward(self, data: TemporalData):

        if self.reduce_his_length:
            # print('before drop data.x.sum', data.x.sum())
            # print('before drop data.padding_mask.sum', data.padding_mask.sum())
            data = drop_his(self.valid_observation_length, self.historical_steps, \
                                                        self.reduce_his_length, self.random_his_length, \
                                                        self.random_interpolate_zeros, data)

        if self.rotate:
            rotate_mat = torch.empty(data.num_nodes, 2, 2, device=self.device)
            sin_vals = torch.sin(data['rotate_angles'])
            cos_vals = torch.cos(data['rotate_angles'])
            rotate_mat[:, 0, 0] = cos_vals
            rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals
            rotate_mat[:, 1, 1] = cos_vals
            if data.y is not None:
                data.y = torch.bmm(data.y, rotate_mat)
            data['rotate_mat'] = rotate_mat
        else:
            data['rotate_mat'] = None

        # if self.reduce_his_length:
        #     data.x = dropped_x
        #     data.padding_mask = dropped_padding_mask

        local_embed = self.local_encoder(data=data)
        global_embed = self.global_interactor(data=data, local_embed=local_embed)
        y_hat, pi = self.decoder(local_embed=local_embed, global_embed=global_embed)
        # for x in [y_hat, pi]:
        #     print('sum is', x.sum())       
        return y_hat, pi

    def training_step(self, data, batch_idx):
        # if 'traffic_controls' not in data[0]:
        #     data = self.convert_av2_to_av1_format(data)
        if self.reduce_his_length: 
            valid_observation_length = self.valid_observation_length
            if self.random_his_length:
                valid_observation_length = random.randint(1, self.historical_steps)
            if self.random_interpolate_zeros:
                # Make sure agent is visible at least at current frame
                set_zeros = random.sample(range(1, self.historical_steps), self.historical_steps-valid_observation_length)
                set_zeros.sort()
            batch_size = len(data)
            # print('Start forgetting history in training, remain ', valid_observation_length,' steps.......')
            x = data['x']
            padding_mask = data['padding_mask'] 
            agent_index = data['agent_index']
            device = x.device
            for bz_idx in range(batch_size):
                for set_zero_idx in set_zeros:
                    agent_idx = agent_index[bz_idx]
                    x[agent_idx][set_zeros] = torch.zeros(self.historical_steps-valid_observation_length, 2, device=device)
                    padding_mask[agent_idx][set_zeros] = torch.ones(self.historical_steps-valid_observation_length, device=device).bool()

        y_hat, pi = self(data)
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        valid_steps = reg_mask.sum(dim=-1)
        cls_mask = valid_steps > 0
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        soft_target = F.softmax(-l2_norm[:, cls_mask] / valid_steps[cls_mask], dim=0).t().detach()
        cls_loss = self.cls_loss(pi[cls_mask], soft_target)
        loss = reg_loss + cls_loss
        self.log('train_reg_loss', reg_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        return loss



    def validation_step(self, data, batch_idx):

        if self.reduce_his_length: 
            valid_observation_length = self.valid_observation_length
            if self.random_his_length:
                valid_observation_length = random.randint(1, self.historical_steps)
            if self.random_interpolate_zeros:
                # Make sure agent is visible at least at current frame
                set_zeros = random.sample(range(1, self.historical_steps), self.historical_steps-valid_observation_length)
                set_zeros.sort()
            batch_size = len(data)
            # print('Start forgetting history in training, remain ', valid_observation_length,' steps.......')
            x = data['x']
            padding_mask = data['padding_mask'] 
            agent_index = data['agent_index']
            device = x.device
            for bz_idx in range(batch_size):
                agent_idx = agent_index[bz_idx]
                x[agent_idx][set_zeros] = torch.zeros(self.historical_steps-valid_observation_length, 2, device=device)
                padding_mask[agent_idx][set_zeros] = torch.ones(self.historical_steps-valid_observation_length, device=device).bool()
        data.bos_mask[:, 0] = ~data.padding_mask[:, 0]
        data.bos_mask[:, 1: self.historical_steps] = data.padding_mask[:, : self.historical_steps-1] & ~data.padding_mask[:, 1: self.historical_steps]

        y_hat, pi = self(data)
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        self.log('val_reg_loss', reg_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

        y_hat_agent = y_hat[:, data['agent_index'], :, : 2]
        y_agent = data.y[data['agent_index']]
        fde_agent = torch.norm(y_hat_agent[:, :, -1] - y_agent[:, -1], p=2, dim=-1)
        best_mode_agent = fde_agent.argmin(dim=0)
        y_hat_best_agent = y_hat_agent[best_mode_agent, torch.arange(data.num_graphs)]
        self.minADE.update(y_hat_best_agent, y_agent)
        self.minFDE.update(y_hat_best_agent, y_agent)
        self.minMR.update(y_hat_best_agent, y_agent)
        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0), sync_dist=True)
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0), sync_dist=True)
        self.log('val_minMR', self.minMR, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0), sync_dist=True)

    def test_step(self, data, batch_idx) -> None:
        traj, prob = self(data)

        agent_traj = traj[:, data["agent_index"], :, :2].transpose(1, 0)
        agent_pi = prob[data["agent_index"]]
        rot_mat = data["rotate_mat"][data["agent_index"]].transpose(-1, -2)
        position = (
            data["positions"][data["agent_index"], self.historical_steps - 1, :2].unsqueeze(1).unsqueeze(1)
        )
        agent_traj = torch.matmul(agent_traj, rot_mat.unsqueeze(1)) + position

        self.submission_handler.format_data(data, agent_traj, agent_pi)

    def on_test_end(self) -> None:
        self.submission_handler.generate_submission_file()

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM, nn.GRU)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('HiVT')
        parser.add_argument('--historical_steps', type=int, default=20)
        parser.add_argument('--future_steps', type=int, default=30)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--rotate', type=bool, default=True)
        parser.add_argument('--node_dim', type=int, default=2)
        parser.add_argument('--edge_dim', type=int, default=2)
        parser.add_argument('--embed_dim', type=int, default=64)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--num_temporal_layers', type=int, default=4)
        parser.add_argument('--num_global_layers', type=int, default=3)
        parser.add_argument('--local_radius', type=float, default=50)
        parser.add_argument('--parallel', type=bool, default=False)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=64)
        return parent_parser
