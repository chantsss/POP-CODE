from itertools import chainfrom itertools import compressfrom pathlib import Pathfrom typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as Ffrom torch_geometric.data import Batchfrom torch_geometric.data import HeteroData 
from losses import MixtureNLLLossfrom losses import NLLLossfrom metrics.qc_metrics import Brierfrom metrics.qc_metrics import MRfrom metrics.qc_metrics import minADEfrom metrics.qc_metrics import minAHEfrom metrics.qc_metrics import minFDEfrom metrics.qc_metrics import minFHEfrom models.qcnet.qcnet_decoder import QCNetDecoderfrom models.qcnet.qcnet_encoder import QCNetEncoder

try:
    from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission
except ImportError:
    ChallengeSubmission = object


class QCNet(pl.LightningModule):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 output_head: bool,
                 num_historical_steps: int,
                 num_future_steps: int,
                 num_modes: int,
                 num_recurrent_steps: int,
                 num_freq_bands: int,
                 num_map_layers: int,
                 num_agent_layers: int,
                 num_dec_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 pl2pl_radius: float,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_t2m_steps: Optional[int],
                 pl2m_radius: float,
                 a2m_radius: float,
                 lr: float,
                 weight_decay: float,
                 T_max: int,
                 submission_dir: str,
                 submission_file_name: str,
                reduce_his_length: bool = False,
                random_his_length: bool = False,
                random_interpolate_zeros: bool = False,
                valid_observation_length: int = 50,
                recons: bool = False,
                use_recons_cross_attention: bool = False,
                num_modes_recons:int = 1,
                recons_w: float = 0.5,
                distill_w: float = 0.5,
                distill: bool = False,
                distill_out: bool = False,
                use_recons_concat: bool = False,
                teacher_ckpt_path: str = '',
                 **kwargs) -> None:
        super(QCNet, self).__init__()
        self.save_hyperparameters()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.num_recurrent_steps = num_recurrent_steps
        self.num_freq_bands = num_freq_bands
        self.num_map_layers = num_map_layers
        self.num_agent_layers = num_agent_layers
        self.num_dec_layers = num_dec_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.pl2pl_radius = pl2pl_radius
        self.time_span = time_span
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_t2m_steps = num_t2m_steps
        self.pl2m_radius = pl2m_radius
        self.a2m_radius = a2m_radius
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.submission_dir = submission_dir
        self.submission_file_name = submission_file_name
        self.recons = recons
        self.use_recons_cross_attention = use_recons_cross_attention
        self.num_modes_recons = num_modes_recons
        self.recons_w = recons_w
        self.distill_w = distill_w
        self.teacher_ckpt_path = teacher_ckpt_path
        self.distill = distill
        self.distill_out = distill_out
        self.use_recons_concat = use_recons_concat

        if self.distill:
            if self.teacher_ckpt_path != '':
                print('loading teacher checkpoint...')
                self.load_teacher()
            else:
                print('Please input teacher checkpoint...')
                


        self.encoder = QCNetEncoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pl2pl_radius=pl2pl_radius,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_map_layers=num_map_layers,
            num_agent_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.decoder = QCNetDecoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            output_head=output_head,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            num_modes=num_modes,
            num_recurrent_steps=num_recurrent_steps,
            num_t2m_steps=num_t2m_steps,
            pl2m_radius=pl2m_radius,
            a2m_radius=a2m_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_dec_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            recons=recons,
            use_recons_cross_attention=use_recons_cross_attention,
            num_modes_recons=num_modes_recons,
            use_recons_concat = use_recons_concat
        )

        self.reg_loss = NLLLoss(component_distribution=['laplace'] * output_dim + ['von_mises'] * output_head,
                                reduction='none')
        self.cls_loss = MixtureNLLLoss(component_distribution=['laplace'] * output_dim + ['von_mises'] * output_head,
                                       reduction='none')

        self.Brier = Brier(max_guesses=6)
        self.minADE = minADE(max_guesses=6)
        self.minAHE = minAHE(max_guesses=6)
        self.minFDE = minFDE(max_guesses=6)
        self.minFHE = minFHE(max_guesses=6)
        self.MR = MR(max_guesses=6)

        if self.recons:
            self.Brier_recons = Brier(max_guesses=self.num_modes_recons)
            self.minADE_recons = minADE(max_guesses=self.num_modes_recons)
            self.minAHE_recons = minAHE(max_guesses=self.num_modes_recons)
            self.minFDE_recons = minFDE(max_guesses=self.num_modes_recons)
            self.minFHE_recons = minFHE(max_guesses=self.num_modes_recons)
            self.MR_recons = MR(max_guesses=self.num_modes_recons)

        self.test_predictions = dict()
        self.reduce_his_length = reduce_his_length
        self.random_his_length = random_his_length
        self.random_interpolate_zeros = random_interpolate_zeros
        self.valid_observation_length = valid_observation_length

    def load_teacher(self):

        self.teacher = QCNet.load_from_checkpoint(checkpoint_path=self.teacher_ckpt_path)
        self.distill_criterion = nn.MSELoss()

    def forward(self, data: HeteroData):

        teacher_feature = None
        if self.distill:
            with torch.no_grad():
                encode_feature = self.teacher.encoder(data)
                distill_out = self.teacher.decoder.distill_forward(data, encode_feature, self.distill_out)
                teacher_feature =  {'scene_enc_x_a': encode_feature['x_a'],
                                    'scene_enc_x_pl': encode_feature['x_pl'],
                                    'm_propose': distill_out['m_propose'],
                                    'm_refine': distill_out['m_refine'],
                                    'distill_out': distill_out}

        data['agent']['predict_mask_his'] = torch.zeros_like(data['agent']['predict_mask'])
        if self.reduce_his_length:
            data = self.drop_his(self.valid_observation_length, self.num_historical_steps, \
                                            self.reduce_his_length, self.random_his_length, \
                                            self.random_interpolate_zeros, data, drop_all=False, drop_for_recons=True)

        scene_enc = self.encoder(data)
        pred = self.decoder(data, scene_enc, self.recons)
        pred['student_feature'] = {'scene_enc_x_a': scene_enc['x_a'],
                                    'scene_enc_x_pl': scene_enc['x_pl'],
                                    'm_propose': pred['m_propose'],
                                    'm_refine': pred['m_refine']}
        pred['teacher_feature'] = teacher_feature

        return pred

    def training_step(self,
                      data,
                      batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
            data['agent']['agent_index'] += data['agent']['ptr'][:-1]
        pred = self(data)
        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        cls_mask = data['agent']['predict_mask'][:, -1]
        if self.recons:
            reg_mask_recons = data['agent']['predict_mask_his'][:, :self.num_historical_steps]

        if self.output_head:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['loc_propose_head'],
                                      pred['scale_propose_pos'][..., :self.output_dim],
                                      pred['conc_propose_head']], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
            if self.recons:
                traj_propose_recons = torch.cat([pred['reconstructions']['loc_propose_pos'][..., :self.output_dim],
                                        pred['reconstructions']['loc_propose_head'],
                                        pred['reconstructions']['scale_propose_pos'][..., :self.output_dim],
                                        pred['reconstructions']['conc_propose_head']], dim=-1)
                # traj_refine_recons = torch.cat([pred['reconstructions']['loc_refine_pos'][..., :self.output_dim],
                #                         pred['reconstructions']['loc_refine_head'],
                #                         pred['reconstructions']['scale_refine_pos'][..., :self.output_dim],
                #                         pred['reconstructions']['conc_refine_head']], dim=-1)

        else:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['scale_propose_pos'][..., :self.output_dim]], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
            if self.recons:
                traj_propose_recons = torch.cat([pred['reconstructions']['loc_propose_pos'][..., :self.output_dim],
                                        pred['reconstructions']['scale_propose_pos'][..., :self.output_dim]], dim=-1)
                # traj_refine_recons = torch.cat([pred['reconstructions']['loc_refine_pos'][..., :self.output_dim],
                #                         pred['reconstructions']['scale_refine_pos'][..., :self.output_dim]], dim=-1)

        pi = pred['pi']
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)
        l2_norm = (torch.norm(traj_propose[..., :self.output_dim] -
                              gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=-1)
        traj_propose_best = traj_propose[torch.arange(traj_propose.size(0)), best_mode]
        traj_refine_best = traj_refine[torch.arange(traj_refine.size(0)), best_mode]
        reg_loss_propose = self.reg_loss(traj_propose_best,
                                         gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_propose = reg_loss_propose.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_propose = reg_loss_propose.mean()
        reg_loss_refine = self.reg_loss(traj_refine_best,
                                        gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_refine = reg_loss_refine.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_refine = reg_loss_refine.mean()
        cls_loss = self.cls_loss(pred=traj_refine[:, :, -1:].detach(),
                                 target=gt[:, -1:, :self.output_dim + self.output_head],
                                 prob=pi,
                                 mask=reg_mask[:, -1:]) * cls_mask
        cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)

        loss = 0
        if self.recons:
            # recons
            gt_recons = torch.cat([data['agent']['target_recons'][..., :self.output_dim], data['agent']['target_recons'][..., -1:]], dim=-1)
            l2_norm_recons = (torch.norm(traj_propose_recons[..., :self.output_dim] -
                                gt_recons[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask_recons.unsqueeze(1)).sum(dim=-1)
            best_mode_recons = l2_norm_recons.argmin(dim=-1)
            traj_propose_best_recons = traj_propose_recons[torch.arange(traj_propose_recons.size(0)), best_mode_recons]
            # traj_refine_best_recons = traj_refine_recons[torch.arange(traj_refine_recons.size(0)), best_mode_recons]
            reg_loss_propose_recons = self.reg_loss(traj_propose_best_recons,
                                            gt_recons[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask_recons
            reg_loss_propose_recons = reg_loss_propose_recons.sum(dim=0) / reg_mask_recons.sum(dim=0).clamp_(min=1)
            reg_loss_propose_recons = reg_loss_propose_recons.mean()
            # reg_loss_refine_recons = self.reg_loss(traj_refine_best_recons,
            #                                 gt_recons[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask_recons
            # reg_loss_refine_recons = reg_loss_refine_recons.sum(dim=0) / reg_mask_recons.sum(dim=0).clamp_(min=1)
            # reg_loss_refine_recons = reg_loss_refine_recons.mean()
            # cls_loss = self.cls_loss(pred=traj_refine_recons[:, :, -1:].detach(),
            #                          target=gt_recons[:, -1:, :self.output_dim + self.output_head],
            #                          prob=pi_recons,
            #                          mask=reg_mask[:, -1:]) * cls_mask
            # cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)

            self.log('train_reg_loss_propose_recons', reg_loss_propose_recons, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
            # self.log('train_reg_loss_refine_recons', train_reg_loss_propose_recons, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
            
            loss = loss + (reg_loss_propose_recons) * self.recons_w

        self.log('train_reg_loss_propose', reg_loss_propose, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_reg_loss_refine', reg_loss_refine, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_cls_loss', cls_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        
        if self.distill:
            distill_loss = 0
            # features = ['scene_enc_x_a', 'scene_enc_x_pl', 'm_propose', 'm_refine']
            features = ['m_propose', 'm_refine']
            student_out = pred['student_feature']
            teacher_out = pred['teacher_feature']
            
            if self.distill_out:
                features = ['loc_propose_pos', 'loc_propose_head', 'scale_propose_pos', 'conc_propose_head',
                            'loc_refine_pos', 'loc_refine_head', 'scale_refine_pos', 'conc_refine_head', 'pi']
                teacher_out = pred['teacher_feature']['distill_out']
                student_out = pred
            
            for feature in features:
                distill_loss += self.distill_criterion(student_out[feature], teacher_out[feature].detach())
                
            loss = loss + distill_loss * self.distill_w
            self.log('train_distill_loss', distill_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)

        loss = loss + reg_loss_propose + reg_loss_refine + cls_loss
        return loss

    def validation_step(self,
                        data,
                        batch_idx):
        with torch.no_grad():
            if isinstance(data, Batch):
                data['agent']['av_index'] += data['agent']['ptr'][:-1]
                data['agent']['agent_index'] += data['agent']['ptr'][:-1]
            pred = self(data)
            reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
            cls_mask = data['agent']['predict_mask'][:, -1]
            if self.recons:
                reg_mask_recons = data['agent']['predict_mask_his'][:, :self.num_historical_steps]

            if self.output_head:
                traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                        pred['loc_propose_head'],
                                        pred['scale_propose_pos'][..., :self.output_dim],
                                        pred['conc_propose_head']], dim=-1)
                traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                        pred['loc_refine_head'],
                                        pred['scale_refine_pos'][..., :self.output_dim],
                                        pred['conc_refine_head']], dim=-1)
                if self.recons:
                    traj_propose_recons = torch.cat([pred['reconstructions']['loc_propose_pos'][..., :self.output_dim],
                                            pred['reconstructions']['loc_propose_head'],
                                            pred['reconstructions']['scale_propose_pos'][..., :self.output_dim],
                                            pred['reconstructions']['conc_propose_head']], dim=-1)
                    # traj_refine_recons = torch.cat([pred['reconstructions']['loc_refine_pos'][..., :self.output_dim],
                    #                         pred['reconstructions']['loc_refine_head'],
                    #                         pred['reconstructions']['scale_refine_pos'][..., :self.output_dim],
                    #                         pred['reconstructions']['conc_refine_head']], dim=-1)

            else:
                traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                        pred['scale_propose_pos'][..., :self.output_dim]], dim=-1)
                traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                        pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
                if self.recons:
                    traj_propose_recons = torch.cat([pred['reconstructions']['loc_propose_pos'][..., :self.output_dim],
                                            pred['reconstructions']['scale_propose_pos'][..., :self.output_dim]], dim=-1)
                    # traj_refine_recons = torch.cat([pred['reconstructions']['loc_refine_pos'][..., :self.output_dim],
                    #                         pred['reconstructions']['scale_refine_pos'][..., :self.output_dim]], dim=-1)

            pi = pred['pi']
            gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)
            l2_norm = (torch.norm(traj_propose[..., :self.output_dim] -
                                gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
            best_mode = l2_norm.argmin(dim=-1)
            traj_propose_best = traj_propose[torch.arange(traj_propose.size(0)), best_mode]
            traj_refine_best = traj_refine[torch.arange(traj_refine.size(0)), best_mode]
            reg_loss_propose = self.reg_loss(traj_propose_best,
                                            gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
            reg_loss_propose = reg_loss_propose.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
            reg_loss_propose = reg_loss_propose.mean()
            reg_loss_refine = self.reg_loss(traj_refine_best,
                                            gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
            reg_loss_refine = reg_loss_refine.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
            reg_loss_refine = reg_loss_refine.mean()
            cls_loss = self.cls_loss(pred=traj_refine[:, :, -1:].detach(),
                                    target=gt[:, -1:, :self.output_dim + self.output_head],
                                    prob=pi,
                                    mask=reg_mask[:, -1:]) * cls_mask
            cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)
            self.log('val_reg_loss_propose', reg_loss_propose, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
                    sync_dist=True)
            self.log('val_reg_loss_refine', reg_loss_refine, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
                    sync_dist=True)
            self.log('val_cls_loss', cls_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

            if self.recons:
                # recons
                gt_recons = torch.cat([data['agent']['target_recons'][..., :self.output_dim], data['agent']['target_recons'][..., -1:]], dim=-1)
                l2_norm_recons = (torch.norm(traj_propose_recons[..., :self.output_dim] -
                                    gt_recons[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask_recons.unsqueeze(1)).sum(dim=-1)
                best_mode_recons = l2_norm_recons.argmin(dim=-1)
                traj_propose_best_recons = traj_propose_recons[torch.arange(traj_propose_recons.size(0)), best_mode_recons]
                # traj_refine_best_recons = traj_refine_recons[torch.arange(traj_refine_recons.size(0)), best_mode_recons]
                reg_loss_propose_recons = self.reg_loss(traj_propose_best_recons,
                                                gt_recons[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask_recons
                reg_loss_propose_recons = reg_loss_propose_recons.sum(dim=0) / reg_mask_recons.sum(dim=0).clamp_(min=1)
                reg_loss_propose_recons = reg_loss_propose_recons.mean()
                # reg_loss_refine_recons = self.reg_loss(traj_refine_best_recons,
                #                                 gt_recons[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask_recons
                # reg_loss_refine_recons = reg_loss_refine_recons.sum(dim=0) / reg_mask_recons.sum(dim=0).clamp_(min=1)
                # reg_loss_refine_recons = reg_loss_refine_recons.mean()
                # cls_loss = self.cls_loss(pred=traj_refine_recons[:, :, -1:].detach(),
                #                          target=gt_recons[:, -1:, :self.output_dim + self.output_head],
                #                          prob=pi_recons,
                #                          mask=reg_mask[:, -1:]) * cls_mask
                # cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)
                self.log('val_reg_loss_propose_recons', reg_loss_propose_recons, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
                        sync_dist=True)

            if self.distill:
                distill_loss = 0
                # features = ['scene_enc_x_a', 'scene_enc_x_pl', 'm_propose', 'm_refine']
                features = ['m_propose', 'm_refine']
                student_out = pred['student_feature']
                teacher_out = pred['teacher_feature']
                
                if self.distill_out:
                    features = ['loc_propose_pos', 'loc_propose_head', 'scale_propose_pos', 'conc_propose_head',
                                'loc_refine_pos', 'loc_refine_head', 'scale_refine_pos', 'conc_refine_head', 'pi']
                    teacher_out = pred['teacher_feature']['distill_out']
                    student_out = pred
                
                for feature in features:
                    distill_loss += self.distill_criterion(student_out[feature], teacher_out[feature].detach())

                # loss = loss + distill_loss * self.distill_w
                self.log('val_distill_loss', distill_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
                        sync_dist=True)


            if self.dataset == 'argoverse_v2':
                eval_mask = data['agent']['category'] == 3
            else:
                raise ValueError('{} is not a valid dataset'.format(self.dataset))
            valid_mask_eval = reg_mask[eval_mask]
            traj_eval = traj_refine[eval_mask, :, :, :self.output_dim + self.output_head]
            if not self.output_head:
                traj_2d_with_start_pos_eval = torch.cat([traj_eval.new_zeros((traj_eval.size(0), self.num_modes, 1, 2)),
                                                        traj_eval[..., :2]], dim=-2)
                motion_vector_eval = traj_2d_with_start_pos_eval[:, :, 1:] - traj_2d_with_start_pos_eval[:, :, :-1]
                head_eval = torch.atan2(motion_vector_eval[..., 1], motion_vector_eval[..., 0])
                traj_eval = torch.cat([traj_eval, head_eval.unsqueeze(-1)], dim=-1)
            pi_eval = F.softmax(pi[eval_mask], dim=-1)
            gt_eval = gt[eval_mask]

            if self.recons:
                valid_mask_eval_recons = reg_mask_recons[eval_mask]
                traj_eval_recons = traj_propose_recons[eval_mask, :, :, :self.output_dim + self.output_head]
                if not self.output_head:
                    traj_2d_with_start_pos_eval_recons = torch.cat([traj_eval_recons.new_zeros((traj_eval_recons.size(0), self.num_modes_recons, 1, 2)),
                                                            traj_eval_recons[..., :2]], dim=-2)
                    motion_vector_eval_recons = traj_2d_with_start_pos_eval_recons[:, :, 1:] - traj_2d_with_start_pos_eval_recons[:, :, :-1]
                    head_eval_recons = torch.atan2(motion_vector_eval_recons[..., 1], motion_vector_eval_recons[..., 0])
                    traj_eval_recons = torch.cat([traj_eval_recons, head_eval_recons.unsqueeze(-1)], dim=-1)
                # pi_eval_recons = pi_eval[:, 0].unsqueeze(-1)
                pi_eval_recons = torch.ones_like(pi_eval[:, :self.num_modes_recons].unsqueeze(-1))
                gt_eval_recons = gt_recons[eval_mask]

                self.Brier_recons.update(pred=traj_eval_recons[..., :self.output_dim], target=gt_eval_recons[..., :self.output_dim], prob=pi_eval_recons,
                                valid_mask=valid_mask_eval_recons)
                self.minADE_recons.update(pred=traj_eval_recons[..., :self.output_dim], target=gt_eval_recons[..., :self.output_dim], prob=pi_eval_recons,
                                valid_mask=valid_mask_eval_recons)
                self.minAHE_recons.update(pred=traj_eval_recons, target=gt_eval_recons, prob=pi_eval_recons, valid_mask=valid_mask_eval_recons)
                self.minFDE_recons.update(pred=traj_eval_recons[..., :self.output_dim], target=gt_eval_recons[..., :self.output_dim], prob=pi_eval_recons,
                                valid_mask=valid_mask_eval_recons)
                self.minFHE_recons.update(pred=traj_eval_recons, target=gt_eval_recons, prob=pi_eval_recons, valid_mask=valid_mask_eval_recons)
                self.MR_recons.update(pred=traj_eval_recons[..., :self.output_dim], target=gt_eval_recons[..., :self.output_dim], prob=pi_eval_recons,
                            valid_mask=valid_mask_eval_recons)
                # self.log('val_Brier', self.Brier, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
                self.log('val_minADE_recons', self.minADE_recons, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval_recons.size(0))
                self.log('val_minAHE_recons', self.minAHE_recons, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval_recons.size(0))
                self.log('val_minFDE_recons', self.minFDE_recons, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval_recons.size(0))
                self.log('val_minFHE_recons', self.minFHE_recons, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval_recons.size(0))
                self.log('val_MR_recons', self.MR_recons, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval_recons.size(0))


            self.Brier.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                            valid_mask=valid_mask_eval)
            self.minADE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                            valid_mask=valid_mask_eval)
            self.minAHE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval)
            self.minFDE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                            valid_mask=valid_mask_eval)
            self.minFHE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval)
            self.MR.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                        valid_mask=valid_mask_eval)
            self.log('val_Brier', self.Brier, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
            self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
            self.log('val_minAHE', self.minAHE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
            self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
            self.log('val_minFHE', self.minFHE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
            self.log('val_MR', self.MR, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))

    def test_step(self,
                  data,
                  batch_idx):
        with torch.no_grad():
            if isinstance(data, Batch):
                data['agent']['av_index'] += data['agent']['ptr'][:-1]
            pred = self(data)
            if self.output_head:
                traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                        pred['loc_refine_head'],
                                        pred['scale_refine_pos'][..., :self.output_dim],
                                        pred['conc_refine_head']], dim=-1)
            else:
                traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                        pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
            pi = pred['pi']
            if self.dataset == 'argoverse_v2':
                eval_mask = data['agent']['category'] == 3
            else:
                raise ValueError('{} is not a valid dataset'.format(self.dataset))
            origin_eval = data['agent']['position'][eval_mask, self.num_historical_steps - 1]
            theta_eval = data['agent']['heading'][eval_mask, self.num_historical_steps - 1]
            cos, sin = theta_eval.cos(), theta_eval.sin()
            rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device=self.device)
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = sin
            rot_mat[:, 1, 0] = -sin
            rot_mat[:, 1, 1] = cos
            traj_eval = torch.matmul(traj_refine[eval_mask, :, :, :2],
                                    rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
            pi_eval = F.softmax(pi[eval_mask], dim=-1)

            traj_eval = traj_eval.cpu().numpy()
            pi_eval = pi_eval.cpu().numpy()
            if self.dataset == 'argoverse_v2':
                eval_id = list(compress(list(chain(*data['agent']['id'])), eval_mask))
                # if isinstance(data, Batch):
                #     for i in range(data.num_graphs):
                #         self.test_predictions[data['scenario_id'][i]] = (pi_eval[i], {eval_id[i]: traj_eval[i]})
                # else:
                #     self.test_predictions[data['scenario_id']] = (pi_eval[0], {eval_id[0]: traj_eval[0]})
                if isinstance(data, Batch):
                    for i in range(data.num_graphs):
                        self.test_predictions[data['scenario_id'][i]] = {eval_id[i]: (traj_eval[i], pi_eval[i])}
                        if data['scenario_id'][i] == '0000b329-f890-4c2b-93f2-7e2413d4ca5b':
                            print('Problem is here')
                else:
                    self.test_predictions[data['scenario_id']] = {eval_id[0]: (traj_eval[0], pi_eval[0])}
            else:
                raise ValueError('{} is not a valid dataset'.format(self.dataset))

    def on_test_end(self):
        if self.dataset == 'argoverse_v2':
            ChallengeSubmission(self.test_predictions).to_parquet(
                Path(self.submission_dir) / f'{self.submission_file_name}.parquet')
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
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
        parser = parent_parser.add_argument_group('QCNet')
        parser.add_argument('--dataset', type=str, required=True)
        parser.add_argument('--input_dim', type=int, default=2)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--output_dim', type=int, default=2)
        parser.add_argument('--output_head', action='store_true')
        parser.add_argument('--num_historical_steps', type=int, required=True)
        parser.add_argument('--num_future_steps', type=int, required=True)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--num_recurrent_steps', type=int, required=True)
        parser.add_argument('--num_freq_bands', type=int, default=64)
        parser.add_argument('--num_map_layers', type=int, default=1)
        parser.add_argument('--num_agent_layers', type=int, default=1)
        parser.add_argument('--num_dec_layers', type=int, default=1)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--head_dim', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--pl2pl_radius', type=float, required=True)
        parser.add_argument('--time_span', type=int, default=None)
        parser.add_argument('--pl2a_radius', type=float, required=True)
        parser.add_argument('--a2a_radius', type=float, required=True)
        parser.add_argument('--num_t2m_steps', type=int, default=None)
        parser.add_argument('--pl2m_radius', type=float, required=True)
        parser.add_argument('--a2m_radius', type=float, required=True)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=64)
        parser.add_argument('--submission_dir', type=str, default='./')
        parser.add_argument('--submission_file_name', type=str, default='submission')


        return parent_parser


    def drop_his(self, valid_observation_length, historical_steps, reduce_his_length, random_his_length, 
                random_interpolate_zeros, data, drop_all=False, drop_for_recons=True):

        if reduce_his_length: 
            valid_observation_length = valid_observation_length
            if random_his_length:
                valid_observation_length = torch.randint(low=1, high=historical_steps+1, size=(1,)).item()
            if random_interpolate_zeros:
                # Make sure agent is visible at least at current frame
                indices = torch.arange(1, historical_steps)
                shuffle = torch.randperm(historical_steps - 1)
                set_zeros = indices[shuffle][:historical_steps - valid_observation_length]
            
            if len(set_zeros) != 0:
                # data_dict = data.to_dict()
                batch_size = len(data)

                if drop_all:
                    indices = data['agent']['agent_index']
                    data['agent']['valid_mask_o'] = data['agent']['valid_mask'].clone()
                    for bz_idx in range(batch_size):
                        agent_idx = indices[bz_idx]

                        # Drop position
                        data['agent']['position'][agent_idx][set_zeros] = data['agent']['position'][agent_idx][set_zeros].fill_(0)
                        # Drop velocity
                        data['agent']['velocity'][agent_idx][set_zeros] = data['agent']['velocity'][agent_idx][set_zeros].fill_(0)
                        # Drop heading
                        data['agent']['heading'][agent_idx][set_zeros] = data['agent']['heading'][agent_idx][set_zeros].fill_(0)

                        # Drop valid mask, used in Encoder
                        data['agent']['valid_mask'][agent_idx][set_zeros] = data['agent']['valid_mask'][agent_idx][set_zeros].fill_(False)
                        data['agent']['valid_mask'][agent_idx, 1: historical_steps] = (
                                data['agent']['valid_mask'][agent_idx, :historical_steps - 1] &
                                data['agent']['valid_mask'][agent_idx, 1: historical_steps])
                        data['agent']['valid_mask'][agent_idx, 0] = data['agent']['valid_mask'][agent_idx, 0].fill_(False)
                    
                    # # predict_mask used in Decoder 
                    # Make predict mask, make the states originally valid but set zeros to 1

                    recons_predict_mask = data['agent']['valid_mask_o'][..., :self.num_historical_steps] & \
                           ~data['agent']['valid_mask'][..., :self.num_historical_steps]

                    # Make sure not predict current frame
                    recons_predict_mask[:, 0] = recons_predict_mask[:, 0].fill_(False)
                    data['agent']['predict_mask_his'][..., :self.num_historical_steps] = recons_predict_mask

                else:

                    # indices = data['agent']['agent_index']
                    data['agent']['valid_mask_o'] = data['agent']['valid_mask'].clone()
                    # for bz_idx in range(batch_size):
                    # agent_idx = indices[bz_idx]

                    # Drop position
                    data['agent']['position'][:, set_zeros] = data['agent']['position'][:, set_zeros].fill_(0)
                    # Drop velocity
                    data['agent']['velocity'][:, set_zeros] = data['agent']['velocity'][:, set_zeros].fill_(0)
                    # Drop heading
                    data['agent']['heading'][:, set_zeros] = data['agent']['heading'][:, set_zeros].fill_(0)

                    # Drop valid mask, used in Encoder
                    data['agent']['valid_mask'][:, set_zeros] = data['agent']['valid_mask'][:, set_zeros].fill_(False)
                    data['agent']['valid_mask'][:, 1: historical_steps] = (
                            data['agent']['valid_mask'][:, :historical_steps - 1] &
                            data['agent']['valid_mask'][:, 1: historical_steps])

                    # Make predict mask, make the states originally valid but set zeros to 1

                    recons_predict_mask = data['agent']['valid_mask_o'][..., :self.num_historical_steps] & \
                           ~data['agent']['valid_mask'][..., :self.num_historical_steps]

                    # Make sure not predict current frame
                    recons_predict_mask[:, 0] = recons_predict_mask[:, 0].fill_(False)
                    data['agent']['predict_mask_his'][..., :self.num_historical_steps] = recons_predict_mask

        return data











