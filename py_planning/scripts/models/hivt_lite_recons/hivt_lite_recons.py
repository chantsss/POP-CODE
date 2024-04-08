
#
# import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchmetrics import MetricCollection
# import copy
from models.common.utils import TemporalData
# from metrics import MR, berierFDE, minADE, minFDE, avgMinADE, avgMinFDE
# from metrics.analysis_metric import AnalysisMetric
# from submission.submission_base import SubmissionBase
from .decoder import GaussianDecoder, MLPDecoderfrom .global_interactor import GlobalInteractorfrom .local_encoder import LocalEncoderfrom .losses.gaussian_nll_loss import GaussianNLLLossfrom .losses.laplace_nll_loss import LaplaceNLLLossfrom .losses.multipath_loss import MultipathLaplaceLoss, MultipathLossfrom .losses.nll_cls_loss import NllClsLossfrom .losses.soft_target_cross_entropy_loss import SoftTargetCrossEntropyLossfrom models.common.utils import drop_hisfrom models.hivt_lite import HiVTLitefrom torch_geometric.data import Batch

class HiVTLiteRcons(nn.Module):
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
        gaussian: bool = False,
        use_multipath_loss: bool = False,
        local_radius: float = 50,
        historical_steps_recons: int = 50,
        future_steps_recons: int = 50,
        num_modes_recons: int = 6,
        rotate_recons: bool = True,
        node_dim_recons: int = 2,
        edge_dim_recons: int = 2,
        dim_recons: int = 128,
        num_heads_recons: int = 8,
        dropout_recons: float = 0.1,
        nat_backbone_recons: bool = False,
        pos_embed_recons: bool = False,
        nll_cls_recons: bool = False,
        num_temporal_layers_recons: int = 4,
        num_global_layers_recons: int = 3,
        gaussian_recons: bool = False,
        use_multipath_loss_recons: bool = False,
        local_radius_recons: float = 50,
        use_cross_attention_by_adding: bool = False,
        use_cross_attention_by_layer: bool = False,
        cross_attention_at_local_embedding: bool = False,
        use_correlation_w: bool = False,
        parallel: bool = False,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        T_max: int = 64,
        reduce_his_length: bool = False,
        random_his_length: bool = False,
        random_interpolate_zeros: bool = False,
        valid_observation_length: int = 20,
        drop_pre_recons: bool = False,
        warm_up_epoch: int = 0,
        only_train_recons: bool = False,
        freeze_recons: bool = False,
        muliti_task: bool = False,
        drop_all_agent: bool = False,
        add_init: bool = False,
        distillation: bool = False,
        distillinit: bool = False,
        distillrefine: bool = False,
        decode_init: bool = True,
        use_cross_attention_by_concat: bool = False,
        feature_w: float = 0.5,
        recons_w: float = 0.5,
        refine_w: float = 0.5,
        init_w: float = 0.5,
        train_init: bool= False,
        train_refine: bool = False,
        train_recons: bool = False,
        teacher_ckpt_path: str = '',
        submission_handler=None,
    ) -> None:
        super(HiVTLiteRcons, self).__init__()
        # self.save_hyperparameters(ignore=["submission_handler"])

        self.historical_steps = historical_steps
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.rotate = rotate
        self.parallel = parallel

        self.historical_steps_recons = historical_steps_recons
        self.future_steps_recons = future_steps_recons
        self.num_modes_recons = num_modes_recons
        self.rotate_recons = rotate_recons
        self.epoch = 0

        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.nll_cls = nll_cls
        self.gaussian = gaussian
        self.use_multipath_loss = use_multipath_loss
        self.validate_mode = False
        self.use_cross_attention_by_adding = use_cross_attention_by_adding
        self.use_cross_attention_by_layer = use_cross_attention_by_layer
        self.cross_attention_at_local_embedding = cross_attention_at_local_embedding
        self.use_correlation_w = use_correlation_w
        self.drop_pre_recons = drop_pre_recons
        self.only_train_recons = only_train_recons
        self.freeze_recons = freeze_recons
        self.freeze_cross = True
        self.warm_up_epoch = warm_up_epoch
        self.muliti_task = muliti_task
        self.drop_all_agent = drop_all_agent
        self.add_init = add_init
        self.distillation = distillation
        self.distillrefine = distillrefine
        self.distillinit = distillinit
        self.feature_w = feature_w
        self.decode_init = decode_init
        self.use_cross_attention_by_concat = use_cross_attention_by_concat
        self.recons_w = recons_w
        self.refine_w = refine_w
        self.init_w = init_w
        self.stopped_distill_once = False

        self.train_init = train_init
        self.train_refine =  train_refine
        self.train_recons = train_recons
        self.simu = False

        if self.distillation:
            if teacher_ckpt_path != '':
                print('loading teacher checkpoint...')
                self.teacher = HiVTLiteRcons.load_from_checkpoint(checkpoint_path=teacher_ckpt_path)
            else:
                print('Please input teacher checkpoint...')

        # metrics = MetricCollection([minADE(), minFDE(), MR(), berierFDE()])
        # ma_metrics = MetricCollection([avgMinADE(), avgMinFDE()]) 
        # ma_metrics = MetricCollection([avgMinADE()]) 
        # self.val_metrics = metrics.clone(prefix="val_")
        # self.val_metrics_ma = ma_metrics.clone(prefix="maval_")

        # if submission_handler:
        #     self.submission_handler: SubmissionBase = submission_handler

        self.reduce_his_length = reduce_his_length
        self.random_his_length = random_his_length
        self.random_interpolate_zeros = random_interpolate_zeros
        self.valid_observation_length = valid_observation_length

        if not self.muliti_task:
            self.local_encoder_recons = LocalEncoder(
                historical_steps=historical_steps_recons,
                node_dim=node_dim_recons,
                edge_dim=edge_dim_recons,
                embed_dim=dim_recons,
                num_heads=num_heads_recons,
                dropout=dropout_recons,
                local_radius=local_radius_recons,
                parallel=parallel,
                # nat_backbone=nat_backbone_recons,
                # pos_embed=pos_embed_recons,
            )

            self.global_interactor_recons = GlobalInteractor(
                historical_steps=historical_steps_recons,
                embed_dim=dim_recons,
                edge_dim=edge_dim_recons,
                num_modes=num_modes_recons,
                num_heads=num_heads_recons,
                num_layers=num_global_layers_recons,
                dropout=dropout_recons,
                rotate=rotate_recons,
            )

        if self.gaussian:
            self.decoder_recons = GaussianDecoder(
                local_channels=dim_recons,
                global_channels=dim_recons,
                future_steps=future_steps_recons,
                num_modes=num_modes_recons,
                use_rho=False,
            )
        else:
            self.decoder_recons = MLPDecoder(
                local_channels=dim_recons,
                global_channels=dim_recons,
                future_steps=historical_steps_recons,
                num_modes=num_modes_recons,
                uncertain=True,
                recons=self.train_recons,
            )

            if self.nll_cls:
                self.reg_loss = LaplaceNLLLoss(reduction="mean")
                self.cls_loss = NllClsLoss(reduction="mean")
            elif self.use_multipath_loss:
                self.multipath_loss = MultipathLaplaceLoss()
            else:
                self.reg_loss = LaplaceNLLLoss(reduction="mean")
                self.cls_loss = SoftTargetCrossEntropyLoss(reduction="mean")

        if not self.only_train_recons:
            if self.use_cross_attention_by_layer:
                self.cross_attention_layer_global = torch.nn.MultiheadAttention(embed_dim = dim, num_heads = num_heads, dropout = dropout)
                if self.cross_attention_at_local_embedding:
                    self.cross_attention_layer_local = torch.nn.MultiheadAttention(embed_dim = dim, num_heads = num_heads, dropout = dropout)


            self.local_encoder = LocalEncoder(
                historical_steps=historical_steps,
                node_dim=node_dim,
                edge_dim=edge_dim,
                embed_dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                local_radius=local_radius,
                parallel=parallel, 
                # nat_backbone=nat_backbone,
                # pos_embed=pos_embed,
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
                if self.decode_init:
                    self.decoder_init = GaussianDecoder(
                        local_channels=dim,
                        global_channels=dim,
                        future_steps=future_steps,
                        num_modes=num_modes,
                        use_rho=False,
                    )
                if not self.muliti_task:
                    if self.use_cross_attention_by_concat:
                        self.decoder_refine = GaussianDecoder(
                            local_channels=dim*2,
                            global_channels=dim*2,
                            future_steps=future_steps,
                            num_modes=num_modes,
                            use_rho=False,
                        )      
                    else:
                        self.decoder_refine = GaussianDecoder(
                            local_channels=dim,
                            global_channels=dim,
                            future_steps=future_steps,
                            num_modes=num_modes,
                            use_rho=False,
                        )
                self.multipath_loss = MultipathLoss(use_rho=False)
                
            else:
                if self.decode_init:
                    self.decoder_init = MLPDecoder(
                        local_channels=dim,
                        global_channels=dim,
                        future_steps=future_steps,
                        num_modes=num_modes,
                        uncertain=True,
                    )
                if not self.muliti_task:
                    if self.use_cross_attention_by_concat:
                        self.decoder_refine = MLPDecoder(
                            local_channels=dim*2,
                            global_channels=dim*2,
                            future_steps=future_steps,
                            num_modes=num_modes,
                            uncertain=True,
                        )
                    else:
                        self.decoder_refine = MLPDecoder(
                            local_channels=dim,
                            global_channels=dim,
                            future_steps=future_steps,
                            num_modes=num_modes,
                            uncertain=True,
                        )

    @staticmethod
    def load_from_checkpoint(checkpoint_path):
        # 加载模型参数
        state_dict = torch.load(checkpoint_path)
        import os 
        parent_dir = os.path.dirname(os.path.dirname(checkpoint_path)) + '/config.yaml'
        import yaml
        # 读取配置文件
        with open(parent_dir, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        model = HiVTLiteRcons(**config['model_args'])
        model.load_state_dict(state_dict['model_state_dict'])
        return model

    def distill_forward(self, data: TemporalData):

        teacher_local_embed = self.teacher.local_encoder(data)
        teacher_global_embed = self.teacher.global_interactor(data=data, local_embed=teacher_local_embed)

        return teacher_local_embed, teacher_global_embed

    def forward(self, data: TemporalData):
        
        device = data['x'].device
         
        if 'is_intersections' not in data:
            data['turn_directions'] = torch.zeros(data['traffic_controls'].shape, device = device)
            data['traffic_controls'] = torch.zeros(data['traffic_controls'].shape, device = device)
            data['is_intersections'] = torch.zeros(data['traffic_controls'].shape, device = device)
        
        if self.rotate:
            rotate_mat = torch.empty(data['num_nodes'], 2, 2, device = device)
            sin_vals = torch.sin(data["rotate_angles"])
            cos_vals = torch.cos(data["rotate_angles"])
            rotate_mat[:, 0, 0] = cos_vals
            rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals
            rotate_mat[:, 1, 1] = cos_vals
            rotate_mat = rotate_mat
            if data['y'] is not None:  # [N, 30, 2]
                data['y'] = torch.bmm(data['y'][..., :2], rotate_mat)
            data["rotate_mat"] = rotate_mat
        else:
            data["rotate_mat"] = None
        
        # data['x'] = data['x'][..., :2]
        # data.positions = data.positions[..., :2]

        if 'x_type' not in data:
            data["x_type"] = torch.zeros(data['num_nodes'], dtype=torch.long)
        
        if 'x_category' not in data:
            data["x_category"] = torch.ones(data['num_nodes'], dtype=torch.long) * 2


        y_hat_init = None
        pi_init = None
        y_hat_recons = None
        pi_recons = None
        y_hat_refine = None
        pi_refine = None
        teacher_local_embed = None
        teacher_global_embed = None
        local_embed = None
        global_embed = None
        global_embed_refine = None
        local_embed_refine = None

        # Teacher model
        if self.distillation and not self.simu:
            with torch.no_grad():
                teacher_local_embed, teacher_global_embed = self.distill_forward(data)
        
        # Drop process
        data['his_pred_padding_mask'] = torch.ones_like(data['padding_mask']).bool()
        if self.reduce_his_length:
            if self.reduce_his_length and self.random_his_length and self.random_interpolate_zeros\
                and 'padding_mask_masked' in data:
                data['padding_mask'] = data['padding_mask_masked']
                data['x'] = data['x_masked']
                
            else:
                data = drop_his(self.valid_observation_length, self.historical_steps, \
                                                self.reduce_his_length, self.random_his_length, \
                                                self.random_interpolate_zeros, data, drop_all=self.drop_all_agent, drop_for_recons=True)


        # Init forward
        local_embed = self.local_encoder(data=data)
        global_embed = self.global_interactor(data=data, local_embed=local_embed)
        if self.train_init:
            if self.decode_init:
                y_hat_init, pi_init = self.decoder_init(local_embed=local_embed, global_embed=global_embed)


        # Recons forward
        if not self.muliti_task:
            if self.freeze_recons:
                with torch.no_grad():
                    local_embed_recons = self.local_encoder_recons(data=data)
                    global_embed_recons = self.global_interactor_recons(data=data, local_embed=local_embed_recons)
                    y_hat_recons, pi_recons = self.decoder_recons(local_embed=local_embed_recons, global_embed=global_embed_recons)
            else:
                if self.train_recons:
                    local_embed_recons = self.local_encoder_recons(data=data)
                    global_embed_recons = self.global_interactor_recons(data=data, local_embed=local_embed_recons)
                    y_hat_recons, pi_recons = self.decoder_recons(local_embed=local_embed_recons, global_embed=global_embed_recons)

        if self.only_train_recons:

            return  {
            'y_hat_init' : y_hat_init,
            'pi_init' : pi_init,
            'y_hat_recons': y_hat_recons,
            'pi_recons' : pi_recons,
            'y_hat_refine' : y_hat_refine,
            'pi_refine' : pi_refine,
            'teacher_local_embed': teacher_local_embed,
            'teacher_global_embed' : teacher_global_embed,
            'local_embed' : local_embed,
            'global_embed' : global_embed,
            'global_embed_refine' : global_embed_refine,
            'local_embed_refine' : local_embed_refine,

            }
        
        # Multi task forward
        if self.muliti_task:
            if self.train_recons:
                y_hat_recons, pi_recons = self.decoder_recons(local_embed=local_embed, global_embed=global_embed)
                pi_recons = torch.ones_like(pi_init, device=pi_init.device)
            if self.distillation:
                return  {
                'y_hat_init' : y_hat_init,
                'pi_init' : pi_init,
                'y_hat_recons': y_hat_recons,
                'pi_recons' : pi_recons,
                'y_hat_refine' : y_hat_refine,
                'pi_refine' : pi_refine,
                'teacher_local_embed': teacher_local_embed,
                'teacher_global_embed' : teacher_global_embed,
                'local_embed' : local_embed,
                'global_embed' : global_embed,
                'global_embed_refine' : global_embed_refine,
                'local_embed_refine' : local_embed_refine,

                }

        # Cross attention
        if self.use_cross_attention_by_layer:
            global_embed_refine, _ = self.cross_attention_layer_global(global_embed, global_embed_recons, global_embed_recons)
            if self.cross_attention_at_local_embedding:
                local_embed_refine, _ = self.cross_attention_layer_local(local_embed, local_embed_recons, local_embed_recons)
            else:
                local_embed_refine = local_embed
            
        elif self.use_cross_attention_by_adding:
            global_embed_refine = global_embed_recons + global_embed
            if self.cross_attention_at_local_embedding:
                local_embed_refine = local_embed_recons + local_embed
            else:
                local_embed_refine = local_embed
        
        elif self.use_cross_attention_by_concat:
            global_embed_refine = torch.cat([global_embed_recons, global_embed], -1)
            local_embed_refine = torch.cat([local_embed_recons, local_embed], -1)
        
        else:
            local_embed_refine = local_embed
            global_embed_refine = global_embed

        # Decoder refinement
        if self.train_refine:
            y_hat_refine, pi_refine = self.decoder_refine(local_embed=local_embed_refine, global_embed=global_embed_refine)
        
        if self.add_init:
            y_hat_refine = y_hat_refine + y_hat_init.detach()      

        if self.simu:
            y = y_hat_init
            pi = pi_init
            agent_traj = (
                y[:, data["agent_index"], :, :2]
                .view(self.num_modes, -1, self.future_steps, 2)
                .transpose(1, 0)
            )  # [N, F, 30, 2]
            agent_pi = pi[data["agent_index"]].view(-1, self.num_modes)  # [N, F]
            rot_mat = (
                data["rotate_mat"][data["agent_index"]].transpose(-1, -2).view(-1, 2, 2)
            )  # [N, 2, 2]
            position = data["positions"][data["agent_index"], self.historical_steps-1, :2].view(-1, 1, 1, 2)
            agent_traj = torch.matmul(agent_traj, rot_mat.unsqueeze(1)) + position

        return  {
                'y_hat_init' : y_hat_init,
                'pi_init' : pi_init,
                'y_hat_recons': y_hat_recons,
                'pi_recons' : pi_recons,
                'y_hat_refine' : y_hat_refine,
                'pi_refine' : pi_refine,
                'teacher_local_embed': teacher_local_embed,
                'teacher_global_embed' : teacher_global_embed,
                'local_embed' : local_embed,
                'global_embed' : global_embed,
                'global_embed_refine' : global_embed_refine,
                'local_embed_refine' : local_embed_refine,
                'traj': agent_traj if self.simu else None,
                'probs': agent_pi if self.simu else None,
                }

    def print_weight(self):
        param_list = []
        for name, param in self.named_parameters():
            param_list.append(param.sum())
            print(str(name)+'.sum is', param.sum())
        print('param sum is', sum(param_list))

    # def on_train_epoch_start(self):

        # if self.freeze_recons:
        #     self.local_encoder_recons.eval()
        #     self.global_interactor_recons.eval()
        #     self.decoder_recons.eval()
        
    #     print('on_train_epoch_start ......')
    #     if self.stopped_distill_once:
    #         print('on_train_epoch_start pass ......')
    #         pass
    #     else:
    #         len_dataset = len(self.trainer.datamodule.train_dataloader())
    #         if self.global_step >= (5 * len_dataset/4):
    #             print('len_dataset is......', len_dataset)
    #             print('global step now is......', self.global_step)
    #             print('Stop distillation now......')
    #             self.distillation = False
    #             self.teacher = torch.nn.Sequential()
    #             self.stopped_distill_once = True
    #         else:
    #             pass

    #     if self.only_train_recons:
    #         self.eval()
    #         self.local_encoder_recons.train()
    #         self.global_interactor_recons.train()
    #         self.decoder_recons.train()

    #     else:
    #         self.train()

    #         if self.global_step >= self.warm_up_epoch * len(self.trainer.datamodule.train_dataloader()):
    #             self.freeze_cross = False

    #         if self.freeze_cross:
    #             self.decoder_refine.eval()
    #             if self.use_cross_attention_by_layer:
    #                 self.cross_attention_layer_global.eval()
    #                 if self.cross_attention_at_local_embedding:
    #                     self.cross_attention_layer_local.eval()




    # def on_train_batch_start(self, batch, batch_idx):

    #     if not self.only_train_recons:
    #         if self.freeze_cross:
    #             self.decoder_refine.eval()
    #             if self.use_cross_attention_by_layer:
    #                 self.cross_attention_layer_global.eval()
    #                 if self.cross_attention_at_local_embedding:
    #                     self.cross_attention_layer_local.eval()
    #         else:
    #             self.decoder_refine.train()
    #             if self.use_cross_attention_by_layer:
    #                 self.cross_attention_layer_global.train()
    #                 if self.cross_attention_at_local_embedding:
    #                     self.cross_attention_layer_local.train()

    def training_step(self, data, predictions):
        # laplace nll loss
        # if self.distillation:
        #     y_hat, pi, y_hat_recons, pi_recons, y_hat_refine, pi_refine, teacher_local_embed, teacher_global_embed, local_embed, global_embed = self(data)
        # else:
        #     y_hat, pi, y_hat_recons, pi_recons, y_hat_refine, pi_refine = self(data)
        [y_hat, pi, y_hat_recons, pi_recons,  y_hat_refine, pi_refine, \
         teacher_local_embed, teacher_global_embed, local_embed, \
            global_embed, local_embed_refine, global_embed_refine, _, _] = [predictions[key] for key in predictions.keys()]
                    
        if self.distillrefine:
            local_embed = local_embed_refine
            global_embed = global_embed_refine

        self.device = data['x'].device
        batch_size = len(data)
        reg_mask = ~data["padding_mask"][:, self.historical_steps:]
        scored_mask = data["x_category"] >= 2
        reg_mask[~scored_mask, :] = False
        valid_steps = reg_mask.sum(dim=-1)
        cls_mask = valid_steps > 0

        if y_hat != None:
            # init predictions
            l2_norm = (torch.norm(y_hat[:, :, :, :2] - data['y'], p=2, dim=-1) * reg_mask).sum(
                dim=-1
            )  # [F, N]
            best_mode = l2_norm.argmin(dim=0)
            y_hat_best = y_hat[best_mode, torch.arange(data['num_nodes'])]

        if y_hat_refine != None:
            # refine predictions
            l2_norm_refine = (torch.norm(y_hat_refine[:, :, :, :2] - data['y'], p=2, dim=-1) * reg_mask).sum(
                dim=-1)  # [F, N]
            best_mode_refine = l2_norm_refine.argmin(dim=0)
            y_hat_best_refine = y_hat_refine[best_mode_refine, torch.arange(data['num_nodes'])]

        # recons predictions
        if y_hat_recons!= None:
            reg_mask_recons = ~data["his_pred_padding_mask"][:, :self.historical_steps_recons]
            reg_mask_recons[~scored_mask, :] = False
            valid_steps_recons = reg_mask_recons.sum(dim=-1)
            cls_mask_recons = valid_steps_recons > 0
            l2_norm_recons = (torch.norm(y_hat_recons[:, :, :, :2] - data['x'], p=2, dim=-1) \
                            * reg_mask_recons).sum(dim=-1)  # [F, N]
            best_mode_recons = l2_norm_recons.argmin(dim=0)
            y_hat_best_recons = y_hat_recons[best_mode_recons, torch.arange(data['num_nodes'])]

        # if self.nll_cls:
        #     if y_hat != None:
        #         # init loss
        #         pis = (pi.transpose(1, 0).unsqueeze(-1).repeat(1, 1, self.future_steps)[:, reg_mask])
        #         preds = y_hat[:, reg_mask].detach()  # [F, n, 4]
        #         cls_loss = self.cls_loss(preds, pis, data['y'][reg_mask])
        #         reg_loss = self.reg_loss(y_hat_best[reg_mask], data['y'][reg_mask])
        #     else:
        #         cls_loss = torch.zeros(1, device=self.device)
        #         reg_loss = torch.zeros(1, device=self.device)

        #     if y_hat_refine != None:
        #         # refine loss
        #         pis_refine = (pi_refine.transpose(1, 0).unsqueeze(-1).repeat(1, 1, self.future_steps)[:, reg_mask])
        #         preds_refine = y_hat_refine[:, reg_mask].detach()  # [F, n, 4]
        #         cls_loss_refine = self.cls_loss(preds_refine, pis_refine, data['y'][reg_mask])
        #         reg_loss_refine = self.reg_loss(y_hat_best_refine[reg_mask], data['y'][reg_mask])
        #     else:
        #         cls_loss_refine = torch.zeros(1, device=self.device)
        #         reg_loss_refine = torch.zeros(1, device=self.device)

        #     # recons loss
        #     if y_hat_recons!= None:
        #         pis_recons = (pi_recons.transpose(1, 0).unsqueeze(-1).repeat(1, 1, self.future_steps_recons)[:, reg_mask_recons])
        #         preds_recons = y_hat_recons[:, reg_mask_recons].detach()  # [F, n, 4]
        #         cls_loss_recons = self.cls_loss(preds_recons, pis_recons, data['x'][reg_mask_recons])
        #         reg_loss_recons = self.reg_loss(y_hat_best_recons[reg_mask_recons], data['x'][reg_mask_recons])
        
        # elif self.use_multipath_loss:
        #     if y_hat != None:
        #         # init loss
        #         log_pi = F.log_softmax(pi, dim=-1)
        #         log_pi_best = (
        #             log_pi[torch.arange(data['num_nodes']), best_mode]
        #             .unsqueeze(-1)
        #             .repeat(1, self.future_steps)
        #         )  # [N, 30]
        #         loss, reg_loss, cls_loss = self.multipath_loss(
        #             y_hat_best[reg_mask], log_pi_best[reg_mask], data['y'][reg_mask]
        #         )
        #     else:
        #         cls_loss = torch.zeros(1, device=self.device)
        #         reg_loss = torch.zeros(1, device=self.device)

        #     if y_hat_refine != None:
        #         # refine loss
        #         log_pi_refine = F.log_softmax(pi_refine, dim=-1)
        #         log_pi_best_refine = (
        #             log_pi_refine[torch.arange(data['num_nodes']), best_mode_refine]
        #             .unsqueeze(-1)
        #             .repeat(1, self.future_steps)
        #         )  # [N, 30]
        #         loss_refine, reg_loss_refine, cls_loss_refine = self.multipath_loss(
        #             y_hat_best_refine[reg_mask], log_pi_best_refine[reg_mask], data['y'][reg_mask]
        #         )
            
        #     else:
        #         cls_loss_refine = torch.zeros(1, device=self.device)
        #         reg_loss_refine = torch.zeros(1, device=self.device)


        #     # recons loss
        #     if y_hat_recons!= None:
        #         log_pi_recons = F.log_softmax(pi_recons, dim=-1)
        #         log_pi_best_recons = (
        #             log_pi_recons[torch.arange(data['num_nodes']), best_mode_recons]
        #             .unsqueeze(-1)
        #             .repeat(1, self.future_steps_recons)
        #         )  # [N, 30]
        #         loss_recons, reg_loss_recons, cls_loss_recons = self.multipath_loss(
        #             y_hat_best_recons[reg_mask_recons], log_pi_best_recons[reg_mask_recons], data['x'][reg_mask_recons]
        #         )


        # else:
        if y_hat != None:
            # init loss 
            soft_target = (
                F.softmax(-l2_norm[:, cls_mask] / valid_steps[cls_mask], dim=0)
                .t()
                .detach()
            )
            cls_loss = self.cls_loss(pi[cls_mask], soft_target)
            reg_loss = self.reg_loss(y_hat_best[reg_mask], data['y'][reg_mask])
        else:
            cls_loss = torch.zeros(1, device=self.device)
            reg_loss = torch.zeros(1, device=self.device)

        if y_hat_refine != None:
            # refine loss
            soft_target_refine = (
                F.softmax(-l2_norm_refine[:, cls_mask] / valid_steps[cls_mask], dim=0)
                .t()
                .detach()
            )
            cls_loss_refine = self.cls_loss(pi_refine[cls_mask], soft_target_refine)
            reg_loss_refine = self.reg_loss(y_hat_best_refine[reg_mask], data['y'][reg_mask])
        
        else:
            cls_loss_refine = torch.zeros(1, device=self.device)
            reg_loss_refine = torch.zeros(1, device=self.device)


        # recons loss
        recons_ade = torch.zeros(1, device=self.device)
        if y_hat_recons!= None:
            soft_target_recons = (
                F.softmax(-l2_norm_recons[:, cls_mask_recons] / valid_steps_recons[cls_mask_recons], dim=0)
                .t()
                .detach()
            )
            # cls_loss_recons = self.cls_loss(pi_recons[cls_mask_recons], soft_target_recons)
            cls_loss_recons = torch.zeros(1, device=self.device)
            reg_loss_recons = self.reg_loss(y_hat_best_recons[reg_mask_recons], data['x'][reg_mask_recons])
            
            recons_ade = l2_norm_recons[best_mode_recons, torch.arange(data['num_nodes'])].mean()
            

        else:
            cls_loss_recons = torch.zeros(1, device=self.device)
            reg_loss_recons = torch.zeros(1, device=self.device)

        
        w = (recons_ade + 1).item() if self.use_correlation_w else 1

        loss = w * (self.init_w * (reg_loss + cls_loss) + self.refine_w * (reg_loss_refine + cls_loss_refine)) \
            + self.recons_w * (reg_loss_recons + cls_loss_recons)

        feature_loss = torch.zeros(1, device=self.device)
        if self.distillation:
            feature_loss_local = nn.MSELoss()(local_embed, teacher_local_embed.detach())
            feature_loss_global = nn.MSELoss()(global_embed, teacher_global_embed.detach())
            feature_loss = feature_loss_global + feature_loss_local
        
        loss = loss + self.feature_w * feature_loss

        # self.log("train/reg_loss", reg_loss.item(), prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)
        # self.log("train/cls_loss", cls_loss.item(), on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)
        # self.log("train/reg_loss_refine", reg_loss_refine.item(), prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)
        # self.log("train/cls_loss_refine", cls_loss_refine.item(), on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)
        # self.log("train/reg_loss_recons", reg_loss_recons.item(), prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)
        # self.log("train/cls_loss_recons", cls_loss_recons.item(), on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)
        # self.log("train/loss", loss.item(), prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)
        # self.log("train/feature_loss", feature_loss.item(), prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)

        return loss, reg_loss, cls_loss, reg_loss_refine, cls_loss_refine, reg_loss_recons, cls_loss_recons, feature_loss

    # def on_validation_start(self) -> None:
    #     if self.validate_mode:
    #         self.analysis_metrics = AnalysisMetric()
    #         self.analysis_metrics_refine = AnalysisMetric('_refine')
    #         self.analysis_metrics_recons = AnalysisMetric('_recons')

    def validation_step(self, data, batch_idx):

        if self.distillation:
            y_hat, pi, y_hat_recons, pi_recons, y_hat_refine, pi_refine, teacher_local_embed, teacher_global_embed, local_embed, global_embed = self(data)
        else:
            y_hat, pi, y_hat_recons, pi_recons, y_hat_refine, pi_refine = self(data)

        batch_size = len(data)
        reg_mask = ~data["padding_mask"][:, self.historical_steps :]
        scored_mask = data["x_category"] >= 2
        reg_mask[~scored_mask, :] = False

        if y_hat != None:
            # init predictions
            l2_norm = (torch.norm(y_hat[:, :, :, :2] - data['y'], p=2, dim=-1) * reg_mask).sum(
                dim=-1
            )  # [F, N]
            best_mode = l2_norm.argmin(dim=0)
            y_hat_best = y_hat[best_mode, torch.arange(data['num_nodes'])]

        if y_hat_refine != None:

            # refine predictions
            l2_norm_refine = (torch.norm(y_hat_refine[:, :, :, :2] - data['y'], p=2, dim=-1) * reg_mask).sum(
                dim=-1)  # [F, N]
            best_mode_refine = l2_norm_refine.argmin(dim=0)
            y_hat_best_refine = y_hat_refine[best_mode_refine, torch.arange(data['num_nodes'])]

        # recons predictions
        reg_mask_recons = ~data["padding_mask"][:, :self.historical_steps_recons]
        reg_mask_recons[~scored_mask, :] = False
        l2_norm_recons = (torch.norm(y_hat_recons[:, :, :, :2] - data['x'], p=2, dim=-1) \
                          * reg_mask_recons).sum(dim=-1)  # [F, N]
        best_mode_recons = l2_norm_recons.argmin(dim=0)
        y_hat_best_recons = y_hat_recons[best_mode_recons, torch.arange(data['num_nodes'])]

        # if self.use_multipath_loss:
        #     if y_hat != None:
        #         # init loss
        #         log_pi = F.log_softmax(pi, dim=-1)
        #         log_pi_best = (
        #             log_pi[torch.arange(data['num_nodes']), best_mode]
        #             .unsqueeze(-1)
        #             .repeat(1, self.future_steps)
        #         )  # [N, 30]
        #         loss, reg_loss, cls_loss = self.multipath_loss(
        #             y_hat_best[reg_mask], log_pi_best[reg_mask], data['y'][reg_mask]
        #         )
        #     else:
        #         cls_loss = torch.zeros(1, device=self.device)
        #         reg_loss = torch.zeros(1, device=self.device)

        #     if y_hat_refine != None:
        #         # refine loss
        #         log_pi_refine = F.log_softmax(pi_refine, dim=-1)
        #         log_pi_best_refine = (
        #             log_pi_refine[torch.arange(data['num_nodes']), best_mode_refine]
        #             .unsqueeze(-1)
        #             .repeat(1, self.future_steps)
        #         )  # [N, 30]
        #         loss_refine, reg_loss_refine, cls_loss_refine = self.multipath_loss(
        #             y_hat_best_refine[reg_mask], log_pi_best_refine[reg_mask], data['y'][reg_mask]
        #         )

        #     else:
        #         cls_loss_refine = torch.zeros(1, device=self.device)
        #         reg_loss_refine = torch.zeros(1, device=self.device)

        #     # recons loss
        #     log_pi_recons = F.log_softmax(pi_recons, dim=-1)
        #     log_pi_best_recons = (
        #         log_pi_recons[torch.arange(data['num_nodes']), best_mode_recons]
        #         .unsqueeze(-1)
        #         .repeat(1, self.future_steps_recons)
        #     )  # [N, 30]
        #     loss_recons, reg_loss_recons, cls_loss_recons = self.multipath_loss(
        #         y_hat_best_recons[reg_mask_recons], log_pi_best_recons[reg_mask_recons], data['x'][reg_mask_recons]
        #     )

        #     recons_ade = l2_norm_recons[best_mode_recons, torch.arange(data['num_nodes'])].mean()
        #     w = (recons_ade + 1).item() if self.use_correlation_w else 1
            
        #     if self.num_modes_recons == 1:
        #         loss = w * (self.init_w * (reg_loss + cls_loss) + self.refine_w * (reg_loss_refine + cls_loss_refine)) \
        #             + self.recons_w * reg_loss_recons                 
        #     else:
        #         loss = w * (self.init_w * (reg_loss + cls_loss) + self.refine_w * (reg_loss_refine + cls_loss_refine)) \
        #             + self.recons_w * (reg_loss_recons + cls_loss_recons) 

        #     feature_loss = torch.zeros(1, device=self.device)
        #     if self.distillation:
        #         feature_loss_local = nn.MSELoss()(local_embed, teacher_local_embed.detach())
        #         feature_loss_global = nn.MSELoss()(global_embed, teacher_global_embed.detach())
        #         feature_loss = feature_loss_global + feature_loss_local
            
        #     loss = loss + self.feature_w * feature_loss

        #     self.log("val/reg_loss", reg_loss.item(), prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        #     self.log("val/cls_loss", cls_loss.item(), on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        #     self.log("val/reg_loss_refine", reg_loss_refine.item(), prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        #     self.log("val/cls_loss_refine", cls_loss_refine.item(), on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        #     self.log("val/reg_loss_recons", reg_loss_recons.item(), prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        #     self.log("val/cls_loss_recons", cls_loss_recons.item(), on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        #     self.log("val/loss", loss.item(), on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        #     self.log("val/feature_loss", feature_loss.item(), on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)

        # else:
        if y_hat!=None:
            reg_loss = self.reg_loss(y_hat_best[reg_mask], data['y'][reg_mask])
            self.log("val/reg_loss", reg_loss.item(), prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        
        if y_hat_refine!=None:
            reg_loss_refine = self.reg_loss(y_hat_best_refine[reg_mask], data['y'][reg_mask])
            self.log("val/reg_loss_refine", reg_loss_refine.item(), prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)

        reg_loss_recons = self.reg_loss(y_hat_best_recons[reg_mask_recons], data['x'][reg_mask_recons])
        self.log("val/reg_loss_recons", reg_loss_recons.item(), prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)

        feature_loss = torch.zeros(1, device=self.device)
        if self.distillation:
            feature_loss_local = nn.MSELoss()(local_embed, teacher_local_embed.detach())
            feature_loss_global = nn.MSELoss()(global_embed, teacher_global_embed.detach())
            feature_loss = feature_loss_global + feature_loss_local
        self.log("val/feature_loss", feature_loss.item(), prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)

        if y_hat != None:
            # init prediction metrics
            y_hat_agent = y_hat[:, data["agent_index"], :, :2]
            pi_hat_agent = pi[data["agent_index"]]
            y_agent = data['y'][data["agent_index"]]
            outputs = {"traj": y_hat_agent.transpose(1, 0), "prob": pi_hat_agent}
            metrics = self.val_metrics(outputs, y_agent)
            metrics_ma = self.val_metrics_ma(y_hat, pi, data['y'], reg_mask)
            for k, v in metrics.items():
                metrics[k] = v.item()
            for k, v in metrics_ma.items():
                metrics_ma[k] = v.item()

        if y_hat_refine != None:
            # refine prediction metrics
            y_hat_agent_refine = y_hat_refine[:, data["agent_index"], :, :2]
            pi_hat_agent_refine = pi_refine[data["agent_index"]]
            y_agent_refine = data['y'][data["agent_index"]]
            outputs_refine = {"traj": y_hat_agent_refine.transpose(1, 0), "prob": pi_hat_agent_refine}
            metrics_refine = self.val_metrics(outputs_refine, y_agent_refine)
            metrics_refine_ma = self.val_metrics_ma(y_hat_refine, pi_refine, data['y'], reg_mask)
            metrics_refine_ = {}
            metrics_refine_ma_ = {}
            for k, v in metrics_refine.items():
                metrics_refine_[k+'_refine'] = v.item()
            for k, v in metrics_refine_ma.items():
                metrics_refine_ma_[k+'_refine'] = v.item()


        # recons prediction metrics
        y_hat_agent_recons = y_hat_recons[:, data["agent_index"], :, :2]
        pi_recons = torch.ones_like(pi, device=pi.device)
        pi_hat_agent_recons = pi_recons[data["agent_index"]]
        y_agent_recons = data['x'][data["agent_index"]]
        outputs_recons = {"traj": y_hat_agent_recons.transpose(1, 0), "prob": pi_hat_agent_recons}
        metrics_recons = self.val_metrics(outputs_recons, y_agent_recons)
        metrics_recons_ = {}
        for k, v in metrics_recons.items():
            metrics_recons_[k+'_recons'] = v.item()

        # print(metrics)
        if y_hat!=None:
            self.log_dict(
                metrics, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True
            )
            self.log_dict(
                metrics_ma, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True
            )
        if y_hat_refine != None:
            self.log_dict(
                metrics_refine_, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True
            )
            self.log_dict(
                metrics_refine_ma_, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True
            )
        self.log_dict(
            metrics_recons_, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True
        )

        if self.validate_mode:
            if y_hat!=None:
                self.analysis_metrics.log(outputs, y_agent, data["sequence_id"])
            if y_hat_refine != None:
                self.analysis_metrics_refine.log(outputs_refine, y_agent_refine, data["sequence_id"])
            self.analysis_metrics_recons.log(outputs_recons, y_agent_recons, data["sequence_id"])

    def on_validation_end(self) -> None:
        if self.validate_mode:
            print("save analysis metrics")
            if not self.only_train_recons:
                self.analysis_metrics.save("./analysis_metrics.pt")
                self.analysis_metrics_refine.save("./analysis_metrics_refine.pt")
            self.analysis_metrics_recons.save("./analysis_metrics_recons.pt")

    @torch.no_grad()
    def test_step(self, data, batch_idx) -> None:
        if self.distillation:
            y_hat, pi, y_hat_recons, pi_recons, y_hat_refine, pi_refine, _, _, _, _ = self(data)
        else:
            y_hat, pi_hat, _, _, y_hat_refine, pi_refine = self(data)

        if y_hat_refine != None:
            y = y_hat_refine
            pi = pi_refine
        else:
            y = y_hat
            pi = pi_hat

        agent_traj = (
            y[:, data["agent_index"], :, :2]
            .view(self.num_modes, -1, self.future_steps, 2)
            .transpose(1, 0)
        )  # [N, F, 30, 2]
        agent_pi = pi[data["agent_index"]].view(-1, self.num_modes)  # [N, F]
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


def collate_fn(batch):

    # td_batch_list = [TemporalData(**bt) for bt in batch]
    batch_data = Batch.from_data_list(batch)
    batch_data['turn_directions'] = torch.zeros(batch_data['traffic_controls'].shape)
    batch_data['traffic_controls'] = torch.zeros(batch_data['traffic_controls'].shape)
    batch_data['is_intersections'] = torch.zeros(batch_data['traffic_controls'].shape)

    return batch_data