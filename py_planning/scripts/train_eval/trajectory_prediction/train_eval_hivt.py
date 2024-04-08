import torch.optim
import torch.utils.data as torch_datafrom typing import Dictfrom train_eval.trajectory_prediction.initialization import initialize_prediction_network, initialize_metric, initialize_dataset
import torch
import time
import math
import os
import train_eval.utils as train_eval_ufrom models.hivt_lite_recons import HiVTLiteRconsfrom torch_geometric.data import Batchfrom models.common.utils import TemporalData
import numpy as np
import random 

# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def collate_fn(batch):

    # td_batch_list = [TemporalData(**bt) for bt in batch]
    batch_data = Batch.from_data_list(batch)
    batch_data['turn_directions'] = torch.zeros(batch_data['traffic_controls'].shape)
    batch_data['traffic_controls'] = torch.zeros(batch_data['traffic_controls'].shape)
    batch_data['is_intersections'] = torch.zeros(batch_data['traffic_controls'].shape)

    return batch_data


def convert_module_to_double(module):
    for child in module.children():
        convert_module_to_double(child)
        for child_1 in child.children():
            convert_module_to_double(child_1)
            for child_2 in child_1.children():
                convert_module_to_double(child_2)
                for child_3 in child_2.children():
                    convert_module_to_double(child_3)
    if hasattr(module, 'weight') and module.weight is not None:
        module.weight = torch.nn.Parameter(module.weight.double())
    if hasattr(module, 'bias') and module.bias is not None:
        module.bias = torch.nn.Parameter(module.bias.double())



class TrainAndEvaluatorHiVT:
    """
    Train/Evaler class for running train-val loops
    """
    def __init__(self, cfg: Dict, checkpoint_path=None, just_weights=False, writer=None, ld=False):
        """
        Initialize trainer object
        :param cfg: Configuration parameters
            : key dataset: indicates which dataset to process
        :param checkpoint_path: Path to checkpoint with trained weights
        :param just_weights: Load just weights from checkpoint
        :param writer: Tensorboard summary writer
        :param writer: ld for distill
        """
        # Initialize datasets:
        train_set = initialize_dataset(cfg['dataset'], 
            encoder_type=cfg['model_type'], version='train')
        val_set = initialize_dataset(cfg['dataset'], 
            encoder_type=cfg['model_type'], version='eval')
        datasets = {'train': train_set, 'eval': val_set}

        # Initialize dataloaders
        print("Initialize with num_workers={}, batch_size={}.".format(cfg['num_workers'], cfg['batch_size']))
        self.tr_dl = torch_data.DataLoader(datasets['train'], 
                                           cfg['batch_size'], shuffle=True,
                                           num_workers=cfg['num_workers'],
                                        #    prefetch_factor=2 if cfg['num_workers'] > 1 else None,
                                           pin_memory=True,
                                           collate_fn=collate_fn)
        self.val_dl = torch_data.DataLoader(datasets['eval'], 
                                            cfg['batch_size'], shuffle=False,
                                            num_workers=cfg['num_workers'],
                                            # prefetch_factor=2 if cfg['num_workers'] > 1 else None,
                                            pin_memory=True,
                                            collate_fn=collate_fn)

        # Initialize model
        self.model = HiVTLiteRcons(**cfg['model_args'])
        self.model.float().to(device)
        


        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg['optim_args']['lr'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=cfg['optim_args']['scheduler_step'],
            gamma=cfg['optim_args']['scheduler_gamma'])

        # Initialize epochs
        self.current_epoch = 0

        # Initialize losses
        self.losses = [initialize_metric(cfg['losses'][i], cfg['loss_args'][i]) for i in range(len(cfg['losses']))]
        self.loss_weights = cfg['loss_weights']

        # Initialize metrics
        self.train_metrics_init = [initialize_metric(cfg['tr_metrics'][i], cfg['tr_metric_args'][i])
                              for i in range(len(cfg['tr_metrics']))]
        self.val_metrics_init = [initialize_metric(cfg['val_metrics'][i], cfg['val_metric_args'][i])
                            for i in range(len(cfg['val_metrics']))]
        self.train_metrics_refine = [initialize_metric(cfg['tr_metrics'][i], cfg['tr_metric_args'][i])
                              for i in range(len(cfg['tr_metrics']))]
        self.val_metrics_refine = [initialize_metric(cfg['val_metrics'][i], cfg['val_metric_args'][i])
                            for i in range(len(cfg['val_metrics']))]
        self.train_metrics_recons = [initialize_metric(cfg['tr_metrics'][i], cfg['tr_metric_args'][i])
                              for i in range(len(cfg['tr_metrics']))]
        self.val_metrics_recons = [initialize_metric(cfg['val_metrics'][i], cfg['val_metric_args'][i])
                            for i in range(len(cfg['val_metrics']))]
        
        self.val_metric_init = math.inf
        self.min_val_metric_init = math.inf
        self.val_metric_refine = math.inf
        self.min_val_metric_refine = math.inf
        self.val_metric_recons = math.inf
        self.min_val_metric_recons = math.inf
        self.save_with_refine = cfg['optim_args']['save_with_refine']

        self.train_metrics = {
            'init': self.train_metrics_init,
            'refine': self.train_metrics_refine,
            'recons': self.train_metrics_recons,
        } 

        self.val_metrics = {
            'init': self.val_metrics_init,
            'refine': self.val_metrics_refine,
            'recons': self.val_metrics_recons,
        } 

        # Print metrics after these many minibatches to keep track of training
        self.log_period = max(len(self.tr_dl)//cfg['log_freq'], 1)

        # Initialize tensorboard writer
        self.writer = writer
        self.tb_iters = 0

        # Load checkpoint if checkpoint path is provided
        if checkpoint_path is not None:
            print()
            print("Loading checkpoint from " + checkpoint_path + " ...", end=" ")
            if ld:
                self.load_checkpoint_for_distill(checkpoint_path, just_weights=just_weights)
            else:
                self.load_checkpoint(checkpoint_path, just_weights=just_weights)
            print("Done")

        # Generate anchors if using an anchor based trajectory decoder
        # if hasattr(self.model.decoder, 'anchors') and torch.as_tensor(self.model.decoder.anchors == 0).all():
        #     print()
        #     print("Extracting anchors for decoder ...", end=" ")
        #     self.model.decoder.generate_anchors(self.tr_dl.dataset)
        #     print("Done")

    def train(self, num_epochs: int, output_dir: str):
        """
        Main function to train model
        :param num_epochs: Number of epochs to run training for
        :param output_dir: Output directory to store tensorboard logs and checkpoints
        :return:
        """

        # Run training, validation for given number of epochs
        start_epoch = self.current_epoch
        for epoch in range(start_epoch, start_epoch + num_epochs):

            # Set current epoch
            self.current_epoch = epoch
            print()
            print('Epoch (' + str(self.current_epoch + 1) + '/' + str(start_epoch + num_epochs) + ')')

            # Train
            train_epoch_metrics = self.run_epoch('train', self.tr_dl)
            self.print_metrics(train_epoch_metrics, self.tr_dl, mode='train')

            # Validate
            with torch.no_grad():
                val_epoch_metrics = self.run_epoch('eval', self.val_dl)
            self.print_metrics(val_epoch_metrics, self.val_dl, mode='eval')

            # Scheduler step
            self.scheduler.step()

            min_val_metric = self.min_val_metric_init
            # Update validation metric
            if self.save_with_refine:
                self.val_metric = val_epoch_metrics[self.val_metrics['refine'][0].name + '_refine'] / val_epoch_metrics['minibatch_count']
                min_val_metric = self.min_val_metric_refine
            else:
                self.val_metric = val_epoch_metrics[self.val_metrics['init'][0].name + '_init'] / val_epoch_metrics['minibatch_count']
            
            # save best checkpoint when applicable
            if self.val_metric < min_val_metric:
                self.min_val_metric = self.val_metric
                self.save_checkpoint(os.path.join(output_dir, 'checkpoints', 'best.tar'))

            # Save checkpoint
            self.save_checkpoint(os.path.join(output_dir, 'checkpoints', str(self.current_epoch) + '.tar'))

    def eval(self):
        """
        Port function to evaluate model: print the evaluation metrics
        """
        with torch.no_grad():
            
            self.model.reduce_his_length = False
            self.model.random_his_length = False
            self.model.random_interpolate_zeros = False
            self.model.valid_observation_length = 5

            if self.model.reduce_his_length:
                print('reduce_his_length') 
            if self.model.random_his_length:
                print('random_his_length') 
            else: 
                print('valid ob length = ', self.model.valid_observation_length) 
            if self.model.random_interpolate_zeros:
                print('random_interpolate_zeros') 


            val_epoch_metrics = self.run_epoch('eval', self.val_dl)
            # if self.save_with_refine:
            #     return val_epoch_metrics['refine']
            # else:
            #     return val_epoch_metrics['init']
            return val_epoch_metrics

    def run_epoch(self, mode: str, dl: torch_data.DataLoader):
        """
        Runs an epoch for a given dataloader
        :param mode: 'train' or 'eval'
        :param dl: dataloader object
        """
        if mode == 'eval':
            self.model.eval()
        else:
            self.model.train()

        # Initialize epoch metrics
        epoch_metrics = self.initialize_metrics_for_epoch(mode)

        # Main loop
        st_time = time.time()
        for i, data in enumerate(dl):
            # Load data
            data = train_eval_u.send_to_device(
                train_eval_u.convert_double_to_float(data))
            # print("Debug", data['ground_truth']['traj'].shape)
            # data = train_eval_u.send_to_device(train_eval_u.convert_float_to_double(data))

            # Forward pass
            predictions = self.model(data)
            # print("check inputs keys", data['inputs'].keys())

            # Compute loss and backprop if training
            if mode == 'train':
                loss, reg_loss, cls_loss, \
                reg_loss_refine, cls_loss_refine, \
                reg_loss_recons,\
                cls_loss_recons, feature_loss = self.model.training_step(data, predictions)
                self.back_prop(loss)

            # Keep time
            minibatch_time = time.time() - st_time
            st_time = time.time()


            minibatch_metrics_init, minibatch_metrics_refine, \
            minibatch_metrics_recons, epoch_metrics_init, \
            epoch_metrics_refine, epoch_metrics_recons= {}, {}, {}, \
                                                        {}, {}, {}

            # Aggregate metrics
            agent_index = data.agent_index
            data['ground_truth'] = {}
            data['ground_truth']['traj'] = data.y[agent_index]
            data['ground_truth']['masks'] = data.padding_mask[agent_index, 5:]
            pre_out = ['y_hat_init', 'pi_init', 'y_hat_recons', \
             'pi_recons', 'y_hat_refine', 'pi_refine']
            for out in pre_out:
                if predictions[out] != None:
                    if len(predictions[out].shape) > 2:
                        predictions[out] = predictions[out][:, agent_index]
                    else:
                        predictions[out] = predictions[out][agent_index]

            if predictions['y_hat_init'] != None:
                predictions['traj'] = predictions['y_hat_init']
                predictions['probs'] = predictions['pi_init']
                minibatch_metrics_init, epoch_metrics_init = self.aggregate_metrics(epoch_metrics,
                                                              predictions, data['ground_truth'], mode, 'init')
            
            if predictions['y_hat_refine'] != None:
                predictions['traj'] = predictions['y_hat_refine']
                predictions['probs'] = predictions['pi_refine']
                minibatch_metrics_refine, epoch_metrics_refine = self.aggregate_metrics(epoch_metrics,
                                                                        predictions, data['ground_truth'], mode, 'refine')


            if predictions['y_hat_recons'] != None:
                predictions['traj'] = predictions['y_hat_recons']
                predictions['probs'] = predictions['pi_recons']
                data['ground_truth']['traj'] = data.x[agent_index]
                data['ground_truth']['masks'] = data.padding_mask[agent_index, :5]
                minibatch_metrics_recons, epoch_metrics_recons = self.aggregate_metrics(epoch_metrics,
                                                                        predictions, data['ground_truth'], mode, 'recons')

            epoch_metrics['minibatch_count'] += 1
            epoch_metrics['time_elapsed'] += minibatch_time

            # Log minibatch metrics to tensorboard during training
            if mode == 'train':
                if minibatch_metrics_init != None:
                    self.log_tensorboard_train(minibatch_metrics_init) 
                if minibatch_metrics_refine != None:
                    self.log_tensorboard_train(minibatch_metrics_refine)
                if minibatch_metrics_recons != None:
                    self.log_tensorboard_train(minibatch_metrics_recons)

                self.writer.add_scalar('train/' + 'loss', loss, self.tb_iters)
                self.writer.add_scalar('train/' + 'reg_loss', reg_loss, self.tb_iters)
                self.writer.add_scalar('train/' + 'cls_loss', cls_loss, self.tb_iters)
                self.writer.add_scalar('train/' + 'reg_loss_refine', reg_loss_refine, self.tb_iters)
                self.writer.add_scalar('train/' + 'cls_loss_refine', cls_loss_refine, self.tb_iters)
                self.writer.add_scalar('train/' + 'reg_loss_recons', reg_loss_recons, self.tb_iters)
                self.writer.add_scalar('train/' + 'cls_loss_recons', cls_loss_recons, self.tb_iters)
                self.writer.add_scalar('train/' + 'feature_loss', feature_loss, self.tb_iters)
                
                

            # Display metrics at a predefined frequency
            if i % self.log_period == self.log_period - 1:
                if self.save_with_refine:
                    self.print_metrics(epoch_metrics, dl, mode)
                else:
                    self.print_metrics(epoch_metrics, dl, mode)

        # Log val metrics for the complete epoch to tensorboard
        if mode == 'eval':
            if epoch_metrics_init != {}:
                self.log_tensorboard_val(epoch_metrics_init) 
                epoch_metrics = epoch_metrics_init
            if epoch_metrics_refine != {}:
                self.log_tensorboard_val(epoch_metrics_refine)
                epoch_metrics = epoch_metrics_refine
            if epoch_metrics_recons != {}:
                self.log_tensorboard_val(epoch_metrics_recons)
                # epoch_metrics['recons'] = epoch_metrics_recons

        return epoch_metrics

    def compute_loss(self, model_outputs: Dict, ground_truth: Dict) -> torch.Tensor:
        """
        Computes loss given model outputs and ground truth labels
        """
        loss_vals = [loss.compute(model_outputs, ground_truth) for loss in self.losses]
        total_loss = torch.as_tensor(0, device=device).float()
        for n in range(len(loss_vals)):
            total_loss += self.loss_weights[n] * loss_vals[n]

        return total_loss

    def back_prop(self, loss: torch.Tensor, grad_clip_thresh=10):
        """
        Backpropagate loss
        """
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_thresh)
        self.optimizer.step()

    def initialize_metrics_for_epoch(self, mode: str):
        """
        Initialize metrics for epoch
        """
        metrics = self.train_metrics if mode == 'train' else self.val_metrics
        epoch_metrics = {'minibatch_count': 0, 'time_elapsed': 0}
        for specific in metrics.keys():
            for metric in metrics[specific]:
                epoch_metrics[metric.name + '_' + specific] = 0

        return epoch_metrics

    def aggregate_metrics(self, epoch_metrics: Dict, model_outputs: Dict, ground_truth: Dict,
                          mode: str, specific: str = ''):
        """
        Aggregates metrics by minibatch for the entire epoch
        """
        if mode == 'train':
            metrics = self.train_metrics[specific]
        else:
            metrics = self.val_metrics[specific]

        minibatch_metrics = {}
        for metric in metrics:
            minibatch_metrics[metric.name + '_' + specific] = metric.compute(model_outputs, ground_truth).item()

        for metric in metrics:
            epoch_metrics[metric.name + '_' + specific] += minibatch_metrics[metric.name + '_' + specific]

        return minibatch_metrics, epoch_metrics

    def print_metrics(self, epoch_metrics: Dict, dl: torch_data.DataLoader, mode: str):
        """
        Prints aggregated metrics
        """
        metrics = self.train_metrics if mode == 'train' else self.val_metrics
        minibatches_left = len(dl) - epoch_metrics['minibatch_count']
        eta = (epoch_metrics['time_elapsed']/epoch_metrics['minibatch_count']) * minibatches_left
        epoch_progress = int(epoch_metrics['minibatch_count']/len(dl) * 100)
        print('\rTraining:' if mode == 'train' else '\rValidating:', end=" ")
        progress_bar = '['
        for i in range(20):
            if i < epoch_progress // 5:
                progress_bar += '='
            else:
                progress_bar += ' '
        progress_bar += ']'
        print(progress_bar, str(epoch_progress), '%', end=", ")
        print('ETA:', int(eta), end="s, ")
        print('Metrics', end=": { ")
        if self.save_with_refine:
            print_key = 'refine'
        else:
            print_key = 'init'

        for specific in metrics.keys():
            if specific == print_key:
                for metric in metrics[specific]:
                    metric_val = epoch_metrics[metric.name + '_' + specific]/epoch_metrics['minibatch_count']
                    print(metric.name + ':', format(metric_val, '0.2f'), end=", ")
                print('\b\b }', end="\n" if eta == 0 else "")

    def load_checkpoint(self, checkpoint_path, just_weights=False):
        """
        Loads checkpoint from given path
        """
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if not just_weights:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.val_metric = checkpoint['val_metric']
            self.min_val_metric = checkpoint['min_val_metric']


    def load_checkpoint_for_distill(self, checkpoint_path, just_weights=False):
        """
        Loads checkpoint from given path
        """
        checkpoint = torch.load(checkpoint_path)
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = self.model.state_dict()

        # 从预训练的权重中筛选出需要的部分
        # pretrained_dict_ = {k: v for k, v in pretrained_dict.items() if k not in model_dict}

        pretrained_dict_in = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        # 更新模型的权重
        model_dict.update(pretrained_dict_in)
        self.model.load_state_dict(model_dict)
        print('loaded model checkpoint from', checkpoint_path)
        if len(pretrained_dict_in) != len(pretrained_dict):
            print('Note after replacement model still has unmatched para')

        if not just_weights:
            # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.val_metric = checkpoint['val_metric']
            self.min_val_metric = checkpoint['min_val_metric']

    def save_checkpoint(self, checkpoint_path):
        """
        Saves checkpoint to given path
        """
        torch.save({
            'epoch': self.current_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_metric': self.val_metric,
            'min_val_metric': self.min_val_metric
        }, checkpoint_path)

    def log_tensorboard_train(self, minibatch_metrics: Dict, specific: str =''):
        """
        Logs minibatch metrics during training
        """
        if self.writer:
            for metric_name, metric_val in minibatch_metrics.items():
                self.writer.add_scalar('train/' + metric_name + specific, metric_val, self.tb_iters)
        self.tb_iters += 1

    def log_tensorboard_val(self, epoch_metrics, specific: str =''):
        """
        Logs epoch metrics for validation set
        """
        if self.writer:
            for metric_name, metric_val in epoch_metrics.items():
                if metric_name != 'minibatch_count' and metric_name != 'time_elapsed':
                    metric_val /= epoch_metrics['minibatch_count']
                    self.writer.add_scalar('val/' + metric_name + specific, metric_val, self.tb_iters)
