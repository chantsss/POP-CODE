import torch.nn as nn
import torch
import copy

def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif "weight_hh" in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif "weight_hr" in name:
                nn.init.xavier_uniform_(param)
            elif "bias_ih" in name:
                nn.init.zeros_(param)
            elif "bias_hh" in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif "weight_hh" in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif "bias_ih" in name:
                nn.init.zeros_(param)
            elif "bias_hh" in name:
                nn.init.zeros_(param)


def drop_his(valid_observation_length, historical_steps, reduce_his_length, random_his_length, 
            random_interpolate_zeros, data, drop_all=False, drop_for_recons=False):
    
    if reduce_his_length: 
        valid_observation_length = valid_observation_length
        if random_his_length:
            valid_observation_length = torch.randint(low=1, high=historical_steps+1, size=(1,)).item()
        
        if random_interpolate_zeros:
            # Make sure agent is visible at least at current frame
            indices = torch.arange(1, historical_steps)
            shuffle = torch.randperm(historical_steps - 1)
            set_zeros = indices[shuffle][:historical_steps - valid_observation_length]
        else:
            indices = torch.arange(1, historical_steps)
            set_zeros = indices[:historical_steps - valid_observation_length]

        if drop_for_recons:

            data['his_pred_padding_mask'] = torch.zeros_like(data['padding_mask']).bool()
            if len(set_zeros) != 0:
                batch_size = len(data)
                # print('Start forgetting history in training, remain ', valid_observation_length,' steps.......')
                # device = data['x'].device
                padding_mask_origin = data['padding_mask'].clone()
                if not drop_all:
                    for bz_idx in range(batch_size):
                        agent_idx = data['agent_index'][bz_idx]
                        data['x'][agent_idx][set_zeros] = data['x'][agent_idx][set_zeros].fill_(0)
                        data['padding_mask'][agent_idx][set_zeros] = data['padding_mask'][agent_idx][set_zeros].fill_(True)
                        
                else:
                        data['x'][:, set_zeros, :] = data['x'][:, set_zeros, :].fill_(0)
                        data['padding_mask'][:, set_zeros] = data['padding_mask'][:, set_zeros].fill_(True)

                data['his_pred_padding_mask'] = ~padding_mask_origin & data['padding_mask']
        
        else:

            if len(set_zeros) != 0:
                batch_size = len(data)

                if not drop_all:
                    for bz_idx in range(batch_size):
                        agent_idx = data['agent_index'][bz_idx]
                        data['x'][agent_idx][set_zeros] = data['x'][agent_idx][set_zeros].fill_(0)
                        data['padding_mask'][agent_idx][set_zeros] = data['padding_mask'][agent_idx][set_zeros].fill_(True)
                        
                else:
                        data['x'][:, set_zeros, :] = data['x'][:, set_zeros, :].fill_(0)
                        data['padding_mask'][:, set_zeros] = data['padding_mask'][:, set_zeros].fill_(True)


    return data
