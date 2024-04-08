import torch.nn as nn
import torch
import copyfrom typing import List, Optionalfrom torch_geometric.data import Data

class TemporalData(Data):

    def __init__(self,
                 x: Optional[torch.Tensor] = None,
                 positions: Optional[torch.Tensor] = None,
                 edge_index: Optional[torch.Tensor] = None,
                 edge_attrs: Optional[List[torch.Tensor]] = None,
                 y: Optional[torch.Tensor] = None,
                 num_nodes: Optional[int] = None,
                 padding_mask: Optional[torch.Tensor] = None,
                 bos_mask: Optional[torch.Tensor] = None,
                 rotate_angles: Optional[torch.Tensor] = None,
                 lane_vectors: Optional[torch.Tensor] = None,
                 stop_line: Optional[torch.Tensor] = None,
                 cross_walk: Optional[torch.Tensor] = None,
                 turn_directions: Optional[torch.Tensor] = None,
                 traffic_controls: Optional[torch.Tensor] = None,
                 lane_actor_index: Optional[torch.Tensor] = None,
                 lane_actor_vectors: Optional[torch.Tensor] = None,
                 seq_id: Optional[int] = None,
                 **kwargs) -> None:
        if x is None:
            super(TemporalData, self).__init__()
            return
        super(TemporalData, self).__init__(x=x, positions=positions, edge_index=edge_index, y=y, num_nodes=num_nodes,
                                           padding_mask=padding_mask, bos_mask=bos_mask, rotate_angles=rotate_angles,
                                           lane_vectors=lane_vectors, stop_line=stop_line, cross_walk=cross_walk,
                                           turn_directions=turn_directions, traffic_controls=traffic_controls,
                                           lane_actor_index=lane_actor_index, lane_actor_vectors=lane_actor_vectors,
                                           seq_id=seq_id, **kwargs)
        if edge_attrs is not None:
            for t in range(self.x.size(1)):
                self[f'edge_attr_{t}'] = edge_attrs[t]

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'lane_actor_index':
            return torch.tensor([[self['lane_vectors'].size(0)], [self.num_nodes]])
        else:
            return super().__inc__(key, value)

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
            shuffle = torch.randperm(historical_steps - 2)
            set_zeros = indices[shuffle][:historical_steps - valid_observation_length]

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

class SingleInputEmbedding(nn.Module):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super(SingleInputEmbedding, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel),
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)



class MultipleInputEmbedding(nn.Module):
    def __init__(self, in_channels: List[int], out_channel: int) -> None:
        super(MultipleInputEmbedding, self).__init__()
        self.module_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_channel, out_channel),
                    nn.LayerNorm(out_channel),
                    nn.ReLU(inplace=True),
                    nn.Linear(out_channel, out_channel),
                )
                for in_channel in in_channels
            ]
        )
        self.aggr_embed = nn.Sequential(
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel),
        )
        self.apply(init_weights)

    def forward(
        self,
        continuous_inputs: List[torch.Tensor],
        categorical_inputs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        for i in range(len(self.module_list)):
            continuous_inputs[i] = self.module_list[i](continuous_inputs[i])
        output = torch.stack(continuous_inputs).sum(dim=0)  # * sum?
        if categorical_inputs is not None:
            output += torch.stack(categorical_inputs).sum(dim=0)
        return self.aggr_embed(output)
