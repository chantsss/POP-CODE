import torch
import torch.nn as nnfrom typing import Dict

class PriorityPrediction(nn.Module):
  """
  Priority prediction network
  """
  def __init__(self, args: Dict):
    """
    Initialize network structure.
    :param args: dict of arguments
    """
    super().__init__()

    # flatten_input =\
    #   args['dim_traj_layer_in'][0] * args['dim_traj_layer_in'][1] * args['dim_traj_layer_in'][2]
    flatten_input = args['dim_traj_layer_in'] + args['dim_speed_layer_in']

    self.layer_in = nn.Linear(flatten_input, args['dim_traj_layer_mid'])
    self.layer_relu = nn.ReLU(inplace=True)
    self.layer_out = nn.Linear(args['dim_traj_layer_mid'], args['dim_traj_layer_out'])

  def forward(self, inputs: Dict):
    """
    Forward pass for outputs
    """
    input_feats: torch.Tensor= inputs['i_path_feats'] # (batch_num, feat_size)
    input_v0s: torch.Tensor= inputs['i_v0'] # (batch_num, 2)

    # in1 = torch.flatten(input_trajs, start_dim=1)
    in1 = torch.cat((input_feats, input_v0s), dim=1)

    in1 = self.layer_in(in1)
    in1 = self.layer_relu(in1)
    out = self.layer_out(in1)
    out_probs = torch.softmax(out, dim=1)

    # print("out_probs shape", out_probs.shape)

    return out_probs
