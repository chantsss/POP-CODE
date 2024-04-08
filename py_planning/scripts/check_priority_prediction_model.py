import argparse
import os
import envs.config

import torch
import torch.utils.data as torch_data
import matplotlib.pyplot as plt
import numpy as np
from models.priority_prediction.model_factory import PredictionModelFactoryfrom train_eval.priority_prediction.dataset import PriorityDataset
import paper_plot.utils as plot_utils
import paper_plot.functions as plot_func

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="path to prediction network model", required=True)
parser.add_argument("--checkpoint", help="Path to pre-trained or intermediate checkpoint", required=True)
parser.add_argument("--rviz", help="enable rviz (=1) or not (=0)", type=int, default=0)
# parser.add_argument("--rviz_pause", help="rviz pause seconds", default=0.1)
parser.add_argument("--prefix_str", help="string to indicates model files", default="net_pri_pred")
args = parser.parse_args()

cfg, model = PredictionModelFactory().produce(  
    device='cpu', model_path=args.model_path, check_point=args.checkpoint)

dtset= PriorityDataset(version='eval', args=cfg)

dl = torch_data.DataLoader(dtset, batch_size=1, 
                           shuffle=False, num_workers=1,
                           prefetch_factor=2, pin_memory=True)

if args.rviz > 0:
    fig = plt.figure()
    plot_utils.fig_reset()
    fig_ax1 = fig.add_subplot(111)
    # fig_ax2 = fig.add_subplot(122)
    plot_utils.subfig_reset()

    fig_ax1.grid()
    # fig_ax2.grid()

    fig_root = envs.config.get_root2folder(args.model_path, 'rviz')

success = 0.
amount = 0.
for i, data in enumerate(dl):
    inputs = data['inputs']
    ground_truth = data['ground_truth']
    
    is_correct = model.check_result_is_legal(inputs, ground_truth)
    success += is_correct
    amount += 1.0

    if (args.rviz) > 0 and (is_correct < 1e-3):
        # visualize uncorrect results only
        plt.ion()

        input_trajs: torch.Tensor= inputs['i_trajs'] # [1, traj_num, traj_node_num, node_feat_dim]
        input_v0s: torch.Tensor= inputs['i_v0'] # [1, 2]

        ## fig_ax1
        xy0s = input_trajs[0, 0, :, :]
        xy1s = input_trajs[0, 1, :, :]
        xy1s = xy1s[np.abs(xy1s[:, 0]) > 1e-6] # remove 0.0 mask
        if xy1s.shape[0] == 0:
            # print("stop agent with result=", is_correct)
            continue # skip to rviz
        
        x0_min = min(torch.min(xy0s[:, 0]), torch.min(xy1s[:, 0]))
        x0_max = max(torch.max(xy0s[:, 0]), torch.max(xy1s[:, 0]))
        y0_min = min(torch.min(xy0s[:, 1]), torch.min(xy1s[:, 1]))
        y0_max = max(torch.max(xy0s[:, 1]), torch.max(xy1s[:, 1]))

        p1, = fig_ax1.plot(xy0s[:, 0], xy0s[:, 1], '.r')
        p2, = fig_ax1.plot(xy1s[:, 0], xy1s[:, 1], '.b')
        s1, = fig_ax1.plot(xy0s[0, 0], xy0s[0, 1], 'xr')
        s2, = fig_ax1.plot(xy1s[0, 0], xy1s[0, 1], 'xb')
        fig_ax1.set_xlim(x0_min - 2.0, x0_max + 2.0)
        fig_ax1.set_ylim(y0_min - 2.0, y0_max + 2.0)

        fig_ax1.legend(
            [   'traj$_1$ with $v_0$={:.1f}'.format(input_v0s[0, 0]), 
                'traj$_2$ with $v_0$={:.1f}'.format(input_v0s[0, 1])
            ], loc='upper right')

        ## save figure
        fig.savefig(os.path.join(
            fig_root, 'result_{}_c{:.0f}.png'.format(i, is_correct)), bbox_inches='tight')
        # fig.savefig(os.path.join(fig_root, 'result_{}.pdf'.format(i)), bbox_inches='tight')

        ## other operations
        # plt.pause(args.rviz_pause), plt.ioff()
        plt.ioff()
        p1.remove(), p2.remove()
        s1.remove(), s2.remove()
        # pp.remove(), cb.remove()

    print("\rprocess {}/ {}, with success rate={:.3f}%; ".format(
        i, len(dl), (success/amount * 100.0)), end="")
print(" ")
