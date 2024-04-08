import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import envs.config

DATA_VERSION = 'v1.0-mini'
SPLIT_VERSION = 'mini_train'
# DATA_VERSION = 'v1.0-trainval'
# SPLIT_VERSION = 'train'
DATAROOT = os.path.join(envs.config.NUSCENES_DATA_ROOT, DATA_VERSION)

import thirdparty.configfrom nuscenes import NuScenesfrom nuscenes.eval.prediction.splits import get_prediction_challenge_splitfrom torch.utils.data import DataLoader, IterableDataset

nuscenes = NuScenes(DATA_VERSION, dataroot=DATAROOT)
train_set = get_prediction_challenge_split(SPLIT_VERSION, dataroot=DATAROOT)

num_modes = 6
num_trainset_1 = len(train_set) - 1
batch_num = 5
epoch_num = 10000

print("*** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** ")
print("DATAROOT={}".format(DATAROOT))
print("TRAIN DATASET SIZE: {}".format(len(train_set)))
# TRAIN UNIT='4d87aaf2d82549969f1550607ef46a63_faf2ea71b30941329a3c3f3866cec714', 32186
from nuscenes.prediction import PredictHelper
helper = PredictHelper(nuscenes)

import matplotlib.pyplot as pltfrom nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizerfrom nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistoryfrom nuscenes.prediction.input_representation.interface import InputRepresentationfrom nuscenes.prediction.input_representation.combinators import Rasterizer
static_layer_rasterizer = StaticLayerRasterizer(helper)
agent_rasterizer = AgentBoxesWithFadedHistory(helper, seconds_of_history=3.0)
mtp_input_representation = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())

import torchfrom nuscenes.prediction.models.backbone import ResNetBackbonefrom nuscenes.prediction.models.mtp import MTP, MTPLoss
import torch.optim as optim
import random
import copy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
backbone = ResNetBackbone('resnet50')
mtp = MTP(backbone, num_modes=num_modes)
mtp = mtp.to(device)

loss_function = MTPLoss(num_modes, 1.0, 5.0)
optimizer = optim.Adam(mtp.parameters(), lr=0.001)
initial_loss = None

for epoch in range(0, epoch_num):
    batch_sids = [random.randint(0, num_trainset_1) for _ in range(0, batch_num)]

    image_tensors = None
    state_tensors = None
    future_xy_tensors = None
    for idx in batch_sids:
        instance_token, sample_token = train_set[idx].split("_")

        img = mtp_input_representation.make_input_representation(
            instance_token, sample_token)
        image_tensor = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0)

        agent_state_vector = torch.Tensor([[
            helper.get_velocity_for_agent(instance_token, sample_token),                     
            0.0,
            helper.get_heading_change_rate_for_agent(instance_token, sample_token)]])

        future_xy_local = helper.get_future_for_agent(instance_token, sample_token, 
                                                      seconds=6, in_agent_frame=True)
        future_xy_local = torch.tensor([[future_xy_local]])

        if image_tensors == None:
            image_tensors = image_tensor
            state_tensors = agent_state_vector
            future_xy_tensors = future_xy_local
        else:
            image_tensors = torch.cat((image_tensors, image_tensor), dim=0)
            state_tensors = torch.cat((state_tensors, agent_state_vector), dim=0)
            future_xy_tensors = torch.cat((future_xy_tensors, future_xy_local), dim=0)

    # print("check shape=", image_tensors.shape, 
    #       state_tensors.shape, future_xy_tensors.shape)
    image_tensors = image_tensors.float().to(device)
    state_tensors = state_tensors.float().to(device)
    future_xy_tensors = future_xy_tensors.float().to(device)
    image_tensors /= 255.0

    optimizer.zero_grad()

    prediction = mtp(image_tensors, state_tensors)
    loss = loss_function(prediction, future_xy_tensors)
    loss.backward()
    optimizer.step()

    # learning rate decay
    if epoch > 0 and epoch % 250 == 0:
        for p in optimizer.param_groups:
            p["lr"] *= 0.95

    # rviz
    # print(image_tensors.shape) # ([5, 3, 500, 500])
    if epoch > 0 and epoch % 500 == 0:
        for i in range(0, image_tensors.shape[0]):
            image_rviz = image_tensors[i, :, :, :].permute(1, 2, 0).cpu().numpy()
            tar_traj_array = future_xy_tensors[i, 0, :, :].cpu().numpy()
            trajs, probs = loss_function._get_trajectory_and_modes(prediction.cpu())
            trajs = trajs.detach().numpy()[i, :, : ,:]
            probs = probs.detach().numpy()[i, :]

            # torch.Size([12, 2]) (6, 12, 2) (6,)
            # print(tar_traj_array.shape, trajs.shape, probs.shape)
            width = image_rviz.shape[0]
            height = image_rviz.shape[1]
            # print("image_rviz", image_rviz.shape)
            # print(image_rviz[:, 0, 0])

            fig = plt.figure()
            plt.subplot(121)
            plt.imshow(image_rviz)
            plt.subplot(122)
            plt.plot(tar_traj_array[:, 0], tar_traj_array[:, 1], 'r')
            for mode in range(0, trajs.shape[0]):
                plt.plot(trajs[mode, :, 0], trajs[mode, :, 1], 'b')
            plt.show()

    current_loss = loss.cpu().detach().numpy()
    if initial_loss == None:
        initial_loss = copy.copy(current_loss)
    print('\r Epoch [{}/{}], Loss: {:.4f} / Initial {:.4f};'\
          .format(epoch+1, epoch_num, current_loss, initial_loss), 
          end="")

