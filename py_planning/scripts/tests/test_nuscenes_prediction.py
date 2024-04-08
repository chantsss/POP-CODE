import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import envs.config

DATA_VERSION = 'v1.0-mini'
DATAROOT = os.path.join(envs.config.NUSCENES_DATA_ROOT, DATA_VERSION)

import thirdparty.configfrom nuscenes import NuScenesfrom nuscenes.eval.prediction.splits import get_prediction_challenge_split

nuscenes = NuScenes(DATA_VERSION, dataroot=DATAROOT)
train_set = get_prediction_challenge_split("mini_train", dataroot=DATAROOT)

print("*** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** ")
print("DATAROOT={}".format(DATAROOT))
print("TRAIN UNIT: {}, {}".format(train_set[0], len(train_set)))
# TRAIN UNIT='4d87aaf2d82549969f1550607ef46a63_faf2ea71b30941329a3c3f3866cec714', 32186
from nuscenes.prediction import PredictHelper
helper = PredictHelper(nuscenes)

instance_token, sample_token = train_set[300].split("_")
annotation = helper.get_sample_annotation(instance_token, sample_token)
print("*** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** ")
print("UNIT DETAILS", instance_token, sample_token)
print(annotation)
# annotation = 
#   {'token': 'a286c9633fa34da5b978758f348996b0', 
#    'sample_token': '39586f9d59004284a7114a68825e8eec', 
#    'instance_token': 'bc38961ca0ac4b14ab90e547ba79fbb6', 
#    'visibility_token': '4', 
#    'attribute_tokens': ['cb5118da1ab342aa947717dc53544259'], 
#    'translation': [392.945, 1148.426, 0.766], 
#    'size': [1.708, 4.01, 1.631], 
#    'rotation': [-0.5443682117180475, 0.0, 0.0, 0.8388463804957943], 
#    'prev': '16140fbf143d4e26a4a7613cbd3aa0e8', 
#    'next': 'b41e15b89fd44709b439de95dd723617', 
#    'num_lidar_pts': 0, 
#    'num_radar_pts': 0, 
#    'category_name': 'vehicle.car'}

print("*** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** ")
# Meters / second.
print(f"Agent Velocity: {helper.get_velocity_for_agent(instance_token, sample_token)}")
# Meters / second^2.
print(f"Agent Acceleration: {helper.get_acceleration_for_agent(instance_token, sample_token)}")
# Radians / second.
print(f"Agent Heading Change Rate: {helper.get_heading_change_rate_for_agent(instance_token, sample_token)}")

# To get the closest lane to a location, use the get_closest_lane method. 
# To see the internal data representation of the lane, use the get_lane_record method. 
# You can also explore the connectivity of the lanes, with the get_outgoing_lanes 
# and get_incoming_lane methods.from nuscenes.map_expansion.map_api import NuScenesMap
nusc_map = NuScenesMap(map_name='singapore-onenorth', dataroot=DATAROOT)
x, y, yaw = 395, 1095, 0
closest_lane = nusc_map.get_closest_lane(x, y, radius=2)
lane_record = nusc_map.get_arcline_path(closest_lane)
print("*** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** ")
print("LANE SYMBOL=", closest_lane) # 5933500a-f0f2-4d69-9bbc-83b875e4a73e
print("LANE_RECORD=", lane_record)
print("PREDECESSOR_LANE=", nusc_map.get_incoming_lane_ids(closest_lane))
print("NEXT_LANE=", nusc_map.get_outgoing_lane_ids(closest_lane))
# LANE_RECORD = 
#   # [{'start_pose': [421.2419602954602, 1087.9127960414617, 2.739593514975998], 
#       'end_pose': [391.7142849867393, 1100.464077182952, 2.7365754617298705], 
#       'shape': 'LSR', 'radius': 999.999, 
#       'segment_length': [0.23651121617864976, 28.593481378991886, 3.254561444252876]}]
# To help manipulate the lanes, we've added an arcline_path_utils module. 
# For example, something you might want to do is discretize a lane into a sequence of poses.from nuscenes.map_expansion import arcline_path_utils
poses = arcline_path_utils.discretize_lane(lane_record, resolution_meters=1)
print("LANE POSES=", len(poses))
# print(poses)
# LANE POSES= 
#   [(421.2419602954602, 1087.9127960414617, 2.739593514975998), ...]

'''
It is common in the prediction literature to represent the state of an agent as a tensor 
containing information about the semantic map (such as the drivable area and walkways), 
as well the past locations of surrounding agents.

Each paper in the field chooses to represent the input in a slightly different way. 
For example, CoverNet and MTP choose to rasterize the map information and agent locations into 
a three channel RGB image. But Rules of the Road decides to use a "taller" tensor with 
information represented in different channels.

We provide a module called input_representation that is meant to make it easy for you 
to define your own input representation. In short, you need to define your own 
StaticLayerRepresentation, AgentRepresentation, and Combinator.

The StaticLayerRepresentation controls how the static map information is represented. 
The AgentRepresentation controls how the locations of the agents in the scene are represented. 
The Combinator controls how these two sources of information are combined into a single tensor.

For more information, consult input_representation/interface.py. To help get you started, 
we've provided implementations of input representation used in CoverNet and MTP.
'''

import matplotlib.pyplot as pltfrom nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizerfrom nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistoryfrom nuscenes.prediction.input_representation.interface import InputRepresentationfrom nuscenes.prediction.input_representation.combinators import Rasterizer
static_layer_rasterizer = StaticLayerRasterizer(helper)
agent_rasterizer = AgentBoxesWithFadedHistory(helper, seconds_of_history=3.0)
mtp_input_representation = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())

# map_name = helper.get_map_name_from_sample_token(sample_token)
# img = mtp_input_representation.make_input_representation(instance_token, sample_token)
# print("img_ouput.shape=", img.shape) # (500, 500, 3)

img_static = static_layer_rasterizer.make_representation(instance_token, sample_token)
print("*** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** ")
print("img_static.shape=", img_static.shape) # (500, 500, 3)
# > x, y = sample_annotation['translation'][:2]
# > yaw = quaternion_yaw(Quaternion(sample_annotation['rotation']))
# > layer_names = ['drivable_area', 'ped_crossing', 'walkway']
# > masks = self.maps[map_name].get_map_mask(patchbox, angle_in_degrees, self.layer_names, canvas_size=canvas_size)
#   masks.shape = [3, 800, 800]
# > change_color_of_binary_mask(), see how this function do with masks, and how to change color.

# > here, center of image is not at the target_agent.
#   it depends on the value of meters_ahead, meters_left.
# > agent colors:
#   if 'vehicle' in category_name:
#       return 255, 255, 0  # yellow
#   elif 'object' in category_name:
#       return 204, 0, 204  # violet
#   elif 'human' in category_name or 'animal' in category_name:
#       return 255, 153, 51  # orange
#   else:
#       raise ValueError(f"Cannot map {category_name} to a color.")
img_agents = agent_rasterizer.make_representation(instance_token, sample_token)
print("img_agents.shape=", img_agents.shape) # (500, 500, 3)

img = Rasterizer().combine([img_static, img_agents])

fig = plt.figure()
plt.subplot(131)
plt.imshow(img_static)
plt.subplot(132)
plt.imshow(img_agents)
plt.subplot(133)
plt.imshow(img)
plt.show()

import torch
# The second input is a tensor containing the velocity, acceleration, and heading change rate for the agent.
agent_state_vector = torch.Tensor([[helper.get_velocity_for_agent(instance_token, sample_token),
                                    helper.get_acceleration_for_agent(instance_token, sample_token),
                                    helper.get_heading_change_rate_for_agent(instance_token, sample_token)]])
# print(agent_state_vector.shape, agent_state_vector)
#  shape = torch.Size([1, 3]) tensor([[1.7065,    nan, 0.0000]])

# 1. permute the shape (500, 500, 3) > torch.Size([3, 500, 500])
# 2. unsqueeze the shape torch.Size([3, 500, 500]) > torch.Size([1, 3, 500, 500])
image_tensor = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0)
# print("image_tensor shape=", image_tensor.shape)

# We've provided PyTorch implementations for CoverNet and MTP. 
# Below we show, how to make predictions on the previously created input representation.from nuscenes.prediction.models.backbone import ResNetBackbonefrom nuscenes.prediction.models.mtp import MTPfrom nuscenes.prediction.models.covernet import CoverNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Both models take a CNN backbone as a parameter. 
# We've provided wrappers for ResNet and MobileNet v2. In this example, we'll use ResNet50.
backbone = ResNetBackbone('resnet50')
mtp = MTP(backbone, num_modes=6)
# Note that the value of num_modes depends on the size of the lattice used for CoverNet.
covernet = CoverNet(backbone, num_modes=64)

# Output has 50 entries.
# The first 24 are x,y coordinates (in the agent frame) over the next 6 seconds at 2 Hz for the first mode.
# The second 24 are the x,y coordinates for the second mode.
# The last 2 are the logits of the mode probabilities
mtp_output = mtp(image_tensor, agent_state_vector)

# CoverNet outputs a probability distribution over the trajectory set.
# These are the logits of the probabilities
cnt_output = covernet(image_tensor, agent_state_vector)
print("*** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** ")
# 150 = 24 * 6 modes + 6 probabilities
print("mtp_output", mtp_output.shape) # mtp_output torch.Size([1, 150])
print("cnt_output", cnt_output.shape) # cnt_output torch.Size([1, 64])
from torch.utils.data import DataLoader, Datasetfrom typing import List

class TestDataset(Dataset):
    def __init__(self, tokens: List[str], helper: PredictHelper):
        self.tokens = tokens
        self.static_layer_representation = StaticLayerRasterizer(helper)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index: int):

        token = self.tokens[index]
        instance_token, sample_token = token.split("_")

        image = self.static_layer_representation.make_representation(instance_token, sample_token)
        image = torch.Tensor(image).permute(2, 0, 1)
        agent_state_vector = torch.ones((3))
        ground_truth = torch.ones((1, 12, 2))

        ground_truth[:, :, 1] = torch.arange(0, 6, step=0.5)

        return image, agent_state_vector, ground_truth

tokens = ['bd26c2cdb22d4bb1834e808c89128898_ca9a282c9e77460f8360f564131a8af5',
          '085fb7c411914888907f7198e998a951_ca9a282c9e77460f8360f564131a8af5',
          'f4af7fd215ee47aa8b64bac0443d7be8_9ee4020153674b9e9943d395ff8cfdf3']

batch_size = 2
dataset = TestDataset(tokens, helper)
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=1)

for img, agent_state_vector, ground_truth in dataloader:
    print("training format with batch[{}]:".format(batch_size), img.shape, 
                                                    agent_state_vector.shape, ground_truth.shape)
    print("agent_state_vector={}".format(agent_state_vector))
    print("ground_truth={}".format(ground_truth))
    # batch[2]: torch.Size([2, 3, 500, 500]) torch.Size([2, 3]) torch.Size([2, 1, 12, 2])
