import abc
from pyquaternion import Quaternion
import numpy as npfrom typing import Dict, Tuple, Union, Listfrom shapely.geometry import Pointfrom shapely.geometry.polygon import Polygon
import os
import time
from preprocessor.preprocessor import PreprocessInterfacefrom preprocessor.utils import AgentType, TrafficLight, rgba2rgbfrom utils.angle_operation import get_normalized_anglefrom utils.transform import XYYawTransformfrom itertools import permutations, product
import matplotlib.pyplot as plt
import io
import colorsys
import torch

class PreprocessVectorizationHiVT(PreprocessInterface):
    """
    Class for agent prediction, using the vector representation for maps and agents
    """
    def __init__(self, args: Dict):
        super().__init__()

        self.t_h: float= args['t_history']
        self.t_f: float= args['t_future']

        args = args['hivt']
        self.t_interval: float= args['t_interval']
        self.feat_siz: int= args['feature_size']

        self.map_extent = args['map_extent']
        self.polyline_resolution: float= args['polyline_resolution']
        self.polyline_length: int= args['polyline_length']
        self.local_radius: float = args['local_radius']

        self.max_nodes: int= args['num_lane_nodes']
        self.max_vehicles_num: int= args['num_vehicles']
        self.max_pedestrians_num: int= args['num_pedestrians']

        self.t_h_state_num: int= int(self.t_h / self.t_interval) + 1# add one current state
        self.t_f_state_num: int= int(self.t_f / self.t_interval)

    ### Port functions
    def process(self, idx: int, agent_idx: int):
        """
        Function to process data.
        :param idx: data index
        :param agent_idx: agent index
        """

        # Target agent
        ori_pose = self.get_target_agent_global_pose(idx, agent_idx)

        # x, y co-ordinates in agent's frame of reference
        hist = self.get_past_motion_states(idx, ori_pose, agent_idx, in_agent_frame=True)
        hist_zeropadded = np.zeros((self.t_h_state_num, self.feat_siz))
        hist_zeropadded[-hist.shape[0]:] = hist
        target_padding_masks = np.zeros((1, self.t_h_state_num+self.t_f_state_num))
        target_rotate_angle = np.zeros(1)

        if len(hist) < self.t_h_state_num and self.simu != True: # 过滤不够时长的数据
            # print('target agent his len is', len(hist), ' <', self.t_h_state_num)
            return None
        if len(hist) > 1:  # 计算actor的朝向（近似值）
            heading_vector = hist[...,:2][-1] - hist[...,:2][-2]
            target_rotate_angle = np.array([np.arctan2(heading_vector[1], heading_vector[0])])
        else:  # 如果有效时间步骤的数量小于2，则不对该actor进行预测
            target_padding_masks[:, self.t_h_state_num:, :] = 1

        future = self.get_future_motion_states(idx, agent_idx, in_agent_frame=True)
        target_motion_states = np.concatenate((hist,future), 0)
        target_motion_states = target_motion_states[np.newaxis, :]
        


        # Surrounding agent
        # Get vehicles and pedestrian histories for current sample
        vehicles_o, vehicles_o_indices = self.get_surrounding_agents_of_type(idx, agent_idx, AgentType.VEHICLE)
        pedestrians_o, pedestrians_o_indices = self.get_surrounding_agents_of_type(idx, agent_idx, AgentType.PEDESTRIAN)
        # Discard poses outside map extent
        vehicles_o, vehicles_o_indices = self.discard_poses_outside_extent(vehicles_o, vehicles_o_indices)
        pedestrians_o, pedestrians_o_indices = self.discard_poses_outside_extent(pedestrians_o, pedestrians_o_indices)
        
        vehicles_num = len(vehicles_o)
        pedestrians_num = len(pedestrians_o)

        if vehicles_num + pedestrians_num == 0:  # add dummy vehicles for graph building
            vehicles_num = 1
            vehicles_o = np.array([[[0, 0, 0, 0, 0]]])
            vehicles_o_indices = [9999]

        # Convert to fixed size arrays for batching
        vehicles, vehicle_padding_masks, vehicle_rotate_angles = self.list_to_tensor_motions(
            vehicles_o, self.max_vehicles_num, self.t_h_state_num+self.t_f_state_num, self.feat_siz, self.t_h_state_num)
        pedestrians, pedestrian_padding_masks, pedestrian_rotate_angles = self.list_to_tensor_motions(
            pedestrians_o, self.max_pedestrians_num, self.t_h_state_num+self.t_f_state_num, self.feat_siz, self.t_h_state_num)
        
        num_nodes = vehicles_num + pedestrians_num + 1

        agent_states = [target_motion_states]
        agent_masks = [target_padding_masks] 
        agent_angles = [target_rotate_angle]
        if vehicles_num > 0:
            agent_states.append(vehicles)
            agent_masks.append(vehicle_padding_masks)
            agent_angles.append(vehicle_rotate_angles)
        if pedestrians_num > 0:
            agent_states.append(pedestrians)
            agent_masks.append(pedestrian_padding_masks)
            agent_angles.append(pedestrian_rotate_angles)

        x = torch.from_numpy(np.concatenate(agent_states, 0))
        padding_mask = torch.from_numpy(np.concatenate(agent_masks, 0).astype(bool))
        rotate_angles = torch.from_numpy(np.concatenate(agent_angles, 0))

        positions = x.clone()[..., :2]

        x[:, self.t_h_state_num:, :2] = torch.where(
            (padding_mask[:, self.t_h_state_num-1].unsqueeze(-1) | padding_mask[:, self.t_h_state_num:]).unsqueeze(-1),
            torch.zeros(num_nodes, self.t_f_state_num, 2),
            x[:, self.t_h_state_num:, :2] - x[:, self.t_h_state_num-1, :2].unsqueeze(-2),
        )
        x[:, 1:self.t_h_state_num, :2] = torch.where(
            (padding_mask[:, :self.t_h_state_num-1] | padding_mask[:, 1:self.t_h_state_num]).unsqueeze(-1),
            torch.zeros(num_nodes, self.t_h_state_num-1, 2),
            x[:, 1:self.t_h_state_num, :2] - x[:, :self.t_h_state_num-1, :2],
        )
        x[:, 0, :2] = torch.zeros(num_nodes, 2)

        # x = positions[:, :self.t_h_state_num]
        # y = positions[:, self.t_h_state_num:]
        
        # bos_mask为True表示时间步骤t有效且时间步骤t-1无效
        bos_mask = torch.zeros((num_nodes, self.t_h_state_num), dtype=bool)
        bos_mask[:, 0] = ~padding_mask[:, 0]
        bos_mask[:, 1:self.t_h_state_num] = padding_mask[:, :self.t_h_state_num-1] & ~padding_mask[:, 1:self.t_h_state_num]
        x_type = torch.ones((num_nodes))
        x_type[:vehicles_num+1] = 1 # vehicle 0, pedestrian 1
        x_category = torch.ones((num_nodes)) * 2

        if num_nodes == 1 :
            print('Note num_nodes should not be 0')
            return None
        
        edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()




        # lane feature
        # Get lanes around agent within map_extent
        lanes = self.get_lanes_around_agent(ori_pose, self.polyline_resolution)

        # Get relevant polygon layers from the map_api
        polygons = self.get_polygons_around_agent(ori_pose)
        traffic_lights = self.get_traffic_lights_around_agent(ori_pose)

        # Get vectorized representation of lanes, x,y,yaw,stop-line and cross-walk
        lane_node_feats, _ = self.get_lane_node_feats(ori_pose, lanes, polygons, traffic_lights)
        
        # Discard lanes outside map extent
        lane_node_feats_origin = self.discard_poses_outside_extent(lane_node_feats)

        # Add dummy node (0, 0, 0, 0, 0) if no lane nodes are found
        if len(lane_node_feats) == 0:
            lane_node_feats = [np.zeros((1, self.feat_siz))]

        # Convert list of lane node feats to fixed size numpy array and masks
        lane_node_feats, lane_node_masks =\
            self.list_to_tensor_lane(lane_node_feats_origin, self.max_nodes, self.polyline_length, self.feat_siz)

        lane_vectors = []
        lane_positions = []
        stop_line = []
        cross_walk = []
        # non_full_padding_lanes = lane_node_feats[lane_node_masks.sum(-1).sum(-1)<self.polyline_length*self.feat_siz]

        for lane in lane_node_feats:
            lane_vectors.append(lane[1:, :2] - lane[:-1, :2]) # 只计算少于20的
            lane_positions.append(lane[:-1, :2])
            stop_line.append(lane[:-1, 3])
            cross_walk.append(lane[:-1, 4])

        lane_vectors = torch.cat(lane_vectors, dim=0)
        lane_positions = torch.cat(lane_positions, dim=0)
        stop_line = torch.cat(stop_line, dim=0)
        cross_walk = torch.cat(cross_walk, dim=0)

        actor_list = [agent_idx] + vehicles_o_indices + pedestrians_o_indices

        node_inds = [actor_list.index(actor_id) for actor_id in actor_list]

        node_positions = positions[:, self.t_h_state_num-1,:2]

        lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_vectors.size(0)), node_inds))).t().contiguous()
        lane_actor_vectors = \
            lane_positions.repeat_interleave(len(node_inds), dim=0) - node_positions.repeat(lane_vectors.size(0), 1)
        mask = torch.norm(lane_actor_vectors, p=2, dim=-1) < self.local_radius
        lane_actor_index = lane_actor_index[:, mask]
        lane_actor_vectors = lane_actor_vectors[mask]

        data = {
            "x": x[:, :self.t_h_state_num, :2],  # [N, 50, 5]
            "x_type": x_type,  # [N]
            "x_category": x_category,  # [N]
            "positions": positions,  # [N, 110, 2]
            "edge_index": edge_index,  # [2, N x N - 1]
            "y": x[:, self.t_h_state_num:],  # [N, 60, 2]
            "num_nodes": num_nodes,
            "padding_mask": padding_mask,  # [N, 110]
            "bos_mask": bos_mask,  # [N, 50]
            "rotate_angles": rotate_angles,  # [N]
            
            "lane_vectors": lane_vectors,  # [L, 2]
            "lane_positions": lane_positions,
            # "is_intersections": is_intersections,  # [L]
            "turn_directions": stop_line, # [L]
            "traffic_controls": cross_walk, # [L]
            "lane_actor_index": lane_actor_index,  # [2, E_{A-L}]
            "lane_actor_vectors": lane_actor_vectors,  # [E_{A-L}, 2]
            "agent_index": 0,
            "origin": torch.tensor(ori_pose),
            "theta": torch.tensor(ori_pose[-1]),
            # "av_index": av_index,
            # "seq_id": seq_id,
            # "city": city,
            # "scenario_id": scenario_id,
            # "track_id": agent_id,
        }

        if self.enable_rviz:
          data['rviz'] = self.visualize_graph(
            positions, 
            lane_node_feats)
          self.sanity_check(data=data)
        

        return data

    # @staticmethod
    def sanity_check(self, data, pred_res=None, attn_res=None):

        fig = plt.figure(figsize=(10, 10))

        for i, (actor, mask, pos, x_type) in enumerate(
            zip(data['x'], data['padding_mask'], data['positions'], data['x_type'])
        ):
            valid_mask = ~mask[:self.t_h_state_num]
            xy = actor[valid_mask]
            xy = torch.cumsum(-torch.flip(xy, dims=[0]), dim=0) + pos[None, self.t_h_state_num-1]
            xy = torch.cat([pos[None, self.t_h_state_num-1], xy], dim=0)
            xy = xy + pos[None, self.t_h_state_num-1]
            plt.plot(xy[:, 0], xy[:, 1], color="black", linewidth=1)
            plt.scatter(xy[0, 0], xy[0, 1], color="black", linewidth=1)
            plt.text(pos[self.t_h_state_num-1, 0], pos[self.t_h_state_num-1, 1], str(x_type.numpy()), fontsize=5)

        for i, (y, mask, pos) in enumerate(
            zip(data['y'], data['padding_mask'][:, self.t_h_state_num:], data['positions'])
        ):
            if mask.all():
                continue
            xy = y[~mask]
            xy = xy + pos[None, self.t_h_state_num-1]
            if i == data['agent_index']:
                color = "blue"
            else:
                color = "orange"
            plt.plot(xy[:, 0], xy[:, 1], color=color, linewidth=1)
            plt.scatter(xy[-1, 0], xy[-1, 1], color=color)

        # for i, (pos, mask) in enumerate(zip(data['positions'], data['padding_mask'])):
        #     mask = mask[:4]
        #     pos = pos[:4]
        #     if mask.all():
        #         continue
        #     pos = pos[~mask]
        #     if i == data['agent_index']:
        #         color = "cyan"
        #     else:
        #         color = "green"
        #     plt.plot(pos[:, 0], pos[:, 1], color="green", linewidth=1)
        #     plt.scatter(pos[-1, 0], pos[-1, 1], color=color, s=50, zorder=100)

        # if attn_res is not None:
        #     attn, idx = attn_res
        #     agent_pos = data['positions'][data['agent_index'], 4]
        #     for score, src in zip(attn, idx):
        #         src_pos = data['positions'][src, 4]
        #         plt.plot(
        #             [agent_pos[0], src_pos[0]],
        #             [agent_pos[1], src_pos[1]],
        #             color="blue",
        #             linewidth=0.5,
        #         )
        #         plt.text(x=src_pos[0], y=src_pos[1], s=f"{score:.2f}")

        # if pred_res is not None:
        #     trajs, pi = pred_res
        #     for traj in trajs:
        #         plt.plot(traj[:, 0], traj[:, 1], color="red", linewidth=1)
        #         plt.scatter(traj[-1, 0], traj[-1, 1], color="red", alpha=0.5)

        plt.scatter(
            data['lane_positions'][:, 0], data['lane_positions'][:, 1], color="gray", s=10
        )

        plt.axis("equal")
        # with io.BytesIO() as io_buf:
        filename = time.strftime("%Y%m%d-%H%M%S") + ".jpg"
        fig.savefig(filename) # is rgba
        return fig
    
    @staticmethod
    def color_by_yaw(agent_yaw_in_radians: float,
                     lane_yaw_in_radians: float) -> Tuple[float, float, float]:
      """
      Color the pose one the lane based on its yaw difference to the agent yaw.
      :param agent_yaw_in_radians: Yaw of the agent with respect to the global frame.
      :param lane_yaw_in_radians: Yaw of the pose on the lane with respect to the global frame.
      """

      # By adding pi, lanes in the same direction as the agent are colored blue.
      diff_radian = get_normalized_angle(
        agent_yaw_in_radians - lane_yaw_in_radians) + np.pi
      angle = diff_radian * 180/np.pi

      normalized_rgb_color = colorsys.hsv_to_rgb(angle/360, 1., 1.)
      color = [color*255 for color in normalized_rgb_color]

      # To make the return type consistent with Color definition
      return color[0], color[1], color[2]


    def visualize_graph(self, positions, non_full_padding_lanes):
        """
        Function to visualize lane graph.
        """
        plt.close('all')
        fig, ax = plt.subplots()
        ax.imshow(np.zeros((3, 3)), extent=self.map_extent, cmap='gist_gray')

        # Plot edges
        # for src_id, src_feats in enumerate(node_feats):
        #   feat_len = np.sum(np.sum(np.absolute(src_feats), axis=1) != 0)

        #   if feat_len > 0:
        #     src_x = np.mean(src_feats[:feat_len, 0])
        #     src_y = np.mean(src_feats[:feat_len, 1])

        #     for idx, dest_id in enumerate(s_next[src_id]):
        #       edge_t = edge_type[src_id, idx]
        #       visited = evf_gt[src_id, idx]
        #       if 3 > edge_t > 0:
        #         dest_feats = node_feats[int(dest_id)]
        #         feat_len_dest = np.sum(np.sum(np.absolute(dest_feats), axis=1) != 0)
        #         dest_x = np.mean(dest_feats[:feat_len_dest, 0])
        #         dest_y = np.mean(dest_feats[:feat_len_dest, 1])
        #         d_x = dest_x - src_x
        #         d_y = dest_y - src_y

        #         line_style = '-' if edge_t == 1 else '--'
        #         width = 2 if visited else 0.01
        #         alpha = 1 if visited else 0.5

        #         plt.arrow(src_x, src_y, d_x, d_y, color='w', head_width=0.1, length_includes_head=True,
                        #   linestyle=line_style, width=width, alpha=alpha)

        # Plot nodes
        for node_feat in non_full_padding_lanes:
          feat_len = np.sum(np.sum(np.absolute(node_feat), axis=1) != 0)
          if feat_len > 0:
            x = np.mean(node_feat[:feat_len, 0])
            y = np.mean(node_feat[:feat_len, 1])
            yaw = np.arctan2(np.mean(np.sin(node_feat[:feat_len, 2])),
                             np.mean(np.cos(node_feat[:feat_len, 2])))
            c = self.color_by_yaw(0, yaw)
            c = np.asarray(c).reshape(-1, 3) / 255
            s = 50
            ax.scatter(x, y, s, c=c)

        # Plot other agents
        # vehicle_feats = agents['vehicles']
        # vehicle_masks = agents['vehicle_masks']
        # # print("vehicles", vehicle_feats.shape)
        # for vehicle, mask in zip(vehicle_feats, vehicle_masks):
        #     if sum(mask[0]) < 1e-3:
        #         ax.scatter(vehicle[0, 0], vehicle[0, 1], c='w', s=50, marker='X')
        #     else:
        #         break

        # print("fut_xy", fut_xy)
        plt.plot(positions[0, 0, 0], positions[0, 0, 1], 'ro', markersize=4)
        plt.plot(positions[0, :, 0], positions[0, :, 1], color='r', lw=2)

        # save to buffer
        get_img = None
        with io.BytesIO() as io_buf:
          fig.savefig(io_buf, format='raw') # is rgba
          io_buf.seek(0)
          full_view_img = np.frombuffer(io_buf.getvalue(), dtype=np.uint8)
          w, h = fig.canvas.get_width_height()
          full_view_img = full_view_img.reshape((int(h), int(w), -1))
          get_img = rgba2rgb(full_view_img)

        # plt.show()
        return get_img

    ### Functions
    def get_inputs(self, idx: int, agent_idx: int) -> Dict:
        """
        Gets model inputs for agent prediction
        :param idx: data index
        :param agent_idx: agent index
        :return inputs: Dictionary with input representations
        """
        # Target agent
        ori_pose = self.get_target_agent_global_pose(idx, agent_idx)

        # x, y co-ordinates in agent's frame of reference
        hist = self.get_past_motion_states(idx, ori_pose, agent_idx, in_agent_frame=True)
        hist_zeropadded = np.zeros((self.t_h_state_num, self.feat_siz))
        hist_zeropadded[-hist.shape[0]:] = hist
        hist_zeropadded = hist_zeropadded[np.newaxis, :]
        target_padding_masks = np.ones((1, self.t_h_state_num))
        # target_padding_masks[-hist.shape[0]:] = 0
        target_rotate_angle = np.zeros(1)
        

        if len(hist) < self.t_h_state_num and self.simu != True: # 过滤不够时长的数据
            # print('target agent his len is', len(hist), ' <', self.t_h_state_num)
            return None
        if len(hist) > 1:  # 计算actor的朝向（近似值）
            heading_vector = hist[...,:2][-1] - hist[...,:2][-2]
            target_rotate_angle = np.array([np.arctan2(heading_vector[1], heading_vector[0])])

        # else:  # 如果有效时间步骤的数量小于2，则不对该actor进行预测
        #     target_padding_masks[:, self.t_h_state_num:, :] = 1

        # future = self.get_future_motion_states(idx, agent_idx, in_agent_frame=True)
        # target_motion_states = np.concatenate((hist,future), 0)
        # target_motion_states = hist[np.newaxis, :]
        


        # Surrounding agent
        # Get vehicles and pedestrian histories for current sample
        vehicles_o, vehicles_o_indices = self.get_surrounding_agents_of_type_his(idx, agent_idx, AgentType.VEHICLE)
        pedestrians_o, pedestrians_o_indices = self.get_surrounding_agents_of_type_his(idx, agent_idx, AgentType.PEDESTRIAN)
        # Discard poses outside map extent
        vehicles_o, vehicles_o_indices = self.discard_poses_outside_extent(vehicles_o, vehicles_o_indices)
        pedestrians_o, pedestrians_o_indices = self.discard_poses_outside_extent(pedestrians_o, pedestrians_o_indices)
        
        vehicles_num = len(vehicles_o)
        pedestrians_num = len(pedestrians_o)

        if vehicles_num + pedestrians_num == 0:  # add dummy vehicles for graph building
            vehicles_num = 1
            vehicles_o = np.array([[[0, 0, 0, 0, 0]]])
            vehicles_o_indices = [9999]

        # Convert to fixed size arrays for batching
        vehicles, vehicle_padding_masks, vehicle_rotate_angles = self.list_to_tensor_motions(
            vehicles_o, self.max_vehicles_num, self.t_h_state_num, self.feat_siz, self.t_h_state_num)
        pedestrians, pedestrian_padding_masks, pedestrian_rotate_angles = self.list_to_tensor_motions(
            pedestrians_o, self.max_pedestrians_num, self.t_h_state_num, self.feat_siz, self.t_h_state_num)
        
        num_nodes = vehicles_num + pedestrians_num + 1

        agent_states = [hist_zeropadded]
        agent_masks = [target_padding_masks] 
        agent_angles = [target_rotate_angle]
        if vehicles_num > 0:
            agent_states.append(vehicles)
            agent_masks.append(vehicle_padding_masks)
            agent_angles.append(vehicle_rotate_angles)
        if pedestrians_num > 0:
            agent_states.append(pedestrians)
            agent_masks.append(pedestrian_padding_masks)
            agent_angles.append(pedestrian_rotate_angles)

        x = torch.from_numpy(np.concatenate(agent_states, 0))
        padding_mask = torch.from_numpy(np.concatenate(agent_masks, 0).astype(bool))
        rotate_angles = torch.from_numpy(np.concatenate(agent_angles, 0))

        positions = x[..., :2].clone()

        # x[:, 5:] = torch.where(
        #     (padding_mask[:, 4].unsqueeze(-1) | padding_mask[:, 5:]).unsqueeze(-1),
        #     torch.zeros(num_nodes, self.t_f_state_num, 2),
        #     x[:, 5:] - x[:, 4].unsqueeze(-2),
        # )
        x[:, 0, :2] = torch.zeros(num_nodes, 2)
        if len(x) > 1:
            x[:, 1:, :2] = torch.where((padding_mask[:, :self.t_h_state_num-1] | padding_mask[:, 1:]).unsqueeze(-1),\
                                       torch.zeros(num_nodes, self.t_h_state_num-1, 2),\
                                          x[:, 1:, :2] - x[:, :self.t_h_state_num-1, :2])

        # x = positions[:, :self.t_h_state_num]
        # y = positions[:, self.t_h_state_num:]
        
        # bos_mask为True表示时间步骤t有效且时间步骤t-1无效
        bos_mask = torch.zeros((num_nodes, self.t_h_state_num), dtype=bool)
        bos_mask[:, 0] = ~padding_mask[:, 0]
        bos_mask[:, 1:self.t_h_state_num] = padding_mask[:, :self.t_h_state_num-1] & ~padding_mask[:, 1:self.t_h_state_num]
        x_type = torch.ones((num_nodes))
        x_type[:vehicles_num+1] = 1 # vehicle 0, pedestrian 1
        x_category = torch.ones((num_nodes)) * 2

        # if num_nodes == 1 :
        #     print('Note num_nodes should not be 0')
        #     return None
        
        edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()




        # lane feature
        # Get lanes around agent within map_extent
        lanes = self.get_lanes_around_agent(ori_pose, self.polyline_resolution)

        # Get relevant polygon layers from the map_api
        polygons = self.get_polygons_around_agent(ori_pose)
        traffic_lights = self.get_traffic_lights_around_agent(ori_pose)

        # Get vectorized representation of lanes, x,y,yaw,stop-line and cross-walk
        lane_node_feats, _ = self.get_lane_node_feats(ori_pose, lanes, polygons, traffic_lights)
        
        # Discard lanes outside map extent
        lane_node_feats_origin = self.discard_poses_outside_extent(lane_node_feats)

        # Add dummy node (0, 0, 0, 0, 0) if no lane nodes are found
        if len(lane_node_feats) == 0:
            lane_node_feats = [np.zeros((1, self.feat_siz))]

        # Convert list of lane node feats to fixed size numpy array and masks
        lane_node_feats, lane_node_masks =\
            self.list_to_tensor_lane(lane_node_feats_origin, self.max_nodes, self.polyline_length, self.feat_siz)

        lane_vectors = []
        lane_positions = []
        stop_line = []
        cross_walk = []
        # non_full_padding_lanes = lane_node_feats[lane_node_masks.sum(-1).sum(-1)<self.polyline_length*self.feat_siz]

        for lane in lane_node_feats:
            lane_vectors.append(lane[1:, :2] - lane[:-1, :2]) # 只计算少于20的
            lane_positions.append(lane[:-1, :2])
            stop_line.append(lane[:-1, 3])
            cross_walk.append(lane[:-1, 4])

        lane_vectors = torch.cat(lane_vectors, dim=0)
        lane_positions = torch.cat(lane_positions, dim=0)
        stop_line = torch.cat(stop_line, dim=0)
        cross_walk = torch.cat(cross_walk, dim=0)

        actor_list = [agent_idx] + vehicles_o_indices + pedestrians_o_indices

        node_inds = [actor_list.index(actor_id) for actor_id in actor_list]

        # if len(positions) == 5
        node_positions = positions[:, self.t_h_state_num-1,:2]

        lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_vectors.size(0)), node_inds))).t().contiguous()
        lane_actor_vectors = \
            lane_positions.repeat_interleave(len(node_inds), dim=0) - node_positions.repeat(lane_vectors.size(0), 1)
        mask = torch.norm(lane_actor_vectors, p=2, dim=-1) < self.local_radius
        lane_actor_index = lane_actor_index[:, mask]
        lane_actor_vectors = lane_actor_vectors[mask]

        inputs = {
            "x": x[:, :self.t_h_state_num, :2],  # [N, 50, 2]
            "x_type": x_type,  # [N]
            "x_category": x_category,  # [N]
            "positions": positions,  # [N, 110, 2]
            "edge_index": edge_index,  # [2, N x N - 1]
            "y": None,  # [N, 60, 2]
            "num_nodes": num_nodes,
            "padding_mask": padding_mask,  # [N, 110]
            "bos_mask": bos_mask,  # [N, 50]
            "rotate_angles": rotate_angles,  # [N]
            
            "lane_vectors": lane_vectors,  # [L, 2]
            "lane_positions": lane_positions,
            # "is_intersections": is_intersections,  # [L]
            "turn_directions": stop_line, # [L]
            "traffic_controls": cross_walk, # [L]
            "lane_actor_index": lane_actor_index,  # [2, E_{A-L}]
            "lane_actor_vectors": lane_actor_vectors,  # [E_{A-L}, 2]
            "agent_index": 0,
            "origin": torch.tensor(ori_pose),
            "theta": torch.tensor(ori_pose[-1]),
            # "av_index": av_index,
            # "seq_id": seq_id,
            # "city": city,
            # "scenario_id": scenario_id,
            # "track_id": agent_id,
        }

        return inputs

    ### Overrides
    def extract_target_agent_representation(self, idx: int, agent_idx: int) -> np.ndarray:
        """
        Extracts target agent representation
        :param idx: data index
        :param agent_idx: agent index
        :return hist: track history for target agent, shape: [t_h * 2 + 1, feat_siz]
        """
        ori_pose = self.get_target_agent_global_pose(idx, agent_idx)

        # x, y co-ordinates in agent's frame of reference
        hist = self.get_past_motion_states(idx, ori_pose, agent_idx, in_agent_frame=True)

        # Zero pad for track histories shorter than t_h
        hist_zeropadded = np.zeros((self.t_h_state_num, self.feat_siz))
        hist_zeropadded[-hist.shape[0]:] = hist

        hist = hist_zeropadded
        return hist

    def extract_map_representation(self, idx: int, agent_idx: int) -> Union[int, Dict]:
        """
        Extracts map representation
        :param idx: data index
        :param agent_idx: agent index
        :return: Returns an ndarray with lane node features, shape 
                 [max_nodes, polyline_length, feat_siz] and an ndarray of
                 masks of the same shape, with value 1 if the nodes/poses are empty,
        """
        # Get agent representation in global co-ordinates
        global_pose = self.get_target_agent_global_pose(idx, agent_idx)

        # Get lanes around agent within map_extent
        lanes = self.get_lanes_around_agent(global_pose, self.polyline_resolution)

        # Get relevant polygon layers from the map_api
        polygons = self.get_polygons_around_agent(global_pose)
        traffic_lights = self.get_traffic_lights_around_agent(global_pose)

        # Get vectorized representation of lanes
        lane_node_feats, _ = self.get_lane_node_feats(global_pose, lanes, polygons, traffic_lights)

        # Discard lanes outside map extent
        lane_node_feats = self.discard_poses_outside_extent(lane_node_feats)

        # Add dummy node (0, 0, 0, 0, 0) if no lane nodes are found
        if len(lane_node_feats) == 0:
            lane_node_feats = [np.zeros((1, self.feat_siz))]

        # Convert list of lane node feats to fixed size numpy array and masks
        lane_node_feats, lane_node_masks =\
            self.list_to_tensor(lane_node_feats, self.max_nodes, self.polyline_length, self.feat_siz)

        map_representation = {
            'lane_node_feats': lane_node_feats,
            'lane_node_masks': lane_node_masks,
        }

        return map_representation

    # def extract_surrounding_agent_representation(self, idx: int, agent_idx: int) -> \
    #         Union[Tuple[int, int], Dict]:
    #     """
    #     Extracts surrounding agent representation
    #     :param idx: data index
    #     :param agent_idx: agent index
    #     :return: ndarrays with surrounding pedestrian and vehicle track histories and masks for non-existent agents
    #     """

    #     # Get vehicles and pedestrian histories for current sample
    #     vehicles = self.get_surrounding_agents_of_type(idx, agent_idx, AgentType.VEHICLE)
    #     pedestrians = self.get_surrounding_agents_of_type(idx, agent_idx, AgentType.PEDESTRIAN)
    #     # Discard poses outside map extent
    #     vehicles = self.discard_poses_outside_extent(vehicles)
    #     pedestrians = self.discard_poses_outside_extent(pedestrians)
        
    #     # Convert to fixed size arrays for batching
    #     vehicles, vehicle_padding_masks = self.list_to_tensor(
    #         vehicles, self.max_vehicles_num, self.t_h_state_num+self.t_f_state_num, self.feat_siz)
    #     pedestrians, pedestrian_padding_masks = self.list_to_tensor(
    #         pedestrians, self.max_pedestrians_num, self.t_h_state_num+self.t_f_state_num, self.feat_siz)
        
    #     num_nodes = self.max_vehicles_num + self.max_pedestrians_num
    #     positions = np.stack((vehicles, pedestrians), 0)
    #     padding_mask = np.stack((vehicle_padding_masks, pedestrian_padding_masks), 0)
    #     x = positions[:, :self.t_h_state_num]
    #     y = positions[:, self.t_h_state_num:]
    #     # bos_mask为True表示时间步骤t有效且时间步骤t-1无效
    #     bos_mask = np.zeros((num_nodes, self.t_h_state_num))
    #     bos_mask[:, 0] = ~padding_mask[:, 0]
    #     bos_mask[:, 1:self.t_h_state_num] = padding_mask[:, :self.t_h_state_num-1] & ~padding_mask[:, 1:self.t_h_state_num]
    #     x_type = np.ones((num_nodes))
    #     x_type[:self.max_vehicles_num] = 0 # vehicle 1, pedestrian 1
    #     x_category = np.ones((num_nodes)) * 2
    #     edge_index = np.array(list(permutations(range(num_nodes), 2))).T

    #     surrounding_agent_representation = {
    #         "x": x,  # [N, 4, 5]
    #         "x_type": x_type,  # [N]
    #         "x_category": x_category,  # [N]
    #         "positions": positions,  # [N, 110, 2]
    #         "edge_index": edge_index,  # [2, N x N - 1]
    #         "y": y,  # [N, 60, 2]
    #         "num_nodes": num_nodes,
    #         "padding_mask": padding_mask,  # [N, 110]
    #         "bos_mask": bos_mask,  # [N, 50]
    #     }

    #     return surrounding_agent_representation

    def extract_target_representation(self, idx: int, agent_idx: int) -> Union[np.ndarray, Dict]:
        """
        Extracts target representation for target agent
        :param idx: data index
        :param agent_idx: agent index
        """
        return self.get_future_motion_states(idx, agent_idx, 
            involve_full_future=False, in_agent_frame=True)

    ### Abstracts
    @abc.abstractmethod
    def get_past_motion_states(self, idx: int, ori_pose: Tuple, 
                                     agent_idx: int, in_agent_frame: bool) -> np.ndarray:
        '''
        Extract target agent past history
        :param idx: data index
        :param ori_pose: original agent pose
        :param agent_idx: agent index
        :param in_agent_frame: representation in agent frame
        return shape = (valid_history + 1 , feat_siz)
        [
          ...,
          [x, y, yaw, v, yaw_rate, ...]_{t = current - t_interval * 2.0},
          [x, y, yaw, v, yaw_rate, ...]_{t = current - t_interval * 1.0},
          [x, y, yaw, v, yaw_rate, ...]_{t = current},
        ]
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def get_future_motion_states(self, idx: int, agent_idx: int, in_agent_frame: bool,
                                       involve_full_future: bool=False) -> np.ndarray:
        '''
        Extract target agent future history
        :param idx: data index
        :param agent_idx: agent index
        :param in_agent_frame: representation in agent frame
        :param involve_full_future: contain all future states (not just for prediction)
        return shape = (valid_future , feat_siz)
        [
          [x, y, yaw, v, yaw_rate, ...]_{t = current + t_interval * 1.0},
          [x, y, yaw, v, yaw_rate, ...]_{t = current + t_interval * 2.0},
        ]
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def get_target_agent_global_pose(self, idx: int, agent_idx: int) -> Tuple[float, float, float]:
        """
        Returns global pose of target agent
        :param idx: data index
        :param agent_idx: agent index
        :return global_pose: (x, y, yaw) or target agent in global co-ordinates
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_lanes_around_agent(self, global_pose: Tuple[float, float, float], 
                                     polyline_resolution: float) -> Dict:
        """
        Gets lane polylines around the target agent
        :param global_pose: (x, y, yaw) or target agent in global co-ordinates
        :param polyline_resolution: resolution to sample lane centerline poses
        :return lanes: Dictionary of lane polylines (centerline)
            lane_key(str or int) > List[Tuple(float, float, float)]
                                    [(x, y, yaw)_{lane_node0}, (x, y, yaw)_{1}, ...]
        @note yaw value is critical when graph processing.!
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_polygons_around_agent(self, global_pose: Tuple[float, float, float]) -> Dict:
        """
        Gets polygon layers around the target agent e.g. crosswalks, stop lines
        :param global_pose: (x, y, yaw) or target agent in global co-ordinates
        :return polygons: Dictionary of polygon layers, each type as a list of shapely Polygons
            {   
                'stop_line': List[Polygon],
                'ped_crossing': List[Polygon]
            }
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_traffic_lights_around_agent(self, global_pose: Tuple[float, float, float]) -> Dict[TrafficLight, List[Polygon]]:
        """
        Gets traffic light layers around the target agent
        :param global_pose: (x, y, yaw) or target agent in global co-ordinates
        :return dict of traffic lights:
            {   
                TrafficLight: List[Polygon],
            }
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_surrounding_agent_indexs_with_type(self, idx: int, agent_idx: int, agent_type: AgentType) -> List[int]:
        """
        Returns surrounding agents's list of indexs
        :param idx: data index
        :param agent_idx: agent index
        :param agent_type: AgentType
        :return: list of indexs of surrounding agents
        """
        raise NotImplementedError()

    ### Class functions
    def get_lane_node_feats(self, origin: Tuple, lanes: Dict[Union[str, int], List[Tuple]],
                            polygons: Dict[str, List[Polygon]],
                            traffic_lights: Dict[TrafficLight, List[Polygon]]
        ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Generates vector HD map representation in the agent centric frame of reference
        :param origin: (x, y, yaw) of target agent in global co-ordinates
        :param lanes: lane centerline poses in global co-ordinates
        :param polygons: stop-line and cross-walk polygons in global co-ordinates
        :return:
        """

        # Convert lanes to list
        lane_ids = [k for k, v in lanes.items()]
        lanes = [v for k, v in lanes.items()]

        # Get flags indicating whether a lane lies on 
        # 1. stop lines or crosswalks:
        # 2. traffic lights
        # return List[ np.array=(lane_node_num, 2=flags) ]
        lane_flags = self.get_lane_flags(lanes, polygons, traffic_lights)

        # Convert lane polylines to local coordinates: [num, 6, 3]
        lanes = [np.asarray([self.global_to_local(origin, pose) for pose in lane]) for lane in lanes]

        # Concatenate lane poses and lane flags
        lane_node_feats = [np.concatenate((lanes[i], lane_flags[i]), axis=1) for i in range(len(lanes))]

        # Split lane centerlines into smaller segments:
        lane_node_feats, lane_node_ids = self.split_lanes(lane_node_feats, self.polyline_length, lane_ids)

        return lane_node_feats, lane_node_ids

    def get_surrounding_agents_of_type(self, idx: int, agent_idx: int, agent_type: AgentType) -> List[np.ndarray]:
        """
        Returns surrounding agents of a particular class for a given sample
        :param idx: data index
        :param agent_idx: agent index
        :param agent_type: AgentType
        :return: list of ndarrays of agent track histories.
             List[ agents (in assigned type) with data from motion_states() ]
        """
        agent_list = []

        ori_pose = self.get_target_agent_global_pose(idx, agent_idx)

        indexs = self.get_surrounding_agent_indexs_with_type(idx, agent_idx, agent_type)
        for _agent_idx in indexs:
            past_motion_states = self.get_past_motion_states(idx, ori_pose, _agent_idx, in_agent_frame=True)
            future_motion_states = self.get_future_motion_states(idx, agent_idx, in_agent_frame=True)
            motion_states = np.concatenate((past_motion_states, future_motion_states), 0)
            agent_list.append(motion_states)

        return agent_list, indexs
    

    def get_surrounding_agents_of_type_his(self, idx: int, agent_idx: int, agent_type: AgentType) -> List[np.ndarray]:
        """
        Returns surrounding agents of a particular class for a given sample
        :param idx: data index
        :param agent_idx: agent index
        :param agent_type: AgentType
        :return: list of ndarrays of agent track histories.
             List[ agents (in assigned type) with data from motion_states() ]
        """
        agent_list = []

        ori_pose = self.get_target_agent_global_pose(idx, agent_idx)

        indexs = self.get_surrounding_agent_indexs_with_type(idx, agent_idx, agent_type)
        for _agent_idx in indexs:
            past_motion_states = self.get_past_motion_states(idx, ori_pose, _agent_idx, in_agent_frame=True)
            agent_list.append(past_motion_states)

        return agent_list, indexs

    def discard_poses_outside_extent(self, pose_set: List[np.ndarray],
                                     ids: List[str] = None) -> Union[List[np.ndarray],
                                                                     Tuple[List[np.ndarray], List[str]]]:
        """
        Discards lane or agent poses outside predefined extent in target agent's frame of reference.
        :param pose_set: agent or lane polyline poses
        :param ids: annotation record tokens for pose_set. Only applies to lanes.
        :return: Updated pose set
        """
        updated_pose_set = []
        updated_ids = []

        for m, poses in enumerate(pose_set):
            flag = False
            for n, pose in enumerate(poses):
                if self.map_extent[0] <= pose[0] <= self.map_extent[1] and \
                        self.map_extent[2] <= pose[1] <= self.map_extent[3]:
                    flag = True

            if flag:
                updated_pose_set.append(poses)
                if ids is not None:
                    updated_ids.append(ids[m])

        updated_pose_set = updated_pose_set[:self.max_nodes]
        updated_ids = updated_ids[:self.max_nodes]

        if ids is not None:
            return updated_pose_set, updated_ids
        else:
            return updated_pose_set

    @staticmethod
    def global_to_local(origin: Tuple, global_pose: Tuple) -> Tuple:
        """
        Converts pose in global co-ordinates to local co-ordinates.
        :param origin: (x, y, yaw) of origin in global co-ordinates
        :param global_pose: (x, y, yaw) in global co-ordinates
        :return local_pose: (x, y, yaw) in local co-ordinates
        """
        # Unpack
        global_x, global_y, global_yaw = global_pose
        origin_x, origin_y, origin_yaw = origin

        # Translate + Rotate
        inv_origin = XYYawTransform(x=origin_x, y=origin_y, yaw_radian=origin_yaw)
        inv_origin.inverse()
        get_pose = inv_origin.multiply_from_right(
            XYYawTransform(x=global_x, y=global_y, yaw_radian=global_yaw)
        )
        local_pose = (get_pose._x, get_pose._y, get_pose._yaw)

        return local_pose

    @staticmethod
    def split_lanes(lanes: List[np.ndarray], max_len: int, 
                    lane_ids: List[Union[str, int]]) -> Tuple[List[np.ndarray], List[str]]:
        """
        Splits lanes into roughly equal sized smaller segments with defined maximum length
        :param lanes: list of lane poses
        :param max_len: maximum admissible length of polyline
        :param lane_ids: list of lane ID tokens
        :return lane_segments: list of smaller lane segments
                lane_segment_ids: list of lane ID tokens corresponding to original lane that the segment is part of
        """
        lane_segments = []
        lane_segment_ids = []
        for idx, lane in enumerate(lanes):
            n_segments = int(np.ceil(len(lane) / max_len))
            n_poses = int(np.ceil(len(lane) / n_segments))
            for n in range(n_segments):
                lane_segment = lane[n * n_poses: (n+1) * n_poses]
                lane_segments.append(lane_segment)
                lane_segment_ids.append(lane_ids[idx])

        return lane_segments, lane_segment_ids

    def get_lane_flags(self, lanes: List[List[Tuple]], 
                             polygons: Dict[str, List[Polygon]],
                             traffic_lights: Dict[TrafficLight, List[Polygon]]
        ) -> List[np.ndarray]:
        """
        Returns flags indicating whether each pose on lane polylines lies on polygon map layers
        like stop-lines or cross-walks
        :param lanes: list of lane poses
        :param polygons: dictionary of polygon layers
        :return lane_flags: list of ndarrays with flags
        """
        poly_value = self.poly_priority_idx
        tl_value = self.tl_color_to_priority_idx

        lane_flags = [np.zeros((len(lane), 2)) for lane in lanes]
        for lane_num, lane in enumerate(lanes):
            for pose_num, pose in enumerate(lane):
                point = Point(pose[0], pose[1])
                # for n, k in enumerate(polygons.keys()):
                #     # two type of polygon, 
                #     polygon_list = polygons[k]
                #     for polygon in polygon_list:
                #         if polygon.contains(point):
                #             lane_flags[lane_num][pose_num][n] = 1
                #             break
                for poly_key, polygon_list in polygons.items():
                    for polygon in polygon_list:
                        if polygon.contains(point):
                            lane_flags[lane_num][pose_num][0] += poly_value[poly_key]
                            break # one poly_key add lane_node value once

                for light_key, polygon_list in traffic_lights.items():
                    enable_brake = False
                    for polygon in polygon_list:
                        if polygon.contains(point):
                            lane_flags[lane_num][pose_num][1] = tl_value[light_key]
                            enable_brake = True
                            break
                    if enable_brake:
                        break # one traffic light set value of lane_node once

        return lane_flags

    @staticmethod
    def list_to_tensor_motions(feat_list: List[np.ndarray], max_num: int, max_len: int,
                       feat_size: int, t_h_state_num: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts a list of sequential features (e.g. lane polylines or agent history) to fixed size numpy arrays for
        forming mini-batches

        :param feat_list: List of sequential features
        :param max_num: Maximum number of sequences in List
        :param max_len: Maximum length of each sequence
        :param feat_size: Feature dimension
        :return: 1) ndarray of features of shape [max_num, max_len, feat_dim]. Has zeros where elements are missing,
            2) ndarray of binary masks of shape [max_num, max_len, feat_dim]. Has ones where elements are missing.
        """
        max_num = len(feat_list)
        feat_array = np.zeros((max_num, max_len, feat_size))
        mask_array = np.ones((max_num, max_len))
        rotate_angles = np.zeros((max_num))
        for n, feats in enumerate(feat_list):
            feat_array[n, :len(feats), :] = feats
            mask_array[n, :len(feats)] = 0

            if len(feats) > 1:  # 计算actor的朝向（近似值）
                heading_vector = feats[...,:2][-1] - feats[...,:2][-2]
                rotate_angles[n] = np.arctan2(heading_vector[1], heading_vector[0])
            else:  # 如果有效时间步骤的数量小于2，则不对该actor进行预测
                mask_array[n, t_h_state_num:] = 1

        return feat_array, mask_array, rotate_angles

    @staticmethod
    def list_to_tensor_lane(feat_list: List[np.ndarray], max_num: int, max_len: int,
                       feat_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts a list of sequential features (e.g. lane polylines or agent history) to fixed size numpy arrays for
        forming mini-batches

        :param feat_list: List of sequential features
        :param max_num: Maximum number of sequences in List
        :param max_len: Maximum length of each sequence
        :param feat_size: Feature dimension
        :return: 1) ndarray of features of shape [max_num, max_len, feat_dim]. Has zeros where elements are missing,
            2) ndarray of binary masks of shape [max_num, max_len, feat_dim]. Has ones where elements are missing.
        """
        max_num = len(feat_list)
        feat_array = torch.zeros((max_num, max_len, feat_size))
        mask_array = torch.ones((max_num, max_len, feat_size))
        for n, feats in enumerate(feat_list):
            feat_array[n, :len(feats), :] = torch.from_numpy(feats)
            mask_array[n, :len(feats), :] = 0

        return feat_array, mask_array
    
    @staticmethod
    def list_to_tensor(feat_list: List[np.ndarray], max_num: int, max_len: int,
                       feat_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts a list of sequential features (e.g. lane polylines or agent history) to fixed size numpy arrays for
        forming mini-batches

        :param feat_list: List of sequential features
        :param max_num: Maximum number of sequences in List
        :param max_len: Maximum length of each sequence
        :param feat_size: Feature dimension
        :return: 1) ndarray of features of shape [max_num, max_len, feat_dim]. Has zeros where elements are missing,
            2) ndarray of binary masks of shape [max_num, max_len, feat_dim]. Has ones where elements are missing.
        """
        feat_array = np.zeros((max_num, max_len, feat_size))
        mask_array = np.ones((max_num, max_len, feat_size))
        for n, feats in enumerate(feat_list):
            feat_array[n, :len(feats), :] = feats
            mask_array[n, :len(feats), :] = 0

        return feat_array, mask_array

    @staticmethod
    def flip_horizontal(data: Dict):
        """
        Helper function to randomly flip some samples across y-axis for data augmentation
        :param data: Dictionary with inputs and ground truth values.
        :return: data: Dictionary with inputs and ground truth values fligpped along y-axis.
        """
        # Flip target agent
        hist = data['inputs']['target_agent_representation']
        hist[:, 0] = -hist[:, 0]  # x-coord
        hist[:, 4] = -hist[:, 4]  # yaw-rate
        data['inputs']['target_agent_representation'] = hist

        # Flip lane node features
        lf = data['inputs']['map_representation']['lane_node_feats']
        lf[:, :, 0] = -lf[:, :, 0]  # x-coord
        lf[:, :, 2] = -lf[:, :, 2]  # yaw
        data['inputs']['map_representation']['lane_node_feats'] = lf

        # Flip surrounding agents
        vehicles = data['inputs']['surrounding_agent_representation']['vehicles']
        vehicles[:, :, 0] = -vehicles[:, :, 0]  # x-coord
        vehicles[:, :, 4] = -vehicles[:, :, 4]  # yaw-rate
        data['inputs']['surrounding_agent_representation']['vehicles'] = vehicles

        peds = data['inputs']['surrounding_agent_representation']['pedestrians']
        peds[:, :, 0] = -peds[:, :, 0]  # x-coord
        peds[:, :, 4] = -peds[:, :, 4]  # yaw-rate
        data['inputs']['surrounding_agent_representation']['pedestrians'] = peds

        # Flip groud truth trajectory
        fut = data['ground_truth']['traj']
        fut[:, 0] = -fut[:, 0]  # x-coord
        data['ground_truth']['traj'] = fut

        return data
