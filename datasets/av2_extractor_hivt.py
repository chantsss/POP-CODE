import tracebackfrom itertools import permutations, productfrom pathlib import Pathfrom typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as Ffrom av2.datasets.motion_forecasting.data_schema import TrackCategoryfrom av2.map.map_api import ArgoverseStaticMapfrom torch_geometric.data import Data

# from extractors.extractor import Extractor
from .av2_common import OBJECT_TYPE_MAP, load_av2_df
import abc
import os
import pickle
import tracebackfrom pathlib import Pathfrom utils.utils_hivt import TemporalData

class Extractor:
    def __init__(self, save_path: Path = None, mode: str = "train") -> None:
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path
        self.mode = mode

    @abc.abstractmethod
    def get_data(self, file: Path):
        raise NotImplementedError

    def save(self, file: Path):
        assert self.save_path is not None

        try:
            data = self.get_data(file)
            sequence_id = int(file.stem)
            data["sequence_id"] = sequence_id
        except Exception:
            print(traceback.format_exc())
            print("found error while extracting data from {}".format(file))
        save_file = self.save_path / (file.stem + ".pkl")
        with open(save_file, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def analyze(self, data):
        raise NotImplementedError


class Av2ExtractorHiVT(Extractor):
    def __init__(
        self,
        radius: float = 100,
        save_path: Path = None,
        mode: str = "train",
        ignore_type=[5, 6, 7, 8, 9],
        debug: bool = False,
    ) -> None:
        super().__init__(save_path, mode=mode)
        self.radius = radius
        self.debug = debug
        self.ignore_type = ignore_type

    def save(self, file: Path):
        assert self.save_path is not None

        try:
            data = self.get_data(file)
        except Exception:
            print(traceback.format_exc())
            print("found error while extracting data from {}".format(file))
        save_file = self.save_path / (file.stem + ".pt")
        torch.save(data, save_file)

    def get_data(self, file: Path):
        return self.process(file)

    def process(self, raw_path: str) -> Data:
        df, am, scenario_id = load_av2_df(raw_path)

        # 过滤掉在历史时间步骤中看不到的actors
        timestamps = list(np.sort(df["timestep"].unique()))
        cur_df = df[df["timestep"] == timestamps[49]]
        actor_ids = list(cur_df["track_id"].unique())

        agent_id = cur_df["focal_track_id"].values[0]
        actor_ids.remove(agent_id)
        actor_ids = [agent_id] + actor_ids
        num_nodes = len(actor_ids)

        df = df[df["track_id"].isin(actor_ids)]

        
        av_index = actor_ids.index("AV")
        agent_index = actor_ids.index(agent_id)
        
        city = df["city"].values[0]

        local_df = df[df["track_id"] == agent_id].iloc
        # 使场景以AV为中心
        origin = torch.tensor([local_df[49]["position_x"], local_df[49]["position_y"]], dtype=torch.float)
        theta = torch.tensor([local_df[49]["heading"]], dtype=torch.float)        
        rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                                [torch.sin(theta), torch.cos(theta)]])

        # 初始化
        x = torch.zeros(num_nodes, 110, 2, dtype=torch.float)
        x_type = torch.zeros(num_nodes, dtype=torch.long)
        x_category = torch.zeros(num_nodes, dtype=torch.long)
        edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
        padding_mask = torch.ones(num_nodes, 110, dtype=torch.bool)
        bos_mask = torch.zeros(num_nodes, 50, dtype=torch.bool)
        rotate_angles = torch.zeros(num_nodes, dtype=torch.float)
        x_heading = torch.zeros(num_nodes, 110, dtype=torch.float)
        x_type = torch.zeros(num_nodes, dtype=torch.long)


        for actor_id, actor_df in df.groupby("track_id"):
            node_idx = actor_ids.index(actor_id)
            x_type[node_idx] = OBJECT_TYPE_MAP[actor_df["object_type"].values[0]]            
            node_steps = [timestamps.index(timestamp) for timestamp in actor_df["timestep"]]
            padding_mask[node_idx, node_steps] = False
            x_category[node_idx] = actor_df["object_category"].values[0]   
            if padding_mask[node_idx, 49]:  # 不对当前时间步骤看不见的actors进行预测
                padding_mask[node_idx, 50:] = True
            xy = torch.from_numpy(np.stack([actor_df["position_x"].values, actor_df["position_y"].values], axis=-1)).float()
            x[node_idx, node_steps] = torch.matmul(xy - origin, rotate_mat)
            
            heading = torch.from_numpy(actor_df["heading"].values).float()
            x_heading[node_idx, node_steps] = heading - theta            
            
            node_historical_steps = list(filter(lambda node_step: node_step < 50, node_steps))
            if len(node_historical_steps) > 1:  # 计算actor的朝向（近似值）
                heading_vector = x[node_idx, node_historical_steps[-1]] - x[node_idx, node_historical_steps[-2]]
                rotate_angles[node_idx] = torch.atan2(heading_vector[1], heading_vector[0])
            else:  # 如果有效时间步骤的数量小于2，则不对该actor进行预测
                padding_mask[node_idx, 50:] = True

        # bos_mask为True表示时间步骤t有效且时间步骤t-1无效
        bos_mask[:, 0] = ~padding_mask[:, 0]
        bos_mask[:, 1:50] = padding_mask[:, :49] & ~padding_mask[:, 1:50]

        positions = x.clone()
        x[:, 50:] = torch.where((padding_mask[:, 49].unsqueeze(-1) | padding_mask[:, 50:]).unsqueeze(-1),
                                torch.zeros(num_nodes, 60, 2),
                                x[:, 50:] - x[:, 49].unsqueeze(-2))
        x[:, 1:50] = torch.where((padding_mask[:, :49] | padding_mask[:, 1:50]).unsqueeze(-1),
                                torch.zeros(num_nodes, 49, 2),
                                x[:, 1:50] - x[:, :49])
        x[:, 0] = torch.zeros(num_nodes, 2)

        # 获取当前时间步骤的车道特征
        df_49 = df[df["timestep"] == timestamps[49]]
        node_inds_49 = [actor_ids.index(actor_id) for actor_id in df_49["track_id"]]
        node_positions_49 = torch.from_numpy(np.stack([df_49["position_x"].values, df_49["position_y"].values], axis=-1)).float()
        (
            lane_vectors,
            is_intersections,
            lane_actor_index,
            lane_actor_vectors,
            lane_positions,
        ) = self.get_lane_features(am, node_inds_49, node_positions_49, origin, rotate_mat, city, self.radius)

        y = None if self.mode == "test" else x[:, 50:]
        seq_id = os.path.splitext(os.path.basename(raw_path))[0]

        data = {
            "x": x[:, :50],  # [N, 50, 2]
            "x_type": x_type,  # [N]
            "x_category": x_category,  # [N]
            "positions": positions,  # [N, 110, 2]
            "edge_index": edge_index,  # [2, N x N - 1]
            "y": y,  # [N, 60, 2]
            "num_nodes": num_nodes,
            "padding_mask": padding_mask,  # [N, 110]
            "bos_mask": bos_mask,  # [N, 50]
            "rotate_angles": rotate_angles,  # [N]
            "lane_vectors": lane_vectors,  # [L, 2]
            "lane_positions": lane_positions if self.debug else None,
            "is_intersections": is_intersections,  # [L]
            "lane_actor_index": lane_actor_index,  # [2, E_{A-L}]
            "lane_actor_vectors": lane_actor_vectors,  # [E_{A-L}, 2]
            "agent_index": agent_index,
            "origin": origin.unsqueeze(0),
            "theta": theta,
            "av_index": av_index,
            "seq_id": seq_id,
            "city": city,
            "scenario_id": scenario_id,
            "track_id": agent_id,
            
        }

        data = drop_his_proprecess(valid_observation_length=50, historical_steps=50, reduce_his_length=True, random_his_length=True, 
            random_interpolate_zeros=True, data=data, drop_all=False, drop_for_recons=True)


        return TemporalData(**data)

    @staticmethod
    def get_lane_features(am: ArgoverseStaticMap, node_inds: List[int], node_positions: torch.Tensor,
                        origin: torch.Tensor, rotate_mat: torch.Tensor, city: str, radius: float):
        (lane_positions, lane_vectors, is_intersections) = ([], [], [])

        # the map in av2 is limited to 100x100
        lane_ids = am.get_nearby_lane_segments(node_positions[0].numpy(), radius * 3)
        lane_ids = [l.id for l in lane_ids]
        node_positions = torch.matmul(node_positions - origin, rotate_mat).float()

        for lane_id in lane_ids:
            lane_centerline = torch.from_numpy(
                am.get_lane_segment_centerline(lane_id)[:, :2]
            ).float()
            lane_centerline = torch.matmul(lane_centerline - origin, rotate_mat)
            is_intersection = am.lane_is_in_intersection(lane_id)
            lane_positions.append(lane_centerline[:-1])
            lane_vectors.append(lane_centerline[1:] - lane_centerline[:-1])
            count = len(lane_centerline) - 1
            is_intersections.append(
                is_intersection * torch.ones(count, dtype=torch.uint8)
            )

        lane_positions = torch.cat(lane_positions, dim=0)
        lane_vectors = torch.cat(lane_vectors, dim=0)
        is_intersections = torch.cat(is_intersections, dim=0)

        lane_actor_index = (
            torch.LongTensor(
                list(product(torch.arange(lane_vectors.size(0)), node_inds))
            )
            .t()
            .contiguous()
        )  # the first is the source and the second is the target
        lane_actor_vectors = lane_positions.repeat_interleave(
            len(node_inds), dim=0
        ) - node_positions.repeat(lane_vectors.size(0), 1)
        mask = torch.norm(lane_actor_vectors, p=2, dim=-1) < radius
        lane_actor_index = lane_actor_index[:, mask]
        lane_actor_vectors = lane_actor_vectors[mask]

        # add laneGCN?
        return (
            lane_vectors,
            is_intersections,
            lane_actor_index,
            lane_actor_vectors,  # why ?
            lane_positions,
        )

    @staticmethod
    def sanity_check(data, pred_res=None, attn_res=None):
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 10))

        for i, (actor, mask, pos, x_type) in enumerate(
            zip(data.x, data.padding_mask, data.positions, data.x_type)
        ):
            valid_mask = ~mask[:50]
            xy = actor[valid_mask]
            xy = torch.cumsum(-torch.flip(xy, dims=[0]), dim=0) + pos[None, 49]
            xy = torch.cat([pos[None, 49], xy], dim=0)
            plt.plot(xy[:, 0], xy[:, 1], color="blue", linewidth=2)
            plt.scatter(xy[0, 0], xy[0, 1], color="blue", linewidth=2)
            plt.text(pos[49, 0], pos[49, 1], str(x_type.numpy()))

        for i, (y, mask, pos) in enumerate(
            zip(data.y, data.padding_mask[:, 50:], data.positions)
        ):
            if mask.all():
                continue
            xy = y[~mask]
            xy = xy + pos[None, 49]
            if i == data.agent_index:
                color = "purple"
            else:
                color = "orange"
            plt.plot(xy[:, 0], xy[:, 1], color=color, linewidth=1)
            plt.scatter(xy[-1, 0], xy[-1, 1], color=color)

        for i, (pos, mask) in enumerate(zip(data.positions, data.padding_mask)):
            mask = mask[:50]
            pos = pos[:50]
            if mask.all():
                continue
            pos = pos[~mask]
            if i == data.agent_index:
                color = "cyan"
            else:
                color = "green"
            plt.plot(pos[:, 0], pos[:, 1], color="green", linewidth=1)
            plt.scatter(pos[-1, 0], pos[-1, 1], color=color, s=50, zorder=100)

        if attn_res is not None:
            attn, idx = attn_res
            agent_pos = data.positions[data.agent_index, 49]
            for score, src in zip(attn, idx):
                src_pos = data.positions[src, 49]
                plt.plot(
                    [agent_pos[0], src_pos[0]],
                    [agent_pos[1], src_pos[1]],
                    color="blue",
                    linewidth=0.5,
                )
                plt.text(x=src_pos[0], y=src_pos[1], s=f"{score:.2f}")

        if pred_res is not None:
            trajs, pi = pred_res
            for traj in trajs:
                plt.plot(traj[:, 0], traj[:, 1], color="red", linewidth=1)
                plt.scatter(traj[-1, 0], traj[-1, 1], color="red", alpha=0.5)

        plt.scatter(
            data.lane_positions[:, 0], data.lane_positions[:, 1], color="gray", s=10
        )

        plt.axis("equal")
        return fig

def drop_his_proprecess(valid_observation_length, historical_steps, reduce_his_length, random_his_length, 
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

        if drop_for_recons:

            data['his_pred_padding_mask'] = torch.zeros_like(data['padding_mask']).bool()
            if len(set_zeros) != 0:
                data['padding_mask_masked'] = data['padding_mask'].clone()
                data['x_masked'] = data['x'].clone()
                if not drop_all:
                    # for bz_idx in range(batch_size):
                    agent_idx = data['agent_index']
                    data['x_masked'][agent_idx][set_zeros] = data['x_masked'][agent_idx][set_zeros].fill_(0)
                    data['padding_mask_masked'][agent_idx][set_zeros] = data['padding_mask_masked'][agent_idx][set_zeros].fill_(True)
                        
                else:
                    data['x_masked'][:, set_zeros, :] = data['x_masked'][:, set_zeros, :].fill_(0)
                    data['padding_mask_masked'][:, set_zeros] = data['padding_mask_masked'][:, set_zeros].fill_(True)

                data['his_pred_padding_mask'] = ~data['padding_mask'] & data['padding_mask_masked']

            else:
                data['padding_mask_masked'] = data['padding_mask'].clone()
                data['x_masked'] = data['x'].clone()
                data['his_pred_padding_mask'] = ~data['padding_mask'] & data['padding_mask_masked']

        else:

            if len(set_zeros) != 0:

                if not drop_all:
                    # for bz_idx in range(batch_size):
                    agent_idx = data['agent_index']
                    data['x'][agent_idx][set_zeros] = data['x'][agent_idx][set_zeros].fill_(0)
                    data['padding_mask'][agent_idx][set_zeros] = data['padding_mask'][agent_idx][set_zeros].fill_(True)
                        
                else:
                    data['x'][:, set_zeros, :] = data['x'][:, set_zeros, :].fill_(0)
                    data['padding_mask'][:, set_zeros] = data['padding_mask'][:, set_zeros].fill_(True)


    return data