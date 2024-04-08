from typing import List

import numpy as np
import torch
from utils.utils_hivt import TemporalData
from .aug_base import AugBase


class AugHiVT(AugBase):
    def __init__(self, scale: List[float] = [0.8, 1.25], flip_y: float = 0.5) -> None:
        """
        scale: global scale ratio
        flip_y: the probability of flipping the y axis
        """
        super().__init__()
        self.scale = scale
        self.flip_y = flip_y

    def check_data(self, data):
        pass

    def augment(self, data: TemporalData):
        self.check_data(data)

        # global scaling
        scale_ratio = np.random.uniform(self.scale[0], self.scale[1])
        data.x = data.x * scale_ratio
        data.y = data.y * scale_ratio
        data.positions = data.positions * scale_ratio
        data.lane_vectors = data.lane_vectors * scale_ratio
        data.lane_actor_vectors = data.lane_actor_vectors * scale_ratio
        data.origin = data.origin * scale_ratio
        if "lane_positions" in data:
            data.lane_positions = data.lane_positions * scale_ratio

        # flip y axis
        if self.flip_y > np.random.uniform():
            data.x[:, :, 1] = -data.x[:, :, 1]
            data.y[:, :, 1] = -data.y[:, :, 1]
            data.positions[:, :, 1] = -data.positions[:, :, 1]
            data.lane_vectors[:, 1] = -data.lane_vectors[:, 1]
            data.lane_actor_vectors[:, 1] = -data.lane_actor_vectors[:, 1]
            data.origin[0, 1] = -data.origin[0, 1]
            data.rotate_angles = -data.rotate_angles
            data.theta = -data.theta
            if "lane_positions" in data:
                data.lane_positions[:, 1] = -data.lane_positions[:, 1]

        return data
