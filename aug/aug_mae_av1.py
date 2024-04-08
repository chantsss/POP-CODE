from typing import List

import math
import numpy as np
import torch
from .aug_base import AugBase


class AugMAEAv1(AugBase):
    def __init__(
        self, rotate_angle=60.0, flip_prob=0.5, scale=[0.8, 1.2]
    ):
        """
        scale: global scale ratio
        flip_y: the probability of flipping the y axis
        """
        super().__init__()
        self.rotate_angle = rotate_angle * np.pi / 180
        self.flip_prob = flip_prob
        self.scale = scale

    def check_data(self, data):
        pass

    def augment(self, data):
        if self.rotate_angle > 0:
            theta = torch.tensor(np.random.uniform(-1, 1) * self.rotate_angle)
            rotate_mat = torch.tensor(
                [
                    [torch.cos(theta), -torch.sin(theta)],
                    [torch.sin(theta), torch.cos(theta)],
                ],
            )

            for key in ["x", "x_centers", "y", "lane_positions", "lane_centers"]:
                data[key] = torch.matmul(data[key], rotate_mat)

            for key in ["x_angles", "lane_angles", "theta"]:
                data[key] = data[key] + theta

        # global scaling
        scale_ratio = np.random.uniform(self.scale[0], self.scale[1])
        keys = ["x", "y", "x_positions", "x_centers", "lane_positions", "lane_centers"]

        for key in keys:
            data[key] = data[key] * scale_ratio

        # flip y axis
        if self.flip_prob > np.random.uniform():
            for key in keys:
                data[key][..., 1] = -data[key][..., 1]

            for key in ["x_angles", "lane_angles"]:
                data[key] = -data[key]

            data["origin"][..., 1] = -data["origin"][..., 1]

        return data
