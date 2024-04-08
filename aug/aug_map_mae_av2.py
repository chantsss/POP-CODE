from typing import List

import numpy as np
import torch
from .aug_base import AugBase


class AugMapMaeAv2(AugBase):
    def __init__(self, rotate_angle=60 / 180 * np.pi):
        """
        scale: global scale ratio
        flip_y: the probability of flipping the y axis
        """
        super().__init__()
        self.rotate_angle = rotate_angle

    def check_data(self, data):
        pass

    def augment(self, data):
        theta = torch.tensor(np.random.uniform(-1, 1) * self.rotate_angle)
        rotate_mat = torch.tensor(
            [
                [torch.cos(theta), -torch.sin(theta)],
                [torch.sin(theta), torch.cos(theta)],
            ],
        )

        for key in ["lane_positions", "lane_centers"]:
            data[key] = torch.matmul(data[key], rotate_mat)
        data["lane_angles"] += theta

        return data
