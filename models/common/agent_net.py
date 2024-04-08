# Copyright (c) 2020 Uber Technologies, Inc.
# Please check LICENSE for more detail

from fractions import gcd

import torch
import torch.nn as nn
import torch.nn.functional as Ffrom torch import Tensor


class Conv1d(nn.Module):
    def __init__(
        self, n_in, n_out, kernel_size=3, stride=1, norm="GN", ng=32, act=True
    ):
        super(Conv1d, self).__init__()
        assert norm in ["GN", "BN", "SyncBN"]

        self.conv = nn.Conv1d(
            n_in,
            n_out,
            kernel_size=kernel_size,
            padding=(int(kernel_size) - 1) // 2,
            stride=stride,
            bias=False,
        )

        if norm == "GN":
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == "BN":
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit("SyncBN has not been added!")

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class Res1d(nn.Module):
    def __init__(
        self, n_in, n_out, kernel_size=3, stride=1, norm="GN", ng=32, act=True
    ):
        super(Res1d, self).__init__()
        assert norm in ["GN", "BN", "SyncBN"]
        padding = (int(kernel_size) - 1) // 2
        self.conv1 = nn.Conv1d(
            n_in,
            n_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.conv2 = nn.Conv1d(
            n_out, n_out, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.relu = nn.ReLU(inplace=True)

        # All use name bn1 and bn2 to load imagenet pretrained weights
        if norm == "GN":
            self.bn1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.bn2 = nn.GroupNorm(gcd(ng, n_out), n_out)

        if stride != 1 or n_out != n_in:
            if norm == "GN":
                self.downsample = nn.Sequential(
                    nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(gcd(ng, n_out), n_out),
                )
            elif norm == "BN":
                self.downsample = nn.Sequential(
                    nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(n_out),
                )
            else:
                exit("SyncBN has not been added!")
        else:
            self.downsample = None

        self.act = act

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        if self.act:
            out = self.relu(out)
        return out


class ActorNet(nn.Module):
    """
    Actor feature extractor with Conv1D
    """

    def __init__(self, feature_dim: int, d_model: int):
        super(ActorNet, self).__init__()
        norm = "GN"
        ng = 1

        n_in = feature_dim
        n_out = [32, 64, 128]
        blocks = [Res1d, Res1d, Res1d]
        num_blocks = [2, 2, 2]

        groups = []
        for i in range(len(num_blocks)):
            group = []
            if i == 0:
                group.append(blocks[i](n_in, n_out[i], norm=norm, ng=ng))
            else:
                group.append(blocks[i](n_in, n_out[i], stride=2, norm=norm, ng=ng))

            for j in range(1, num_blocks[i]):
                group.append(blocks[i](n_out[i], n_out[i], norm=norm, ng=ng))
            groups.append(nn.Sequential(*group))
            n_in = n_out[i]
        self.groups = nn.ModuleList(groups)

        n = d_model
        lateral = []
        for i in range(len(n_out)):
            lateral.append(Conv1d(n_out[i], n, norm=norm, ng=ng, act=False))
        self.lateral = nn.ModuleList(lateral)

        self.output = Res1d(n, n, norm=norm, ng=ng)

    def forward(self, actors: Tensor) -> Tensor:
        out = actors

        outputs = []
        for i in range(len(self.groups)):
            out = self.groups[i](out)
            outputs.append(out)

        out = self.lateral[-1](outputs[-1])
        for i in range(len(outputs) - 2, -1, -1):
            out = F.interpolate(out, scale_factor=2, mode="linear", align_corners=False)
            out += self.lateral[i](outputs[i])

        out = self.output(out)[:, :, -1]
        return out


class ActorNetPlus(nn.Module):
    """
    Actor feature extractor with Conv1D
    """

    def __init__(self, feature_dim: int, d_model: int):
        super(ActorNetPlus, self).__init__()
        norm = "GN"
        ng = 1

        n_in = feature_dim
        n_out = [32, 64, 128]
        blocks = [Res1d, Res1d, Res1d]
        num_blocks = [2, 2, 2]

        groups = []
        for i in range(len(num_blocks)):
            group = []
            if i == 0:
                group.append(blocks[i](n_in, n_out[i], norm=norm, ng=ng))
            else:
                group.append(blocks[i](n_in, n_out[i], stride=2, norm=norm, ng=ng))

            for j in range(1, num_blocks[i]):
                group.append(blocks[i](n_out[i], n_out[i], norm=norm, ng=ng))
            groups.append(nn.Sequential(*group))
            n_in = n_out[i]
        self.groups = nn.ModuleList(groups)

        n = d_model
        lateral = []
        for i in range(len(n_out)):
            lateral.append(Conv1d(n_out[i], n, norm=norm, ng=ng, act=False))
        self.lateral = nn.ModuleList(lateral)

        self.output = Res1d(n, n, norm=norm, ng=ng)

        self.pos_proj = nn.Linear(2, n)
        self.lstm = nn.LSTM(n, n, 1, batch_first=True)

    def forward(self, actors: Tensor, pos: Tensor) -> Tensor:
        out = actors

        outputs = []
        for i in range(len(self.groups)):
            out = self.groups[i](out)
            outputs.append(out)

        out = self.lateral[-1](outputs[-1])
        for i in range(len(outputs) - 2, -1, -1):
            out = F.interpolate(out, scale_factor=2, mode="linear", align_corners=False)
            out += self.lateral[i](outputs[i])

        out = self.output(out)  # (A, D, T)
        h0 = self.pos_proj(pos).unsqueeze(0)

        print(h0.shape, out.shape)
        _, lstm_out = self.lstm(out.transpose(1, 2), h0)

        return out[:, :, -1] + lstm_out.squeeze(0)
