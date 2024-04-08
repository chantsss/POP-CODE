from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as Ffrom timm.models.layers import trunc_normal_from torch_geometric.data import Batch, Datafrom torch_geometric.nn.conv import MessagePassingfrom torch_geometric.typing import Adj, OptTensor, Sizefrom torch_geometric.utils import softmax, subgraph
from models.common.utils import TemporalDatafrom models.common.nat_actor_net import NatActorNetfrom models.common.utils import init_weights
from .embedding import MultipleInputEmbedding, SingleInputEmbeddingfrom .utils import DistanceDropEdge


class LocalEncoder(nn.Module):
    def __init__(
        self,
        historical_steps: int,
        node_dim: int,
        edge_dim: int,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        parallel: bool = False,
        local_radius: float = 50,
        nat_backbone: bool = False,
        pos_embed: bool = False,
    ) -> None:
        super(LocalEncoder, self).__init__()
        self.historical_steps = historical_steps
        self.parallel = parallel

        self.actor_net = NatActorNet(
            3, 32, seq_len=historical_steps, learnable_pos_encoding=pos_embed
        )

        self.actor_type_embed = nn.Parameter(torch.Tensor(10, embed_dim))

        self.drop_edge = DistanceDropEdge(local_radius)
        self.al_encoder = ALEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        trunc_normal_(self.actor_type_embed, std=0.02)

    def forward(self, data: TemporalData):
        x = torch.bmm(data["x"], data["rotate_mat"])  # (N, T, 2)
        padding_mask = data["padding_mask"][:, : self.historical_steps]  # (N, T)
        actor_feat = torch.cat([x, padding_mask.unsqueeze(-1)], dim=-1)

        out = self.actor_net(actor_feat.permute(0, 2, 1))  # (N, 128)
        
        out += self.actor_type_embed[data["x_type"]]  # (N, 128)

        edge_index, edge_attr = self.drop_edge(
            data["lane_actor_index"], data["lane_actor_vectors"]
        )
        out = self.al_encoder(
            x=(data["lane_vectors"], out),
            edge_index=edge_index,
            edge_attr=edge_attr,
            is_intersections=data["is_intersections"],
            rotate_mat=data["rotate_mat"],
        )
        return out  # [N, D]


class ALEncoder(MessagePassing):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super(ALEncoder, self).__init__(aggr="add", node_dim=0, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.lane_embed = MultipleInputEmbedding(
            in_channels=[node_dim, edge_dim], out_channel=embed_dim
        )
        self.lin_q = nn.Linear(embed_dim, embed_dim)
        self.lin_k = nn.Linear(embed_dim, embed_dim)
        self.lin_v = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.is_intersection_embed = nn.Parameter(torch.Tensor(2, embed_dim))
        nn.init.normal_(self.is_intersection_embed, mean=0.0, std=0.02)
        self.apply(init_weights)

    def forward(
        self,
        x: Tuple[torch.Tensor, torch.Tensor],
        edge_index: Adj,
        edge_attr: torch.Tensor,
        is_intersections: torch.Tensor,
        rotate_mat: Optional[torch.Tensor] = None,
        size: Size = None,
    ) -> torch.Tensor:
        x_lane, x_actor = x
        is_intersections = is_intersections.long()
        x_actor = x_actor + self._mha_block(
            self.norm1(x_actor),
            x_lane,
            edge_index,
            edge_attr,
            is_intersections,
            rotate_mat,
            size,
        )
        x_actor = x_actor + self._ff_block(self.norm2(x_actor))
        return x_actor

    def message(
        self,
        edge_index: Adj,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        is_intersections_j,
        rotate_mat: Optional[torch.Tensor],
        index: torch.Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> torch.Tensor:
        if rotate_mat is None:
            x_j = self.lane_embed(
                [x_j, edge_attr],
                [self.is_intersection_embed[is_intersections_j]],
            )
        else:
            rotate_mat = rotate_mat[edge_index[1]]
            x_j = self.lane_embed(
                [
                    torch.bmm(x_j.unsqueeze(-2), rotate_mat).squeeze(-2),
                    torch.bmm(edge_attr.unsqueeze(-2), rotate_mat).squeeze(-2),
                ],
                [self.is_intersection_embed[is_intersections_j]],
            )
        query = self.lin_q(x_i).view(
            -1, self.num_heads, self.embed_dim // self.num_heads
        )
        key = self.lin_k(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value = self.lin_v(x_j).view(
            -1, self.num_heads, self.embed_dim // self.num_heads
        )
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * key).sum(dim=-1) / scale
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_drop(alpha)
        return value * alpha.unsqueeze(-1)

    def update(self, inputs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x_actor = x[1]
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(x_actor))
        return inputs + gate * (self.lin_self(x_actor) - inputs)

    def _mha_block(
        self,
        x_actor: torch.Tensor,
        x_lane: torch.Tensor,
        edge_index: Adj,
        edge_attr: torch.Tensor,
        is_intersections: torch.Tensor,
        rotate_mat: Optional[torch.Tensor],
        size: Size,
    ) -> torch.Tensor:
        x_actor = self.out_proj(
            self.propagate(
                edge_index=edge_index,
                x=(x_lane, x_actor),
                edge_attr=edge_attr,
                is_intersections=is_intersections,
                rotate_mat=rotate_mat,
                size=size,
            )
        )
        return self.proj_drop(x_actor)

    def _ff_block(self, x_actor: torch.Tensor) -> torch.Tensor:
        return self.mlp(x_actor)
