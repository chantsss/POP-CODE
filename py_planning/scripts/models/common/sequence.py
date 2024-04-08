import torch
import torch.nn as nnfrom models.common.utils import init_weights

"""
A bunch of modules that deal with sequential input.
general i/o:
    sequence input: (batch_size, num_seq, seq_len, dim_in)
    output: (batch, num_seq, dim_out)
"""


class Subgraph(nn.Module):
    def __init__(self, dim, layers: int) -> None:
        """
        Implementation of the subgraph of VectorNet
        """
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.ReLU())
            for _ in range(layers)
        )
        self.apply(init_weights)

    def forward(self, x, mask):
        """make sure that at least one non-padding element is in each sequence, otherwise there will be nan in the output
        x: [batch_size, num_seq, seq_len, dim]
        mask: [batch_size, num_seq, seq_len]
        out:[batch_size, num_seq, dim]
        """
        for layer in self.layers:
            x = layer(x)
            masked_x = x.masked_fill(mask.unsqueeze(-1), float("-inf"))
            x_agg = masked_x.max(dim=-2, keepdim=True)[0]
            assert torch.isfinite(x_agg).all()
            x = torch.cat([x, x_agg.expand_as(x)], dim=-1)
        return torch.max(x, dim=-2).values


class SubgraphLarge(nn.Module):
    def __init__(self, dim, layers: int) -> None:
        """
        Implementation of the subgraph of VectorNet
        """
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())
            for _ in range(layers)
        )
        self.remaps = nn.ModuleList(nn.Linear(2 * dim, dim) for _ in range(layers))
        self.out_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 2 * dim),
            nn.ReLU(),
            nn.Linear(2 * dim, dim),
        )

        self.apply(init_weights)

    def forward(self, x, mask):
        """make sure that at least one non-padding element is in each sequence, otherwise there will be nan in the output
        x: [batch_size, num_seq, seq_len, dim]
        mask: [batch_size, num_seq, seq_len]
        out:[batch_size, num_seq, dim]
        """
        for layer, remap in zip(self.layers, self.remaps):
            shortcut = x
            x = layer(x)
            masked_x = x.masked_fill(mask.unsqueeze(-1), float("-inf"))
            x_agg = masked_x.max(dim=-2, keepdim=True)[0]
            assert torch.isfinite(x_agg).all()
            x = torch.cat([x, x_agg.expand_as(x)], dim=-1)
            x = shortcut + remap(x)

        out = torch.max(x, dim=-2).values  # max_pooling
        out = out + self.out_mlp(out)
        return out


class SubgraphAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class MnMBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        """
        Implementation https://arxiv.org/abs/2207.00738, MnM means Mix & Match
        """
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())
        self.remap = nn.Linear(2 * dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.layerNorm(dim)
        self.apply(init_weights)

    def forward(self, x, mask):
        """
        x: [batch_size, seq_len, dim]
        mask: [batch_size, seq_len]
        out: [batch_size, seq_len, dim]
        """
        mask_x = self.norm1(x).masked_fill(mask.unsqueeze(-1), float("-inf"))
        x_mix = mask_x.max(dim=-2, keepdim=True)[0]
        assert torch.isfinite(x_mix).all()
        x_match = x + self.remap(torch.cat([x, x_mix.expand_as(x)], dim=-1))
        out = x_match + self.mlp(self.norm2(x_match))
        return out
