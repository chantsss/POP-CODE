import copyfrom typing import Optional

import torch
import torch.nn as nnfrom models.common.utils import init_weights


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_feature=None,
        out_features=None,
        act_layer=nn.ReLU,
        dropout=0.0,
    ):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_feature = hidden_feature or in_features
        self.fc1 = nn.Linear(in_features, hidden_feature)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_feature, out_features)
        self.dropout = nn.Dropout(dropout)
        self.apply(init_weights)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super(SinusoidalPositionalEncoding, self).__init__()
        # Positional Encoder
        pe = torch.zeros(max_len, d_model)
        t = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        log_value = torch.log(torch.tensor([1e4])).item()
        omega = torch.exp((-log_value / d_model) * torch.arange(0, d_model, 2).float())
        pe[:, 0::2] = torch.sin(t * omega)
        pe[:, 1::2] = torch.cos(t * omega)
        self.register_buffer("static_embedding", pe.unsqueeze(0))  # (1, n_seq, d_model)

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.static_embedding[:, :seq_len]


class MHA(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout=0.0):
        super(MHA, self).__init__()
        self._q = nn.Linear(dim, dim)
        self._k = nn.Linear(dim, dim)
        self._v = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=dropout
        )
        self.apply(init_weights)

    def forward(self, q, k, v, key_padding_mask=None, attn_mask=None):
        Q = self._q(q)
        K = self._k(k)
        V = self._v(v)

        out, _ = self.attn(
            Q, K, V, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )
        return out


class GatedMHA(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout=0.0):
        super(GatedMHA, self).__init__()
        self._q = nn.Linear(dim, dim)
        self._k = nn.Linear(dim, dim)
        self._v = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=dropout
        )

        self.apply(init_weights)

    def forward(self, q, k, v, key_padding_mask=None, attn_mask=None):
        Q = self._q(q)
        K = self._k(k)
        V = self._v(v)

        out, _ = self.attn(
            Q, K, V, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )
        return out


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class TFEncoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, dropout=0.0) -> None:
        super(TFEncoderLayer, self).__init__()
        self.attn = MHA(dim, num_heads, dropout)
        self.ff = MLP(dim, dim * mlp_ratio, dropout=dropout)
        self.sublayer = clones(SublayerConnection(dim, dropout), 2)
        self.apply(init_weights)

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        x = self.sublayer[0](
            x, lambda x: self.attn(x, x, x, key_padding_mask, attn_mask)
        )
        return self.sublayer[1](x, self.ff)


class TFEncoderLayerSwinv2(nn.Module):
    """Transformer Encoder Layer from swin-v2"""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, dropout=0.0) -> None:
        super(TFEncoderLayerSwinv2, self).__init__()
        self.attn = MHA(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * mlp_ratio, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.apply(init_weights)

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        shortcut = x
        x = self.attn(x, x, x, key_padding_mask, attn_mask)
        x = shortcut + self.norm1(x)
        x = x + self.norm2(self.dropout(self.mlp(x)))
        return x


class TFDecoderLayer(nn.Module):
    """Classic transformer decoder layer, consisting of self-attn, residual, cross-attn"""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, dropout=0.0) -> None:
        super(TFDecoderLayer, self).__init__()
        self.self_attn = MHA(dim, num_heads, dropout)
        self.cross_attn = MHA(dim, num_heads, dropout)
        self.sublayer = clones(SublayerConnection(dim, dropout), 3)
        self.ff = MLP(dim, dim * mlp_ratio, dropout=dropout)
        self.apply(init_weights)

    def forward(
        self,
        x,
        memory,
        src_key_padding_mask=None,
        src_attn_mask=None,
        tgt_key_padding_mask=None,
        tgt_attn_mask=None,
    ):
        m = memory
        x = self.sublayer[0](
            x, lambda x: self.self_attn(x, x, x, src_key_padding_mask, src_attn_mask)
        )
        x = self.sublayer[1](
            x, lambda x: self.cross_attn(x, m, m, tgt_key_padding_mask, tgt_attn_mask)
        )
        return self.sublayer[2](x, self.ff)


class CrossAttentionLayer(nn.Module):
    """shortcut connection + cross attention"""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, dropout=0.0) -> None:
        super(CrossAttentionLayer, self).__init__()
        self.attn = MHA(dim, num_heads, dropout)
        self.ff = MLP(dim, dim * mlp_ratio, dropout=dropout)
        self.sublayer = clones(SublayerConnection(dim, dropout), 2)

    def forward(self, x, enc, key_padding_mask=None, attn_mask=None):
        x = self.sublayer[0](
            x, lambda x: self.attn(x, enc, enc, key_padding_mask, attn_mask)
        )
        return self.sublayer[1](x, self.ff)


class TFEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int,
        num_layers: int,
        encoder_layer: Optional[TFEncoderLayerSwinv2] = TFEncoderLayer,
        dropout=0.0,
    ) -> None:
        super(TFEncoder, self).__init__()
        self.layers = nn.ModuleList(
            encoder_layer(dim, num_heads, mlp_ratio, dropout=dropout)
            for _ in range(num_layers)
        )

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        for layer in self.layers:
            x = layer(x, key_padding_mask, attn_mask)
        return x


class TFDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int,
        num_layers: int,
        dropout=0.0,
    ) -> None:
        super(TFDecoder, self).__init__()
        self.layers = nn.ModuleList(
            TFDecoderLayer(dim, num_heads, mlp_ratio, dropout=dropout)
            for _ in range(num_layers)
        )

    def forward(
        self,
        x,
        memory,
        src_key_padding_mask=None,
        src_attn_mask=None,
        tgt_key_padding_mask=None,
        tgt_attn_mask=None,
    ):
        for layer in self.layers:
            x = layer(
                x,
                memory,
                src_key_padding_mask,
                src_attn_mask,
                tgt_key_padding_mask,
                tgt_attn_mask,
            )
        return x
