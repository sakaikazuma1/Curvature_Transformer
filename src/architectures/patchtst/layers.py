# Minimal version of the original PatchTST without the PE variants unused for classification (Coord1d/2d, lin*, exp*)
# and without moving_avg / series_decomp.

import torch
from torch import nn


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims = dims
        self.contiguous = contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


def get_activation_fn(activation):
    if callable(activation):
        return activation()
    if activation.lower() == "relu":
        return nn.ReLU()
    if activation.lower() == "gelu":
        return nn.GELU()
    raise ValueError(
        f"{activation} is not available. Use 'relu', 'gelu', or a callable."
    )


def positional_encoding(q_len, d_model):
    # Learnable absolute positional embedding.
    w = torch.empty((q_len, d_model))
    nn.init.uniform_(w, -0.02, 0.02)
    return nn.Parameter(w)
