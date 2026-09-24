# RevIN (Kim et al., 2021) normalization, moved out of the PatchTST backbone into the preprocessing.
# The learnable affine weight cannot live in the preprocessing, so only the statistics part is kept.

import torch

REVIN_EPS = 1e-5


def revin_normalize(x: torch.Tensor, subtract_last: bool) -> torch.Tensor:
    """(N, 1, L) -> (N, 1, L)"""
    center = x[..., -1:] if subtract_last else x.mean(dim=-1, keepdim=True)
    stdev = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + REVIN_EPS)
    return (x - center) / stdev
