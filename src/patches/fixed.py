import torch
from torch import nn


def pad_to_patch_grid(
    x: torch.Tensor, patch_size: int, num_patch: float
) -> torch.Tensor:
    """(N, 1, L) -> (N, 1, patch_size*num_patch)"""
    length = x.shape[-1]
    if length % (patch_size * num_patch) == 0:
        return x

    padding_shape = list(x.shape)
    padding_shape[-1] = int(patch_size * num_patch - length)
    padding = torch.zeros(*padding_shape, device=x.device, dtype=x.dtype)
    return torch.cat([x, padding], dim=-1)


class FixedPatcher:
    """Pad the series onto the patch grid, then cut patches of patch_size every stride samples."""

    def __init__(
        self, patch_ratio: float, patch_size: int, stride: int, padding_patch: str | None
    ) -> None:
        self.patch_ratio = patch_ratio
        self.patch_size = patch_size
        self.stride = stride
        # PatchTST's padding_patch="end": replicate the last value for one more stride.
        self.end_padding = (
            nn.ReplicationPad1d((0, stride)) if padding_patch == "end" else nn.Identity()
        )

    def forward(self, series_batch):
        """(N, 1, L) -> (N, num_patches, patch_size), (N, num_patches), 0"""
        padded = self.end_padding(
            pad_to_patch_grid(series_batch, self.patch_size, self.patch_ratio)
        )
        patches = padded.squeeze(1).unfold(-1, self.patch_size, self.stride).contiguous()
        # Every fixed-size patch is full.
        lengths = torch.full(patches.shape[:2], self.patch_size, dtype=torch.int64)
        # No curvature is used, so the coverage is 0.
        return patches, lengths, 0

    def __call__(self, series_batch):
        return self.forward(series_batch)
