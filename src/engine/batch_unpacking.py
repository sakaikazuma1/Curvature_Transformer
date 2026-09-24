from typing import Any

import torch


def unpack_batch(
    batch: Any, device: Any
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Every dataset yields (patches, patch lengths, labels).
    patches, patch_lengths, labels = batch
    return patches.to(device), patch_lengths.to(device), labels.to(device)


def forward_batch(model: Any, batch: Any, device: Any) -> tuple[torch.Tensor, torch.Tensor]:
    patches, patch_lengths, labels = unpack_batch(batch, device)
    return model(patches, patch_lengths=patch_lengths), labels
