from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike


def second_derivative_magnitude(series: ArrayLike) -> np.ndarray:
    values = np.array(series, dtype=float)
    if len(values) < 2:
        return np.zeros_like(values)
    first_derivative = np.gradient(values)
    second_derivative = np.gradient(first_derivative)
    return np.asarray(np.abs(second_derivative))


def split_evenly_by_length(series_length: int, num_patches: int) -> list[int]:
    return [
        int(position) for position in np.linspace(0, series_length, num_patches + 1)
    ]


def split_evenly_by_curvature(
    series: np.ndarray,
    num_patches: int,
    curvature_scale: float,
) -> list[int]:
    if num_patches <= 0:
        return [0, len(series)]

    magnitude = second_derivative_magnitude(series)

    # Adding curvature_scale as a floor keeps a minimum weight even in flat regions.
    weighted_curvature = magnitude + curvature_scale
    total_curvature = np.sum(weighted_curvature)
    if total_curvature == 0 or not np.isfinite(total_curvature):
        return split_evenly_by_length(len(series), num_patches)

    cumulative_curvature = np.cumsum(weighted_curvature)
    curvature_per_patch = total_curvature / num_patches

    boundaries = [0]
    next_target = curvature_per_patch
    for boundary_index in range(num_patches):
        # Interior boundaries exclude the series end. Overlapping the end would make the set-based
        # deduplication drop one patch, so only the last boundary may pick the series end.
        candidates = cumulative_curvature[:-1] if boundary_index < num_patches - 1 and len(series) > 1 else cumulative_curvature
        nearest = int(np.argmin(np.abs(candidates - next_target)))
        boundaries.append(nearest + 1)
        next_target += curvature_per_patch
    boundaries[-1] = len(series)
    return boundaries


def first_duplicated_boundary(boundaries: list[int]) -> int | None:
    seen: set[int] = set()
    for boundary in boundaries[1:-1]:
        if boundary in seen:
            return boundary
        seen.add(boundary)
    return None


def map_to_original_indices(
    boundaries: list[int],
    remaining_indices: np.ndarray,
    locked_boundaries: list[int],
    padded_length: int,
) -> list[int]:
    mapped_boundaries: list[int] = []
    for boundary in boundaries:
        if boundary <= 0:
            mapped_boundaries.append(0)
        elif boundary >= len(remaining_indices):
            mapped_boundaries.append(padded_length)
        else:
            mapped_boundaries.append(int(remaining_indices[boundary]))
    return sorted(set(mapped_boundaries + locked_boundaries + [0, padded_length]))


def find_curvature_boundaries(
    unpadded_series: ArrayLike,
    padded_length: int,
    num_patches: int,
    curvature_scale: float,
) -> list[int]:
    remaining_series = np.array(unpadded_series, dtype=float)
    remaining_indices = np.arange(len(remaining_series))
    remaining_num_patches = num_patches
    locked_boundaries: list[int] = []

    boundaries: list[int] | None = None
    for _ in range(max(1, len(remaining_series))):
        boundaries = split_evenly_by_curvature(
            remaining_series, remaining_num_patches, curvature_scale
        )
        duplicated = first_duplicated_boundary(boundaries)
        if duplicated is None:
            break

        # A duplicated boundary can no longer move, so lock it, drop that sample and redraw the rest.
        if duplicated >= len(remaining_series):
            locked_boundaries.append(padded_length)
            sample_to_remove = len(remaining_series) - 1
        else:
            locked_boundaries.append(int(remaining_indices[duplicated]))
            sample_to_remove = duplicated

        if len(remaining_series) <= 1 or remaining_num_patches <= 1:
            break

        remaining_series = np.delete(remaining_series, sample_to_remove)
        remaining_indices = np.delete(remaining_indices, sample_to_remove)
        remaining_num_patches -= 1

    if boundaries is None:
        boundaries = [0, len(remaining_series)]

    return map_to_original_indices(
        boundaries, remaining_indices, locked_boundaries, padded_length
    )


def overlapping_spans(boundaries: list[int], stride_divisor: int) -> list[tuple[int, int]]:
    """Curvature boundaries (K+1,) -> spans ((K-1)*stride_divisor+1, 2).

    Each curvature interval is split into stride_divisor parts by index, and a patch spans
    stride_divisor consecutive parts, so the stride is 1/stride_divisor of a patch
    (stride_divisor=2: every interval plus the midpoint-to-midpoint patch between neighbours).
    """
    fine_boundaries = [
        start + (end - start) * part // stride_divisor
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
        for part in range(stride_divisor)
    ] + [boundaries[-1]]
    return [
        (fine_boundaries[i], fine_boundaries[i + stride_divisor])
        for i in range(len(fine_boundaries) - stride_divisor)
    ]


def patch_one_series(
    unpadded_series: Any,
    num_patches: int,
    curvature_scale: float,
    stride_divisor: int,
) -> list[np.ndarray]:
    # Add no padding to the series or the patches; slice only real samples.
    if len(unpadded_series) == 0:
        raise ValueError("Cannot patch an empty series")
    boundaries = find_curvature_boundaries(
        unpadded_series=unpadded_series,
        padded_length=len(unpadded_series),
        num_patches=num_patches,
        curvature_scale=curvature_scale,
    )
    # Duplicate boundaries are already removed. Even short series produce no empty or dummy patches.
    values = np.asarray(unpadded_series, dtype=np.float32)
    return [values[start:end] for start, end in overlapping_spans(boundaries, stride_divisor)]


def zero_pad_patches_to_rectangle(patches_per_sample, num_patches, patch_width):
    """After splitting every sample, right-pad with zeros to the fixed width patch_width."""
    longest_patch_length = max(len(p) for sample in patches_per_sample for p in sample)
    if longest_patch_length > patch_width:
        raise ValueError(
            f"patch_width {patch_width} is shorter than the longest patch "
            f"{longest_patch_length}; padding to it would truncate real samples"
        )
    patches = np.zeros((len(patches_per_sample), num_patches, patch_width),
                       dtype=np.float32)
    lengths = np.zeros((len(patches_per_sample), num_patches), dtype=np.int64)
    for i, sample in enumerate(patches_per_sample):
        if len(sample) > num_patches:
            raise ValueError("Boundary count exceeds requested patch count")
        for j, patch in enumerate(sample):
            lengths[i, j] = len(patch)
            patches[i, j, :len(patch)] = patch
    # Unused slots have length 0.
    return patches, lengths


def patch_series_batch(
    unpadded_series_batch: torch.Tensor,
    num_patches: int,
    curvature_scale: float,
    stride_divisor: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the whole dataset into variable-length patches, then align them to the series-length width."""
    # The embedding linear layer needs a fixed input width. A patch can never be longer than the series,
    # so using the series length as the width gives train and test the same width and never truncates.
    patch_width = int(unpadded_series_batch.shape[-1])
    patches_per_sample = [
        patch_one_series(series, num_patches, curvature_scale, stride_divisor)
        for series in unpadded_series_batch
    ]
    num_spans = (num_patches - 1) * stride_divisor + 1
    return zero_pad_patches_to_rectangle(patches_per_sample, num_spans, patch_width)


class DerivativePatcher:
    def __init__(
        self,
        patch_ratio: float,
        curvature_scale: float,
        stride_divisor: int,
    ) -> None:
        if not np.isfinite(patch_ratio) or not 0 < patch_ratio <= 100:
            raise ValueError("patch_ratio must be in (0, 100]")
        if not np.isfinite(curvature_scale) or curvature_scale < 0:
            raise ValueError("curvature_scale (lambda) must be finite and nonnegative")
        if stride_divisor < 1:
            raise ValueError("stride_divisor must be at least 1")
        self.num_patch = int(100 / patch_ratio)
        self.curvature_scale = curvature_scale
        self.stride_divisor = stride_divisor

    def forward(self, series_batch):
        patches, lengths = patch_series_batch(
            series_batch.squeeze(1),
            self.num_patch,
            self.curvature_scale,
            self.stride_divisor,
        )
        # Keep the dataset on the CPU and transfer it per mini-batch during training.
        return (
            torch.from_numpy(patches),
            torch.from_numpy(lengths),
            self.curvature_scale,
        )

    def __call__(self, series_batch):
        return self.forward(series_batch)
