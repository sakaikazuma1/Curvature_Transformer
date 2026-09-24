from argparse import Namespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.preprocess_data import dataset_metadata
from src.preprocess_data.revin import revin_normalize
from src.preprocess_data.raw_file_reader import load_ts_dataset, load_tsv_datasets

STD_EPSILON = 1e-8


def peek_seq_len(dataset_name: str, use_ucr2018: bool) -> int:
    # Only seq_len is needed before the main pass. The data is loaded twice, matching the original behavior.
    if use_ucr2018:
        x_train, _, _, _ = load_tsv_datasets(
            dataset_name, preset_files=True, use_ucr2018=True
        )
    else:
        x_train, _, _, _, _ = load_ts_dataset(dataset_name, delimiter="", dimensions=1)
    return int(x_train.shape[-1])


def load_split(
    args: Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if not args.use_ucr2018:
        return load_ts_dataset(args.dataset_name, delimiter=",", dimensions=1)

    x_train, y_train, x_test, y_test = load_tsv_datasets(
        args.dataset_name, preset_files=True, use_ucr2018=True
    )
    return (
        x_train,
        y_train,
        x_test,
        y_test,
        dataset_metadata.nb_classes(args.dataset_name),
    )


def standardize(values: np.ndarray) -> np.ndarray:
    """(N, L) -> (N, L)"""
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    std = np.where(std < STD_EPSILON, 1.0, std)
    return np.asarray((values - mean) / std)


def normalize_per_sample(
    x_train: torch.Tensor, x_test: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, 1, L) -> (N, 1, L)"""
    train_values = np.asarray(x_train.squeeze(), dtype=np.float32)
    test_values = np.asarray(x_test.squeeze(), dtype=np.float32)

    # Keep a channel axis, shaping it as (N, 1, L) like the other series tensors.
    return (
        torch.from_numpy(standardize(train_values)).unsqueeze(1),
        torch.from_numpy(standardize(test_values)).unsqueeze(1),
    )


def build_dataloaders(
    args: Namespace, patcher: Any
) -> tuple[int, DataLoader, DataLoader, tuple[int, int], float]:
    x_train, y_train, x_test, y_test, num_classes = load_split(args)
    x_train, x_test = normalize_per_sample(x_train, x_test)
    if args.revin:
        x_train = revin_normalize(x_train, args.revin_subtract_last)
        x_test = revin_normalize(x_test, args.revin_subtract_last)

    patched_train, patch_lengths_train, curvature_coverage = patcher(x_train)
    patched_test, patch_lengths_test, _ = patcher(x_test)
    train_dataset = TensorDataset(patched_train, patch_lengths_train, y_train)
    test_dataset = TensorDataset(patched_test, patch_lengths_test, y_test)
    # (num_patches, patch_width) of every sample; the models are sized from it.
    patch_shape = (int(patched_train.shape[1]), int(patched_train.shape[2]))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return num_classes, train_loader, test_loader, patch_shape, curvature_coverage
