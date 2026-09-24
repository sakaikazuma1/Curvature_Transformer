import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from src.preprocess_data import dataset_metadata


def load_values_and_labels(
    data_file: str, label_file: str | None = None, delimiter: str = " "
) -> tuple[np.ndarray, np.ndarray]:
    if label_file:
        values = np.genfromtxt(data_file, delimiter=delimiter)
        labels = np.genfromtxt(label_file, delimiter=delimiter)
        if labels.ndim > 1:
            labels = labels[:, 1]
        return values, labels

    # Without a label_file, column 0 is treated as the label.
    values = np.genfromtxt(data_file, delimiter=delimiter)
    return values[:, 1:], values[:, 0]


def read_train_test_split(
    train_file: str,
    train_label: str | None = None,
    test_file: str | None = None,
    test_label: str | None = None,
    test_split: float = 0.1,
    delimiter: str = " ",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_values, train_labels = load_values_and_labels(
        train_file, train_label, delimiter
    )
    if test_file:
        test_values, test_labels = load_values_and_labels(
            test_file, test_label, delimiter
        )
        return train_values, train_labels, test_values, test_labels

    test_size = int(test_split * float(train_labels.shape[0]))
    return (
        train_values[test_size:],
        train_labels[test_size:],
        train_values[:test_size],
        train_labels[:test_size],
    )


def preset_tsv_paths(dataset_name: str) -> tuple[str, str]:
    return (
        os.path.join("data", dataset_name, f"{dataset_name}_TRAIN.tsv"),
        os.path.join("data", dataset_name, f"{dataset_name}_TEST.tsv"),
    )


def preset_txt_paths(dataset_name: str) -> tuple[str, str, str, str]:
    return (
        os.path.join("data", f"train-{dataset_name}s-data.txt"),
        os.path.join("data", f"train-{dataset_name}s-labels.txt"),
        os.path.join("data", f"test-{dataset_name}s-data.txt"),
        os.path.join("data", f"test-{dataset_name}s-labels.txt"),
    )


def scale_to_unit_range(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    train_max = np.nanmax(x_train)
    train_min = np.nanmin(x_train)
    # Do not use test statistics; reuse the train min/max.
    return (
        2.0 * (x_train - train_min) / (train_max - train_min) - 1.0,
        2.0 * (x_test - train_min) / (train_max - train_min) - 1.0,
    )


def to_series_tensor(values: np.ndarray, dataset_name: str) -> torch.Tensor:
    """(N, dims*timesteps) -> (N, dims, timesteps)"""
    num_dims = dataset_metadata.nb_dims(dataset_name)
    num_timesteps = int(values.shape[1] / num_dims)
    return torch.tensor(
        values.reshape((-1, num_dims, num_timesteps)), dtype=torch.float32
    )


def to_one_hot(labels: np.ndarray, dataset_name: str) -> torch.Tensor:
    """(N,) -> (N, num_classes)"""
    class_indices = dataset_metadata.class_offset(labels, dataset_name)
    return F.one_hot(
        torch.tensor(class_indices, dtype=torch.long),
        num_classes=dataset_metadata.nb_classes(dataset_name),
    )


def load_tsv_datasets(
    dataset_name: str,
    preset_files: bool = False,
    ucr: bool = False,
    use_ucr2018: bool = False,
    train_data_file: str | None = None,
    train_labels_file: str | None = None,
    test_data_file: str | None = None,
    test_labels_file: str | None = None,
    test_split: float = 0,
    delimiter: str = " ",
    normalize_input: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not preset_files:
        if train_data_file is None:
            raise ValueError("preset_files=False のときは train_data_file が必要です")
        x_train, y_train, x_test, y_test = read_train_test_split(
            train_data_file,
            train_labels_file,
            test_data_file,
            test_labels_file,
            test_split=test_split,
            delimiter=delimiter,
        )
    elif ucr:
        train_path, test_path = preset_tsv_paths(dataset_name)
        x_train, y_train, x_test, y_test = read_train_test_split(
            train_path, "", test_path, "", delimiter=","
        )
    elif use_ucr2018:
        train_path, test_path = preset_tsv_paths(dataset_name)
        x_train, y_train, x_test, y_test = read_train_test_split(
            train_path, "", test_path, "", delimiter="\t"
        )
    else:
        train_path, train_label_path, test_path, test_label_path = preset_txt_paths(
            dataset_name
        )
        x_train, y_train, x_test, y_test = read_train_test_split(
            train_path,
            train_label_path,
            test_path,
            test_label_path,
            test_split=test_split,
            delimiter=delimiter,
        )

    if normalize_input:
        x_train, x_test = scale_to_unit_range(x_train, x_test)

    x_train = np.nan_to_num(x_train)
    x_test = np.nan_to_num(x_test)

    return (
        to_series_tensor(x_train, dataset_name),
        to_one_hot(y_train, dataset_name),
        to_series_tensor(x_test, dataset_name),
        to_one_hot(y_test, dataset_name),
    )


def read_ts_file(
    data_file: str, delimiter: str, dimensions: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    series_rows: list[Any] = []
    label_names: list[str] = []

    with open(data_file) as ts_file:
        for line in tqdm(ts_file):
            line = line.strip()
            if line.startswith("@") or line == "":
                continue
            series_text, label_text = line.split(":")
            series_rows.append(list(map(float, series_text.split(delimiter))))
            label_names.append(label_text.strip())

    encoder = LabelEncoder()
    class_indices = encoder.fit_transform(label_names)
    num_classes = len(encoder.classes_)

    values = np.array(series_rows)
    series = torch.tensor(
        values.reshape(values.shape[0], dimensions, values.shape[1]),
        dtype=torch.float32,
    )
    one_hot = F.one_hot(
        torch.tensor(class_indices, dtype=torch.long), num_classes=num_classes
    )
    return series, one_hot, num_classes


def load_ts_dataset(
    dataset_name: str, delimiter: str, dimensions: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    test_path = os.path.join("data", dataset_name, f"{dataset_name}_TEST.ts")
    train_path = os.path.join("data", dataset_name, f"{dataset_name}_TRAIN.ts")
    test_series, test_labels, _ = read_ts_file(
        test_path, delimiter, dimensions
    )
    train_series, train_labels, num_classes = read_ts_file(
        train_path, delimiter, dimensions
    )
    return train_series, train_labels, test_series, test_labels, num_classes
