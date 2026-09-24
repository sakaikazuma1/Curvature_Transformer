import os
from datetime import datetime
from typing import Any

import torch
from tqdm import tqdm

from src.engine.batch_unpacking import forward_batch

RUNNING_LOSS_WINDOW = 10
PROGRESS_BAR_FORMAT = "{n_fmt}/{total_fmt} [{bar}] {percentage:3.0f}%"


def train_one_batch(
    model: Any, optimizer: Any, criterion: Any, batch: Any, device: Any
) -> float:
    optimizer.zero_grad()
    logits, labels = forward_batch(model, batch, device)
    loss = criterion(logits.squeeze(1), labels.argmax(dim=-1))

    loss.backward()
    optimizer.step()
    return float(loss.item())


def train_one_epoch(
    model: Any,
    optimizer: Any,
    criterion: Any,
    train_loader: Any,
    device: Any,
    epoch_index: int,
    progress_bar: Any,
) -> float:
    running_loss = 0.0
    epoch_loss = 0.0
    batch_count = 0

    for batch_index, batch in enumerate(train_loader):
        loss_value = train_one_batch(model, optimizer, criterion, batch, device)

        running_loss += loss_value
        epoch_loss += loss_value
        batch_count += 1

        if batch_index % RUNNING_LOSS_WINDOW == RUNNING_LOSS_WINDOW - 1:
            progress_bar.set_postfix(
                Epoch=epoch_index + 1,
                Batch=f"{batch_index + 1}/{len(train_loader)}",
                Loss=f"{running_loss / RUNNING_LOSS_WINDOW:.3f}",
            )
            running_loss = 0.0

        progress_bar.set_postfix(Epoch=epoch_index + 1, Batch=batch_index + 1)
        progress_bar.update(1)

    if batch_count == 0:
        return 0
    return epoch_loss / batch_count


def run_epochs(
    model: Any,
    optimizer: Any,
    criterion: Any,
    train_loader: Any,
    device: Any,
    epochs: int,
) -> list[float]:
    epoch_losses = []
    total_steps = epochs * len(train_loader)
    with tqdm(
        total=total_steps, bar_format=PROGRESS_BAR_FORMAT, desc="Total Progress"
    ) as progress_bar:
        for epoch_index in range(epochs):
            epoch_losses.append(
                train_one_epoch(
                    model,
                    optimizer,
                    criterion,
                    train_loader,
                    device,
                    epoch_index,
                    progress_bar,
                )
            )
    return epoch_losses


def save_checkpoint(
    model: Any, model_dir_label: str, dataset_name: str, run_id: str
) -> None:
    timestamp = datetime.now().strftime("%m%d_%H%M")
    model_dir = f"./models/{model_dir_label}_{dataset_name}"
    os.makedirs(model_dir, exist_ok=True)

    checkpoint_path = f"{model_dir}/run_{run_id}_{timestamp}.pth"
    torch.save(model.state_dict(), checkpoint_path)

    # Append so a later test run can look up the last saved weights.
    with open(f"{model_dir}/modelspath.txt", "a") as path_file:
        path_file.write(checkpoint_path + "\n")


def append_loss_log(
    out_label: str,
    dataset_name: str,
    curvature_coverage: float,
    epoch_losses: list[float],
) -> None:
    mean_loss = sum(epoch_losses) / len(epoch_losses)
    log_path = f"outputs/txt/{out_label}.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as log_file:
        log_file.write(f"dataset_{dataset_name}_C_{curvature_coverage}_loss:{mean_loss}\n")


class Trainer:
    def __init__(
        self,
        *,
        model: Any,
        optimizer: Any,
        criterion: Any,
        train_loader: Any,
        device: Any,
        epochs: int,
        dataset_name: str,
        model_dir_label: str,
        run_id: str,
        out_label: str,
        curvature_coverage: float,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.device = device
        self.epochs = epochs
        self.dataset_name = dataset_name
        self.model_dir_label = model_dir_label
        self.run_id = run_id
        self.out_label = out_label
        self.curvature_coverage = curvature_coverage

    def run(self) -> None:
        self.model.train()
        epoch_losses = run_epochs(
            self.model,
            self.optimizer,
            self.criterion,
            self.train_loader,
            self.device,
            self.epochs,
        )
        save_checkpoint(
            self.model, self.model_dir_label, self.dataset_name, self.run_id
        )
        print("Finished Training")
        append_loss_log(
            self.out_label, self.dataset_name, self.curvature_coverage, epoch_losses
        )
