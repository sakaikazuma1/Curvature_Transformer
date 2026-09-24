import os
from typing import Any

import torch

from src.engine.batch_unpacking import forward_batch


def resolve_model_path(
    test_model_path: str | None, model_dir_label: str, dataset_name: str
) -> str:
    if test_model_path is not None:
        return test_model_path

    # If not given, use the last weights appended by training.
    model_path_file = f"./models/{model_dir_label}_{dataset_name}/modelspath.txt"
    with open(model_path_file) as path_file:
        return path_file.readlines()[-1].strip()


def load_weights(model: Any, checkpoint_path: str, device: Any) -> None:
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )


def count_correct(model: Any, test_loader: Any, device: Any) -> tuple[int, int]:
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            logits, labels = forward_batch(model, batch, device)
            predictions = torch.argmax(logits.squeeze(1), dim=-1)

            correct += int(torch.sum(predictions == labels.argmax(dim=-1)).item())
            total += len(labels)
    return correct, total


def append_accuracy_log(out_label: str, patch_size: int, accuracy: float) -> None:
    log_path = f"outputs/txt/{out_label}.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as log_file:
        log_file.write(f"patch_size={patch_size},accuracy={accuracy:.4f}\n")


class Evaluator:
    def __init__(
        self,
        *,
        model: Any,
        test_loader: Any,
        device: Any,
        dataset_name: str,
        model_dir_label: str,
        out_label: str,
        patch_size: int,
        test_model_path: str | None = None,
    ) -> None:
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.dataset_name = dataset_name
        self.model_dir_label = model_dir_label
        self.out_label = out_label
        self.patch_size = patch_size
        self.test_model_path = test_model_path

    def run(self) -> None:
        load_weights(
            self.model,
            resolve_model_path(
                self.test_model_path, self.model_dir_label, self.dataset_name
            ),
            self.device,
        )
        self.model.eval()

        correct, total = count_correct(self.model, self.test_loader, self.device)
        accuracy = correct / total
        print(f"Accuracy: {accuracy:.4f}")

        append_accuracy_log(self.out_label, self.patch_size, accuracy)
