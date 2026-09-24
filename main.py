import math
from argparse import Namespace
from typing import Any

from torch import nn, optim

from src.architectures.patchtst import PatchTST
from src.architectures.vit import ViT
from src.args import parse_args
from src.device import get_device
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.modes import build_patcher
from src.preprocess_data.dataloader_builder import build_dataloaders, peek_seq_len


def initial_patch_size(args: Namespace) -> int:
    seq_len = peek_seq_len(args.dataset_name, args.use_ucr2018)
    return math.ceil(int(args.patch_ratio) * int(seq_len) / 100)


def build_patchtst(
    num_patches: int, patch_width: int, num_classes: int, device: Any
) -> PatchTST:
    return PatchTST(
        patch_width=patch_width,
        num_patches=num_patches,
        num_classes=num_classes,
        embedding_dim=64,
        num_transformer_layers=4,
        num_attention_heads=4,
        mlp_hidden_dim=512,
        attention_head_dim=64,
        dropout=0.1,
    ).to(device)


def build_vit(
    num_patches: int, patch_width: int, num_classes: int, device: Any
) -> ViT:
    return ViT(
        patch_width=patch_width,
        num_patches=num_patches,
        num_classes=num_classes,
        embedding_dim=64,
        num_transformer_layers=4,
        num_attention_heads=4,
        mlp_hidden_dim=512,
        attention_head_dim=64,
        dropout=0.1,
        emb_dropout=0.1,
    ).to(device)


def build_model(
    args: Namespace, num_patches: int, patch_width: int, num_classes: int, device: Any
) -> PatchTST | ViT:
    if args.model_arch == "patchtst":
        return build_patchtst(num_patches, patch_width, num_classes, device)
    return build_vit(num_patches, patch_width, num_classes, device)


def main() -> None:
    args = parse_args()
    device = get_device()

    patch_size = initial_patch_size(args)
    num_classes, train_loader, test_loader, (num_patches, patch_width), curvature_coverage = (
        build_dataloaders(args, build_patcher(args, patch_size))
    )

    model = build_model(
        args,
        num_patches=num_patches,
        patch_width=patch_width,
        num_classes=num_classes,
        device=device,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    if args.train:
        Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            device=device,
            epochs=args.epochs,
            dataset_name=args.dataset_name,
            model_dir_label=args.model_dir,
            run_id=args.run_id,
            out_label=args.output_dir,
            curvature_coverage=curvature_coverage,
        ).run()

    if args.test_model_path is not None:
        Evaluator(
            model=model,
            test_loader=test_loader,
            device=device,
            dataset_name=args.dataset_name,
            model_dir_label=args.model_dir,
            out_label=args.output_dir,
            # Logged as before: the fixed-size patch width derived from patch_ratio and the series length.
            patch_size=patch_size,
            test_model_path=args.test_model_path,
        ).run()


if __name__ == "__main__":
    main()
