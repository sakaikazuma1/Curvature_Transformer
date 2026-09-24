import argparse


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Runs test or train model.")

    parser.add_argument("--test_model_path", type=str, default=None)
    parser.add_argument("--train", action="store_true")

    parser.add_argument("--patch_ratio", type=float)

    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--run_id", type=str)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dataset_name", type=str, default="ElectricDevices")
    parser.add_argument("--use_ucr2018", action="store_true")

    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--output_dir", type=str, default="out")
    parser.add_argument("--epochs", type=int, default=50)

    parser.add_argument(
        "--patching_method",
        type=str,
        default="normal",
        choices=["normal", "derivative"],
    )

    parser.add_argument(
        "--model_arch",
        type=str,
        default="vit",
        choices=["vit", "patchtst"],
        help="Backbone selection. vit: existing 1D ViT. patchtst: PatchTST.",
    )

    parser.add_argument(
        "--patchtst_stride",
        type=int,
        default=None,
        help="Stride of the PatchTST patches in samples; patch_size//2 if omitted. "
        "With derivative, each curvature interval is split into patch_size//stride parts.",
    )

    parser.add_argument(
        "--padding_patch",
        type=str,
        default=None,
        choices=["end"],
        help="end: replicate the last value for one more stride before cutting fixed-size patches.",
    )
    parser.add_argument("--revin", action="store_true", help="RevIN normalization (no affine) before patching.")
    parser.add_argument("--revin_subtract_last", action="store_true")

    parser.add_argument("--curvature_scale", type=float, default=3)

    return parser


def parse_args() -> argparse.Namespace:
    return build_argparser().parse_args()
