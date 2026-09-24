from argparse import Namespace

from src.patches.derivative import DerivativePatcher
from src.patches.fixed import FixedPatcher

PATCHING_METHODS = ("normal", "derivative")


def check_patching_method(patching_method: str) -> None:
    if patching_method not in PATCHING_METHODS:
        raise ValueError(
            f"Unknown patching_method: {patching_method!r}. "
            f"Valid choices: {sorted(PATCHING_METHODS)}"
        )


def patch_stride(args: Namespace, patch_size: int) -> int:
    # ViT keeps non-overlapping patches; PatchTST overlaps them as in the original (default patch_size // 2).
    if args.model_arch != "patchtst":
        return patch_size
    if args.patchtst_stride is None:
        return patch_size // 2
    return args.patchtst_stride


def build_derivative_patcher(args: Namespace, stride: int, patch_size: int) -> DerivativePatcher:
    if args.padding_patch is not None:
        raise ValueError("padding_patch applies only to fixed-size patches (patching_method=normal)")
    return DerivativePatcher(
        patch_ratio=args.patch_ratio,
        curvature_scale=args.curvature_scale,
        # A stride of patch_size / k splits each curvature interval into k parts.
        stride_divisor=patch_size // stride,
    )


def build_patcher(args: Namespace, patch_size: int) -> DerivativePatcher | FixedPatcher:
    check_patching_method(args.patching_method)
    stride = patch_stride(args, patch_size)
    if args.patching_method == "derivative":
        return build_derivative_patcher(args, stride, patch_size)
    return FixedPatcher(
        patch_ratio=args.patch_ratio,
        patch_size=patch_size,
        stride=stride,
        padding_patch=args.padding_patch,
    )
