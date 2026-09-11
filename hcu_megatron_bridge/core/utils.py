# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Utility functions used throughout Megatron core"""

from typing import List

import torch

from megatron.core.utils import (
    _missing_tilelang_dsa_kernel_dependencies, 
    _missing_cudnn_dsa_kernel_dependencies,
)


_VALID_DSA_KERNEL_BACKENDS = ("none", "tilelang", "cudnn", "hip")


def _missing_hip_dsa_kernel_dependencies() -> List[str]:
    """Return missing HIP DSA kernel dependencies."""
    missing = []
    try:
        from flash_mla import flash_mla_sparse_fwd  # noqa: F401
    except ImportError:
        missing.append("flash_mla " "(https://github.com/deepseek-ai/FlashMLA/tree/nv_dev)")
    try:
        from hip_dsa import DSA  # noqa: F401
    except ImportError:
        missing.append("HIP DSA namespace not available. Install the hip_dsa package")
    return missing


def _validate_dsa_kernel_backend_dependencies(dsa_kernel_backend: str) -> None:
    """Validate optional fused DSA kernel backend dependencies."""
    if dsa_kernel_backend not in _VALID_DSA_KERNEL_BACKENDS:
        raise ValueError(
            "dsa_kernel_backend must be one of: " f"{', '.join(_VALID_DSA_KERNEL_BACKENDS)}."
        )
    if dsa_kernel_backend == "none":
        return
    if not torch.cuda.is_available():
        raise ValueError(
            f"dsa_kernel_backend={dsa_kernel_backend} requires a CUDA device, "
            "but none is available."
        )

    missing = []
    if dsa_kernel_backend == "tilelang":
        missing = _missing_tilelang_dsa_kernel_dependencies()
    elif dsa_kernel_backend == "cudnn":
        missing = _missing_cudnn_dsa_kernel_dependencies()
    elif dsa_kernel_backend == "hip":
        missing = _missing_hip_dsa_kernel_dependencies()

    if missing:
        raise ValueError(
            f"dsa_kernel_backend={dsa_kernel_backend} requires fused DSA kernels, "
            f"but the following packages are not available: {', '.join(missing)}. "
            "Install them or set dsa_kernel_backend=none to use the PyTorch fallback."
        )