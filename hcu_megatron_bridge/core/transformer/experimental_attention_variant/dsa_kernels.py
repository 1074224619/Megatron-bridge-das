# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Backend-neutral hooks for optional fused DeepSeek sparse attention kernels."""

from megatron.core.transformer.transformer_config import TransformerConfig


_BACKEND_MODULE_NAME_BY_BACKEND = {
    "tilelang": "megatron.core.transformer.experimental_attention_variant.dsa_tilelang_kernels",
    "cudnn": "megatron.core.transformer.experimental_attention_variant.dsa_cudnn_kernels",
    "hip": "hcu_megatron_bridge.core.transformer.experimental_attention_variant.dsa_hip_kernels",
}


def _get_dsa_kernel_backend(config: TransformerConfig) -> str:
    """Return the configured DSA kernel backend."""
    backend = config.dsa_kernel_backend
    if backend != "none" and backend not in _BACKEND_MODULE_NAME_BY_BACKEND:
        raise ValueError("dsa_kernel_backend must be one of: none, tilelang, cudnn, hip")
    return backend