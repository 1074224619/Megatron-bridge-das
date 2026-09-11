# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from typing import Tuple

import torch
from torch import Tensor

import megatron.core.transformer.experimental_attention_variant.dsa_cudnn_kernels as _upstream
from megatron.core.transformer.experimental_attention_variant.dsa_cudnn_kernels import (
    _get_topk_alignment,
    _get_head_padding,
    round_up_to_nearest_multiple,
)


_flash_mla_sparse_fwd = None
run_fused_qk_topk = _upstream.run_fused_qk_topk
run_fused_qk_topk_with_loss = _upstream.run_fused_qk_topk_with_loss
run_fused_absorbed_sparse_attention = _upstream.run_fused_absorbed_sparse_attention
run_fused_dsa_attention = _upstream.run_fused_dsa_attention


def _ensure_flash_mla() -> None:
    """Lazily import the FlashMLA sparse-forward kernel.

    FlashMLA ships ``flash_mla_sparse_fwd`` with a multi-head-KV signature;
    :func:`_dsa_fwd_flash_mla` below adapts the DSA-shape inputs and pads
    ``topk`` to the alignment expected by the active FlashMLA sparse-prefill kernel.
    """
    global _flash_mla_sparse_fwd
    if _flash_mla_sparse_fwd is not None:
        return

    try:
        from flash_mla import flash_mla_sparse_fwd as _fwd
    except ImportError as e:
        raise ImportError(
            "FlashMLA is required for DSA sparse attention forward. "
            "Install from https://github.com/deepseek-ai/FlashMLA/tree/nv_dev "
            "so that `from flash_mla import flash_mla_sparse_fwd` succeeds."
        ) from e
    _flash_mla_sparse_fwd = _fwd


def _ensure_dsa_namespace_hip() -> None:
    """Try cudnn first, fall back to hip_dsa. Writes to the UPSTREAM module's _cudnn_dsa."""
    if _upstream._cudnn_dsa is not None:
        return
    try:
        from cudnn import DSA as _ns
    except ImportError as cudnn_error:
        try:
            from hip_dsa import DSA as _ns
        except ImportError as hip_error:
            raise ImportError(
                "No DSA namespace available. Install nvidia-cudnn-frontend[cutedsl] or hip_dsa. "
                f"(cudnn: {cudnn_error!r}; hip: {hip_error!r})"
            ) from None
    _upstream._cudnn_dsa = _ns 


def _dsa_fwd_flash_mla(
    q: Tensor,
    kv: Tensor,
    topk_idxs: Tensor,
    softmax_scale: float,
    d_v: int,
    attn_sink: Tensor,
    topk_length: Tensor,
) -> Tuple[Tensor, Tensor]:
    """DSA-shaped adapter around :func:`flash_mla.flash_mla_sparse_fwd`.

    Accepts flat tensors with global indices; pads ``topk`` to
    the GPU-specific alignment; returns ``(out, lse)``.
    """
    _ensure_flash_mla()

    topk_count = topk_idxs.shape[-1]
    topk_align = _get_topk_alignment()
    topk_padded = round_up_to_nearest_multiple(topk_count, topk_align)
    if topk_padded != topk_count:
        pad_width = topk_padded - topk_count
        topk_idxs = torch.nn.functional.pad(topk_idxs, (0, pad_width), value=-1)

    kv_3d = kv.unsqueeze(1)  # (total_S_kv, 1, D)  h_kv=1
    indices = topk_idxs.unsqueeze(1)  # (total_S_q, 1, topk_padded) h_kv=1

    actual_num_heads = q.size(1)
    padded_num_heads = _get_head_padding(actual_num_heads)
    if padded_num_heads != actual_num_heads:
        q_padded = q.new_zeros((q.size(0), padded_num_heads, q.size(2)))
        q_padded[:, :actual_num_heads, :] = q
        q = q_padded

        attn_sink_padded = attn_sink.new_full((padded_num_heads,), float("-inf"))
        attn_sink_padded[:actual_num_heads] = attn_sink
        attn_sink = attn_sink_padded

    with torch.cuda.nvtx.range("flash_mla_sparse_fwd"):
        try:
            out, _max_logits, lse = _flash_mla_sparse_fwd(
                q,
                kv_3d,
                indices,
                softmax_scale,
                d_v=d_v,
                attn_sink=attn_sink,
                topk_length=topk_length,
                indexer_topk=0,
            )
        except Exception as e:
            from hip_dsa import sparse_attn_forward
            out, lse = sparse_attn_forward(
                q=q,
                kv=kv_3d,
                attn_sink=attn_sink,
                topk_idxs=topk_idxs,
                softmax_scale=softmax_scale,
                topk_length=topk_length,
            )

    if padded_num_heads != actual_num_heads:
        out = out[:, :actual_num_heads, :]
        lse = lse[:, :actual_num_heads]

    return out, lse
