# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen2.5-VL SFT recipes for the BW1100 (144 GB) cards.

Every public entry builds its own ``ConfigContainer``: the model architecture is
read from the checkpoint's ``config.json`` through ``AutoBridge``, the training,
optimizer and dataset defaults live in this module, and the deployment topology
is the only thing that varies between sizes.  No upstream H100 recipe is
imported, so the recipe layer stays self-contained and the launch scripts only
select a recipe name.

``LOCAL_HF_PATH`` (then ``MODEL_HF_PATH``) redirects the checkpoint and the
dataset processor to the on-disk model directory.  ``AutoBridge`` itself is
already redirected by the HCU patch layer; resolving the path here keeps every
consumer on the same local directory.
"""

from __future__ import annotations

import os

import torch
from megatron.core.transformer.enums import AttnBackend

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _sft_common_vlm
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed

_QWEN25_VL_3B_HF_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"
_QWEN25_VL_7B_HF_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
_QWEN25_VL_32B_HF_PATH = "Qwen/Qwen2.5-VL-32B-Instruct"
_QWEN25_VL_72B_HF_PATH = "Qwen/Qwen2.5-VL-72B-Instruct"

_SEQ_LENGTH = 4096


def _hf_path(default: str) -> str:
    """Resolve the checkpoint directory used by the model and processor."""

    return os.environ.get("LOCAL_HF_PATH") or os.environ.get("MODEL_HF_PATH") or default


def _set_topology(
    cfg: ConfigContainer,
    *,
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    sequence_parallel: bool,
    num_layers_in_first_pipeline_stage: int | None = None,
    num_layers_in_last_pipeline_stage: int | None = None,
) -> None:
    """Apply the model-parallel layout for one BW1100 deployment."""

    cfg.model.tensor_model_parallel_size = tensor_model_parallel_size
    cfg.model.pipeline_model_parallel_size = pipeline_model_parallel_size
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = sequence_parallel
    cfg.model.pipeline_dtype = torch.bfloat16 if pipeline_model_parallel_size > 1 else None
    # Language-model uneven split.  PP rank 0 also carries the vision encoder and
    # the embedding, so it normally gets fewer decoder layers; the middle stages
    # take the rest evenly.  Both values must stay None for a uniform split.
    # Constraints (checked by TransformerConfig): (num_layers - first - last) must
    # be divisible by the middle stage count, and these two fields are mutually
    # exclusive with account_for_*_in_pipeline_split and the custom PP layout.
    if (
        num_layers_in_first_pipeline_stage is not None
        or num_layers_in_last_pipeline_stage is not None
    ):
        cfg.model.num_layers_in_first_pipeline_stage = num_layers_in_first_pipeline_stage
        cfg.model.num_layers_in_last_pipeline_stage = num_layers_in_last_pipeline_stage
        cfg.model.account_for_embedding_in_pipeline_split = False
        cfg.model.account_for_loss_in_pipeline_split = False


def _apply_hcu_policy(cfg: ConfigContainer) -> None:
    """Apply the BW1100/ROCm kernel and RNG policy shared by the Qwen recipes."""

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.use_te_rng_tracker = False
    cfg.model.bias_activation_fusion = False
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = False
    # AttnBackend is an enum and the config compares it by identity, so the
    # plain "auto" string used by upstream's recipes would never match.
    cfg.model.attention_backend = AttnBackend.auto
    cfg.model.cross_entropy_loss_fusion = False
    cfg.rng.te_rng_tracker = False
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "GPU_MAX_HW_QUEUES": 4,
    }


def _sft_cfg(
    hf_default: str,
    *,
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    sequence_parallel: bool,
    train_iters: int = 300_000,
    global_batch_size: int = 32,
    micro_batch_size: int = 2,
    num_layers_in_first_pipeline_stage: int | None = None,
    num_layers_in_last_pipeline_stage: int | None = None,
) -> ConfigContainer:
    """Build a full-parameter SFT configuration for one BW1100 size."""

    cfg = _sft_common_vlm()
    hf_path = _hf_path(hf_default)

    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = _SEQ_LENGTH
    _set_topology(
        cfg,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        sequence_parallel=sequence_parallel,
        num_layers_in_first_pipeline_stage=num_layers_in_first_pipeline_stage,
        num_layers_in_last_pipeline_stage=num_layers_in_last_pipeline_stage,
    )

    # Full SFT trains every component; freezing is only used for alignment.
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False

    _apply_hcu_policy(cfg)

    cfg.train.train_iters = train_iters
    cfg.train.global_batch_size = global_batch_size
    cfg.train.micro_batch_size = micro_batch_size
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 32

    # The SFT learning rate is ~60x lower than the pretraining one: these
    # checkpoints are already instruction tuned and large LRs destroy them.
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=500,
        lr_decay_iters=300_000,
        max_lr=5.0e-6,
        min_lr=1.0e-6,
    )
    # Full-precision optimizer state: precision-aware Adam is not enabled on
    # these cards, so all master tensors stay in fp32.
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # DirectHFSFTDatasetConfig: the runner replaces source/splits according to
    # ``--dataset``, but the processor path must come from the recipe so the
    # image/video processor matches the checkpoint.
    cfg.dataset.seq_length = _SEQ_LENGTH
    cfg.dataset.hf_processor_path = hf_path

    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    cfg.mixed_precision = bf16_mixed()
    return cfg


# =============================================================================
# 3B
# =============================================================================
def qwen25_vl_3b_sft_8gpu_bw1100_bf16_config() -> ConfigContainer:
    """Qwen2.5-VL 3B SFT on one BW1100 node: TP=1, PP=1, DP=8."""

    return _sft_cfg(
        _QWEN25_VL_3B_HF_PATH,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        sequence_parallel=False,
    )


# =============================================================================
# 7B
# =============================================================================
def qwen25_vl_7b_sft_8gpu_bw1100_bf16_config() -> ConfigContainer:
    """Qwen2.5-VL 7B SFT on one BW1100 node: TP=1, PP=1, DP=8."""

    return _sft_cfg(
        _QWEN25_VL_7B_HF_PATH,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        sequence_parallel=False,
    )


# =============================================================================
# 32B
# =============================================================================
def qwen25_vl_32b_sft_8gpu_bw1100_bf16_config() -> ConfigContainer:
    """Qwen2.5-VL 32B SFT on one 8-GPU node: TP=2, PP=4, DP=1."""

    return _sft_cfg(
        _QWEN25_VL_32B_HF_PATH,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
        sequence_parallel=False,
    )


# =============================================================================
# 72B
# =============================================================================
def qwen25_vl_72b_sft_16gpu_bw1100_bf16_config() -> ConfigContainer:
    """Qwen2.5-VL 72B SFT on two BW1100 nodes: TP=4, PP=4, DP=1."""

    return _sft_cfg(
        _QWEN25_VL_72B_HF_PATH,
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=4,
        sequence_parallel=False,
    )


__all__ = [
    "qwen25_vl_3b_sft_8gpu_bw1100_bf16_config",
    "qwen25_vl_7b_sft_8gpu_bw1100_bf16_config",
    "qwen25_vl_32b_sft_8gpu_bw1100_bf16_config",
    "qwen25_vl_72b_sft_16gpu_bw1100_bf16_config",
]
