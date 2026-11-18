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

"""Qwen3-VL recipes for the BW1100 (144 GB) cards.

Every public entry builds its own ``ConfigContainer``: the model architecture is
read from the checkpoint's ``config.json`` through ``AutoBridge``, the training,
optimizer and dataset defaults live in this module, and the deployment topology
is the only thing that varies between sizes.  No upstream H100 recipe is
imported, so the recipe layer stays self-contained and the launch scripts only
select a recipe name.

``LOCAL_HF_PATH`` (then ``MODEL_HF_PATH``) redirects the checkpoint, the dataset
processor and the tokenizer to the on-disk model directory.  ``AutoBridge``
itself is already redirected by the HCU patch layer; resolving the path here
keeps every consumer on the same local directory.
"""

from __future__ import annotations

import os

import torch
from megatron.core.transformer.enums import AttnBackend

from megatron.bridge import AutoBridge
from megatron.bridge.data.builders import MockVLMSFTDatasetConfig
from megatron.bridge.recipes.common import _pretrain_common, _sft_common_vlm
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed

_QWEN3_VL_8B_HF_PATH = "Qwen/Qwen3-VL-8B-Instruct"
_QWEN3_VL_30B_A3B_HF_PATH = "Qwen/Qwen3-VL-30B-A3B-Instruct"
_QWEN3_VL_32B_HF_PATH = "Qwen/Qwen3-VL-32B-Instruct"

_SEQ_LENGTH = 4096


def _env_int(name: str, default: int) -> int:
    """Read an integer toggle, treating an empty value as unset."""

    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    return int(raw_value)


def _hf_path(default: str) -> str:
    """Resolve the checkpoint directory used by the model and processor.

    ``LOCAL_HF_PATH`` comes first: the HCU patch layer already redirects
    ``AutoBridge.from_hf_pretrained`` at it, and reading it here keeps the
    dataset processor and tokenizer on the same local directory.
    """

    return os.environ.get("LOCAL_HF_PATH") or os.environ.get("MODEL_HF_PATH") or default


def _set_topology(
    cfg: ConfigContainer,
    *,
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    sequence_parallel: bool,
    expert_model_parallel_size: int = 1,
    expert_tensor_parallel_size: int = 1,
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
    if expert_model_parallel_size > 1:
        cfg.model.expert_model_parallel_size = expert_model_parallel_size
        # Expert tensor parallelism stays at 1 so the expert grid is EP*PP and
        # the MoE sizes fit the same node count as their dense counterparts.
        cfg.model.expert_tensor_parallel_size = expert_tensor_parallel_size


def _apply_hcu_policy(cfg: ConfigContainer, *, moe: bool = False) -> None:
    """Apply the BW1100/ROCm kernel and RNG policy shared by the Qwen recipes."""

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.use_te_rng_tracker = False
    cfg.model.bias_activation_fusion = False
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = False
    if moe:
        # alltoall is Megatron-Core's built-in dispatcher; the flex/hybridep
        # path needs the HybridEP package, which this cluster does not ship.
        cfg.model.moe_token_dispatcher_type = "alltoall"
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


def _mock_vlm_dataset(hf_path: str, seq_length: int) -> MockVLMSFTDatasetConfig:
    """Build the synthetic image/conversation dataset used by pretrain recipes."""

    return MockVLMSFTDatasetConfig(
        seq_length=seq_length,
        hf_processor_path=hf_path,
        prompt=os.environ.get("MOCK_PROMPT") or "Describe this image.",
        random_seed=_env_int("MOCK_RANDOM_SEED", 0),
        image_size=(
            _env_int("MOCK_IMAGE_WIDTH", 256),
            _env_int("MOCK_IMAGE_HEIGHT", 256),
        ),
        num_images=_env_int("MOCK_NUM_IMAGES", 1),
        num_base_examples=_env_int("MOCK_NUM_BASE_EXAMPLES", 1000),
        num_workers=1,
        dataloader_type="single",
        data_sharding=True,
        pin_memory=True,
        persistent_workers=False,
        pad_to_max_length=True,
    )


def _optimizer(
    *,
    lr_warmup_iters: int,
    lr_decay_iters: int | None,
    max_lr: float,
    min_lr: float,
    overrides: dict | None = None,
) -> tuple:
    """Build the fused-Adam/cosine optimizer pair used by every Qwen3-VL recipe.

    ``lr_decay_iters=None`` keeps the scheduler's "decay over ``train_iters``"
    behaviour, which is what the bring-up recipes use.
    """

    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        max_lr=max_lr,
        min_lr=min_lr,
        **(overrides or {}),
    )
    # Full-precision optimizer state: precision-aware Adam is not enabled on
    # these cards, so all master tensors stay in fp32.
    opt_cfg.use_precision_aware_optimizer = False
    opt_cfg.main_grads_dtype = torch.float32
    opt_cfg.main_params_dtype = torch.float32
    opt_cfg.exp_avg_dtype = torch.float32
    opt_cfg.exp_avg_sq_dtype = torch.float32
    return opt_cfg, scheduler_cfg


def _pretrain_cfg(
    hf_default: str,
    *,
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    sequence_parallel: bool,
    expert_model_parallel_size: int = 1,
    expert_tensor_parallel_size: int = 1,
    moe: bool = False,
    train_iters: int = 300_000,
    global_batch_size: int = 32,
    micro_batch_size: int = 2,
    manual_gc: bool = True,
    overlap_grad_reduce: bool = False,
    overlap_param_gather: bool = False,
    eval_interval: int = 500,
    eval_iters: int = 32,
    lr_warmup_iters: int = 500,
    lr_decay_iters: int | None = 300_000,
    max_lr: float = 3.0e-4,
    min_lr: float = 3.0e-5,
    optimizer_overrides: dict | None = None,
    num_layers_in_first_pipeline_stage: int | None = None,
    num_layers_in_last_pipeline_stage: int | None = None,
) -> ConfigContainer:
    """Build a synthetic-data pretrain configuration for one BW1100 size."""

    cfg = _pretrain_common()
    hf_path = _hf_path(hf_default)

    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = _SEQ_LENGTH
    _set_topology(
        cfg,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        sequence_parallel=sequence_parallel,
        expert_model_parallel_size=expert_model_parallel_size,
        expert_tensor_parallel_size=expert_tensor_parallel_size,
        num_layers_in_first_pipeline_stage=num_layers_in_first_pipeline_stage,
        num_layers_in_last_pipeline_stage=num_layers_in_last_pipeline_stage,
    )

    # Full pretraining: upstream's alignment-stage recipes freeze the language
    # and vision towers and train only the projection, which is a different task.
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False

    _apply_hcu_policy(cfg, moe=moe)

    cfg.train.train_iters = train_iters
    cfg.train.global_batch_size = global_batch_size
    cfg.train.micro_batch_size = micro_batch_size
    cfg.train.manual_gc = manual_gc

    cfg.validation.eval_interval = eval_interval
    cfg.validation.eval_iters = eval_iters

    cfg.optimizer, cfg.scheduler = _optimizer(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        max_lr=max_lr,
        min_lr=min_lr,
        overrides=optimizer_overrides,
    )

    # Synthetic image/conversation data.  The class name says SFT because it
    # emits chat-shaped examples; the caller still selects the pretrain loop.
    cfg.dataset = _mock_vlm_dataset(hf_path, cfg.model.seq_length)
    cfg.tokenizer.tokenizer_model = hf_path
    cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"

    cfg.ddp.overlap_grad_reduce = overlap_grad_reduce
    cfg.ddp.overlap_param_gather = overlap_param_gather

    cfg.mixed_precision = bf16_mixed()
    return cfg


def _sft_cfg(
    hf_default: str,
    *,
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    sequence_parallel: bool,
    expert_model_parallel_size: int = 1,
    expert_tensor_parallel_size: int = 1,
    moe: bool = False,
    train_iters: int = 50,
    global_batch_size: int = 32,
    micro_batch_size: int = 2,
    lr_warmup_iters: int = 10,
    lr_decay_iters: int | None = 50,
    max_lr: float = 5.0e-5,
    min_lr: float = 5.0e-6,
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
        expert_model_parallel_size=expert_model_parallel_size,
        expert_tensor_parallel_size=expert_tensor_parallel_size,
        num_layers_in_first_pipeline_stage=num_layers_in_first_pipeline_stage,
        num_layers_in_last_pipeline_stage=num_layers_in_last_pipeline_stage,
    )

    # Full SFT trains every component; freezing is only used for alignment.
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False

    if moe:
        # Grouped GEMM stays on for MoE SFT (as in the reference recipe); the
        # platform policy only forces the router/permute fusions off.
        cfg.model.moe_grouped_gemm = True

    _apply_hcu_policy(cfg, moe=moe)

    cfg.train.train_iters = train_iters
    cfg.train.global_batch_size = global_batch_size
    cfg.train.micro_batch_size = micro_batch_size
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 10

    cfg.optimizer, cfg.scheduler = _optimizer(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        max_lr=max_lr,
        min_lr=min_lr,
    )

    # DirectHFSFTDatasetConfig: the runner replaces source/splits according to
    # ``--dataset``, but the processor path and packing default come from here.
    cfg.dataset.seq_length = _SEQ_LENGTH
    cfg.dataset.hf_processor_path = hf_path
    cfg.dataset.defer_in_batch_packing_to_step = True

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
# 8B
# =============================================================================
def qwen3_vl_8b_pretrain_8gpu_bw1100_bf16_mock_config() -> ConfigContainer:
    """Qwen3-VL 8B pretrain on one BW1100 node: TP=1, PP=1, DP=8."""

    return _pretrain_cfg(
        _QWEN3_VL_8B_HF_PATH,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        sequence_parallel=False,
    )


def qwen3_vl_8b_sft_8gpu_bw1100_bf16_config() -> ConfigContainer:
    """Qwen3-VL 8B SFT on one BW1100 node: TP=1, PP=1, DP=8."""

    return _sft_cfg(
        _QWEN3_VL_8B_HF_PATH,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        sequence_parallel=False,
    )


# =============================================================================
# 30B-A3B
# =============================================================================
def qwen3_vl_30b_a3b_pretrain_16gpu_bw1100_bf16_mock_config() -> ConfigContainer:
    """Qwen3-VL 30B-A3B pretrain on two BW1100 nodes: TP=1, PP=2, EP=8, SP."""

    return _pretrain_cfg(
        _QWEN3_VL_30B_A3B_HF_PATH,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        sequence_parallel=True,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=1,
        moe=True,
    )


def qwen3_vl_30b_a3b_sft_16gpu_bw1100_bf16_config() -> ConfigContainer:
    """Qwen3-VL 30B-A3B SFT on two BW1100 nodes: TP=1, PP=2, EP=8."""

    return _sft_cfg(
        _QWEN3_VL_30B_A3B_HF_PATH,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        sequence_parallel=False,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=1,
        moe=True,
    )


# =============================================================================
# 32B (dense)
# =============================================================================
def qwen3_vl_32b_pretrain_8gpu_bw1100_bf16_mock_config() -> ConfigContainer:
    """Qwen3-VL 32B pretrain on one 8-GPU node: TP=2, PP=4, SP.

    This is the short bring-up configuration: the learning rate starts low
    (3e-5), the scheduler decays over ``TRAIN_ITERS`` and evaluation is turned
    off because no held-out split exists for the synthetic corpus.
    """

    return _pretrain_cfg(
        _QWEN3_VL_32B_HF_PATH,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
        sequence_parallel=True,
        train_iters=_env_int("TRAIN_ITERS", 50),
        global_batch_size=_env_int("GLOBAL_BATCH_SIZE", 32),
        micro_batch_size=_env_int("MICRO_BATCH_SIZE", 1),
        lr_decay_iters=None,
    )


def qwen3_vl_32b_sft_8gpu_bw1100_bf16_config() -> ConfigContainer:
    """Qwen3-VL 32B SFT on one 8-GPU node: TP=2, PP=4, SP.

    Full SFT keeps the language tower, vision tower and projection trainable;
    the runner replaces the dataset source according to ``--dataset`` while this
    recipe fixes the processor path and packing behaviour.
    """

    return _sft_cfg(
        _QWEN3_VL_32B_HF_PATH,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
        sequence_parallel=True,
        train_iters=_env_int("TRAIN_ITERS", 50),
        global_batch_size=_env_int("GLOBAL_BATCH_SIZE", 32),
        micro_batch_size=_env_int("MICRO_BATCH_SIZE", 1),
        lr_decay_iters=None,
    )


__all__ = [
    "qwen3_vl_8b_pretrain_8gpu_bw1100_bf16_mock_config",
    "qwen3_vl_8b_sft_8gpu_bw1100_bf16_config",
    "qwen3_vl_30b_a3b_pretrain_16gpu_bw1100_bf16_mock_config",
    "qwen3_vl_30b_a3b_sft_16gpu_bw1100_bf16_config",
    "qwen3_vl_32b_pretrain_8gpu_bw1100_bf16_mock_config",
    "qwen3_vl_32b_sft_8gpu_bw1100_bf16_config",
]
