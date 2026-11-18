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

"""Qwen3-VL recipes for the BW1000 cards.

BW1000 has less memory per card than BW1100, so tensor parallelism is doubled
for every size while the pipeline layout, expert placement, datasets and
training hyper-parameters stay identical.  The configurations themselves live in
the BW1100 module; this module only widens TP, so there is a single place to
maintain the Qwen3-VL settings.
"""

from __future__ import annotations

from megatron.bridge.training.config import ConfigContainer

from hcu_megatron.bridge.recipes.qwen.bw1100.qwen3_vl import (
    qwen3_vl_30b_a3b_pretrain_16gpu_bw1100_bf16_mock_config,
    qwen3_vl_30b_a3b_sft_16gpu_bw1100_bf16_config,
    qwen3_vl_32b_pretrain_8gpu_bw1100_bf16_mock_config,
    qwen3_vl_32b_sft_8gpu_bw1100_bf16_config,
    qwen3_vl_8b_pretrain_8gpu_bw1100_bf16_mock_config,
    qwen3_vl_8b_sft_8gpu_bw1100_bf16_config,
)


def _widen_tp(cfg: ConfigContainer, tensor_model_parallel_size: int) -> ConfigContainer:
    """Return a copy-free BW1100 config with the BW1000 tensor split applied."""

    cfg.model.tensor_model_parallel_size = tensor_model_parallel_size
    return cfg


# =============================================================================
# 8B
# =============================================================================
def qwen3_vl_8b_pretrain_8gpu_bw1000_bf16_mock_config() -> ConfigContainer:
    """Qwen3-VL 8B pretrain on one BW1000 node: TP=2, PP=1, DP=4."""

    return _widen_tp(qwen3_vl_8b_pretrain_8gpu_bw1100_bf16_mock_config(), 2)


def qwen3_vl_8b_sft_8gpu_bw1000_bf16_config() -> ConfigContainer:
    """Qwen3-VL 8B SFT on one BW1000 node: TP=2, PP=1, DP=4."""

    return _widen_tp(qwen3_vl_8b_sft_8gpu_bw1100_bf16_config(), 2)


# =============================================================================
# 30B-A3B
# =============================================================================
def qwen3_vl_30b_a3b_pretrain_16gpu_bw1000_bf16_mock_config() -> ConfigContainer:
    """Qwen3-VL 30B-A3B pretrain on two BW1000 nodes: TP=2, PP=2, EP=8, SP."""

    return _widen_tp(qwen3_vl_30b_a3b_pretrain_16gpu_bw1100_bf16_mock_config(), 2)


def qwen3_vl_30b_a3b_sft_16gpu_bw1000_bf16_config() -> ConfigContainer:
    """Qwen3-VL 30B-A3B SFT on two BW1000 nodes: TP=2, PP=2, EP=8."""

    return _widen_tp(qwen3_vl_30b_a3b_sft_16gpu_bw1100_bf16_config(), 2)


# =============================================================================
# 32B (dense)
# =============================================================================
def qwen3_vl_32b_pretrain_16gpu_bw1000_bf16_mock_config() -> ConfigContainer:
    """Qwen3-VL 32B pretrain on two BW1000 nodes: TP=4, PP=4, DP=1."""

    return _widen_tp(qwen3_vl_32b_pretrain_8gpu_bw1100_bf16_mock_config(), 4)


def qwen3_vl_32b_sft_16gpu_bw1000_bf16_config() -> ConfigContainer:
    """Qwen3-VL 32B SFT on two BW1000 nodes: TP=4, PP=4, DP=1."""

    return _widen_tp(qwen3_vl_32b_sft_8gpu_bw1100_bf16_config(), 4)


__all__ = [
    "qwen3_vl_8b_pretrain_8gpu_bw1000_bf16_mock_config",
    "qwen3_vl_8b_sft_8gpu_bw1000_bf16_config",
    "qwen3_vl_30b_a3b_pretrain_16gpu_bw1000_bf16_mock_config",
    "qwen3_vl_30b_a3b_sft_16gpu_bw1000_bf16_config",
    "qwen3_vl_32b_pretrain_16gpu_bw1000_bf16_mock_config",
    "qwen3_vl_32b_sft_16gpu_bw1000_bf16_config",
]
