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

"""Qwen2.5-VL SFT recipes for the BW1000 cards.

BW1000 has less memory per card than BW1100, so tensor parallelism is doubled
for every size while the pipeline layout, datasets and training
hyper-parameters stay identical.  The configurations themselves live in the
BW1100 module; this module only widens TP, so there is a single place to
maintain the Qwen2.5-VL settings.
"""

from __future__ import annotations

from megatron.bridge.training.config import ConfigContainer

from hcu_megatron.bridge.recipes.qwen.bw1100.qwen25_vl import (
    qwen25_vl_32b_sft_8gpu_bw1100_bf16_config,
    qwen25_vl_3b_sft_8gpu_bw1100_bf16_config,
    qwen25_vl_72b_sft_16gpu_bw1100_bf16_config,
    qwen25_vl_7b_sft_8gpu_bw1100_bf16_config,
)


def _widen_tp(cfg: ConfigContainer, tensor_model_parallel_size: int) -> ConfigContainer:
    """Return a copy-free BW1100 config with the BW1000 tensor split applied."""

    cfg.model.tensor_model_parallel_size = tensor_model_parallel_size
    return cfg


def qwen25_vl_3b_sft_8gpu_bw1000_bf16_config() -> ConfigContainer:
    """Qwen2.5-VL 3B SFT on one BW1000 node: TP=2, PP=1, DP=4."""

    return _widen_tp(qwen25_vl_3b_sft_8gpu_bw1100_bf16_config(), 2)


def qwen25_vl_7b_sft_8gpu_bw1000_bf16_config() -> ConfigContainer:
    """Qwen2.5-VL 7B SFT on one BW1000 node: TP=2, PP=1, DP=4."""

    return _widen_tp(qwen25_vl_7b_sft_8gpu_bw1100_bf16_config(), 2)


def qwen25_vl_32b_sft_16gpu_bw1000_bf16_config() -> ConfigContainer:
    """Qwen2.5-VL 32B SFT on two BW1000 nodes: TP=4, PP=4, DP=1."""

    return _widen_tp(qwen25_vl_32b_sft_8gpu_bw1100_bf16_config(), 4)


def qwen25_vl_72b_sft_32gpu_bw1000_bf16_config() -> ConfigContainer:
    """Qwen2.5-VL 72B SFT on four BW1000 nodes: TP=8, PP=4, DP=1."""

    return _widen_tp(qwen25_vl_72b_sft_16gpu_bw1100_bf16_config(), 8)


__all__ = [
    "qwen25_vl_3b_sft_8gpu_bw1000_bf16_config",
    "qwen25_vl_7b_sft_8gpu_bw1000_bf16_config",
    "qwen25_vl_32b_sft_16gpu_bw1000_bf16_config",
    "qwen25_vl_72b_sft_32gpu_bw1000_bf16_config",
]
