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
# ruff: noqa: F401
"""Compatibility aliases for the Qwen2.5-VL recipes.

Each platform keeps its own tuning in ``qwen.bw1000.qwen25_vl`` and
``qwen.bw1100.qwen25_vl``; this module only re-exports the public names, the
same top-level alias layout used by ``qwen35.py`` / ``qwen35_vl.py``.
"""

from __future__ import annotations

from hcu_megatron.bridge.recipes.qwen.bw1000.qwen25_vl import (
    qwen25_vl_32b_sft_16gpu_bw1000_bf16_config,
    qwen25_vl_3b_sft_8gpu_bw1000_bf16_config,
    qwen25_vl_72b_sft_32gpu_bw1000_bf16_config,
    qwen25_vl_7b_sft_8gpu_bw1000_bf16_config,
)
from hcu_megatron.bridge.recipes.qwen.bw1100.qwen25_vl import (
    qwen25_vl_32b_sft_8gpu_bw1100_bf16_config,
    qwen25_vl_3b_sft_8gpu_bw1100_bf16_config,
    qwen25_vl_72b_sft_16gpu_bw1100_bf16_config,
    qwen25_vl_7b_sft_8gpu_bw1100_bf16_config,
)

__all__ = [
    "qwen25_vl_3b_sft_8gpu_bw1000_bf16_config",
    "qwen25_vl_7b_sft_8gpu_bw1000_bf16_config",
    "qwen25_vl_32b_sft_16gpu_bw1000_bf16_config",
    "qwen25_vl_72b_sft_32gpu_bw1000_bf16_config",
    "qwen25_vl_3b_sft_8gpu_bw1100_bf16_config",
    "qwen25_vl_7b_sft_8gpu_bw1100_bf16_config",
    "qwen25_vl_32b_sft_8gpu_bw1100_bf16_config",
    "qwen25_vl_72b_sft_16gpu_bw1100_bf16_config",
]
