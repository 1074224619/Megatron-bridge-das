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

import torch


def deepseek_v4_supports_bw_fused_kernels() -> bool:
    """Return whether DSv4 BW-only fused kernels should default on."""
    if not torch.cuda.is_available():
        return False

    major, _minor = torch.cuda.get_device_capability()
    return major >= 9