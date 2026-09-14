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

from argparse import ArgumentParser

from ..feature import AbstractFeature


class DeepSeekV4Feature(AbstractFeature):

    def __init__(self):
        super().__init__('deepseekv4', optimization_level=0)


    def register_args(self, parser: ArgumentParser):
        pass
    

    def register_patches(self, patch_manager, args):

        # major of device capability on rocm is 9
        from hcu_megatron.bridge.models.deepseek.deepseek_v4_bridge import deepseek_v4_supports_bw_fused_kernels

        patch_manager.register_patch('megatron.bridge.models.deepseek.deepseek_v4_bridge.deepseek_v4_supports_blackwell_fused_kernels',
                                    deepseek_v4_supports_bw_fused_kernels)
