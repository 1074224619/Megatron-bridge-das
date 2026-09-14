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


class MegatronBasicFeature(AbstractFeature):

    def __init__(self):
        super().__init__('megatron-basic', optimization_level=0)


    def register_args(self, parser: ArgumentParser):
        pass


    def register_patches(self, patch_manager, args):
        
        # load model from a local path if LOCAL_HF_PATH is set
        from hcu_megatron.bridge.models.conversion.auto_bridge import from_hf_pretrained_wrapper
        
        patch_manager.register_patch('megatron.bridge.models.conversion.auto_bridge.AutoBridge.from_hf_pretrained',
                                    remove_origin_wrappers=True)
        patch_manager.register_patch('megatron.bridge.models.conversion.auto_bridge.AutoBridge.from_hf_pretrained',
                                    from_hf_pretrained_wrapper,
                                    apply_wrapper=True)
        patch_manager.register_patch('megatron.bridge.models.conversion.auto_bridge.AutoBridge.from_hf_pretrained',
                                    classmethod,
                                    apply_wrapper=True)

