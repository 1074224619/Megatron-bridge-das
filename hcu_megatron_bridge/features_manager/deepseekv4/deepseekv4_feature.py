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
from ...training.arguments import remove_original_params


class DeepSeekV4Feature(AbstractFeature):

    def __init__(self):
        super().__init__('deepseekv4', optimization_level=0)


    def register_args(self, parser: ArgumentParser):
        remove_original_params(parser, ["dsa_kernel_backend"])
        group = parser.add_argument_group(title='extra training args')
    
        group.add_argument('--dsa-kernel-backend', type=str, choices=['none', 'tilelang', 'cudnn', 'hip'],
                        default=None,
                        help='Fused DSA kernel backend. When omitted, resolves to "cudnn" for '
                        'experimental_attention_variant="dsv4_hybrid" launches and to "none" '
                        'otherwise.',)
    

    def register_patches(self, patch_manager, args):
        self.register_patch(patch_manager, args)
        self.apply_profiler_annotate(patch_manager)


    def register_patch(self, patch_manager, args):

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


        # major of device capability on rocm is 9
        from hcu_megatron.bridge.models.deepseek.deepseek_v4_bridge import deepseek_v4_supports_bw_fused_kernels

        patch_manager.register_patch('megatron.bridge.models.deepseek.deepseek_v4_bridge.deepseek_v4_supports_blackwell_fused_kernels',
                                    deepseek_v4_supports_bw_fused_kernels)


        # megatron core -- hip fused dsa kernel
        from ...core.utils import (
            _VALID_DSA_KERNEL_BACKENDS, 
            _validate_dsa_kernel_backend_dependencies,
        )
        from ...core.transformer.experimental_attention_variant.dsa_kernels import (
            _BACKEND_MODULE_NAME_BY_BACKEND, 
            _get_dsa_kernel_backend,
        )
        from ...core.transformer.experimental_attention_variant.dsa_hip_kernels import (
            _dsa_fwd_flash_mla,
            _ensure_dsa_namespace_hip,  
        )
        from ...core.transformer.experimental_attention_variant.csa_fused_compressor import (
            _SUPPORTED_COMPUTE_CAPABILITY, 
            _get_frontend,
        )
        from ...core.transformer.experimental_attention_variant.csa_kernels import (
            _csa_fwd_flash_mla,
            _ensure_dsa_namespace,
        )

        patch_manager.register_patch('megatron.core.utils._VALID_DSA_KERNEL_BACKENDS',
                                    _VALID_DSA_KERNEL_BACKENDS)
        patch_manager.register_patch('megatron.core.utils._validate_dsa_kernel_backend_dependencies',
                                    _validate_dsa_kernel_backend_dependencies)
        patch_manager.register_patch('megatron.core.transformer.experimental_attention_variant.dsa_kernels._BACKEND_MODULE_NAME_BY_BACKEND',
                                    _BACKEND_MODULE_NAME_BY_BACKEND)
        patch_manager.register_patch('megatron.core.transformer.experimental_attention_variant.dsa_kernels._get_dsa_kernel_backend',
                                    _get_dsa_kernel_backend)
        patch_manager.register_patch('megatron.core.transformer.experimental_attention_variant.dsa_cudnn_kernels._dsa_fwd_flash_mla',
                                    _dsa_fwd_flash_mla)
        patch_manager.register_patch('megatron.core.transformer.experimental_attention_variant.dsa_cudnn_kernels._ensure_dsa_namespace',
                                    _ensure_dsa_namespace_hip)
        patch_manager.register_patch('megatron.core.transformer.experimental_attention_variant.csa_fused_compressor._SUPPORTED_COMPUTE_CAPABILITY',
                                    _SUPPORTED_COMPUTE_CAPABILITY)
        patch_manager.register_patch('megatron.core.transformer.experimental_attention_variant.csa_fused_compressor._get_frontend',
                                    _get_frontend)
        patch_manager.register_patch('megatron.core.transformer.experimental_attention_variant.csa_kernels._csa_fwd_flash_mla',
                                    _csa_fwd_flash_mla)
        patch_manager.register_patch('megatron.core.transformer.experimental_attention_variant.csa_kernels._ensure_dsa_namespace',
                                    _ensure_dsa_namespace)


        # megatron core -- hip fused mhc kernel
        from ...core.fusions.fused_mhc_kernels import (
            _mhc_backend_status, 
            fused_sinkhorn, 
            fused_h_aggregate, 
            fused_h_post_bda, 
            fused_proj_rms_compute_h,
        )

        patch_manager.register_patch('megatron.core.fusions.fused_mhc_kernels._mhc_backend_status', 
                                    _mhc_backend_status)
        patch_manager.register_patch('megatron.core.fusions.fused_mhc_kernels.fused_sinkhorn', 
                                    fused_sinkhorn)
        patch_manager.register_patch('megatron.core.fusions.fused_mhc_kernels.fused_h_aggregate', 
                                    fused_h_aggregate)
        patch_manager.register_patch('megatron.core.fusions.fused_mhc_kernels.fused_h_post_bda',
                                    fused_h_post_bda)
        patch_manager.register_patch('megatron.core.fusions.fused_mhc_kernels.fused_proj_rms_compute_h',
                                    fused_proj_rms_compute_h)


    def apply_profiler_annotate(self, patch_manager):
        import importlib.util

        from hcu_megatron_bridge.utils import profiler_annotate_patch
        from megatron.core.fusions.fused_mhc_kernels import is_cutile_available

        # `fused_mhc` is a HIP/ROCm-only package; on NVIDIA it isn't installed,
        # and Patch.parse_path would raise ModuleNotFoundError on those entries.
        has_fused_mhc = importlib.util.find_spec('fused_mhc') is not None
        has_cutile = is_cutile_available()

        targets = {
            # unfused mhc
            'megatron.core.transformer.hyper_connection.SinkhornKnopp.forward': 'HyperConnection::native_sinkhorn_forward',
            'megatron.core.transformer.hyper_connection.SinkhornKnopp.backward': 'HyperConnection::native_sinkhorn_backward',
            'megatron.core.transformer.hyper_connection.native_h_aggregate': 'HyperConnection::native_h_aggregate',
            'megatron.core.fusions.fused_mhc_kernels._torch_h_aggregate_bwd': 'HyperConnection::torch_h_aggregate_bwd',
            'megatron.core.transformer.hyper_connection.native_h_post_bda': 'HyperConnection::native_h_post_bda',
            'megatron.core.fusions.fused_mhc_kernels._torch_h_post_bda_bwd': 'HyperConnection::torch_h_post_bda_bwd',
            'megatron.core.fusions.fused_mhc_kernels._torch_proj_rms_compute_h': 'HyperConnection::torch_proj_rms_compute_h',
            'megatron.core.transformer.hyper_connection.HyperConnectionModule._projection_and_get_norm': 'HyperConnection::projection_and_get_norm',
            'megatron.core.transformer.hyper_connection.HyperConnectionModule._compute_h': 'HyperConnection::compute_h',

            # unfused dsa
            'megatron.core.transformer.experimental_attention_variant.csa.unfused_compressed_sparse_attn': 'CompressedSparseAttention::unfused_compressed_sparse_attn',

            # fused mhc --- hip
            'fused_mhc.FusedSinkhorn.forward': 'HipFusedSinkhorn::forward',
            'fused_mhc.FusedSinkhorn.backward': 'HipFusedSinkhorn::backward',
            'fused_mhc.FusedHAggregate.forward': 'HipFusedHAggregate::forward',
            'fused_mhc.FusedHAggregate.backward': 'HipFusedHAggregate::backward',
            'fused_mhc.FusedHPostBDA.forward': 'HipFusedHPostBDA::forward',
            'fused_mhc.FusedHPostBDA.backward': 'HipFusedHPostBDA::backward',
            'fused_mhc.FusedProjRmsComputeH.forward': 'HipFusedProjRmsComputeH::forward',
            'fused_mhc.FusedProjRmsComputeH.backward': 'HipFusedProjRmsComputeH::backward',

            # fused mhc --- triton
            'megatron.core.fusions.fused_mhc_kernels._triton_sinkhorn_fwd': 'TritonFusedSinkhorn::forward',
            'megatron.core.fusions.fused_mhc_kernels._triton_sinkhorn_bwd': 'TritonFusedSinkhorn::backward',
            'megatron.core.fusions.fused_mhc_kernels._triton_h_aggregate_fwd': 'TritonFusedHAggregate::forward',
            'megatron.core.fusions.fused_mhc_kernels._triton_h_post_bda_fwd': 'TritonFusedHPostBDA::forward',
            'megatron.core.fusions.fused_mhc_kernels._triton_h_post_bda_bwd': 'TritonFusedHPostBDA::backward',

            # fused mhc --- cutile
            'megatron.core.fusions.fused_mhc_kernels._cutile_sinkhorn_fwd': 'CutileFusedSinkhorn::forward',
            'megatron.core.fusions.fused_mhc_kernels._cutile_sinkhorn_bwd': 'CutileFusedSinkhorn::backward',
            'megatron.core.fusions.fused_mhc_kernels._cutile_h_aggregate_fwd': 'CutileFusedHAggregate::forward',
            'megatron.core.fusions.fused_mhc_kernels._cutile_h_aggregate_bwd': 'CutileFusedHAggregate::backward',
            'megatron.core.fusions.fused_mhc_kernels._cutile_h_post_bda_fwd': 'CutileFusedHPostBDA::forward',
            'megatron.core.fusions.fused_mhc_kernels._cutile_h_post_bda_bwd': 'CutileFusedHPostBDA::backward',
            'megatron.core.fusions.fused_mhc_kernels._cutile_proj_rms_compute_h_fwd': 'CutileFusedProjRmsComputeH::forward',
            'megatron.core.fusions.fused_mhc_kernels._cutile_fused_compute_h_proj_rms_bwd': 'CutileFusedProjRmsComputeH::backward',

            # fused dsa
            'megatron.core.transformer.experimental_attention_variant.csa_kernels.CSASparseAttnFunc.forward': 'CSASparseAttnFunc::fused_compressed_sparse_attn_forward',
            'megatron.core.transformer.experimental_attention_variant.csa_kernels.CSASparseAttnFunc.backward': 'CSASparseAttnFunc::fused_compressed_sparse_attn_backward',
            'megatron.core.transformer.experimental_attention_variant.csa_kernels.FusedCSAIndexerSparseAttnFunc.forward': 'FusedCSAIndexerSparseAttnFunc::fused_compressed_sparse_attn_forward',
            'megatron.core.transformer.experimental_attention_variant.csa_kernels.FusedCSAIndexerSparseAttnFunc.backward': 'FusedCSAIndexerSparseAttnFunc::fused_compressed_sparse_attn_backward',
        }

        for path, message in targets.items():
            if path.startswith('fused_mhc.') and not has_fused_mhc:
                continue
            if not has_cutile:
                continue

            patch_manager.register_patch(path, profiler_annotate_patch(path, message), apply_wrapper=True)