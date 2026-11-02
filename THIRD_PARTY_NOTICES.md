# 第三方源码来源清单（Third-Party Notices）

本文件登记 Megatron-Bridge-das 及其适配层中源自第三方的源码文件与组件，包括项目名称、仓库 URL、固定 Commit/版本、Copyright、许可证、本地路径和 HYGON 修改情况。

本仓库自身以上游 **Megatron-Bridge**（Apache-2.0）为基线进行二次开发，仓库级归属与许可证信息见 [README.md](README.md)。第三方文件一律以其真实来源的许可证为准，不由本仓库的许可证推定覆盖。

## 1. 上游项目（Git submodule）

| 项目 | 仓库 URL | 固定 Commit/版本 | Copyright | 许可证 | 本地路径 | HYGON 修改 |
| --- | --- | --- | --- | --- | --- | --- |
| Megatron-Bridge | https://github.com/NVIDIA-NeMo/Megatron-Bridge.git | `51885cf132b2814188b6855c25a8588254274c2a`（分支 `v0.6.0`，Tag `v0.6.0`） | Copyright (c) NVIDIA CORPORATION & AFFILIATES | Apache-2.0 | `3rdparty/Megatron-Bridge` | 无。该子模块为固定上游基线，保持原样，不做本地修改 |
| Megatron-LM | https://github.com/NVIDIA/Megatron-LM.git | `cd4afffa648426a959dc7cb1e24b5ce7d0c3ff54` | Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved. | BSD-3-Clause | `3rdparty/Megatron-Bridge/3rdparty/Megatron-LM` | 无。该子模块为固定上游基线，保持原样，不做本地修改 |

上述两个子模块均以 Gitlink（mode `160000`）固定到上表 Commit，发布流程不进行浮动拉取。上游许可证全文分别位于 `3rdparty/Megatron-Bridge/LICENSE` 与 `3rdparty/Megatron-Bridge/3rdparty/Megatron-LM/LICENSE`。

## 2. 适配层中源自第三方的源码文件

说明：

- 下列文件位于本仓库补丁层，均为在上游或第三方同名/同族文件基础上二次开发所得。Apache-2.0 衍生文件已保留原 Copyright 声明并追加 HYGON Copyright 与修改说明；BSD-3-Clause 衍生文件按要求保留原始声明，不追加 SPDX 标识与修改说明。
- “固定 Commit/版本”列给出该文件的对照基线版本；标注“基线无同名文件”的条目表示该文件对本仓库固定的基线版本取不到同名文件，需在发布前按整改清单确认精确来源 Commit，确认后更新本表。
- HYGON 仅对其原创修改部分主张版权，原第三方部分的版权与许可证归属不因二次开发而改变。
- 标注“基线无同名文件”的条目中，部分为 HYGON 新增的模块文件。它们按上游同名族模块的结构与同族许可证编写，此处登记用于来源追溯和许可证归属说明，不改变其 HYGON 原创性质。

### 2.1 源自 NVIDIA Megatron-Bridge（Apache-2.0）

固定版本：`51885cf132b2814188b6855c25a8588254274c2a`（分支 `r0.6.0`，Tag `v0.6.0`）
仓库 URL：https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
Copyright：Copyright (c) NVIDIA CORPORATION & AFFILIATES
许可证：Apache-2.0

| # | 本地路径 | 上游对应路径 | HYGON 修改 |
| ---: | --- | --- | --- |
| 1 | `run_recipe.py` | `scripts/training/run_recipe.py` | 保留 NVIDIA 声明，追加 HYGON Copyright；扩展 recipe 入口与运行时配置解析 |
| 2 | `scripts/conversion/arguments.py` | `scripts/conversion/arguments.py` | 保留 NVIDIA 声明，追加 HYGON Copyright 与修改说明；适配 HCU 后端参数 |
| 3 | `scripts/conversion/gpu_backend.py` | `scripts/conversion/gpu_backend.py` | 保留 NVIDIA 声明，追加 HYGON Copyright 与修改说明；适配 HCU 运行时分支 |
| 4 | `scripts/conversion/run_conversion.py` | `scripts/conversion/run_conversion.py` | 保留 NVIDIA 声明，追加 HYGON Copyright 与修改说明；适配 HCU 转换流程 |
| 5 | `scripts/conversion/setup_conversion.py` | `scripts/conversion/setup_conversion.py` | 保留 NVIDIA 声明，追加 HYGON Copyright 与修改说明；适配 HCU 补丁装载 |
| 6 | `scripts/conversion/convert.sh` | `scripts/conversion/convert.sh` | 适配 HCU 启动流程；原上游 Copyright 与 Apache-2.0 声明需按合规整改清单恢复后再视为完成 |
| 7 | `examples/qwen/prepare_coco_caption_source.py` | 基线无同名文件（Megatron-Bridge 示例脚本族） | 保留 NVIDIA 声明，追加 HYGON Copyright；适配本地图文数据组织 |
| 8 | `examples/qwen/README_coco_energon.md` | 基线无同名文件（Megatron-Bridge 示例文档族） | 保留 NVIDIA 声明，追加 HYGON Copyright；补充本地数据准备说明 |
| 9 | `tests/unit_tests/examples/test_prepare_coco_caption_source.py` | 基线无同名文件（Megatron-Bridge 单元测试族） | 保留 NVIDIA 声明，追加 HYGON Copyright；补充本地路径与用例 |
| 10 | `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/__init__.py` | `src/megatron/bridge/recipes/__init__.py` | 保留 NVIDIA 声明，追加 HYGON Copyright；注册 HYGON 新增 recipe |
| 11 | `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/deepseek/__init__.py` | `src/megatron/bridge/recipes/deepseek/__init__.py` | 保留 NVIDIA 声明，追加 HYGON Copyright；注册 BW1000 recipe |
| 12 | `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/deepseek/deepseek_v4.py` | `src/megatron/bridge/recipes/deepseek/deepseek_v4.py` | 保留 NVIDIA 声明，追加 HYGON Copyright；兼容别名指向 HYGON BW1000 实现 |
| 13 | `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/deepseek/bw1000/__init__.py` | 基线无同名文件（`deepseek/bw1000` 为 HYGON 新增目录） | 追加 HYGON Copyright；导出 BW1000 DeepSeek-V4 recipe |
| 14 | `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/deepseek/bw1000/deepseek_v4.py` | 基线无同名文件（`deepseek/bw1000` 为 HYGON 新增目录） | 保留 NVIDIA 声明，追加 HYGON Copyright；新增 BW1000 预训练/SFT 配置 |
| 15 | `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/glm/__init__.py` | `src/megatron/bridge/recipes/glm/__init__.py` | 保留 NVIDIA 声明，追加 HYGON Copyright；注册 BW1000 recipe |
| 16 | `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/glm/glm52.py` | `src/megatron/bridge/recipes/glm/glm45.py`（同族文件） | 保留 NVIDIA 声明，追加 HYGON Copyright；兼容别名指向 HYGON BW1000 实现 |
| 17 | `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/glm/glm53.py` | 基线无同名文件（Megatron-Bridge GLM recipe 族） | 保留 NVIDIA 声明，追加 HYGON Copyright；兼容别名指向 HYGON BW1000 实现 |
| 18 | `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/glm/bw1000/__init__.py` | 基线无同名文件（`glm/bw1000` 为 HYGON 新增目录） | 追加 HYGON Copyright；导出 BW1000 GLM recipe |
| 19 | `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/glm/bw1000/glm52.py` | 基线无同名文件（`glm/bw1000` 为 HYGON 新增目录） | 保留 NVIDIA 声明，追加 HYGON Copyright；新增 BW1000 预训练/SFT 配置 |
| 20 | `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/glm/bw1000/glm53.py` | 基线无同名文件（`glm/bw1000` 为 HYGON 新增目录） | 保留 NVIDIA 声明，追加 HYGON Copyright；新增 BW1000 预训练配置 |
| 21 | `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/minimax/__init__.py` | `src/megatron/bridge/recipes/minimax/__init__.py` | 保留 NVIDIA 声明，追加 HYGON Copyright；注册 BW1000 recipe |
| 22 | `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/minimax/minimax_m3.py` | `src/megatron/bridge/recipes/minimax/minimax_m3.py` | 保留 NVIDIA 声明，追加 HYGON Copyright；兼容别名指向 HYGON BW1000 实现 |
| 23 | `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/minimax/bw1000/__init__.py` | 基线无同名文件（`minimax/bw1000` 为 HYGON 新增目录） | 保留 NVIDIA 声明，追加 HYGON Copyright；导出 BW1000 MiniMax recipe |
| 24 | `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/minimax/bw1000/minimax_m3.py` | 基线无同名文件（`minimax/bw1000` 为 HYGON 新增目录） | 保留 NVIDIA 声明，追加 HYGON Copyright；新增 BW1000 预训练配置 |
| 25 | `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/qwen/__init__.py` | `src/megatron/bridge/recipes/qwen/__init__.py` | 保留 NVIDIA 声明，追加 HYGON Copyright；注册 BW1000 与 Qwen3.8 recipe |
| 26 | `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/qwen/qwen35.py` | `src/megatron/bridge/recipes/qwen/qwen35.py` | 保留 NVIDIA 声明，追加 HYGON Copyright；兼容别名指向 HYGON BW1000 实现 |
| 27 | `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/qwen/qwen35_vl.py` | 基线无同名文件（`qwen/qwen35_vl.py` 为 HYGON 新增文件） | 保留 NVIDIA 声明，追加 HYGON Copyright；新增多模态 recipe 兼容入口 |
| 28 | `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/qwen/qwen38.py` | 基线无同名文件（Megatron-Bridge Qwen recipe 族） | 保留 NVIDIA 声明，追加 HYGON Copyright；兼容别名指向 HYGON BW1000 实现 |
| 29 | `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/qwen/bw1000/__init__.py` | 基线无同名文件（`qwen/bw1000` 为 HYGON 新增目录） | 保留 NVIDIA 声明，追加 HYGON Copyright；导出 BW1000 Qwen recipe |
| 30 | `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/qwen/bw1000/qwen35.py` | 基线无同名文件（`qwen/bw1000` 为 HYGON 新增目录） | 保留 NVIDIA 声明，追加 HYGON Copyright；新增 BW1000 预训练配置 |
| 31 | `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/qwen/bw1000/qwen35_vl.py` | 基线无同名文件（`qwen/bw1000` 为 HYGON 新增目录） | 追加 HYGON Copyright；新增 BW1000 多模态预训练配置 |
| 32 | `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/qwen/bw1000/qwen38.py` | 基线无同名文件（`qwen/bw1000` 为 HYGON 新增目录） | 保留 NVIDIA 声明，追加 HYGON Copyright；新增 BW1000 预训练配置 |

### 2.2 源自 NVIDIA Megatron-LM（BSD-3-Clause）

固定版本：`cd4afffa648426a959dc7cb1e24b5ce7d0c3ff54`（本仓库 `3rdparty/Megatron-Bridge` 内嵌基线）
仓库 URL：https://github.com/NVIDIA/Megatron-LM.git
Copyright：Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
许可证：BSD-3-Clause

BSD-3-Clause 要求随发布保留完整版权声明、许可条件与免责声明，不要求逐文件追加 SPDX 标识。下列文件保留原 NVIDIA Copyright，并追加 HYGON Copyright 标识 HYGON 的原创修改部分。

| # | 本地路径 | 上游对应路径 | HYGON 修改 |
| ---: | --- | --- | --- |
| 33 | `hcu_megatron_bridge/core/utils.py` | `megatron/core/utils.py` | 保留 NVIDIA 声明，追加 HYGON Copyright；扩展 DSA 后端选择逻辑 |
| 34 | `hcu_megatron_bridge/training/arguments.py` | `megatron/training/arguments.py` | 保留 NVIDIA 声明，追加 HYGON Copyright；适配参数解析与参数裁剪 |
| 35 | `hcu_megatron_bridge/core/transformer/experimental_attention_variant/dsa_kernels.py` | `megatron/core/transformer/experimental_attention_variant/dsa_kernels.py` | 保留 NVIDIA 声明，追加 HYGON Copyright；适配 HCU 内核路径 |
| 36 | `hcu_megatron_bridge/core/fusions/fused_mhc_kernels.py` | 基线无同名文件（Megatron-LM fusions 族）；对照来源待最终确认 | 保留 NVIDIA 声明，追加 HYGON Copyright；适配 HCU 融合内核 |
| 37 | `hcu_megatron_bridge/core/transformer/experimental_attention_variant/csa_kernels.py` | 基线无同名文件（Megatron-LM experimental attention 族）；对照来源待最终确认 | 保留 NVIDIA 声明，追加 HYGON Copyright；适配 HCU 内核路径 |
| 38 | `hcu_megatron_bridge/core/transformer/experimental_attention_variant/csa_fused_compressor.py` | 基线无同名文件（Megatron-LM experimental attention 族）；对照来源待最终确认 | 保留 NVIDIA 声明，追加 HYGON Copyright；适配 HCU 融合压缩实现 |
| 39 | `hcu_megatron_bridge/core/transformer/experimental_attention_variant/dsa_hip_kernels.py` | 基线无同名文件（Megatron-LM experimental attention 族）；对照来源待最终确认 | 保留 NVIDIA 声明，追加 HYGON Copyright；适配 HCU 内核后端 |

## 3. 许可证全文位置

| 许可证 | 全文位置 |
| --- | --- |
| Apache-2.0 | `3rdparty/Megatron-Bridge/LICENSE` |
| BSD-3-Clause | `3rdparty/Megatron-Bridge/3rdparty/Megatron-LM/LICENSE` |

发布包须随附上述许可证全文；若上游附带 `NOTICE` 文件，须一并保留。

## 4. 维护要求

- 新增或升级第三方源码时，必须在本文件登记固定 Commit/版本、许可证与本地路径，并确认许可证属于 MIT、BSD-3-Clause、Apache-2.0、BSD-2-Clause、0BSD 等自动准入范围。
- 第三方文件保留原文件与原声明，不批量补 HYGON 文件头；确有 HYGON 修改时，仅追加修改说明或 HYGON Copyright（覆盖 HYGON 原创部分）。
- 更新上游子模块时，同步更新本文件第 1 节的固定 Commit/版本及 README 中的上游归属信息。
