# Megatron-Bridge-das 项目结构与 Bridge 适配/调用流程

本文档面向初次接触本仓库的开发者，说明 Megatron-Bridge-das 是如何在 NVIDIA 的
[Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)
之上做 **HCU（海光 DCU / ROCm 生态）适配** 的，重点梳理：

1. 仓库的整体目录结构；
2. `hcu_megatron_bridge` 适配层是如何挂接到上游 Megatron-Bridge 的；
3. 从 `examples/` 下的启动脚本到最终训练循环的完整调用链路。

> 本文档不涉及测试脚本的执行，仅做静态代码/结构分析。

---

## 1. 项目定位

`README.md`（仓库根目录）中说明：

> Megatron-Bridge-das 是基于 Megatron-Bridge 的一个 **patch 版训练库**，
> 面向 HCU 生态提供大模型训练解决方案。

也就是说，这个仓库本身**不是**一套独立的训练框架，而是：

- 以 git submodule 的方式整份引入上游 `Megatron-Bridge`（见 `3rdparty/Megatron-Bridge`，对应
  `.gitmodules` 中的 `3rdparty/Megatron-Bridge` → `https://github.com/NVIDIA-NeMo/Megatron-Bridge.git`）；
- 在仓库自己的 `hcu_megatron_bridge/` 中实现一套 **运行时 Monkey-Patch 框架**，
  在进程启动时把 ROCm/DCU 专用的算子、Bridge 子类、特性开关"打"进上游 Megatron-Bridge /
  Megatron-Core 的模块里；
- 通过 `run_recipe.py`（根目录）和 `examples/*/train_*.sh` 这一组启动脚本，
  把上游的 Recipe 驱动的训练入口（`scripts/training/run_recipe.py`）包装成一组
  国产化、开箱即用的训练脚本。

## 2. 顶层目录结构

```text
Megatron-bridge-das/
├── 3rdparty/
│   └── Megatron-Bridge/        # git submodule，NVIDIA 官方 Megatron-Bridge 源码（未修改）
├── data_preparation/           # 数据预处理脚本
├── docs/                       # 文档（本文件所在目录），含各模型的 quick start
├── env/
│   └── env.sh                  # 统一的运行环境变量 + PYTHONPATH 拼装 + recipe 注册表替换
├── examples/                   # 各模型家族的训练启动脚本（入口）
│   ├── deepseek/
│   ├── glm/
│   ├── minimax/
│   └── qwen/
├── hcu_megatron_bridge/        # ★ HCU 适配层核心代码（本仓库的主要贡献）
│   ├── core/                   # 对 megatron.core 的算子/kernel 替换实现
│   ├── features_manager/       # "Feature → Patch" 注册机制
│   ├── src/hcu_megatron/bridge/
│   │   ├── models/             # 对 AutoBridge / 具体模型 Bridge 的猴子补丁与扩展
│   │   └── recipes/             # 新增模型家族（qwen35/qwen35_vl/qwen38/deepseek/glm/minimax）的 Recipe
│   ├── training/arguments.py   # 对 megatron.training 参数解析的包装
│   ├── megatron_adaptor.py     # 适配层入口：集中触发所有补丁的注册与应用
│   ├── patch_utils.py          # 通用的函数/类级别猴子补丁工具（Patch / MegatronPatchesManager）
│   └── utils.py
├── scripts/conversion/         # HF ↔ Megatron 权重转换脚本（NeMo Run 驱动，兼容 Slurm/本地/手动）
├── tests/
├── run_recipe.py               # 训练总入口：转发到上游 run_recipe.py，并在 ROCm 下触发适配层
├── requirements.txt
└── version
```

关键点：**上游代码一行不改**，所有定制都发生在 `hcu_megatron_bridge/`（通过运行期 Patch）
和 `env/env.sh`（通过构建期文件替换 + 环境变量）两处。

## 3. examples/ 入口脚本解读

`examples/` 下按模型家族分组，每个子目录至少有一个 `train_<model>_<task>.sh`，
典型结构（以 `examples/deepseek/train_deepseekv4_flash_pretrain.sh` 为例）：

1. **定位路径**：通过 `CURRENT_DIR`/`MEGATRON_PATH` 计算出仓库根目录；
2. **声明支持的 Recipe 名单**（`SUPPORTED_RECIPES`），这些名字直接对应
   `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/**` 里导出的 Python 函数名；
3. **解析启动参数**：`RECIPE`、`MODE`（`dryrun`/`train`）、`HOSTFILE`、`NODES`；
4. **拼装一组 ConfigContainer overrides**（`DATA_CONFIG`/`TRAIN_CONFIG`/`OPTIMIZER_CONFIG`/
   `PARALLEL_CONFIG`/`MODEL_CONFIG`/`CKPT_AND_LOGGING_CONFIG`/`PROFILING_CONFIG`），
   每一项都是 `key.path=value` 形式的字符串，最终会被当作命令行尾部参数传给 `run_recipe.py`；
5. **多机拓扑**：若提供 hostfile，用首行主机做 rendezvous master，并生成
   `hostfile_slots` 给 `mpirun` 使用；否则单机走 `torchrun`；
6. **关键一步（DeepSeek 专属）**：

   ```bash
   cd ${MEGATRON_PATH}/3rdparty/Megatron-Bridge
   ./scripts/switch_mcore.sh dev
   ```

   训练前先切换子模块内嵌的 Megatron-LM 到 `dev` 分支，说明部分模型（如 DeepSeek-V4-Flash）
   依赖上游尚在开发分支上的 Megatron-Core 特性；
7. **执行**：
   - `dryrun` 模式：直接跑 `python run_recipe.py --recipe ... --mode pretrain --dry-run <overrides>`，
     只解析配置、落盘 YAML，不做分布式初始化；
   - `train` 模式：单机 `torchrun ... run_recipe.py ...`；多机则用 `mpirun` 在每个节点上各起一个
     `torchrun`（`--node_rank` 由 `OMPI_COMM_WORLD_RANK` 在远端求值），并在远端 source
     `/opt/dtk/env.sh` 和本仓库的 `env/env.sh` 之后再执行。

`examples/qwen/run_qwen.sh` 是一个更"瘦"的 wrapper：它只是把
`LOCAL_HF_PATH` / `DATA_PATH` / `TRAIN_ITERS` 等常用环境变量集中放在一处（可被外部环境覆盖），
再 `exec` 到同目录下真正的 `train_qwen35_vl_pretrain.sh`，本身不直接接触 Bridge/Recipe 逻辑。

### 命令行到训练配置的下钻路径

```
examples/<family>/train_*.sh
  └─ (设置 RECIPE/MODE/并行度等 shell 变量, 拼成 KEY=VALUE overrides)
     └─ torchrun / mpirun
        └─ python run_recipe.py --recipe <RECIPE> --mode pretrain <KEY=VALUE ...>
```

## 4. 训练总入口：根目录 `run_recipe.py`

仓库根的 `run_recipe.py` 本身**不是**训练逻辑的实现者，而是一层"路由 + ROCm 触发器"：

```python
SCRIPT_DIR = Path(__file__).resolve().parent / "3rdparty" / "Megatron-Bridge" / "scripts" / "training"
sys.path.insert(0, str(SCRIPT_DIR))          # 把上游训练脚本目录加进 sys.path
...
if torch.version.hip is not None:
    from hcu_megatron_bridge import megatron_adaptor   # ★ ROCm 下才触发适配层
```

它从上游 `3rdparty/Megatron-Bridge/scripts/training/` 和 `scripts/common/` 下 import：

- `recipe_metadata`：识别 `--recipe` 是"库 Recipe"还是"Benchmark Recipe"，推断训练模式；
- `recipe_runner`：`load_recipe` / `apply_cli_overrides` / `run_config` 等，真正执行
  config 解析、ConfigContainer 覆盖、forward-step 选择、调用 `pretrain`/`finetune`；
- `benchmark_parallelism`：并行拓扑换算（弱扩展 global batch size 等）。

`main()` 的核心流程：

1. `_load_selected_recipe` → 调用上游 `recipe_runner.load_recipe(recipe_name)`，
   本质是 `getattr(megatron.bridge.recipes, recipe_name)()`，返回一个
   `ConfigContainer`（内含 `model/dataset/train/optimizer/scheduler/checkpoint/logger/profiling` 等子配置）；
2. 依次应用：`--dataset` 预设、`deterministic` 开关、CLI 的常用便捷参数
   （`-tp/-pp/-cp/-ms/-gb/...`，见文件顶部文档字符串中的映射表）、以及尾部任意
   `key.path=value` 的 **ConfigContainer overrides**；
3. 做一系列"同步"（sequence length、pipeline layout、PEFT/CP 不变量等）；
4. 解析 `forward_step`（例如 `dsv4_step` → `megatron.bridge.models.deepseek.deepseek_v4_step.forward_step`）；
5. 调用 `recipe_runner.run_config(config, mode, step_func, dryrun, ...)`，
   该函数内部再去调上游 `megatron.bridge.training.pretrain` / `finetune`，
   这才是真正的训练循环入口（本仓库未重写该循环本身）。

**重要细节**：ROCm 适配只发生在 `import hcu_megatron_bridge.megatron_adaptor` 这一行，
且只有检测到 `torch.version.hip is not None`（即运行在 ROCm/HIP 构建的 PyTorch 上）才会触发；
在 NVIDIA GPU 上这一步是跳过的，脚本行为与纯上游 Megatron-Bridge 完全一致。

## 5. `hcu_megatron_bridge`：适配层的核心机制

### 5.1 补丁框架：`patch_utils.py`

`MegatronPatchesManager` + `Patch` 是一套通用的"按点分路径（dotted path）replace 符号"的
猴子补丁系统（注释显示思路借鉴自 Ascend MindSpeed 的适配方案）：

- `Patch.parse_path("a.b.c.Foo.bar", ...)`：按模块路径逐段 `import_module`，
  找到最终的类/函数对象，必要时可以 `create_dummy` 创建占位模块/占位函数
  （用于兼容某些可选依赖缺失，例如 ROCm 专属的 `fused_mhc` 包在 NVIDIA 环境不存在）；
- `register_patch(orig_path, new_func_or_cls, apply_wrapper=..., remove_origin_wrappers=...)`：
  注册一个"原符号 → 新符号"的替换规则，支持：
  - 整函数/类替换；
  - 以装饰器形式叠加 wrapper（`apply_wrapper=True`，常用于加一层函数包装而不是整体替换）；
  - 剥离原有的 `staticmethod`/`classmethod`/`functools.wraps` 包装后再重新包装；
  - 替换函数内部的闭包常量（`patch_inner_func`，用于替换嵌套函数）；
- `apply_patches()`：遍历所有已注册的 `Patch`，执行真正的 `setattr`，同时扫描
  `sys.modules` 里所有已经 `from ... import ...` 过该符号的模块，把它们持有的引用也一起替换掉
  （否则"补丁生效前已经导入"的模块看到的还是旧函数）；
- `remove_patches()` / `repatch()`：撤销所有补丁并重新注册（测试/重复调用场景使用）。

### 5.2 Feature 注册机制：`features_manager/`

`AbstractFeature`（`features_manager/feature.py`）定义了每个"特性"的生命周期：

- `feature_name` + `optimization_level`：`optimization_level == 0` 的特性会设置
  `default_patches = True`，即**无条件默认开启**；`optimization_level == 2` 的特性
  则只有在命令行显式打开对应 flag（`--<feature_name>`）时才生效；
- `register_args` / `pre_validate_args` / `validate_args` / `post_validate_args`：
  给 `megatron.training` 的参数解析器挂钩子，扩充/校验 CLI 参数；
- `register_patches(patch_manager, args)`：真正调用 `patch_manager.register_patch(...)`
  把该特性相关的符号替换规则注册进 `MegatronPatchesManager`。

当前注册的特性列表（`features_manager/__init__.py` 中的 `ADAPTOR_FEATURES`）：

| Feature | 默认开启 | 作用 |
|---|---|---|
| `MegatronBasicFeature` | 是 | 让 `AutoBridge.from_hf_pretrained` 支持 `LOCAL_HF_PATH` 环境变量覆盖，离线加载本地 HF 权重 |
| `DeepSeekV4Feature` | 是 | 把 `deepseek_v4_supports_blackwell_fused_kernels` 替换为 ROCm 版本的能力探测（`device_capability.major >= 9`） |
| `ProfilerFeature` | 是 | 给 MHC/DSA 相关的大量 kernel 函数包一层 `profiler_annotate_patch`，用于打 profiling 标注；对 ROCm 独有的 `fused_mhc`/cutile 分支做存在性探测以跳过不适用环境 |
| `DSAFeature` | 是 | 新增 `--dsa-kernel-backend` CLI 选项（`none/tilelang/cudnn/hip`），并把 DSA（稀疏注意力）相关的多个 kernel 后端实现替换为 HIP 版本 |
| `HyperConnectionFeature` | 是 | 把 Hyper-Connection（MHC）相关的 `fused_sinkhorn`/`fused_h_aggregate`/`fused_h_post_bda`/`fused_proj_rms_compute_h` 替换为 ROCm/HIP 实现 |

### 5.3 适配层入口：`megatron_adaptor.py`

```python
def patch_features():
    adaptor_args = get_adaptor_args()
    for feature in ADAPTOR_FEATURES:
        if (getattr(adaptor_args, feature.feature_name, None) and feature.optimization_level == 2) \
           or feature.default_patches:
            feature.register_patches(MegatronPatchesManager, adaptor_args)
    MegatronPatchesManager.apply_patches()

patch_features()   # 模块级别语句：import 即执行
```

因为 `patch_features()` 是**模块级别**调用，所以只要执行了
`from hcu_megatron_bridge import megatron_adaptor`（即 `run_recipe.py` 在检测到
`torch.version.hip` 时做的那一步），就会立刻：

1. 用一个独立的 `ArgumentParser` 解析一遍适配层专属参数（`get_adaptor_args`，与
   `megatron.training` 的主参数解析器分离，`parse_known_args` 容忍未知参数）；
2. 遍历所有 `ADAPTOR_FEATURES`，决定哪些该生效；
3. 调用各自的 `register_patches` 把替换规则注册进 `MegatronPatchesManager`；
4. 统一 `apply_patches()`，完成对上游 `megatron.bridge.*` / `megatron.core.*` 模块的运行期替换。

另外，`hcu_megatron_bridge/training/arguments.py` 里的 `process_adaptor_args` /
`parse_args` 是对 `megatron.training.arguments` 的包装，确保**主训练参数解析器**也能
识别适配层新增的 CLI flag（如 `--dsa-kernel-backend`）。

### 5.4 Bridge / Recipe 扩展：`src/hcu_megatron/bridge/`

这是 PYTHONPATH 上真正可被 `import hcu_megatron.bridge...` 的包（注意顶层包名是
`hcu_megatron`，不是 `hcu_megatron_bridge`；二者是两套不同的 Python 包，详见下节的
PYTHONPATH 说明）：

- `models/conversion/auto_bridge.py`：`from_hf_pretrained_wrapper` —— 一个装饰器，
  在调用上游 `AutoBridge.from_hf_pretrained(model_id, ...)` 之前，如果设置了
  `LOCAL_HF_PATH` 环境变量，就用它替换掉传入的第二个参数（HF 模型路径/ID），
  从而实现"同一行脚本代码，离线环境下自动切到本地权重目录"；这正是
  `MegatronBasicFeature` 注册的补丁所使用的实现；
- `models/deepseek/deepseek_v4_bridge.py`：`deepseek_v4_supports_bw_fused_kernels` ——
  用 `torch.cuda.get_device_capability().major >= 9` 判断是否应开启 DeepSeek-V4
  的"BW-only"（原指 NVIDIA Blackwell）融合 kernel；在 ROCm/DCU 上这是复用同一个
  major-version 判定接口，替换掉上游针对 Blackwell 架构的判断逻辑；
- `recipes/<family>/`：新增的模型家族 Recipe，与上游 Recipe 同样是"返回
  `ConfigContainer` 的工厂函数"，按 `bw1000`（带宽/拓扑代号）等再分子目录，
  例如 `recipes/qwen35/bw1000/qwen35.py` 导出
  `qwen35_text_35b_a3b_pretrain_32gpu_bw1000_bf16_config`；
  `recipes/qwen35/qwen35.py` 则是保留旧名字的兼容别名层；
- `recipes/__init__.py`：**关键胶水文件**。它先 `from megatron.bridge.recipes.<family> import *`
  星号导入上游全部模型家族的 Recipe，再追加
  `from hcu_megatron.bridge.recipes.<family> import *`（deepseek/glm/minimax/qwen35/
  qwen35_vl/qwen38），让这个文件成为"上游 Recipe ∪ HCU 新增 Recipe"的**超集**。

### 5.5 `core/`：底层 kernel 实现

`hcu_megatron_bridge/core/` 存放具体的 ROCm/HIP kernel 实现，被上面的 Feature 补丁引用：

- `fusions/fused_mhc_kernels.py`：Hyper-Connection 相关的融合算子（Sinkhorn、H-aggregate、
  H-post-BDA、proj-rms-compute-h 的 HIP 版本）；
- `transformer/experimental_attention_variant/`：DSA（稀疏注意力变体）相关 kernel，
  区分 `csa_*`（压缩稀疏注意力）与 `dsa_*`（含 `dsa_hip_kernels.py`，HIP 后端实现）。

这些文件本身是纯粹的算子实现，不直接与 Bridge/Recipe 耦合；通过
`features_manager/transformer/{dsa_feature,hyper_connection_feature}.py` 里的
`register_patches` 调用，被接到上游 `megatron.core.*` 对应符号上。

## 6. 构建期的"文件替换"：`env/env.sh`

与运行期的猴子补丁并列的另一种适配手段，是**构建期直接覆盖源文件**：

```bash
cp ${MEGATRON_PATH}/hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/__init__.py \
   ${MEGATRON_PATH}/3rdparty/Megatron-Bridge/src/megatron/bridge/recipes/
```

即：每次 `source env/env.sh`（训练/多机 `mpirun` 远端都会 source 它），
都会把第 5.4 节提到的"胶水 `__init__.py`"**物理覆盖**到上游 submodule 的
`megatron/bridge/recipes/__init__.py` 上。这是因为上游的
`find_library_recipe` 是通过 `getattr(megatron.bridge.recipes, recipe_name)` 做
Recipe 名称解析的，只有把 HCU 新增的 Recipe 符号真正合并进那个被 import 的模块对象里，
`--recipe qwen35_text_35b_a3b_pretrain_32gpu_bw1000_bf16_config` 之类的名字才能被上游
`run_recipe.py` 代码路径认出来。

`env.sh` 同时负责：

- 拼装 `PYTHONPATH`：

  ```
  ${MEGATRON_PATH}/hcu_megatron_bridge/src
  : ${MEGATRON_PATH}/3rdparty/Megatron-Bridge/src
  : ${MEGATRON_PATH}/3rdparty/Megatron-Bridge/3rdparty/Megatron-LM
  ```

  三段路径分别对应：HCU 新增 Bridge/Recipe 包、上游 Megatron-Bridge 包、
  上游内嵌的 Megatron-LM（Megatron-Core）包；
- 设定一批 NCCL/RCCL/ROCm 相关环境变量（`NCCL_ALGO`、`HSA_FORCE_FINE_GRAIN_PCIE`、
  `ROCSHMEM_IB_GID_INDEX` 等），并根据 `/sys/class/infiniband` 下设备名自动判断
  `IB_TYPE`（`mlnx` 或 `shca`，国产 IB 网卡）来设置不同的 `NCCL_IB_HCA` 等参数。

## 7. 完整调用链路总览

```
┌─────────────────────────────────────────────────────────────────────┐
│ examples/<family>/train_*.sh                                        │
│  · 选定 RECIPE / MODE / 并行度 / 数据集路径等                         │
│  · source env/env.sh  →  cp HCU recipes/__init__.py 覆盖上游同名文件  │
│                           →  拼 PYTHONPATH  →  设 NCCL/IB 环境变量     │
│  · torchrun / mpirun 启动分布式进程                                   │
└───────────────┬───────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ run_recipe.py（仓库根目录）                                           │
│  · sys.path 注入上游 scripts/training, scripts/common                │
│  · if torch.version.hip: import hcu_megatron_bridge.megatron_adaptor │
│        └─ 触发 patch_features() → 遍历 ADAPTOR_FEATURES →            │
│           MegatronPatchesManager.register_patch(...) →               │
│           apply_patches()：对 megatron.bridge.* / megatron.core.*    │
│           的若干符号做运行期替换（AutoBridge.from_hf_pretrained、      │
│           DeepSeek 能力探测、DSA/MHC 融合 kernel、profiler 标注）       │
│  · parse_args → 解析 --recipe/--model/--mode 及 KEY=VALUE overrides  │
│  · recipe_runner.load_recipe(recipe_name)                            │
│        └─ getattr(megatron.bridge.recipes, recipe_name)()            │
│           （此时 megatron.bridge.recipes 已是被 HCU 覆盖过的超集模块） │
│  · apply_cli_overrides / sync_* 系列：落地 CLI 覆盖与配置一致性检查     │
│  · recipe_runner.run_config(config, mode, step_func, ...)            │
└───────────────┬───────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 上游 megatron.bridge.training.pretrain / finetune（未改动）           │
│  · 内部通过 AutoBridge（已被 HCU 补丁影响行为）加载/转换 HF 权重       │
│  · 调用被 HCU 替换过的 Megatron-Core 算子（DSA/MHC/DeepSeek kernel）   │
│  · 标准 Megatron 训练循环（forward/backward/optimizer/checkpoint）     │
└─────────────────────────────────────────────────────────────────────┘
```

## 8. 权重转换：`scripts/conversion/`

与训练路径并列的另一条支线是 HF ↔ Megatron 权重转换，走的是同一种"包一层 launcher，
内部仍复用上游逻辑"的模式：

- `scripts/conversion/convert.sh`：`exec python setup_conversion.py "$@"`，
  `cd` 到 `3rdparty/Megatron-Bridge` 执行；
- `setup_conversion.py`：基于 NeMo Run 构建 Local/Slurm Executor 和 `run.Script` 任务，
  支持 `--executor {local,slurm,manual}` 和 `--device {cpu,gpu}`；关键是把
  `PYTHONPATH` 指向本仓库 `3rdparty/Megatron-Bridge/src` + 上游内嵌
  `Megatron-LM`，并额外加入未被 patch 的 `3rdparty/Megatron-Bridge/scripts/conversion`
  目录，让 `run_conversion.py` 能找到未修改的 `cpu_backend.py`/`utils.py` 等同级模块；
- `run_conversion.py`（本仓库的 patched 版本）：实际的转换 worker，`import cpu_backend`
  / `gpu_backend`（来自上游目录，经由上一步追加的 `sys.path`），调用
  `AutoBridge`（同样受 `MegatronBasicFeature` 补丁影响）完成 HF → Megatron 或
  Megatron → HF 的转换，以及 round-trip 校验。

该支线与训练支线共享同一个 Bridge 补丁集合（因为二者都要 `AutoBridge.from_hf_pretrained`），
但转换脚本不经过 `run_recipe.py`，而是独立触发（`run_conversion.py` 并未显式 import
`hcu_megatron_bridge.megatron_adaptor`；补丁生效与否取决于其导入链是否间接拉起该模块，
实际以运行环境里是否已 `import hcu_megatron_bridge` 为准）。

## 9. 关键设计要点小结

1. **零侵入上游**：`3rdparty/Megatron-Bridge` 作为只读 submodule，所有定制都在仓库自己的
   `hcu_megatron_bridge/` 和 `env/env.sh` 中完成，便于后续跟上游同步升级。
2. **两种"打补丁"手段并用**：
   - 运行期猴子补丁（`patch_utils.py` + `features_manager/*`），用于替换函数/类/算子；
   - 构建期文件覆盖（`env.sh` 覆盖 `recipes/__init__.py`），用于让新增 Recipe 对上游
     "按名字查找"的机制可见。
3. **Feature 可插拔**：新增一个 ROCm 特性只需要新写一个 `AbstractFeature` 子类，
   实现 `register_args`/`register_patches`，塞进 `ADAPTOR_FEATURES` 列表即可，
   不需要改动 `megatron_adaptor.py` 或 `patch_utils.py` 本身。
4. **ROCm 专属触发**：几乎所有适配逻辑只在 `torch.version.hip is not None` 时生效，
   在 NVIDIA 环境下运行同一套脚本会完全退化为原生 Megatron-Bridge 行为。
5. **Recipe 命名即契约**：`examples/*/train_*.sh` 里的 `SUPPORTED_RECIPES` 字符串必须与
   `hcu_megatron/bridge/recipes/**` 导出的函数名严格一致，这是连接"启动脚本"与
   "Python 配置工厂函数"的唯一纽带。
