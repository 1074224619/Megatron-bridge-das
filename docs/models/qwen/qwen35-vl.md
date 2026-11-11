<!-- Copyright (c) 2026 Hygon Information Technology Co., Ltd. -->

# Qwen3.5-VL 系列

Qwen3.5-VL 多模态预训练（pretrain）性能测试，覆盖 dense 和 MoE 两类模型，
支持 BW1000 / BW1100 两个平台。每个尺寸都有两个变体：

- **Energon recipe**：读真实 Energon 格式数据集
- **Mock recipe**：生成确定性的合成图文样本，用于分布式冒烟测试

所有 recipe 都不加载预训练权重，只测性能。

recipe 放在 `hcu_megatron` patch 层里。`env/env.sh` 每次启动时会把 patch 层的
`recipes/__init__.py` 复制到上游目录，runner 靠这个解析 recipe 名字；仓库里
`3rdparty/` 下的文件不会被修改。

- Recipes：[hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/qwen/bw1100/qwen35_vl.py](../../../hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/qwen/bw1100/qwen35_vl.py)
- 注册入口：[hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/__init__.py](../../../hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/__init__.py)
- 执行入口：[run_recipe.py](../../../run_recipe.py)
- Launcher：[examples/qwen/train_qwen35_vl_pretrain.sh](../../../examples/qwen/train_qwen35_vl_pretrain.sh)

## Supported

`节点数` 是 recipe 名字里标称的部署规模，由 TP/PP/EP 的最小几何规模决定。
BW1000 与 BW1100 的拓扑一致（144G 卡尚未针对 VL 单独调参）。

| 模型 | 类型 | TP | PP | EP | SP | 序列长度 | 节点 | 卡数 |
|---|---|---:|---:|---:|---|---:|---:|---:|
| 2B | dense | 1 | 1 | 1 | No | 4096 | 1 | 8 |
| 4B | dense | 2 | 1 | 1 | Yes | 4096 | 1 | 8 |
| 9B | dense | 2 | 2 | 1 | Yes | 2048 | 1 | 8 |
| 27B | dense | 4 | 2 | 1 | Yes | 4096 | 1 | 8 |
| 35B-A3B | MoE | 1 | 2 | 8 | No | 4096 | 2 | 16 |
| 122B-A10B | MoE | 4 | 2 | 4 | Yes | 4096 | 1 | 8 |
| 397B-A17B | MoE | 4 | 4 | 8 | Yes | 4096 | 4 | 32 |

对应的 recipe 名（以 BW1100 为例）：

| 模型 | Energon recipe | Mock recipe |
|---|---|---|
| 2B | `qwen35_vl_2b_pretrain_8gpu_bw1100_bf16_config` | `qwen35_vl_2b_pretrain_8gpu_bw1100_bf16_mock_config` |
| 4B | `qwen35_vl_4b_pretrain_8gpu_bw1100_bf16_config` | `qwen35_vl_4b_pretrain_8gpu_bw1100_bf16_mock_config` |
| 9B | `qwen35_vl_9b_pretrain_8gpu_bw1100_bf16_config` | `qwen35_vl_9b_pretrain_8gpu_bw1100_bf16_mock_config` |
| 27B | `qwen35_vl_27b_pretrain_8gpu_bw1100_bf16_config` | `qwen35_vl_27b_pretrain_8gpu_bw1100_bf16_mock_config` |
| 35B-A3B | `qwen35_vl_35b_a3b_pretrain_16gpu_bw1100_bf16_config` | `qwen35_vl_35b_a3b_pretrain_16gpu_bw1100_bf16_mock_config` |
| 122B-A10B | `qwen35_vl_122b_a10b_pretrain_8gpu_bw1100_bf16_config` | `qwen35_vl_122b_a10b_pretrain_8gpu_bw1100_bf16_mock_config` |
| 397B-A17B | `qwen35_vl_397b_a17b_pretrain_32gpu_bw1100_bf16_config` | `qwen35_vl_397b_a17b_pretrain_32gpu_bw1100_bf16_mock_config` |

把名字里的 `bw1100` 换成 `bw1000` 就是 BW1000 平台的 recipe，拓扑完全相同。

并行取值说明：

- 序列并行（SP）要求 TP > 1，所以 4B / 9B / 27B / 122B / 397B 用 TP2 或 TP4，
  而 35B-A3B 是 TP1 拓扑，SP 必须关掉。
- dense 模型的 EP 恒为 1；MoE 模型靠 EP 把专家摊到多卡。
- TP 必须整除 attention head 数，PP 必须整除层数，ETP×EP×PP 必须整除总卡数。

架构参数（层数、hidden size、head 数等）由 `AutoBridge` 从模型自己的
`config.json` 读取，recipe 里刻意不重复声明，避免和权重对不上。上表的
`序列长度` 是训练长度，不是模型的最大位置数。

所有 recipe 都是 BF16 + Transformer Engine，micro/global batch size 1/32，
50 步，Adam（betas `0.9/0.95`），weight decay `0.1`，cosine 学习率
`3e-5 → 3e-6`，warmup 1 步。评估关闭（`validation.eval_iters=0`），因为都是
短程 bring-up。三个 tower 全部参与训练、没有冻结，这点和上游对齐阶段的
recipe 不同。


## 1、环境准备

### 1.1 [光源社区](https://developer.sourcefind.cn/servicelist)拉取 HCU 基础 docker 镜像

### 1.2 创建容器

参考命令如下（请按需修改，正确挂载工作目录，修改正确镜像名称）：

```shell
docker run -it \
       --name megatron-bridge_test \
       --shm-size=64G \
       --device=/dev/kfd \
       --device=/dev/mkfd \
       --device=/dev/dri \
       --cap-add=SYS_PTRACE \
       --security-opt seccomp=unconfined \
       --ulimit memlock=-1:-1 \
       --ipc=host \
       --network=host \
       --workdir=/workspace \
       --privileged \
       -v /opt/hyhal:/opt/hyhal:ro \
       -v /home:/home \
       REPOSITORY:TAG \
       /bin/bash
```

### 1.3 拉取代码仓

```shell
cd /workspace

git clone --recurse-submodules https://github.com/HYGON-AI/Megatron-bridge-das.git
```

### 1.4 安装依赖

```shell
cd /workspace/Megatron-bridge-das
pip install -r requirements.txt
```

## 2、数据处理

### 2.1 Mock 数据

mock recipe 自己生成合成图文样本，无需任何数据集准备，设置
`MOCK_NUM_BASE_EXAMPLES` 控制样本数即可。

### 2.2 Energon 数据

Energon recipe 需要一个已经准备好的 Energon 数据集，路径通过 `DATA_PATH`
（或按尺寸的 `QWEN35_<SIZE>_DATA_PATH`，如 `QWEN35_9B_DATA_PATH`）指定。
数据集的 task encoder、`pad_to_max_length` 等由 recipe 内部固定：

- 数据类型：`EnergonDatasetConfig`，`dataloader_type="external"`
- task encoder：`QwenVLEnergonTaskEncoderConfig`（processor 从
  `LOCAL_HF_PATH` 加载，与权重保持同一目录）
- `do_validation` / `do_test` 关闭，`enable_in_batch_packing` 关闭

下面是把单条样本含多张图片的数据集转成 Energon 的例子,用的是 [TIGER-Lab/Mantis-Instruct](https://huggingface.co/datasets/TIGER-Lab/Mantis-Instruct) 的一个子集。

1. 从 modelscope 下载 TIGER-Lab/Mantis-Instruct 的 `llava_665k_multi` 子集并解压图片目录(注意:需要 44GB 磁盘空间):

    ```
    modelscope download --dataset TIGER-Lab/Mantis-Instruct \
      --include "llava_665k_multi/*" \
      --repo-type dataset  \
      --local_dir ./Mantis-Instruct  
    ```

2. 转换并索引下载好的子集。这个脚本会写出确定性的 train/validation 分片、调用 Energon 的准备 API,并写出 `.nv-meta/dataset.yaml`:

    ```bash
    uv run python 3rdparty/Megatron-Bridge/examples/models/qwen/qwen3_vl/prepare_mantis_energon.py \
        --source-dir /path/to/Mantis-Instruct-LLaVA \
        --output-dir /path/to/Mantis-Instruct-LLaVA/wds \
        --max-samples-per-tar 10000 \
        --validation-fraction 0.01
    ```

3. 在标准的 Energon 启动命令里设置 `dataset.path=/path/to/Mantis-Instruct-LLaVA/wds`。不需要交互式选择类,也不需要手工替换 YAML。

## 3、模型准备

```shell
pip install modelscope
cd /workspace

# 以 2B 为例
modelscope download --model Qwen/Qwen3.5-2B --local_dir ./Qwen3.5-2B
```

或者[点击链接](https://modelscope.cn/models)搜索 `Qwen/Qwen3.5-<尺寸>`，
手动下载模型文件后上传至 `/workspace` 路径下。

VL 的 processor 和 tokenizer 也从 `LOCAL_HF_PATH` 读取，所以必须是完整的
HuggingFace 目录（含 `config.json`、`tokenizer.json`、`preprocessor_config.json`），
不能只放权重。

## 4、启动任务

### 4.1 参数填写

（1）配置 hostfile 文件

进入 `/workspace/Megatron-bridge-das/examples/qwen`，填写 `hostfile`，示例如下
（多个节点填写多行）：

```shell
127.0.0.1
127.0.0.2
```

可以填写节点 ip 或者节点主机名。单机运行时不需要 hostfile。

（2）设置环境变量

```shell
PORT=""   # mpirun 跨节点 ssh 用的端口，需与容器内 sshd 端口一致（默认 25951）
export LOCAL_HF_PATH=/path/to/Qwen3.5-VL-2B
```

### 4.2 pretrain

```shell
docker exec -it "your container name" bash
cd /workspace/Megatron-bridge-das/examples/qwen

export LOCAL_HF_PATH=/workspace/Qwen3.5-2B
export PORT=25951

./train_qwen35_vl_pretrain.sh <RECIPE> <MODE> [HOSTFILE] [NODES] [KEY=VALUE ...]
```

- `RECIPE`：脚本里 `SUPPORTED_RECIPES` 列出的名字，见上面的表格
- `MODE`：`dryrun` 或 `train`
- `HOSTFILE`：hostfile 路径，单机不需要
- `NODES`：使用的节点数，从 hostfile 里取前 N 行
- `KEY=VALUE`：余下参数会最后转发给 runner，可覆盖脚本里设置的项

launcher 会从 recipe 名字里读回尺寸和数据后端，所以拓扑、序列长度和数据集
接线始终与 recipe 一致。

单机 mock 冒烟测试：

```shell
export LOCAL_HF_PATH=/workspace/Qwen3.5-2B
MOCK_NUM_BASE_EXAMPLES=100 ./train_qwen35_vl_pretrain.sh \
  qwen35_vl_2b_pretrain_8gpu_bw1100_bf16_mock_config train
```

单机 Energon 训练：

```shell
export LOCAL_HF_PATH=/workspace/Qwen3.5-4B
export DATA_PATH=/data/energon/qwen35-4b
./train_qwen35_vl_pretrain.sh \
  qwen35_vl_4b_pretrain_8gpu_bw1100_bf16_config train
```

多机训练（35B-A3B，2 节点）：

```shell
export LOCAL_HF_PATH=/workspace/Qwen3.5-35B-A3B
export PORT=25951
./train_qwen35_vl_pretrain.sh \
  qwen35_vl_35b_a3b_pretrain_16gpu_bw1100_bf16_mock_config train ./hostfile 2
```

### 4.3 dryrun

只解析并校验配置，不占卡、不需要数据集：

```shell
export LOCAL_HF_PATH=/workspace/Qwen3.5-2B
./train_qwen35_vl_pretrain.sh \
  qwen35_vl_2b_pretrain_8gpu_bw1100_bf16_mock_config dryrun
```

解析出的完整配置会写到
`examples/qwen/<RECIPE>_tp<>_pp<>_cp<>/<RECIPE>.yaml`。

### 4.4 可覆盖的环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LOCAL_HF_PATH` | `/path/to/Qwen3.5-VL-2B` | 模型目录，必填 |
| `PORT` | `25951` | mpirun 跨节点 ssh 端口，需与容器 sshd 端口一致 |
| `BRIDGE_ENV_SCRIPT` | `env/env.sh` | 集群环境脚本；本仓库内可用 `skills/common/env.sh` |
| `DATA_PATH` | 空 | Energon 数据集路径；Energon recipe 必填 |
| `CHECKPOINT_PATH` | 自动生成 | 日志和 tensorboard 的输出目录 |
| `TRAIN_ITERS` | `50` | 训练步数 |
| `GLOBAL_BATCH_SIZE` | `32` | global batch size |
| `MICRO_BATCH_SIZE` | `1` | micro batch size |
| `SEQ_LENGTH` | 9B 为 `2048`，其余 `4096` | 训练序列长度 |
| `TENSOR_PARALLEL` | 按尺寸 | 覆盖 TP |
| `PIPELINE_PARALLEL` | 按尺寸 | 覆盖 PP |
| `CONTEXT_PARALLEL` | `1` | 覆盖 CP |
| `EXPERT_PARALLEL` | 按尺寸 | 覆盖 EP（MoE） |
| `EXPERT_TENSOR_PARALLEL` | `1` | 覆盖 ETP |
| `SEQUENCE_PARALLEL` | 按尺寸 | 覆盖 SP |
| `GPUS` | `8` | 每节点卡数 |
| `PROFILING` | 空 | 设为 `torch` 时采集 PyTorch profile |

Mock 数据相关：`MOCK_PROMPT`、`MOCK_NUM_IMAGES`、`MOCK_IMAGE_WIDTH`、
`MOCK_IMAGE_HEIGHT`、`MOCK_RANDOM_SEED`、`MOCK_NUM_BASE_EXAMPLES`。

`MASTER_ADDR` / `MASTER_PORT` 默认 `127.0.0.1` / `29501`，可通过环境变量覆盖。

Tail 的 `KEY=VALUE` 会最后转发，优先级高于脚本内的配置块：

```shell
./train_qwen35_vl_pretrain.sh \
  qwen35_vl_2b_pretrain_8gpu_bw1100_bf16_mock_config dryrun \
  model.tensor_model_parallel_size=2 train.train_iters=10
```

部分参数还有按尺寸的兼容写法，recipe 会作为兜底读取：
`QWEN35_4B_DATA_PATH`、`QWEN35_9B_HF_PATH`、`QWEN35_9B_CHECKPOINT_PATH` 等。
同名通用变量优先级更高。

micro batch size 请通过 `MICRO_BATCH_SIZE` 设置，不要用 runner 的 `-mb`。
`ConfigContainer` 要求 `EnergonDatasetConfig.micro_batch_size` 与
`train.micro_batch_size` 一致，recipe 用同一个变量驱动两者；`-mb` 只会改动
其中一边，配置会被拒绝。

### 4.5 直接调用 run_recipe.py

launcher 只是一层包装，任何上游参数都可以直接传：

```shell
source env/env.sh

python run_recipe.py \
  --recipe qwen35_vl_2b_pretrain_8gpu_bw1100_bf16_config \
  --mode pretrain --dry-run \
  dataset.path=/data/energon/qwen35-2b \
  checkpoint.save=/data/checkpoints/qwen35-2b
```

`source env/env.sh` 是必需的：它把 HCU recipe 索引复制进上游目录，并把
`hcu_megatron_bridge/src` 加进 `PYTHONPATH`，否则 recipe 名字解析不到。
路径请传绝对路径——runner 会先 chdir 到 `3rdparty/Megatron-Bridge`。

### 4.6 笔记

- **检查点**：launcher 不读写 checkpoint（`checkpoint.load` / `checkpoint.save`
  都是 `null`），这是性能测试，不落盘、也不会误加载历史权重。
- **`LOCAL_HF_PATH` 的作用**：HCU patch 层会包装
  `AutoBridge.from_hf_pretrained` 并重定向到 `LOCAL_HF_PATH`，所以权重一定从
  本地目录加载。但这个包装只覆盖权重；VL 还需要 processor 和 tokenizer，因此
  recipe 自己也读 `LOCAL_HF_PATH` 来填 `hf_processor_path` 和
  `tokenizer.tokenizer_model`，否则 processor 会回落到 hub 名字、而权重来自
  本地，两者不一致。
- **MoE 的 dispatcher**：固定 `alltoall`（Megatron-Core 内置 NCCL 实现）。
  不要改成 `flex` + `hybridep`——该路径依赖 HybridEP 包，BW 集群镜像没有，
  一改就会在构造 `TransformerLayer` 时报
  `ImportError: HybridEP is not installed`。
- **多机**：hostfile 第一行是 rendezvous master。launcher 会生成一份
  `slots=1` 的 `hostfile_slots`，每节点一个 mpirun 进程，各自拉起本节点的
  torchrun，`--node_rank` 由远端从 `OMPI_COMM_WORLD_RANK` 解析。远端 shell 会
  重新 source `env.sh`，recipe 从环境读模型、数据、batch 和 mock 参数，所以
  这些变量都用 `-x` 透传。
- **batch 约束**：数据并行随节点数增长，`GLOBAL_BATCH_SIZE` 必须能被
  `MICRO_BATCH_SIZE × DP` 整除。launcher 在启动前检查并报出具体数字，而不是
  让训练在 setup 阶段失败。

## 5、测试

```bash
pytest tests/unit_tests/hcu/recipes/
```

这些测试会 stub 掉 `AutoBridge`，只检查 recipe 自身的决策，不需要下载权重。
