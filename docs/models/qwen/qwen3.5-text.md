# Qwen3.5 Text 系列

Qwen3.5 纯文本预训练（pretrain）性能测试，覆盖 dense 和 MoE 两类模型，
支持 BW1000 / BW1100 两个平台。所有 recipe 走 mock（随机初始化 + 合成数据）
只测性能，不加载预训练权重。

## Supported

BW1100 为 144G 显存卡；`节点数` 是 recipe 名字里标称的部署规模。

| 模型 | 平台 | Recipe | 节点 | 卡数 | TP/PP/EP |
|---|---|---|---:|---:|---|
| 0.8B | BW1000 | `qwen35_text_0_8b_pretrain_8gpu_bw1000_bf16_config` | 1 | 8 | 1/1/1 |
| 0.8B | BW1100 | `qwen35_text_0_8b_pretrain_8gpu_bw1100_bf16_config` | 1 | 8 | 1/1/1 |
| 2B | BW1000 | `qwen35_text_2b_pretrain_8gpu_bw1000_bf16_config` | 1 | 8 | 1/1/1 |
| 2B | BW1100 | `qwen35_text_2b_pretrain_8gpu_bw1100_bf16_config` | 1 | 8 | 1/1/1 |
| 4B | BW1000 | `qwen35_text_4b_pretrain_8gpu_bw1000_bf16_config` | 1 | 8 | 2/1/1 |
| 4B | BW1100 | `qwen35_text_4b_pretrain_8gpu_bw1100_bf16_config` | 1 | 8 | 2/1/1 |
| 9B | BW1000 | `qwen35_text_9b_pretrain_8gpu_bw1000_bf16_config` | 1 | 8 | 2/1/1 |
| 9B | BW1100 | `qwen35_text_9b_pretrain_8gpu_bw1100_bf16_config` | 1 | 8 | 2/1/1 |
| 27B | BW1000 | `qwen35_text_27b_pretrain_8gpu_bw1000_bf16_config` | 1 | 8 | 4/2/1 |
| 27B | BW1100 | `qwen35_text_27b_pretrain_8gpu_bw1100_bf16_config` | 1 | 8 | 2/2/1 |
| 35B-A3B (MoE) | BW1000 | `qwen35_text_35b_a3b_pretrain_32gpu_bw1000_bf16_config` | 4 | 32 | 1/4/8 |
| 35B-A3B (MoE) | BW1100 | `qwen35_text_35b_a3b_pretrain_16gpu_bw1100_bf16_config` | 2 | 16 | 1/2/8 |
| 122B-A10B (MoE) | BW1000 | `qwen35_text_122b_a10b_pretrain_32gpu_bw1000_bf16_config` | 4 | 32 | 4/2/4 |
| 122B-A10B (MoE) | BW1100 | `qwen35_text_122b_a10b_pretrain_32gpu_bw1100_bf16_config` | 4 | 32 | 4/2/4 |
| 397B-A17B (MoE) | BW1000 | `qwen35_text_397b_a17b_pretrain_64gpu_bw1000_bf16_config` | 8 | 64 | 4/4/8 |
| 397B-A17B (MoE) | BW1100 | `qwen35_text_397b_a17b_pretrain_64gpu_bw1100_bf16_config` | 8 | 64 | 4/4/8 |

precision 统一 BF16（Transformer Engine），micro batch size 1，
global batch size 64，训练 50 步，Adam，cosine 学习率。

并行取值说明：

- dense 模型 `EP` 必须是 1，否则 Megatron-Core 直接拒绝
  （`num_moe_experts must be non None to use expert-parallel`）。
- MoE 模型的专家靠 `EP` 摊到多卡上。
- BW1100 的 27B（TP2/PP2）和 35B-A3B（TP1/PP2/EP8）比 BW1000 更省卡，
  因为 144G 单卡能装下更多层。
- TP 必须整除 attention head 数，PP 必须整除层数。

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

pretrain 使用 mock 的数据集，无需处理。

## 3、模型准备

每个尺寸单独下载到 `/workspace` 下，`LOCAL_HF_PATH` 指向对应目录即可：

```shell
pip install modelscope
cd /workspace

# 按需选择尺寸，下面以 9B 为例
modelscope download --model Qwen/Qwen3.5-9B --local_dir ./Qwen3.5-9B
```

或者[点击链接](https://modelscope.cn/models)搜索 `Qwen/Qwen3.5-<尺寸>`，
手动下载模型文件后上传至 `/workspace` 路径下。

注意目录名大小写：35B-A3B 的官方目录名是 `Qwen3.5-35B-A3B`，其余尺寸为
小写 `qwen3.5-<尺寸>`（如 `qwen3.5-9b`）。

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
PORT=""   # mpirun 跨节点 ssh 用的端口，需与容器内 sshd 端口一致
          # 在各节点的容器内执行 /usr/sbin/sshd -p <port> 启动，例如 -p 25951
export LOCAL_HF_PATH=/path/to/Qwen3.5-9B
```

### 4.2 pretrain

```shell
docker exec -it "your container name" bash
cd /workspace/Megatron-bridge-das/examples/qwen

export LOCAL_HF_PATH=/workspace/qwen3.5-9b
export PORT=25951

./train_qwen35_text_pretrain.sh <RECIPE> <MODE> [HOSTFILE] [NODES]
```

- `RECIPE`：脚本里 `SUPPORTED_RECIPES` 列出的名字，见上面的表格
- `MODE`：`dryrun` 或 `train`
- `HOSTFILE`：hostfile 路径，单机不需要
- `NODES`：使用的节点数，从 hostfile 里取前 N 行

单机 8 卡示例：

```shell
export LOCAL_HF_PATH=/workspace/qwen3.5-9b
./train_qwen35_text_pretrain.sh \
  qwen35_text_9b_pretrain_8gpu_bw1100_bf16_config train
```

多机示例（2 节点跑 35B-A3B）：

```shell
export LOCAL_HF_PATH=/workspace/Qwen3.5-35B-A3B
export PORT=25951
./train_qwen35_text_pretrain.sh \
  qwen35_text_35b_a3b_pretrain_16gpu_bw1100_bf16_config train ./hostfile 2
```

### 4.3 dryrun

只解析并校验配置，不占卡、不启动训练，用于确认路径和拓扑：

```shell
export LOCAL_HF_PATH=/workspace/qwen3.5-9b
./train_qwen35_text_pretrain.sh \
  qwen35_text_9b_pretrain_8gpu_bw1100_bf16_config dryrun
```

解析出的完整配置会写到
`examples/qwen/<RECIPE>_tp<>_pp<>_ep<>_cp<>/<RECIPE>.yaml`。

### 4.4 可覆盖的环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LOCAL_HF_PATH` | `/path/to/Qwen3.5-35B-A3B` | 模型目录，必填 |
| `PORT` | `12345` | mpirun 跨节点 ssh 端口，多机必填为容器 sshd 端口 |
| `BRIDGE_ENV_SCRIPT` | `env/env.sh` | 集群环境脚本；本仓库内可用 `skills/common/env.sh` |
| `TENSOR_PARALLEL` | 按 recipe | 覆盖 TP |
| `PIPELINE_PARALLEL` | 按 recipe | 覆盖 PP |
| `EXPERT_PARALLEL` | 按 recipe | 覆盖 EP（仅 MoE） |
| `EXPERT_TENSOR_PARALLEL` | `1` | 覆盖 ETP |
| `CONTEXT_PARALLEL` | `1` | 覆盖 CP |
| `GPUS` | `8` | 每节点卡数 |
| `PROFILING` | 空 | 设为 `torch` 时采集 PyTorch profile |

`MASTER_ADDR` / `MASTER_PORT` 在脚本内是固定值（`127.0.0.1` / `29508`），
不读环境变量。

`TRAIN_ITERS` / `GLOBAL_BATCH_SIZE` / `MICRO_BATCH_SIZE` / `SEQ_LENGTH`
目前是脚本内的固定赋值（50 / 64 / 1 / 4096），不能通过环境变量改；
需要改时直接编辑脚本，或绕过 launcher 直接调 `run_recipe.py`。

### 4.5 直接调用 run_recipe.py

launcher 只是入口脚本的一层包装。任何上游参数都可以直接传：

```shell
source env/env.sh

python run_recipe.py \
  --recipe qwen35_text_9b_pretrain_8gpu_bw1100_bf16_config \
  --mode pretrain --dry-run \
  train.train_iters=20 train.global_batch_size=128
```

`source env/env.sh` 是必需的：它把 HCU recipe 索引复制进上游目录，
并把 `hcu_megatron_bridge/src` 加进 `PYTHONPATH`，否则 recipe 名字解析不到。
路径请传绝对路径——runner 会先 chdir 到 `3rdparty/Megatron-Bridge`。

### 4.6 笔记

- MoE 模型的 token dispatcher 固定为 `alltoall`（Megatron-Core 内置的 NCCL
  实现）。不要改成 `flex` + `hybridep`：该路径依赖 HybridEP 包，BW 集群镜像
  里没有，一改就会在构造 `TransformerLayer` 时
  `ImportError: HybridEP is not installed`。
- `checkpoint.load` / `checkpoint.save` 都是 `null`，不读也不写权重，
  避免占用磁盘和误加载历史 checkpoint。
- 多机跑时 hostfile 第一行是 rendezvous master。launcher 会生成一份
  `slots=1` 的 `hostfile_slots`，每节点一个 mpirun 进程，各自拉起本节点的
  torchrun，`--node_rank` 由远端从 `OMPI_COMM_WORLD_RANK` 解析。
