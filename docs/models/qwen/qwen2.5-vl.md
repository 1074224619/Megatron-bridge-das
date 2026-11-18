<!-- Copyright (c) 2026 Hygon Information Technology Co., Ltd. -->

# Qwen2.5-VL 系列

Qwen2.5-VL 多模态模型的 HCU 启动脚本，位于 `examples/qwen`：

- `train_qwen25_vl_sft.sh`：全参 SFT

## Supported

recipe 名中的 `bw1100` / `bw1000` 表示计算卡平台，`8gpu` / `16gpu` / `32gpu`
表示该切分至少需要的卡数；增加节点只扩大 DP。

BW1100（144 GB 卡）：

| 模型 | recipe | 默认 GPU 数 | 默认并行 |
|---|---|---|---|
| Qwen2.5-VL-3B | `qwen25_vl_3b_sft_8gpu_bw1100_bf16_config` | 8 | TP=1, PP=1 |
| Qwen2.5-VL-7B | `qwen25_vl_7b_sft_8gpu_bw1100_bf16_config` | 8 | TP=1, PP=1 |
| Qwen2.5-VL-32B | `qwen25_vl_32b_sft_8gpu_bw1100_bf16_config` | 8 | TP=2, PP=4 |
| Qwen2.5-VL-72B | `qwen25_vl_72b_sft_16gpu_bw1100_bf16_config` | 16 | TP=4, PP=4 |

BW1000：

| 模型 | recipe | 默认 GPU 数 | 默认并行 |
|---|---|---|---|
| Qwen2.5-VL-3B | `qwen25_vl_3b_sft_8gpu_bw1000_bf16_config` | 8 | TP=2, PP=1 |
| Qwen2.5-VL-7B | `qwen25_vl_7b_sft_8gpu_bw1000_bf16_config` | 8 | TP=2, PP=1 |
| Qwen2.5-VL-32B | `qwen25_vl_32b_sft_16gpu_bw1000_bf16_config` | 16 | TP=4, PP=4 |
| Qwen2.5-VL-72B | `qwen25_vl_72b_sft_32gpu_bw1000_bf16_config` | 32 | TP=8, PP=4 |

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

启动脚本会先 `source /opt/dtk/env.sh`，再 `source env/env.sh`，不需要手工复制 recipe 注册文件。

## 2、数据准备

`train_qwen25_vl_sft.sh` 通过 `--dataset` 选择公开数据集，默认 `cord-v2`：

| preset | 说明 |
|---|---|
| `cord-v2` | 票据 OCR 图文数据（默认） |
| `medpix` | 医学影像 VQA |
| `raven` | 视觉推理 |
| `llava-video-178k` | 视频指令数据 |
| `rdr` | 文档推理 |
| `local-vlm` | 本地 JSON/JSONL 图文数据 |

这些 preset 是 Megatron-Bridge 的 `DirectHFSFTDatasetConfig`，会从
HuggingFace 拉取（或使用本地 HF cache），需要能访问数据源或预先下载。

`local-vlm` 需要给出本地数据文件，作为尾部的 `KEY=VALUE` 覆盖传入：

以cord-v1为例:
1. 下载数据集: `modelscope download --dataset zhangs669/cord-v1_20260331065252  --local_dir ./cord-v1`
2. 转成local-vlm格式: `python Megatron-bridge-das/data_preparation/convert_cordv1_to_vlmsft_jsonl.py --src-dir ./cord-v1/data --dst-dir ./cord-v1/sft`
3. 配置sft的train/val/test参数, 开始训练(详细命令参考下文sft命令):

## 3、模型准备

```shell
pip install modelscope
cd /workspace

# 以 3B 为例，其余尺寸替换模型名即可
modelscope download --model Qwen/Qwen2.5-VL-3B-Instruct --local_dir ./Qwen2.5-VL-3B-Instruct
```

## 4、启动任务

### 4.1 参数填写

（1）配置 hostfile 文件

进入 `/workspace/Megatron-bridge-das/examples/qwen`，填写 `hostfile`，示例如下
（多个节点填写多行）：

```shell
127.0.0.1
127.0.0.2
```

可以填节点 IP 或主机名。单机运行时不需要 hostfile；`16gpu` / `32gpu`
的 recipe 需要多机（8 卡/节点）。

（2）设置环境变量

```shell
PORT=25951   # mpirun 跨节点 ssh 用的端口，需与容器内 sshd 端口一致
export LOCAL_HF_PATH=/path/to/Qwen2.5-VL-3B-Instruct
```

### 4.2 SFT

```shell
docker exec -it "your container name" bash
cd /workspace/Megatron-bridge-das/examples/qwen

export LOCAL_HF_PATH=/workspace/Qwen2.5-VL-3B-Instruct
export PRETRAINED_CHECKPOINT=/workspace/Qwen2.5-VL-3B-Instruct
export PORT=25951

./train_qwen25_vl_sft.sh <RECIPE> <MODE> [HOSTFILE] [NODES] [KEY=VALUE ...]
```

- `RECIPE`：脚本里 `SUPPORTED_RECIPES` 列出的名字，见上面的表格
- `MODE`：`dryrun` 或 `sft`
- `HOSTFILE`：hostfile 路径，单机不需要
- `NODES`：使用的节点数，从 hostfile 里取前 N 行
- `KEY=VALUE`：余下参数最后转发给 runner，可覆盖脚本里设置的项

3B 单机 cord-v2：

```shell
cd Megatron-bridge-das/examples/qwen/
LOCAL_HF_PATH=/path/to/Qwen2.5-VL-3B-Instruct DATASET=local-vlm bash train_qwen25_vl_sft.sh qwen25_vl_3b_sft_8gpu_bw1100_bf16_config sft hostfile 1 \
  dataset.source.load_kwargs.data_files.train=./cord-v1/sft/train.jsonl \
  dataset.validation_source.load_kwargs.data_files.validation=./cord-v1/sft/validation.jsonl \
  dataset.do_validation=true \
  validation.eval_iters=5 \
  dataset.test_source.load_kwargs.data_files.test=./cord-v1/sft/test.jsonl \
  dataset.do_test=true
```

### 4.3 dryrun

只解析并校验配置，不占卡、不加载数据：

```shell
export LOCAL_HF_PATH=/workspace/Qwen2.5-VL-3B-Instruct
export PRETRAINED_CHECKPOINT=/workspace/Qwen2.5-VL-3B-Instruct
./train_qwen25_vl_sft.sh qwen25_vl_3b_sft_8gpu_bw1100_bf16_config dryrun
```

解析出的完整配置会写到
`examples/qwen/<RECIPE>/<RECIPE>.yaml`。
