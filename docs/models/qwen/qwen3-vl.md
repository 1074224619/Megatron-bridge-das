<!-- Copyright (c) 2026 Hygon Information Technology Co., Ltd. -->

# Qwen3-VL 系列

Qwen3-VL 多模态模型在 HCU 上的启动脚本，位于 `examples/qwen`：

- `train_qwen3_vl_pretrain.sh`：预训练（mock 合成数据）
- `train_qwen3_vl_sft.sh`：全参 SFT

## Supported

recipe 名中的 `bw1100` / `bw1000` 表示计算卡平台，`8gpu` / `16gpu`
表示该切分至少需要的卡数；增加节点只扩大 DP。

### Pretrain（mock 数据）

BW1100（144 GB 卡）：

| 模型 | recipe | 默认 GPU 数 | 默认并行 |
|---|---|---|---|
| Qwen3-VL-8B | `qwen3_vl_8b_pretrain_8gpu_bw1100_bf16_mock_config` | 8 | TP=1, PP=1 |
| Qwen3-VL-30B-A3B | `qwen3_vl_30b_a3b_pretrain_16gpu_bw1100_bf16_mock_config` | 16 | TP=1, PP=2, EP=8, ETP=1, SP |
| Qwen3-VL-32B | `qwen3_vl_32b_pretrain_8gpu_bw1100_bf16_mock_config` | 8 | TP=2, PP=4, SP |

BW1000：

| 模型 | recipe | 默认 GPU 数 | 默认并行 |
|---|---|---|---|
| Qwen3-VL-8B | `qwen3_vl_8b_pretrain_8gpu_bw1000_bf16_mock_config` | 8 | TP=2, PP=1 |
| Qwen3-VL-30B-A3B | `qwen3_vl_30b_a3b_pretrain_16gpu_bw1000_bf16_mock_config` | 16 | TP=2, PP=2, EP=8, ETP=1, SP |
| Qwen3-VL-32B | `qwen3_vl_32b_pretrain_16gpu_bw1000_bf16_mock_config` | 16 | TP=4, PP=4, SP |

所有 recipe 都是 mock 版本，数据集由 `MockVLMSFTDatasetConfig` 生成确定性的合成图文样本，不读取真实数据；用于分布式冒烟和性能测试。

如果要用真实数据做预训练，需要自行基于 Energon 数据集编写 recipe（可参考`3rdparty/Megatron-Bridge/examples/models/qwen/qwen3_vl`）。

### SFT

BW1100（144 GB 卡）：

| 模型 | recipe | 默认 GPU 数 | 默认并行 |
|---|---|---|---|
| Qwen3-VL-8B | `qwen3_vl_8b_sft_8gpu_bw1100_bf16_config` | 8 | TP=1, PP=1 |
| Qwen3-VL-30B-A3B | `qwen3_vl_30b_a3b_sft_16gpu_bw1100_bf16_config` | 16 | TP=1, PP=2, EP=8, ETP=1 |
| Qwen3-VL-32B | `qwen3_vl_32b_sft_8gpu_bw1100_bf16_config` | 8 | TP=2, PP=4, SP |

BW1000：

| 模型 | recipe | 默认 GPU 数 | 默认并行 |
|---|---|---|---|
| Qwen3-VL-8B | `qwen3_vl_8b_sft_8gpu_bw1000_bf16_config` | 8 | TP=2, PP=1 |
| Qwen3-VL-30B-A3B | `qwen3_vl_30b_a3b_sft_16gpu_bw1000_bf16_config` | 16 | TP=2, PP=2, EP=8, ETP=1 |
| Qwen3-VL-32B | `qwen3_vl_32b_sft_16gpu_bw1000_bf16_config` | 16 | TP=4, PP=4, SP |

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

### 2.1 Pretrain：mock 数据

mock recipe 自己生成合成图文样本，无需数据集准备，可用的环境变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MOCK_NUM_BASE_EXAMPLES` | `1000` | 合成的图文样本数 |
| `MOCK_PROMPT` | `Describe this image.` | 文本提示 |
| `MOCK_NUM_IMAGES` | `1` | 每条样本的图片数 |
| `MOCK_IMAGE_SIZE` | `[256,256]` | 图片分辨率 |
| `MOCK_RANDOM_SEED` | `0` | 随机种子（保证可复现） |
| `SEQ_LENGTH` | `4096` | 序列长度 |

### 2.2 SFT 数据

`train_qwen3_vl_sft.sh` 通过 `--dataset` 选择公开数据集，默认 `cord-v2`：

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

# 以 8B 为例，其余尺寸替换模型名即可
modelscope download --model Qwen/Qwen3-VL-8B-Instruct --local_dir ./Qwen3-VL-8B-Instruct
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

可以填节点 IP 或主机名。单机运行时不需要 hostfile。

（2）设置环境变量

```shell
PORT=25951   # mpirun 跨节点 ssh 用的端口，需与容器内 sshd 端口一致
export LOCAL_HF_PATH=/path/to/Qwen3-VL-8B-Instruct
```

### 4.2 pretrain

```shell
docker exec -it "your container name" bash
cd /workspace/Megatron-bridge-das/examples/qwen

export LOCAL_HF_PATH=/workspace/Qwen3-VL-8B-Instruct
export PORT=25951

./train_qwen3_vl_pretrain.sh <RECIPE> <MODE> [HOSTFILE] [NODES] [KEY=VALUE ...]
```

- `RECIPE`：脚本里 `SUPPORTED_RECIPES` 列出的名字，见上面的表格
- `MODE`：`dryrun` 或 `train`
- `HOSTFILE`：hostfile 路径，单机不需要
- `NODES`：使用的节点数，从 hostfile 里取前 N 行
- `KEY=VALUE`：余下参数最后转发给 runner，可覆盖脚本里设置的项

单机 8B mock 冒烟测试：

```shell
export LOCAL_HF_PATH=/workspace/Qwen3-VL-8B-Instruct
MOCK_NUM_BASE_EXAMPLES=100 ./train_qwen3_vl_pretrain.sh \
  qwen3_vl_8b_pretrain_8gpu_bw1100_bf16_mock_config train
```

### 4.3 SFT

```shell
export LOCAL_HF_PATH=/workspace/Qwen3-VL-8B-Instruct
export PRETRAINED_CHECKPOINT=/workspace/Qwen3-VL-8B-Instruct
export PORT=25951

./train_qwen3_vl_sft.sh <RECIPE> <MODE> [HOSTFILE] [NODES] [KEY=VALUE ...]
```

`MODE` 为 `dryrun` 或 `sft`。

8B 单机 cord-v1：

```shell
cd Megatron-bridge-das/examples/qwen/
LOCAL_HF_PATH=/path/to/Qwen3-VL-8B-Instruct DATASET=local-vlm bash train_qwen3_vl_sft.sh qwen3_vl_8b_sft_8gpu_bw1100_bf16_config sft hostfile 1 \
  dataset.source.load_kwargs.data_files.train=./cord-v1/sft/train.jsonl \
  dataset.validation_source.load_kwargs.data_files.validation=./cord-v1/sft/validation.jsonl \
  dataset.do_validation=true \
  validation.eval_iters=5 \
  dataset.test_source.load_kwargs.data_files.test=./cord-v1/sft/test.jsonl \
  dataset.do_test=true
```

### 4.4 dryrun

只解析并校验配置，不占卡、不加载数据：

```shell
export LOCAL_HF_PATH=/workspace/Qwen3-VL-8B-Instruct
./train_qwen3_vl_pretrain.sh qwen3_vl_8b_pretrain_8gpu_bw1100_bf16_mock_config dryrun

export PRETRAINED_CHECKPOINT=/workspace/Qwen3-VL-8B-Instruct
./train_qwen3_vl_sft.sh qwen3_vl_8b_sft_8gpu_bw1100_bf16_config dryrun
```

解析出的完整配置会写到
`examples/qwen/<RECIPE>/<RECIPE>.yaml`。
