#!/bin/bash
# ==============================================================================
# Qwen3-VL 预训练冒烟测试（mpirun + torchrun）
#
# 并行切分默认值与 recipe 一致，由下面的 SIZE/PLATFORM 表给出；可用环境变量
# TENSOR_PARALLEL / PIPELINE_PARALLEL / EXPERT_PARALLEL / EXPERT_TENSOR_PARALLEL /
# CONTEXT_PARALLEL 覆盖。RECIPE 名中的 bw1100 / bw1000 即平台，8gpu / 16gpu 是
# 最小部署卡数。SP 仍取 recipe 的值，可用尾部 model.sequence_parallel=... 覆盖。
# ==============================================================================

set -euo pipefail
CURRENT_DIR=$( cd $( dirname $0 ) && pwd )
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))

# 默认环境
# PORT 是 mpirun 跨节点 ssh 使用的端口，必须与容器内 sshd 端口一致
# （site.json 的 container.ssh_port），不是默认的 22。
PORT="${PORT:-25951}"
export LOCAL_HF_PATH=${LOCAL_HF_PATH:-/path/to/Qwen3-VL-8B-Instruct}
BRIDGE_ENV_SCRIPT="${BRIDGE_ENV_SCRIPT:-${MEGATRON_PATH}/env/env.sh}" # 集群环境脚本（NCCL/IB 变量、PYTHONPATH、HCU recipe 注册）；可用 BRIDGE_ENV_SCRIPT=<path> 覆盖

# ===================================== 启动模式 =====================================
RECIPE=${1:-}
MODE=${2:-}
HOSTFILE=${3:-}
NODES=${4:-}
# 前四个位置参数固定为 RECIPE/MODE/HOSTFILE/NODES，其余参数全部作为 KEY=VALUE 覆盖。
EXTRA_ARGS=("${@:5}")

PROFILING="" # 设为 "torch" 时在 rank 0 和 4 上采集 PyTorch profile

RED="\033[31m"
YELLOW="\033[33m"
RESET="\033[0m"

SUPPORTED_RECIPES=(
    qwen3_vl_8b_pretrain_8gpu_bw1100_bf16_mock_config
    qwen3_vl_8b_pretrain_8gpu_bw1000_bf16_mock_config
    qwen3_vl_30b_a3b_pretrain_16gpu_bw1100_bf16_mock_config
    qwen3_vl_30b_a3b_pretrain_16gpu_bw1000_bf16_mock_config
    qwen3_vl_32b_pretrain_8gpu_bw1100_bf16_mock_config
    qwen3_vl_32b_pretrain_16gpu_bw1000_bf16_mock_config
)

# 校验 RECIPE 参数
if [[ -z ${RECIPE} ]]; then
    echo -e "${RED}ERROR: missing recipe argument.${RESET}"
    echo -e "${YELLOW}Supported recipes:${RESET}"
    printf "  - %s\n" "${SUPPORTED_RECIPES[@]}"
    echo ""
    echo "Usage: $0 <RECIPE> <MODE> [HOSTFILE] [NODES] [KEY=VALUE ...]"
    echo "  MODE: dryrun | train"
    echo "  Parallel overrides (environment): TENSOR_PARALLEL, PIPELINE_PARALLEL,"
    echo "    EXPERT_PARALLEL, EXPERT_TENSOR_PARALLEL, CONTEXT_PARALLEL"
    exit 1
fi

# 从 recipe 名中解析模型尺寸/平台：
#   qwen3_vl_<size-slug>_pretrain_<N>gpu_bw<plat>_bf16_mock_config
# 在第一个 "_pretrain" 处截断，保证 30b_a3b 这类 MoE 名字完整。
SIZE=${RECIPE#qwen3_vl_}
SIZE=${SIZE%%_pretrain*}
SIZE_UPPER=$(echo "${SIZE}" | tr '[:lower:]' '[:upper:]')
PLATFORM=$(echo "${RECIPE}" | sed -n 's/^.*_\(bw[0-9]\{4\}\)_bf16.*$/\1/p')
if [[ -z ${SIZE} || -z ${PLATFORM} ]]; then
    echo -e "${RED}ERROR: cannot parse size/platform from recipe '${RECIPE}'.${RESET}"
    exit 1
fi

# ===================================== 多机拓扑 =====================================
GPUS=${GPUS:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29501}

# NNODES 依次取 hostfile 行数、NODES 参数、1。
# TP/PP 按尺寸固定，增加节点只会扩大数据并行。
if [[ -n ${HOSTFILE} ]]; then
    [[ -f ${HOSTFILE} ]] || { echo -e "${RED}ERROR: ${HOSTFILE} not found.${RESET}"; exit 1; }
    HOSTFILE=$(cd $(dirname ${HOSTFILE}) && pwd)/$(basename ${HOSTFILE})
    AVAILABLE_NODES=$(grep -cve '^[[:space:]]*$' "${HOSTFILE}" || true)
    NNODES=${NODES:-${AVAILABLE_NODES}}
    if (( NNODES < 1 || NNODES > AVAILABLE_NODES )); then
        echo -e "${RED}ERROR: node count ${NNODES} must be between 1 and ${AVAILABLE_NODES} (hosts in ${HOSTFILE}).${RESET}"
        exit 1
    fi
    # 第一台机器作为 rendezvous master；mpirun 把 local rank 0 放在这里。
    MASTER_ADDR=$(grep -ve '^[[:space:]]*$' "${HOSTFILE}" | sed -n 1p | awk '{print $1}')
else
    NNODES=${NODES:-1}
    if (( NNODES > 1 )); then
        echo -e "${RED}ERROR: multi-node run requires a hostfile.${RESET}"
        exit 1
    fi
fi
WORLD_GPUS=$((NNODES * GPUS))

# 每个节点起一个 mpirun 进程（slots=1），由它为本节点 GPU 启动 torchrun。
# 只有多机分支会用到这个文件；单机没有 hostfile，给 grep 传空文件会让脚本在 set -e 下退出。
HOSTFILE_SLOTS="${CURRENT_DIR}/hostfile_slots"
if [[ -n ${HOSTFILE} ]]; then
    grep -ve '^[[:space:]]*$' "${HOSTFILE}" | sed -n "1,${NNODES}p" | awk '{print $1" slots=1"}' > "${HOSTFILE_SLOTS}"
fi

# ===================================== 数据配置 =====================================
# 这些 recipe 使用 MockVLMSFTDatasetConfig 生成合成样本，下面的变量控制样本内容；
# dataset 和 model 的序列长度保持一致。
SEQ_LENGTH=${SEQ_LENGTH:-4096}
MOCK_PROMPT=${MOCK_PROMPT:-Describe_this_image.}
MOCK_NUM_IMAGES=${MOCK_NUM_IMAGES:-1}
MOCK_NUM_BASE_EXAMPLES=${MOCK_NUM_BASE_EXAMPLES:-1000}
MOCK_RANDOM_SEED=${MOCK_RANDOM_SEED:-0}
MOCK_IMAGE_SIZE=${MOCK_IMAGE_SIZE:-[256,256]}
# 导出给远端进程：HCU recipe 会在构造配置时读取这些环境变量。
export LOCAL_HF_PATH SEQ_LENGTH MOCK_PROMPT MOCK_NUM_IMAGES MOCK_NUM_BASE_EXAMPLES MOCK_RANDOM_SEED MOCK_IMAGE_SIZE

DATA_CONFIG=(
    "dataset.seq_length=${SEQ_LENGTH}"
    "dataset.hf_processor_path=${LOCAL_HF_PATH}"
    "dataset.prompt=${MOCK_PROMPT}"
    "dataset.num_images=${MOCK_NUM_IMAGES}"
    "dataset.num_base_examples=${MOCK_NUM_BASE_EXAMPLES}"
    "dataset.random_seed=${MOCK_RANDOM_SEED}"
    "dataset.image_size=${MOCK_IMAGE_SIZE}"
)

# ===================================== 训练配置 =====================================
TRAIN_ITERS=${TRAIN_ITERS:-50}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
export TRAIN_ITERS GLOBAL_BATCH_SIZE MICRO_BATCH_SIZE

TRAIN_CONFIG=(
    train.train_iters=${TRAIN_ITERS}
    train.global_batch_size=${GLOBAL_BATCH_SIZE}
    train.micro_batch_size=${MICRO_BATCH_SIZE}
    validation.eval_iters=0
    "tokenizer.tokenizer_model=${LOCAL_HF_PATH}"
)

# ===================================== 优化器配置 =====================================
OPTIMIZER_CONFIG=(
    # recipe 已自带 Adam + cosine（lr 3e-4）；需要时再在这里追加覆盖项。
)

# ===================================== 并行配置 =====================================
# 默认表与 recipe 保持一致（bw1100 用窄切分，bw1000 把 TP 翻倍）。
# 每个值都可以用环境变量单独覆盖：
#   TENSOR_PARALLEL / PIPELINE_PARALLEL / EXPERT_PARALLEL /
#   EXPERT_TENSOR_PARALLEL / CONTEXT_PARALLEL
case "${SIZE}" in
    8b)       TP_DEFAULT=1; PP_DEFAULT=1; EP_DEFAULT=1 ;;
    30b_a3b)  TP_DEFAULT=1; PP_DEFAULT=2; EP_DEFAULT=8 ;;
    32b)      TP_DEFAULT=2; PP_DEFAULT=4; EP_DEFAULT=1 ;;
    *)  echo -e "${RED}ERROR: unknown model size '${SIZE}' in ${RECIPE}.${RESET}"; exit 1 ;;
esac
if [[ ${PLATFORM} == bw1000 ]]; then
    case "${SIZE}" in
        8b|30b_a3b) TP_DEFAULT=2 ;;
        32b)        TP_DEFAULT=4 ;;
    esac
fi
TP="${TENSOR_PARALLEL:-${TP_DEFAULT}}"
PP="${PIPELINE_PARALLEL:-${PP_DEFAULT}}"
EP="${EXPERT_PARALLEL:-${EP_DEFAULT}}"
ETP="${EXPERT_TENSOR_PARALLEL:-1}"
CP="${CONTEXT_PARALLEL:-1}"

# world_size 必须同时容纳稠密和专家两套并行网格。
if (( WORLD_GPUS % (TP * PP * CP) != 0 )); then
    echo -e "${RED}ERROR: world size ${WORLD_GPUS} is not divisible by TP*PP*CP = ${TP}*${PP}*${CP}.${RESET}"
    exit 1
fi
if (( EP > 1 && WORLD_GPUS % (ETP * EP * PP) != 0 )); then
    echo -e "${RED}ERROR: world size ${WORLD_GPUS} is not divisible by ETP*EP*PP = ${ETP}*${EP}*${PP}.${RESET}"
    exit 1
fi

PARALLEL_CONFIG=(
    model.tensor_model_parallel_size=${TP}
    model.pipeline_model_parallel_size=${PP}
    model.expert_model_parallel_size=${EP}
    model.expert_tensor_parallel_size=${ETP}
    model.context_parallel_size=${CP}
)

# ===================================== 模型配置 =====================================
MODEL_CONFIG=(
    model.seq_length=${SEQ_LENGTH}
)

# ===================================== HCU 兼容配置 =====================================
# 兼容开关（cuda_graph / fusion / dispatcher）由 recipe 设置，
# 需要临时覆盖时放在尾部 KEY=VALUE 里。

# ===================================== Checkpoint 与日志配置 =====================================
# 性能测试不读也不写 checkpoint，避免在磁盘留下文件，也避免误加载旧 checkpoint。
SAVE_DIR=${CHECKPOINT_PATH:-${CURRENT_DIR}/${RECIPE}}
CONFIG_DUMP=${SAVE_DIR}/${RECIPE}.yaml

CKPT_AND_LOGGING_CONFIG=(
    checkpoint.load=null
    checkpoint.save=null
    checkpoint.save_interval=100
    logger.log_interval=1
    logger.logging_level=40
    logger.log_throughput=true
    "logger.tensorboard_dir=${SAVE_DIR}/tb_logs"
    "logger.save_config_filepath=${CONFIG_DUMP}"
)

# ===================================== Profiler 配置 =====================================
PROFILING_CONFIG=(
    profiling.use_pytorch_profiler=true
    profiling.profile_ranks=[0,4]
    profiling.profile_step_start=10
    profiling.profile_step_end=11
    profiling.pytorch_profiler_collect_callstack=true
)

# ===================================== 摘要 =====================================
mkdir -p ${SAVE_DIR}/train_logs
echo "======================================"
echo -e "${YELLOW}Qwen3-VL ${SIZE_UPPER} pretrain${RESET}"
echo "======================================"
echo "Mode:        ${MODE}"
echo "Profiling:   ${PROFILING:-off}"
echo "Recipe:      ${RECIPE}"
echo "Nodes:       ${NNODES}${HOSTFILE:+ (${HOSTFILE})}"
echo "GPUs:        ${GPUS}/node, ${WORLD_GPUS} total   (TP=${TP}, PP=${PP}, EP=${EP}, ETP=${ETP}, CP=${CP})"
echo "Iters:       ${TRAIN_ITERS}"
echo "GBS/MBS:     ${GLOBAL_BATCH_SIZE} / ${MICRO_BATCH_SIZE}"
echo "Seq length:  ${SEQ_LENGTH}"
echo "Model:       ${LOCAL_HF_PATH}"
echo "Dataset:     mock (${MOCK_NUM_BASE_EXAMPLES} synthetic examples, ${MOCK_NUM_IMAGES} image(s) each)"
echo "Save dir:    ${SAVE_DIR}   (no checkpoint load/save)"
echo "Master:      ${MASTER_ADDR}:${MASTER_PORT}"
echo "======================================"

# ===================================== 执行 =====================================
RECIPE_ARGS=(
    "${MEGATRON_PATH}/run_recipe.py"
    --recipe "${RECIPE}"
    --mode pretrain
)
if [[ ${MODE} == dryrun ]]; then
    RECIPE_ARGS+=(--dry-run)
fi
RECIPE_ARGS+=(
    "${DATA_CONFIG[@]}"
    "${TRAIN_CONFIG[@]}"
    "${OPTIMIZER_CONFIG[@]}"
    "${PARALLEL_CONFIG[@]}"
    "${MODEL_CONFIG[@]}"
    "${CKPT_AND_LOGGING_CONFIG[@]}"
)
if [[ ${PROFILING} == "torch" ]]; then
    RECIPE_ARGS+=("${PROFILING_CONFIG[@]}")
fi
# 尾部 KEY=VALUE 放在最后，优先级高于上面的配置块。
RECIPE_ARGS+=("${EXTRA_ARGS[@]}")

cd ${MEGATRON_PATH}/3rdparty/Megatron-Bridge
./scripts/switch_mcore.sh main
cd ${CURRENT_DIR}

# 每个分支都先 source /opt/dtk/env.sh：否则 torch 看不到 HCU，会报
# "RuntimeError: No CUDA GPUs are available"。
env_args=(
    -x PATH
    -x LIBRARY_PATH
    -x LD_LIBRARY_PATH
    -x PYTHONPATH
    -x HYHAL_PATH
    -x ROCM_PATH
    -x PYTHONWARNINGS
    -x TRANSFORMERS_VERBOSITY
    -x LOCAL_HF_PATH
    -x SEQ_LENGTH
    -x TRAIN_ITERS
    -x GLOBAL_BATCH_SIZE
    -x MICRO_BATCH_SIZE
    -x MOCK_PROMPT
    -x MOCK_NUM_IMAGES
    -x MOCK_NUM_BASE_EXAMPLES
    -x MOCK_RANDOM_SEED
    -x MOCK_IMAGE_SIZE
    -x MASTER_ADDR="${MASTER_ADDR}"
    -x MASTER_PORT="${MASTER_PORT}"
)
SAVE_LOG=${SAVE_DIR}/train_logs/mpi_$(date +%F-%H%M).log

case "${MODE}" in
    dryrun)
        # shellcheck source=/dev/null
        source /opt/dtk/env.sh
        # shellcheck source=/dev/null
        source "${BRIDGE_ENV_SCRIPT}"
        python "${RECIPE_ARGS[@]}"
        echo "======================================"
        echo "Dry run complete. Resolved config: ${CONFIG_DUMP}"
        echo "======================================"
        ;;

    train)
        TORCHRUN_ARGS=(
            --nnodes=${NNODES}
            --nproc_per_node=${GPUS}
            --master_addr=${MASTER_ADDR}
            --master_port=${MASTER_PORT}
        )

        if (( NNODES == 1 )); then
            # shellcheck source=/dev/null
            source /opt/dtk/env.sh
            # shellcheck source=/dev/null
            source "${BRIDGE_ENV_SCRIPT}"
            torchrun "${TORCHRUN_ARGS[@]}" "${RECIPE_ARGS[@]}" 2>&1 | tee "${SAVE_LOG}"
        else
            # 每个节点一个 mpirun 进程（slots=1），各自启动本节点的 torchrun。
            # 远端命令要拼成字符串，让 --node_rank 保持字面量，直到在远端展开
            # （mpirun 会设置 OMPI_COMM_WORLD_RANK）。
            TORCHRUN_QUOTED="$(printf '%q ' "${TORCHRUN_ARGS[@]}")"
            RECIPE_QUOTED="$(printf '%q ' "${RECIPE_ARGS[@]}")"
            REMOTE_CMD="torchrun --node_rank=\${OMPI_COMM_WORLD_RANK} ${TORCHRUN_QUOTED} ${RECIPE_QUOTED}"

            mpirun -np ${NNODES} --hostfile ${HOSTFILE_SLOTS} \
                --allow-run-as-root \
                --bind-to none \
                --mca plm_rsh_no_tree_spawn 1 \
                --mca plm_rsh_args "-p ${PORT}" \
                "${env_args[@]}" \
                bash -c "
                source /opt/dtk/env.sh && \
                source ${BRIDGE_ENV_SCRIPT} && \
                ${REMOTE_CMD}" 2>&1 | tee "${SAVE_LOG}"
        fi
        echo "======================================"
        echo "Training finished. Logs: ${SAVE_DIR}"
        echo "======================================"
        ;;

    *)
        echo -e "${RED}ERROR: unknown mode '${MODE}'. Use 'dryrun' or 'train'.${RESET}" >&2
        exit 1
        ;;
esac
