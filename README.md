# usage

```Shell
# Single-Node
#
#   dryrun     : rank-local config resolution only, no GPUs touched, dumps YAML.
#                Run this FIRST -- it catches recipe/override errors in seconds.
#   train      : real torchrun launch on GPUS GPUs, TRAIN_ITERS steps, mock data.
#
cd examples/deepseekv4
bash train_deepseekv4_pro_pretrain.sh deepseek_v4_pro_pretrain_8gpu_bw1100_bf16_config dryrun
bash train_deepseekv4_pro_pretrain.sh deepseek_v4_pro_pretrain_8gpu_bw1100_bf16_config train

# Multi-Node
#     list one hostname/IP per line
#     cat > hostfile <<'EOF'
#     node01
#     node02
#     EOF
cd examples/deepseekv4
bash train_deepseekv4_pro_pretrain.sh deepseek_v4_pro_pretrain_16gpu_bw1000_bf16_config train hostfile 2
# ==============================================================================
```
