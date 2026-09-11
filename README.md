# usage

```Shell
# Single-Node
#
#   dryrun     : rank-local config resolution only, no GPUs touched, dumps YAML.
#                Run this FIRST -- it catches recipe/override errors in seconds.
#   train      : real torchrun launch on GPUS GPUs, TRAIN_ITERS steps, mock data.
#
cd examples/deepseekv4
bash train_deepseekv4_pro_pretrain.sh dryrun
bash train_deepseekv4_pro_pretrain.sh train

# Multi-Node
#     list one hostname/IP per line
#     cat > hostfile <<'EOF'
#     node01
#     node02
#     EOF
cd examples/deepseekv4
bash train_deepseekv4_pro_pretrain.sh train hostfile 2
# ==============================================================================
```
