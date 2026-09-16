# A/B 基座模型配置 —— serve.sh / bench_*.sh / run_ab.sh 都 source 这个。
#
# 压测场景与参数保持与文档一致，只换基座：
#   4k / 6k 输入, 256 输出, 32 并发, 160 prompts, TP4, seed 1234
#
# 当前基座 Qwen3.5-9B 的选型依据：
#   产线 Qwen3.6-35B-A3B : qwen3_5_moe, hidden 2048, key_dim 2048, value_dim 4096
#   Qwen3.5-9B           : qwen3_5 (dense), hidden 4096, key_dim 2048, value_dim 4096
# 同一条 qwen3_5 代码路径，且 in_proj_qkvz 的 output_sizes=[2048,2048,4096,4096]
# 与产线一致。dense 无 MoE -> AscendFusedMoEWithLoRA 不构造 -> 断言不触发。
#
# 回到原基座：export MODEL_PATH=/models/Qwen3.6-35B-A3B IS_MOE=1
export MODEL_PATH="${MODEL_PATH:-/models/Qwen3.5-9B}"
export SERVED_NAME="${SERVED_NAME:-Qwen36}"
# IS_MOE=1 时 serve.sh 才加 EP / shared-expert overlap 相关参数。
# dense 模型加这些会报错或行为未定义。
export IS_MOE="${IS_MOE:-0}"
