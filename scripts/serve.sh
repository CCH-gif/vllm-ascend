#!/bin/bash
# 用法: serve.sh <base|triton> [额外参数...]
#   base   = 原版 AscendC LoRA 算子 (lora_ops.py.stock)
#   triton = Triton LoRA 算子     (lora_ops.py)
set -x
LORA_IMPL=$1; shift || true
LORA_DIR=/vllm-workspace/vllm-ascend/vllm_ascend/lora
# 参考实现放在 /tmp/e2e/ref/（不在源码树里，树里 lora_ops.py 就是 triton 版）
REF_DIR=/tmp/e2e/ref
case "$LORA_IMPL" in
  base)   cp -f $REF_DIR/lora_ops.py.stock  $LORA_DIR/lora_ops.py ;;
  triton) cp -f $REF_DIR/lora_ops.py.triton $LORA_DIR/lora_ops.py ;;
  *) echo "usage: $0 <base|triton> [extra vllm args]"; exit 2 ;;
esac
# 换实现必须清掉字节码，否则 __pycache__ 里是上一版
rm -rf $LORA_DIR/__pycache__
source /tmp/e2e/env.sh
source /tmp/e2e/model.sh
# MoE 专属参数，dense 基座下必须去掉：--enable-expert-parallel 对没有专家的
# 模型无意义，multistream_overlap_shared_expert 在没有 shared expert 时会报错。
if [ "${IS_MOE}" = "1" ]; then
  _MOE=(--enable-expert-parallel)
  _ADDCFG='{"enable_cpu_binding": true, "multistream_overlap_shared_expert": true, "fuse_muls_add": true, "enable_npugraph_ex": true}'
else
  _MOE=()
  _ADDCFG='{"enable_cpu_binding": true, "fuse_muls_add": true, "enable_npugraph_ex": true}'
fi
echo "[serve] MODEL_PATH=$MODEL_PATH SERVED_NAME=$SERVED_NAME IS_MOE=$IS_MOE" >&2
# LoRA 开关：设了 LORA_ARGS 就加进去（形如 --enable-lora --max-loras 2 ...）
if [ -n "${LORA_ARGS:-}" ]; then read -r -a _LORA <<< "$LORA_ARGS"; else _LORA=(); fi
# profiler：PROFILE=0 可关。挂上它运行期零开销——AsyncLLM 只构造对象
# (async_llm.py:188)，真正 .start() 在 :908，即调 /start_profile 时才发生。
# 要抓 trace：curl -X POST localhost:8001/start_profile ... /stop_profile
# 目录必须绝对路径，否则落在服务进程 cwd。
if [ "${PROFILE:-1}" = "1" ]; then
  PDIR="${PROFILE_DIR:-/tmp/e2e/profiling}"; mkdir -p "$PDIR"
  _PROF=(--profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"$PDIR\", \"torch_profiler_with_stack\": false}")
else
  _PROF=()
fi
export ASCEND_RT_VISIBLE_DEVICES="${VISIBLE_DEVICES:-0,1,2,3}"
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000

vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 \
  --port 8001 \
  --tensor-parallel-size ${TP_SIZE:-4} \
  --max-num-seqs 32 \
  --max-model-len ${MAXLEN:-131072} \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --async-scheduling \
  --trust-remote-code \
  "${_MOE[@]}" \
  --enable-prompt-tokens-details \
  --no-enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enable-chunked-prefill \
  --mamba-ssm-cache-dtype bfloat16 \
  --compilation-config \
  '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config \
  "$_ADDCFG" \
  "${_PROF[@]}" "${_LORA[@]}" "$@"
