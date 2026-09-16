#!/bin/bash
# 真服务 profiler A/B：回答「triton 的 LoRA 算子到底有没有被 aclgraph 捕获」。
#
# 为什么必须上真服务：孤立微基准测的是「只含 1 个算子、连回放 100 次」的吞吐口径，
# 和真图里「288 个算子夹在 GEMM 依赖链上」的延迟口径根本不是一回事，两者矛盾
# （微基准说 triton 快 0.774ms/token，端到端说慢 1.45ms/token）。只有真服务的
# trace 能给出：每个 LoRA 内核出现**多少次**、每次**多久**。
#   - 若每步 288 次 → 没被捕获，eager 重发，主机派发就是账面缺口
#   - 若次数固定且很少 → 被捕获了，问题在设备侧内核本身
#
# 条件与冻结场景一致（4k in / 256 out / seed 1234 / round-robin / lora_moe），
# 只把 prompts 从 160 降到 8 —— trace 只要够多 decode step，不需要跑满时长。
# 正式采样前先跑 3 条暖机，否则 trace 里全是图捕获的噪声。
#
# 只按 setsid 进程组收，绝不按进程名 pkill（多卡共享机器，卡 4 以上是别人的作业）。
set -eu
export LORA_ARGS="--lora-target-modules q_proj k_proj v_proj o_proj experts --enable-lora --lora-modules mylora=/tmp/e2e/lora_moe --max-loras 2 --max-lora-rank 16"
export VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 MAXLEN=131072
export MODEL_PATH=/models/Qwen3-30B-A3B
export SERVED_NAME=Qwen3-30B
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export IS_MOE=0
export PROFILE=1

LOG=/tmp/e2e/profile_cmp.log
: > "$LOG"

bench() {  # bench <n_prompts>
  vllm bench serve \
    --backend openai-chat --model "$MODEL_PATH" \
    --base-url http://localhost:8001 --endpoint /v1/chat/completions \
    --num-prompts "$1" --trust-remote-code \
    --dataset-name random --ignore-eos --seed 1234 \
    --served-model-name "$SERVED_NAME" \
    --random-input-len 4096 --random-output-len 256 --random-range-ratio 0 \
    --max-concurrency 1 \
    --lora-modules mylora --lora-assignment round-robin
}

run_impl() {
  local IMPL="$1"
  local PDIR=/tmp/e2e/profiling_$IMPL
  rm -rf "$PDIR"; mkdir -p "$PDIR"
  export PROFILE_DIR="$PDIR"
  source /tmp/e2e/env.sh
  source /tmp/e2e/model.sh

  local SLOG=/tmp/e2e/profile_${IMPL}_serve.log
  : > "$SLOG"
  setsid nohup /tmp/e2e/serve.sh "$IMPL" > "$SLOG" 2>&1 </dev/null &
  local SPID=$!
  echo "[$(date '+%H:%M:%S')] $IMPL serve pgid=$SPID" >> "$LOG"

  local ok=0
  for _ in $(seq 1 400); do
    curl -sf --max-time 3 http://localhost:8001/health >/dev/null 2>&1 && { ok=1; break; }
    kill -0 "$SPID" 2>/dev/null || { echo "$IMPL SERVE DIED" >> "$LOG"; tail -40 "$SLOG" >> "$LOG"; return 1; }
    sleep 5
  done
  [ "$ok" = 1 ] || { echo "$IMPL 等不到 ready" >> "$LOG"; return 1; }
  echo "[$(date '+%H:%M:%S')] $IMPL ready" >> "$LOG"
  sleep 15

  # 暖机：先把 aclgraph 各档 capture 完，否则 trace 里全是捕获噪声
  echo "[$(date '+%H:%M:%S')] $IMPL 暖机 3 条" >> "$LOG"
  bench 3 >> "$LOG" 2>&1

  echo "[$(date '+%H:%M:%S')] $IMPL 开 profiler" >> "$LOG"
  curl -sf -X POST http://localhost:8001/start_profile >> "$LOG" 2>&1
  bench 8 >> "$LOG" 2>&1
  curl -sf -X POST http://localhost:8001/stop_profile >> "$LOG" 2>&1
  echo "[$(date '+%H:%M:%S')] $IMPL 关 profiler，trace 在 $PDIR" >> "$LOG"

  kill -TERM -- "-$SPID" 2>/dev/null
  for _ in $(seq 1 30); do
    curl -sf --max-time 2 http://localhost:8001/health >/dev/null 2>&1 || break
    sleep 2
  done
  kill -KILL -- "-$SPID" 2>/dev/null
  sleep 10
}

run_impl base
run_impl triton
echo "[$(date '+%H:%M:%S')] 全部完成" >> "$LOG"
