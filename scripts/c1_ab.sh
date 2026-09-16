#!/bin/bash
# c1_4k 单点 A/B，用于快速迭代修复。两台服务各起一次，各跑一遍 c1_4k（160 prompts）。
# 约 45 分钟（2 × [起服务 ~3min + 跑 19min]）。
#
# 为什么单点：c1 是 20 个点里唯一劣化的，且已确认是设备侧的串行归约延迟；
# 改内核后先用它判断方向，方向对了再补跑全表。
# 复现性已验：base 两个样本 1099.10/1107.56（离散 0.77%），triton 三个样本
# 1122.14/1123.94/1125.07（离散 0.26%），两组区间不重叠 → 这个单点能分辨 ~1% 的差。
#
# 只按 setsid 进程组收，绝不按进程名 pkill（多卡共享机器，卡 4 以上是别人的作业）。
set -eu
export LORA_ARGS="--lora-target-modules q_proj k_proj v_proj o_proj experts --enable-lora --lora-modules mylora=/tmp/e2e/lora_moe --max-loras 2 --max-lora-rank 16"
export VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 MAXLEN=131072
export MODEL_PATH=/models/Qwen3-30B-A3B
export SERVED_NAME=Qwen3-30B
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export IS_MOE=0
export PROFILE=0

OUTDIR="${OUTDIR:-/tmp/e2e/results/c1ab}"
mkdir -p "$OUTDIR"
LOG=$OUTDIR/c1_ab.log
: > "$LOG"

run_impl() {
  local IMPL="$1" TAG="${2:-$1}"
  source /tmp/e2e/env.sh
  source /tmp/e2e/model.sh

  local SLOG=$OUTDIR/serve_${TAG}.log
  : > "$SLOG"
  setsid nohup /tmp/e2e/serve.sh "$IMPL" > "$SLOG" 2>&1 </dev/null &
  local SPID=$!
  echo "[$(date '+%H:%M:%S')] $TAG serve pgid=$SPID" >> "$LOG"

  local ok=0
  for _ in $(seq 1 400); do
    curl -sf --max-time 3 http://localhost:8001/health >/dev/null 2>&1 && { ok=1; break; }
    kill -0 "$SPID" 2>/dev/null || { echo "$TAG SERVE DIED" >> "$LOG"; tail -40 "$SLOG" >> "$LOG"; return 1; }
    sleep 5
  done
  [ "$ok" = 1 ] || { echo "$TAG 等不到 ready" >> "$LOG"; return 1; }
  echo "[$(date '+%H:%M:%S')] $TAG ready" >> "$LOG"
  sleep 15

  echo "[$(date '+%H:%M:%S')] 开始 $TAG c1_4k" >> "$LOG"
  vllm bench serve \
    --backend openai-chat --model "$MODEL_PATH" \
    --base-url http://localhost:8001 --endpoint /v1/chat/completions \
    --num-prompts 160 --trust-remote-code \
    --dataset-name random --ignore-eos --seed 1234 \
    --served-model-name "$SERVED_NAME" \
    --random-input-len 4096 --random-output-len 256 --random-range-ratio 0 \
    --max-concurrency 1 \
    --lora-modules mylora --lora-assignment round-robin \
    > "$OUTDIR/${TAG}_c1_4k.txt" 2>&1
  echo "[$(date '+%H:%M:%S')] 完成 $TAG: $(grep -oP 'Benchmark duration \(s\):\s+\K[0-9.]+' "$OUTDIR/${TAG}_c1_4k.txt")" >> "$LOG"

  kill -TERM -- "-$SPID" 2>/dev/null
  for _ in $(seq 1 30); do
    curl -sf --max-time 2 http://localhost:8001/health >/dev/null 2>&1 || break
    sleep 2
  done
  kill -KILL -- "-$SPID" 2>/dev/null
  sleep 10
}

for impl in "$@"; do
  run_impl "$impl" "$impl"
done
echo "[$(date '+%H:%M:%S')] 全部完成" >> "$LOG"
