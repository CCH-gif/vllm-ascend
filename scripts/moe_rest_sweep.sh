#!/bin/bash
# 补齐 MoE 冻结场景剩下的所有并发点。口径：每点都不许劣化。
#
# 已有（当前代码）：base c1_4k 1107.56 / c32_4k 354.02 / c1_6k；
#                   triton c1_4k 1123.94 / c32_4k 320.13。
# 本脚本补：
#   base   : c1_4k 重跑(验 1107.56 可不可复现) + c4/c8/c16_4k + c4/c8/c16/c32_6k
#   triton : c4/c8/c16_4k + c1/c4/c8/c16/c32_6k
# base 那台的 c1_4k 重跑是关键：c1 是唯一显示劣化的点，而它只有一次采样。
#
# 两台服务各起一次，中间不重启。参数与 moe160_* 完全一致：
# IS_MOE=0、PROFILE=0、160 prompts、seed 1234、round-robin、lora_moe。
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

LOG=/tmp/e2e/moe_rest_sweep.log
: > "$LOG"

bench() {  # bench <outdir> <in_len> <conc> <label>
  local OUT="$1" in_len="$2" conc="$3" label="$4"
  echo "########## $label (in=$in_len c=$conc prompts=160) ##########" > "$OUT/$label.txt"
  echo "[$(date '+%H:%M:%S')] 开始 $label (in=$in_len c=$conc)" >> "$LOG"
  vllm bench serve \
    --backend openai-chat --model "$MODEL_PATH" \
    --base-url http://localhost:8001 --endpoint /v1/chat/completions \
    --num-prompts 160 --trust-remote-code \
    --dataset-name random --ignore-eos --seed 1234 \
    --served-model-name "$SERVED_NAME" \
    --random-input-len "$in_len" --random-output-len 256 --random-range-ratio 0 \
    --max-concurrency "$conc" \
    --lora-modules mylora --lora-assignment round-robin \
    >> "$OUT/$label.txt" 2>&1
  echo "[$(date '+%H:%M:%S')] 完成 $label" >> "$LOG"
}

run_impl() {  # run_impl <base|triton> <label...>
  local IMPL="$1"; shift
  local OUT=/tmp/e2e/results/rest_$IMPL
  mkdir -p "$OUT"
  source /tmp/e2e/env.sh
  source /tmp/e2e/model.sh

  local SLOG=/tmp/e2e/rest_${IMPL}_serve.log
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

  # 三元组消费：4096 1 c1_4k_rep → bench <out> <in> <conc> <label>
  # 不能用 `for spec in "$@"` —— 那是按**单词**迭代，会把 "4096" 单独喂给 bench，
  # bench 拿不到 4 个参数，set -u 下 $3 未绑定直接退出（上一轮就是这么空跑 24 分钟的）。
  while [ "$#" -ge 3 ]; do
    bench "$OUT" "$1" "$2" "$3"
    shift 3
  done

  echo "[$(date '+%H:%M:%S')] $IMPL 跑完，收服务" >> "$LOG"
  kill -TERM -- "-$SPID" 2>/dev/null
  for _ in $(seq 1 30); do
    curl -sf --max-time 2 http://localhost:8001/health >/dev/null 2>&1 || break
    sleep 2
  done
  kill -KILL -- "-$SPID" 2>/dev/null
  sleep 10
}

# base：c1_4k 重跑放最前（先拿到关键数），再 4k，再 6k
# c1_6k 也要跑：moe160_base/c1_6k.txt 是空文件，之前那次跑挂了。
run_impl base \
  4096 1  c1_4k_rep \
  4096 4  c4_4k \
  4096 8  c8_4k \
  4096 16 c16_4k \
  4096 32 c32_4k \
  6144 1  c1_6k \
  6144 4  c4_6k \
  6144 8  c8_6k \
  6144 16 c16_6k \
  6144 32 c32_6k

# triton：4k 全跑（含 c1/c32，让整张表同源）+ 6k 全跑
run_impl triton \
  4096 1  c1_4k \
  4096 4  c4_4k \
  4096 8  c8_4k \
  4096 16 c16_4k \
  4096 32 c32_4k \
  6144 1  c1_6k \
  6144 4  c4_6k \
  6144 8  c8_6k \
  6144 16 c16_6k \
  6144 32 c32_6k

echo "[$(date '+%H:%M:%S')] 全部完成" >> "$LOG"
