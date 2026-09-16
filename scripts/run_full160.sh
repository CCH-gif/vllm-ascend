#!/bin/bash
# 全点 160 prompts 的完整 A/B（用户要求：六点只有并发不同）。
#   c ∈ {1,4,8,16,32} 输入 4096，c=32 输入 6144；每点 160 prompts。
# 与上一轮的差别：c<32 原来是 16 prompts，这轮统一 160。
#
# 收进程只按本脚本 setsid 出来的进程组 ($SPID == PGID)：
#   kill -TERM -- "-$SPID" / kill -KILL -- "-$SPID"
# 绝不按进程名 pkill —— 这台机器多卡共享，按名杀会打到别人的作业。
set -u
export LORA_ARGS="--lora-target-modules q_proj k_proj v_proj o_proj in_proj_qkv in_proj_z out_proj gate_proj up_proj down_proj --enable-lora --lora-modules mylora=/tmp/e2e/lora_fixed --max-loras 2 --max-lora-rank 16"
export VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 MAXLEN=131072
source /tmp/e2e/model.sh

sweep() {  # sweep <outdir>
  local OUT=$1
  bench() {  # bench <in_len> <conc> <label>
    echo "########## $3 (in=$1 c=$2 prompts=160) ##########" > "$OUT/$3.txt"
    vllm bench serve \
      --backend openai-chat --model "$MODEL_PATH" \
      --base-url http://localhost:8001 --endpoint /v1/chat/completions \
      --num-prompts 160 --trust-remote-code \
      --dataset-name random --ignore-eos --seed 1234 \
      --served-model-name "$SERVED_NAME" \
      --random-input-len "$1" --random-output-len 256 --random-range-ratio 0 \
      --max-concurrency "$2" \
      --lora-modules mylora --lora-assignment round-robin \
      >> "$OUT/$3.txt" 2>&1
  }
  bench 4096 1  c1_4k
  bench 4096 4  c4_4k
  bench 4096 8  c8_4k
  bench 4096 16 c16_4k
  bench 4096 32 c32_4k
  bench 6144 32 c32_6k
}

run_impl() {  # run_impl <base|triton>
  local IMPL=$1
  local OUT=/tmp/e2e/results/full160_$IMPL
  rm -rf "$OUT"; mkdir -p "$OUT"
  setsid nohup /tmp/e2e/serve.sh "$IMPL" > "/tmp/e2e/full160_$IMPL.log" 2>&1 &
  local SPID=$!
  echo "[$(date '+%H:%M:%S')] $IMPL serve pgid=$SPID" >> /tmp/e2e/full160_driver.log

  for _ in $(seq 1 360); do
    curl -sf --max-time 3 http://localhost:8001/health >/dev/null 2>&1 && break
    kill -0 "$SPID" 2>/dev/null || { echo "[$IMPL] SERVE DIED" >> /tmp/e2e/full160_driver.log; return 1; }
    sleep 5
  done
  echo "[$(date '+%H:%M:%S')] $IMPL ready" >> /tmp/e2e/full160_driver.log

  sweep "$OUT"

  echo "[$(date '+%H:%M:%S')] $IMPL sweep done, stopping" >> /tmp/e2e/full160_driver.log
  kill -TERM -- "-$SPID" 2>/dev/null
  for _ in $(seq 1 30); do
    curl -sf --max-time 2 http://localhost:8001/health >/dev/null 2>&1 || break
    sleep 2
  done
  kill -KILL -- "-$SPID" 2>/dev/null
  sleep 10
  echo "[$(date '+%H:%M:%S')] $IMPL stopped" >> /tmp/e2e/full160_driver.log
}

: > /tmp/e2e/full160_driver.log
run_impl base
run_impl triton
echo "[$(date '+%H:%M:%S')] ALL DONE" >> /tmp/e2e/full160_driver.log
