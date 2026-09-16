#!/bin/bash
# 补测 6k 的 c1/c4/c8/c16（两边 AscendC base + Triton）。参数与 full160 完全一致。
# 结果写入 /tmp/e2e/results/full160_{base,triton}/c{1,4,8,16}_6k.txt
# 只按 setsid 进程组收，绝不按进程名 pkill。
set -u
export LORA_ARGS="--lora-target-modules q_proj k_proj v_proj o_proj in_proj_qkv in_proj_z out_proj gate_proj up_proj down_proj --enable-lora --lora-modules mylora=/tmp/e2e/lora_fixed --max-loras 2 --max-lora-rank 16"
export VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 MAXLEN=131072
source /tmp/e2e/model.sh
LOG=/tmp/e2e/6k_driver.log
: > "$LOG"

run_one() {  # run_one <base|triton>
  local IMPL=$1
  local OUT=/tmp/e2e/results/full160_$IMPL
  mkdir -p "$OUT"
  echo "[$(date '+%H:%M:%S')] $IMPL 起服务" >> "$LOG"
  setsid nohup /tmp/e2e/serve.sh "$IMPL" > "/tmp/e2e/6k_$IMPL.log" 2>&1 &
  local SPID=$!
  local ok=0
  for _ in $(seq 1 360); do
    curl -sf --max-time 3 http://localhost:8001/health >/dev/null 2>&1 && { ok=1; break; }
    kill -0 "$SPID" 2>/dev/null || { echo "[$IMPL] SERVE DIED" >> "$LOG"; tail -30 "/tmp/e2e/6k_$IMPL.log" >> "$LOG"; return 1; }
    sleep 5
  done
  [ "$ok" = 1 ] || { echo "[$IMPL] 等不到 ready" >> "$LOG"; return 1; }
  echo "[$(date '+%H:%M:%S')] $IMPL ready" >> "$LOG"

  for C in 1 4 8 16; do
    echo "[$(date '+%H:%M:%S')] $IMPL 开跑 c${C}_6k" >> "$LOG"
    echo "########## c${C}_6k (in=6144 c=$C prompts=160) ##########" > "$OUT/c${C}_6k.txt"
    vllm bench serve \
      --backend openai-chat --model "$MODEL_PATH" \
      --base-url http://localhost:8001 --endpoint /v1/chat/completions \
      --num-prompts 160 --trust-remote-code \
      --dataset-name random --ignore-eos --seed 1234 \
      --served-model-name "$SERVED_NAME" \
      --random-input-len 6144 --random-output-len 256 --random-range-ratio 0 \
      --max-concurrency "$C" \
      --lora-modules mylora --lora-assignment round-robin \
      >> "$OUT/c${C}_6k.txt" 2>&1
    echo "[$(date '+%H:%M:%S')] $IMPL c${C}_6k 完成" >> "$LOG"
  done

  kill -TERM -- "-$SPID" 2>/dev/null
  for _ in $(seq 1 30); do
    curl -sf --max-time 2 http://localhost:8001/health >/dev/null 2>&1 || break
    sleep 2
  done
  kill -KILL -- "-$SPID" 2>/dev/null
  sleep 10
  echo "[$(date '+%H:%M:%S')] $IMPL 收进程完成" >> "$LOG"
}

run_one base   || { echo "BASE FAILED" >> "$LOG"; exit 1; }
run_one triton || { echo "TRITON FAILED" >> "$LOG"; exit 1; }
echo "[$(date '+%H:%M:%S')] 补测全部完成" >> "$LOG"
