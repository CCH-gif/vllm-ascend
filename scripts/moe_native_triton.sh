#!/bin/bash
# MoE (Qwen3-30B-A3B) 十点全量，**只跑 triton，TRITON_LORA_NATIVE=1**。
#
# 目的：把 MoE 那列从 python-key 派发切到 PrivateUse1，与 dense 9B 同一配置。
#   rest_triton/（报告头条数据）= 默认(python-key)，不设该变量
#   本脚本                      = TRITON_LORA_NATIVE=1（PrivateUse1，与 AscendC 同构）
#
# base 不跑：AscendC 是参考列，与这些 triton 环境变量无关；rest_base/ 已是同日
# 同 session 的有效测量，直接当分母。
#
# 参数与 moe_rest_sweep.sh 逐字一致：IS_MOE=0、PROFILE=0、160 prompts、seed 1234、
# round-robin、lora_moe、--lora-target-modules q/k/v/o/experts。
#
# 收进程：只按本脚本 setsid 出来的进程组，外加组内精确 PID。
# 绝不按进程名 pkill —— 这台机器多卡共享，卡 4 以上是别人的作业。
set -u
export LORA_ARGS="--lora-target-modules q_proj k_proj v_proj o_proj experts --enable-lora --lora-modules mylora=/tmp/e2e/lora_moe --max-loras 2 --max-lora-rank 16"
export VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 MAXLEN=131072
export MODEL_PATH=/models/Qwen3-30B-A3B
export SERVED_NAME=Qwen3-30B
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export IS_MOE=0
export PROFILE=0
export TRITON_LORA_NATIVE=1          # 关键：走 PrivateUse1 注册
source /tmp/e2e/env.sh
source /tmp/e2e/model.sh

OUT=/tmp/e2e/results/rest_triton_native
SLOG=/tmp/e2e/rest_triton_native_serve.log
LOG=/tmp/e2e/rest_triton_native.log
rm -rf "$OUT"; mkdir -p "$OUT"
: > "$SLOG"; : > "$LOG"
echo "[$(date '+%H:%M:%S')] TRITON_LORA_NATIVE='${TRITON_LORA_NATIVE:-<unset>}'" >> "$LOG"

teardown() {
  kill -TERM -- "-$SPID" 2>/dev/null
  # `kill -- -PGID` 带不走被 reparent 到 PID 1 的 vLLM/Worker，按 PGID 点名再收一遍
  local pids p
  pids=$(ps -eo pid,pgid --no-headers | awk -v g="$SPID" '$2==g{print $1}')
  for p in $pids; do kill -TERM "$p" 2>/dev/null; done
  for _ in $(seq 1 30); do
    curl -sf --max-time 2 http://localhost:8001/health >/dev/null 2>&1 || break
    sleep 2
  done
  pids=$(ps -eo pid,pgid --no-headers | awk -v g="$SPID" '$2==g{print $1}')
  kill -KILL -- "-$SPID" 2>/dev/null
  for p in $pids; do kill -KILL "$p" 2>/dev/null; done
  sleep 10
  pids=$(ps -eo pid,pgid --no-headers | awk -v g="$SPID" '$2==g{print $1}' | tr '\n' ' ')
  echo "[$(date '+%H:%M:%S')] 收工，残留 PGID=$SPID 进程: ${pids:-无}" >> "$LOG"
}
SPID=""
trap 'teardown' EXIT INT TERM

bench() {  # bench <in_len> <conc> <label>
  echo "########## $3 (in=$1 c=$2 prompts=160) ##########" > "$OUT/$3.txt"
  echo "[$(date '+%H:%M:%S')] 开始 $3 (in=$1 c=$2)" >> "$LOG"
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
  local d t
  d=$(grep -oP 'Benchmark duration \(s\):\s+\K[0-9.]+' "$OUT/$3.txt" 2>/dev/null)
  t=$(grep -oP 'Output token throughput \(tok/s\):\s+\K[0-9.]+' "$OUT/$3.txt" 2>/dev/null)
  echo "[$(date '+%H:%M:%S')] 完成 $3 dur=${d:-NA} tok/s=${t:-NA}" >> "$LOG"
  # 服务中途死了就停，别空跑
  curl -sf --max-time 3 http://localhost:8001/health >/dev/null 2>&1 || {
    echo "[$3] 服务已死，中止 sweep" >> "$LOG"; tail -30 "$SLOG" >> "$LOG"; return 1; }
  return 0
}

setsid nohup /tmp/e2e/serve.sh triton > "$SLOG" 2>&1 </dev/null &
SPID=$!
echo "[$(date '+%H:%M:%S')] serve pgid=$SPID" >> "$LOG"

ok=0
for _ in $(seq 1 400); do
  curl -sf --max-time 3 http://localhost:8001/health >/dev/null 2>&1 && { ok=1; break; }
  kill -0 "$SPID" 2>/dev/null || { echo "SERVE DIED" >> "$LOG"; tail -40 "$SLOG" >> "$LOG"; exit 1; }
  sleep 5
done
[ "$ok" = 1 ] || { echo "等不到 ready" >> "$LOG"; exit 1; }
echo "[$(date '+%H:%M:%S')] ready" >> "$LOG"

# 断言实际走了 PrivateUse1：每个 worker 打一行，TP4 期望 4
ACT=$(grep -c 'native kPrivateUse1 impls ACTIVE' "$SLOG" 2>/dev/null || true)
UNAV=$(grep -c 'native impls unavailable' "$SLOG" 2>/dev/null || true)
echo "[$(date '+%H:%M:%S')] 派发路径断言: ACTIVE=$ACT UNAVAIL=$UNAV（PrivateUse1 期望 ACTIVE>=1 且 UNAVAIL=0）" >> "$LOG"
if [ "${ACT:-0}" -lt 1 ] || [ "${UNAV:-0}" -ne 0 ]; then
  echo "!! 派发路径不符预期，中止（数据不可用）" >> "$LOG"
  grep -i 'triton-lora' "$SLOG" >> "$LOG" 2>/dev/null
  exit 1
fi
sleep 15

for C in 1 4 8 16 32; do bench 4096 "$C" "c${C}_4k" || break; done
for C in 1 4 8 16 32; do bench 6144 "$C" "c${C}_6k" || break; done

echo "[$(date '+%H:%M:%S')] ALL DONE" >> "$LOG"
