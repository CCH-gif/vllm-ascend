#!/bin/bash
# dense (Qwen3.5-9B) 十点全量，**只跑 triton，默认配置**。
#
# 目的：把 TRITON_LORA_NATIVE 的作用从「bgmv 修复」里分离出来。
#   ab_native_triton/  = 同一份代码 + TRITON_LORA_NATIVE=1（PrivateUse1 注册）
#   本脚本             = 同一份代码 + 默认（python-key 注册）
# 除这一个变量外逐字相同（同 adapter、同 target-modules、同 PUNICA_LORA_TRACE=1），
# 所以两者的差就是这个变量的差。
#
# base 不跑：AscendC 是参考列，改 triton 代码它一分不变，且其十点数值已在
# /tmp/e2e/测试报告.md §3.1（c1_4k 50.44 … c32_6k 260.28），跨 session 复现
# 0.1~0.4%（50.23 / 50.29 / 50.44 三次独立测量）。重跑纯属浪费机时。
#
# 收进程：只按本脚本 setsid 出来的进程组，外加组内精确 PID。
# 绝不按进程名 pkill —— 这台机器多卡共享，卡 4 以上是别人的作业。
set -u
export LORA_ARGS="--lora-target-modules q_proj k_proj v_proj o_proj in_proj_qkv in_proj_z out_proj gate_proj up_proj down_proj --enable-lora --lora-modules mylora=/tmp/e2e/lora_fixed --max-loras 2 --max-lora-rank 16"
export VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 MAXLEN=131072
export PROFILE=0
export PUNICA_LORA_TRACE=1
source /tmp/e2e/env.sh
source /tmp/e2e/model.sh
export TRITON_LORA_NATIVE=0          # 关键：走 python-key 注册（serve.sh 默认是 1，必须显式覆盖）

OUT=/tmp/e2e/results/dense_def_triton
SLOG=/tmp/e2e/dense_def_triton_serve.log
LOG=/tmp/e2e/dense_def_triton.log
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
  local d
  d=$(grep -oP 'Benchmark duration \(s\):\s+\K[0-9.]+' "$OUT/$3.txt" 2>/dev/null)
  local t
  t=$(grep -oP 'Output token throughput \(tok/s\):\s+\K[0-9.]+' "$OUT/$3.txt" 2>/dev/null)
  echo "[$(date '+%H:%M:%S')] 完成 $3 dur=${d:-NA} tok/s=${t:-NA}" >> "$LOG"
  # 服务中途死了就停，别像上轮那样空跑 24 分钟
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

# 断言实际走了哪条派发路径：默认配置下应当一次 ACTIVE 都不打印
ACT=$(grep -c 'native kPrivateUse1 impls ACTIVE' "$SLOG" 2>/dev/null || true)
UNAV=$(grep -c 'native impls unavailable' "$SLOG" 2>/dev/null || true)
echo "[$(date '+%H:%M:%S')] 派发路径断言: ACTIVE=$ACT UNAVAIL=$UNAV（默认配置期望 0/0）" >> "$LOG"
sleep 15

for C in 1 4 8 16 32; do bench 4096 "$C" "c${C}_4k" || break; done
for C in 1 4 8 16 32; do bench 6144 "$C" "c${C}_6k" || break; done

echo "[$(date '+%H:%M:%S')] ALL DONE" >> "$LOG"
