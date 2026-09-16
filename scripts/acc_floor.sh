#!/bin/bash
# 精度尺子的"本底噪声"对照：同一个 base(AscendC) 服务内，同参数跑两遍 probe。
#
# 为什么需要：base 跨 session 跑两遍只有 89.65% token 一致率、位置0 的 logprob
# 差中位 2.1e-2（bf16 ulp 量级）。那到底是"每次起服务导致的"（编译/memory 布局）
# 还是"每次调度导致的"（连续批组成不同 → 求和顺序不同）？同服务内跑两遍就能分开。
#
# 有了这个本底，base-vs-triton 的一致率才有得比：落在本底内就是"分辨不出"。
#
# 只按 setsid 进程组收，绝不按进程名 pkill（多卡共享机器）。
set -u
export LORA_ARGS="--lora-target-modules q_proj k_proj v_proj o_proj in_proj_qkv in_proj_z out_proj gate_proj up_proj down_proj --enable-lora --lora-modules mylora=/tmp/e2e/lora_fixed --max-loras 2 --max-lora-rank 16"
export VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 MAXLEN=131072
export PROFILE=0
unset TRITON_LORA_NATIVE 2>/dev/null || true
source /tmp/e2e/env.sh
source /tmp/e2e/model.sh

LOG=/tmp/e2e/acc_floor_driver.log
OUT=/tmp/e2e/acc_floor
rm -rf "$OUT"; mkdir -p "$OUT"
: > "$LOG"

echo "[$(date '+%H:%M:%S')] 等 acc_full 收工" >> "$LOG"
for _ in $(seq 1 300); do
  grep -q "ALL DONE" /tmp/e2e/acc_full_driver.log && break
  sleep 15
done
echo "[$(date '+%H:%M:%S')] acc_full 结束，等 30s 让进程组退干净" >> "$LOG"
sleep 30

setsid nohup /tmp/e2e/serve.sh basecount > /tmp/e2e/acc_floor_serve.log 2>&1 &
SPID=$!
sleep 20
CPID=$(ps -eo pid,ppid,pgid,args | awk -v p="$SPID" '$2==p {print $3}' | head -1)
echo "[$(date '+%H:%M:%S')] serve pgid=$SPID child=$CPID" >> "$LOG"

ok=0
for _ in $(seq 1 400); do
  curl -sf --max-time 3 http://localhost:8001/health >/dev/null 2>&1 && { ok=1; break; }
  kill -0 "$SPID" 2>/dev/null || { echo "SERVE DIED" >> "$LOG"; tail -30 /tmp/e2e/acc_floor_serve.log >> "$LOG"; exit 1; }
  sleep 5
done
[ "$ok" = 1 ] || { echo "等不到 ready" >> "$LOG"; exit 1; }
echo "[$(date '+%H:%M:%S')] ready" >> "$LOG"
sleep 15

for r in 1 2; do
  echo "[$(date '+%H:%M:%S')] base run$r 开始" >> "$LOG"
  python3 /tmp/e2e/accuracy_probe.py "base_run$r" "$OUT/base_run$r.json" \
    --lora mylora --n 64 --len 4096 --out-tokens 8 --concurrency 32 --seed 1234 \
    >> "$LOG" 2>&1
  echo "[$(date '+%H:%M:%S')] base run$r rc=$?" >> "$LOG"
done

kill -TERM -- "-$SPID" 2>/dev/null
[ -n "${CPID:-}" ] && kill -TERM -- "-$CPID" 2>/dev/null
for _ in $(seq 1 30); do
  curl -sf --max-time 2 http://localhost:8001/health >/dev/null 2>&1 || break
  sleep 2
done
kill -KILL -- "-$SPID" 2>/dev/null
[ -n "${CPID:-}" ] && kill -KILL -- "-$CPID" 2>/dev/null
sleep 10
echo "[$(date '+%H:%M:%S')] 收工" >> "$LOG"
