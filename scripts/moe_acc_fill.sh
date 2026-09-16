#!/bin/bash
# 补跑 moe_acc.sh 因偶发 HTTP 400 丢掉的数据点。
#
# 背景：triton 侧 c=4/c=8 两点上，每个并发波次里有 2~4 个请求被 API 层以
# 400 拒掉（服务端日志只有 "400 Bad Request"，不记原因），accuracy_probe.py
# 因为 ex.map 立刻抛出而整点崩溃、不落盘。base 侧 10 点零个 400。
# 探针已改成"重试 6 次 + 打印响应体"，所以后续点自愈；已经丢掉的两点用本脚本补。
#
# 只按 setsid 进程组收，绝不按进程名 pkill（多卡共享机器）。
set -u
export LORA_ARGS="--lora-target-modules q_proj k_proj v_proj o_proj experts --enable-lora --lora-modules mylora=/tmp/e2e/lora_moe --max-loras 2 --max-lora-rank 16"
export VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 MAXLEN=131072
export MODEL_PATH=/models/Qwen3-30B-A3B
export SERVED_NAME=Qwen3-30B
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export IS_MOE=0
export PROFILE=0
source /tmp/e2e/env.sh
source /tmp/e2e/model.sh

OUTDIR=/tmp/e2e/acc_moe
IMPL=${1:-triton}
LOG=/tmp/e2e/moe_acc_fill_driver.log
: > "$LOG"

# 端口被占说明还有 serve 在跑，直接退出而不是抢
if curl -sf --max-time 2 http://localhost:8001/health >/dev/null 2>&1; then
  echo "端口 8001 已有人在用，先等它结束" >> "$LOG"; exit 1
fi

# 找出缺哪些点（base/triton 两侧都查，缺谁补谁）
MISSING=""
for L in 4096 6144; do
  for C in 1 4 8 16 32; do
    [ -f "$OUTDIR/${IMPL}_${L}_${C}.json" ] || MISSING="$MISSING$L $C"$'\n'
  done
done
MISSING=$(printf '%s' "$MISSING" | sed '/^$/d')
if [ -z "$MISSING" ]; then echo "没有缺口，收工" >> "$LOG"; exit 0; fi
echo "[$(date '+%H:%M:%S')] 需要补的点：" >> "$LOG"
echo "$MISSING" >> "$LOG"

SLOG=/tmp/e2e/moe_acc_fill_serve.log
: > "$SLOG"
setsid nohup /tmp/e2e/serve.sh "$IMPL" > "$SLOG" 2>&1 </dev/null &
SPID=$!
echo "[$(date '+%H:%M:%S')] $IMPL serve pgid=$SPID" >> "$LOG"

ok=0
for _ in $(seq 1 400); do
  curl -sf --max-time 3 http://localhost:8001/health >/dev/null 2>&1 && { ok=1; break; }
  kill -0 "$SPID" 2>/dev/null || { echo "SERVE DIED" >> "$LOG"; tail -40 "$SLOG" >> "$LOG"; exit 1; }
  sleep 5
done
[ "$ok" = 1 ] || { echo "等不到 ready" >> "$LOG"; exit 1; }
echo "[$(date '+%H:%M:%S')] ready" >> "$LOG"
sleep 15

while read -r L C; do
  [ -n "${C:-}" ] || continue
  echo "[$(date '+%H:%M:%S')] $IMPL len=$L c=$C 开始" >> "$LOG"
  python3 /tmp/e2e/accuracy_probe.py "$IMPL" "$OUTDIR/${IMPL}_${L}_${C}.json" \
    --lora mylora --n 64 --len "$L" --out-tokens 8 --concurrency "$C" --seed 1234 \
    >> "$LOG" 2>&1
  # 先取 $?，再跑 date —— 否则 $(date) 的退出码会把 $? 覆盖成 0
  rc=$?
  echo "[$(date '+%H:%M:%S')] $IMPL len=$L c=$C rc=$rc" >> "$LOG"
done <<< "$MISSING"

kill -TERM -- "-$SPID" 2>/dev/null
for _ in $(seq 1 30); do
  curl -sf --max-time 2 http://localhost:8001/health >/dev/null 2>&1 || break
  sleep 2
done
kill -KILL -- "-$SPID" 2>/dev/null
sleep 10
echo "[$(date '+%H:%M:%S')] 补跑收工" >> "$LOG"
