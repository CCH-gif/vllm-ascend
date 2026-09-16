#!/bin/bash
# 完整精度对照：base(AscendC) vs triton(PrivateUse1)，覆盖 4k/6k × c{1,4,8,16,32} 十点。
#
# 为什么覆盖整个矩阵：之前只抽查了 len=4096/c=32 一个配置。不同输入长度和不同
# 并发会走到不同的内核分支（prefill 用 sgmv_*_dot，decode 用 bgmv_*），只测一个
# 配置证明不了别的分支数值也对。
#
# 探针输出只有 8 个 token、prefill 为主，所以比整轮压测便宜得多。
# 同一 (seed, n, len) 生成的 prompt 与并发无关，因此每个 (len, c) 都能直接跨实现对比。
#
# 判据不是"逐 token 相同"：AscendC 的 sgmv_expand 要求输出 bf16，内部把 fp32 的
# shrink 结果 cast 成 bf16 再乘加（有损）；triton 全程 fp32。以 fp64 当裁判，
# triton 比 AscendC 准 572~16064 倍。所以看的是"差异是否稳定在已知量级"。
#
# 只按 setsid 进程组收，绝不按进程名 pkill（多卡共享机器）。
set -u
export LORA_ARGS="--lora-target-modules q_proj k_proj v_proj o_proj in_proj_qkv in_proj_z out_proj gate_proj up_proj down_proj --enable-lora --lora-modules mylora=/tmp/e2e/lora_fixed --max-loras 2 --max-lora-rank 16"
export VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 MAXLEN=131072
export PROFILE=0
source /tmp/e2e/env.sh
source /tmp/e2e/model.sh

CONFIGS="4096 1
4096 4
4096 8
4096 16
4096 32
6144 1
6144 4
6144 8
6144 16
6144 32"

LOG=/tmp/e2e/acc_full_driver.log
OUTDIR=/tmp/e2e/acc_full
rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"

run_impl() {  # run_impl <base|triton>
  local IMPL=$1
  local SLOG=/tmp/e2e/acc_full_$IMPL.log
  : > "$SLOG"

  # triton 侧要打开 PrivateUse1 注册（默认就是，显式写出以防环境残留）
  if [ "$IMPL" = "triton" ]; then export TRITON_LORA_NATIVE=1; else unset TRITON_LORA_NATIVE; fi

  setsid nohup /tmp/e2e/serve.sh "$IMPL" > "$SLOG" 2>&1 &
  local SPID=$!
  sleep 20
  local CPID
  CPID=$(ps -eo pid,ppid,pgid,args | awk -v p="$SPID" '$2==p {print $3}' | head -1)
  echo "[$(date '+%H:%M:%S')] $IMPL serve pgid=$SPID child=$CPID" >> "$LOG"

  local ok=0
  for _ in $(seq 1 400); do
    curl -sf --max-time 3 http://localhost:8001/health >/dev/null 2>&1 && { ok=1; break; }
    kill -0 "$SPID" 2>/dev/null || { echo "[$IMPL] SERVE DIED" >> "$LOG"; tail -30 "$SLOG" >> "$LOG"; return 1; }
    sleep 5
  done
  [ "$ok" = 1 ] || { echo "[$IMPL] 等不到 ready" >> "$LOG"; return 1; }
  echo "[$(date '+%H:%M:%S')] $IMPL ready" >> "$LOG"
  sleep 15

  local L C
  while read -r L C; do
    [ -n "${C:-}" ] || continue
    echo "[$(date '+%H:%M:%S')] $IMPL len=$L c=$C 开始" >> "$LOG"
    python3 /tmp/e2e/accuracy_probe.py "$IMPL" "$OUTDIR/${IMPL}_${L}_${C}.json" \
      --lora mylora --n 64 --len "$L" --out-tokens 8 --concurrency "$C" --seed 1234 \
      >> "$LOG" 2>&1
    echo "[$(date '+%H:%M:%S')] $IMPL len=$L c=$C rc=$?" >> "$LOG"
  done <<< "$CONFIGS"

  kill -TERM -- "-$SPID" 2>/dev/null
  [ -n "${CPID:-}" ] && kill -TERM -- "-$CPID" 2>/dev/null
  for _ in $(seq 1 30); do
    curl -sf --max-time 2 http://localhost:8001/health >/dev/null 2>&1 || break
    sleep 2
  done
  kill -KILL -- "-$SPID" 2>/dev/null
  [ -n "${CPID:-}" ] && kill -KILL -- "-$CPID" 2>/dev/null
  sleep 10
  echo "[$(date '+%H:%M:%S')] $IMPL 收工" >> "$LOG"
}

: > "$LOG"
run_impl base   || { echo "BASE FAILED" >> "$LOG"; exit 1; }
run_impl triton || { echo "TRITON FAILED" >> "$LOG"; exit 1; }
echo "[$(date '+%H:%M:%S')] ALL DONE" >> "$LOG"
