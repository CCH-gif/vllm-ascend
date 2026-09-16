#!/bin/bash
# MoE 精度 10 点配对报告。acc_stats.py 一次只吃一对文件，这里逐点喂。
#
# 关键：不要只看"整体 token 一致率"。acc_stats.py 把它拆成
#   1) 数值噪声 —— 首次分叉之前的 |Δlogprob|（跨实现可比，干净）
#   2) 级联     —— 一次近 tie 翻转后后面全部不可比
# 必须与 moe_acc_floor.sh 量出的本底比：base 自己跑两遍也只有 ~90%。
set -u
DIR=${1:-/tmp/e2e/acc_moe}
OUT=${2:-/tmp/e2e/moe_acc_report.txt}
: > "$OUT"
for L in 4096 6144; do
  for C in 1 4 8 16 32; do
    A="$DIR/base_${L}_${C}.json"
    B="$DIR/triton_${L}_${C}.json"
    {
      echo "########## len=$L c=$C ##########"
      if [ -f "$A" ] && [ -f "$B" ]; then
        python3 /tmp/e2e/acc_stats.py "$A" "$B"
      else
        echo "  数据缺失 (base=$([ -f "$A" ] && echo y || echo n) triton=$([ -f "$B" ] && echo y || echo n))"
        echo
      fi
    } >> "$OUT" 2>&1
  done
done
cat "$OUT"
