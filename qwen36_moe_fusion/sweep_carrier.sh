#!/bin/bash
# Fused-v1 (native gate_up+swiglu) operating envelope + A/B across M. Real shape K=2048 N=1024 E=256.
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/ascend-toolkit/latest/lib64
PY=/usr/local/python3.11.13/bin/python3
DIR=/workspace/ascend-operator-agent-v2-gitcode6.18/qwen36_moe_fusion
cd "$DIR"
devA=0; devB=3
: > bench_A_carrier.json
# list "M U"
A="8192 256|4096 256|2048 256|1024 256|512 256|256 128"
B="1024 128|512 128|256 64|128 64|128 32|64 32|64 16"
run_list(){ local dev=$1; local out=$3; : > "$out"; local IFS='|'; for pair in $2; do
  local M=${pair%% *}; local U=${pair##* }
  echo "=== dev$dev M$M U$U ===" >> "$out"
  local line; line=$(timeout 220 $PY -u bench_one.py "$dev" "$M" "$U" 2>/dev/null | grep RESULT)
  if [ -n "$line" ]; then echo "$line" >> "$out"; else echo "{\"M\":$M,\"U\":$U,\"STATUS\":\"FAIL_OR_TIMEOUT\"}" >> "$out"; fi
done; }
run_list $devA "$A" bench_A_carrier_A.json &
run_list $devB "$B" bench_A_carrier_B.json &
wait
cat bench_A_carrier_A.json bench_A_carrier_B.json > bench_A_carrier.json
echo SWEEPDONE
