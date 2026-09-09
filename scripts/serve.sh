#!/bin/bash
# Triton LoRA serving — recommended (high-concurrency-safe) config.
#
# HOW TO INSTALL INTO YOUR OWN vllm-ascend SOURCE TREE (drop-in, no upstream edit):
#   1. Copy these 6 files from this repo's ROOT into <vllm-ascend>/vllm_ascend/lora/:
#        OVERWRITE: lora_ops.py                       (routes the LoRA ops to Triton)
#        ADD:       lora_ops_triton.py  lora_ops_triton_kernels.py
#                   lora_cpp_launcher.cpp  lora_cpp_launcher.cpython-312-aarch64-linux-gnu.so
#                   lora_native_ops.cpp
#   2. C++: lora_native_ops.so auto-builds on first import (see _native_build() in
#      lora_ops_triton.py; needs g++ + torch/CANN dev headers on PATH). The committed
#      lora_cpp_launcher.so is prebuilt for python3.12 + aarch64 — if your python
#      version/arch differs, rebuild lora_cpp_launcher.cpp with the same g++ flags
#      that _native_build() uses.
#   3. Version: built against recent vllm-ascend main (V1 engine). On an older version
#      whose PunicaWrapper LoRA call signatures differ, adjust the wrappers in lora_ops.py.
#   4. Below: replace /models/Qwen3.5-27B and the openscad LoRA path with yours. After
#      boot, verify LoRA is LIVE (openscad vs base output must differ on a CODE-domain
#      prompt — a prose prompt can be a false negative).
#
# WHY the two non-obvious flags (measured 2026-09-09, Qwen3.5-27B + openscad
# LoRA on Ascend 910B4 64 GB):
#
#   1. --max-num-seqs 12
#      Resident decode capacity = KV pool / ~256 KV per sequence ~= 12. A larger
#      value makes the scheduler over-admit the 13th sequence -> preempts running
#      decodes -> thrash fixed point: "Running 12 / Waiting N / KV cache 88.9% /
#      gen ~22 tok/s" (healthy is ~100). Capping at the resident capacity makes
#      overflow requests queue cleanly instead of preempting.
#
#   2. --cudagraph-capture-sizes 1 2 4 8 12   (SPACE separated, not commas)
#      Lowering --max-num-seqs alone makes vLLM auto-shrink capture sizes to the
#      largest power-of-2 <= n -> [1,2,4,8], so decode batches 9..12 fall off the
#      graph onto a ~1 s/step slow path and throughput gets WORSE. Passing the
#      sizes explicitly overrides the default derivation so the graph covers the
#      full resident batch.
#
# Measured with scripts/bench_concurrency.py (same load): C16 25.4 -> ~82, C32
# 24.1 -> ~78 tok/s. The native AscendC backend collapses identically under the
# old config -> the defect is serving-level, not the Triton kernel.

exec >> /tmp/serve_triton.log 2>&1
echo "===== TRITON LORA SERVE START $(date) ====="
export ASCEND_RT_VISIBLE_DEVICES=0      # NPU to bind
export TRITON_LORA_CPP=1                # route LoRA ops through the C++ launcher
export TRITON_LORA_TIME=0               # 0 = no per-op timing logging
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
exec vllm serve /models/Qwen3.5-27B \
  --served-model-name qwen3.5-27B --host 0.0.0.0 --port 7519 \
  --max-model-len 256 --max-num-batched-tokens 512 \
  --max-num-seqs 12 --cudagraph-capture-sizes 1 2 4 8 12 \
  --gpu-memory-utilization 0.98 --trust-remote-code \
  --enable-lora --max-loras 2 --max-lora-rank 16 \
  --lora-modules '{"name":"openscad","path":"/models/qwen35-27b-openscad-lora"}'
