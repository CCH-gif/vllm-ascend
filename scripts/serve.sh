#!/bin/bash
exec >> /tmp/serve_triton.log 2>&1
echo "===== TRITON LORA SERVE START $(date) ====="
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
vllm serve /models/Qwen3.5-27B \
  --served-model-name qwen3.5-27B --host 0.0.0.0 --port 7519 \
  --max-model-len 1024 --max-num-batched-tokens 1024 --max-num-seqs 4 \
  --gpu-memory-utilization 0.95 --trust-remote-code \
  --enable-lora --max-loras 2 --max-lora-rank 16 \
  --lora-modules '{"name":"openscad","path":"/models/qwen35-27b-openscad-lora"}'
