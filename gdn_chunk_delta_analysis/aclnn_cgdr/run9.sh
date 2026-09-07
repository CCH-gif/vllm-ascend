#!/bin/bash
# Clean minimal runtime env for CANN 9.1 extracted root (/tmp/cann91) on host 910B.
# Usage: run9.sh <python> <script> [args...]
B=/tmp/cann91/root/usr/local/Ascend/cann-9.1.0
unset LD_LIBRARY_PATH ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_OPP_PATH ASCEND_AICPU_PATH \
      TOOLCHAIN_HOME ASCEND_CUSTOM_OPP_PATH ASCEND_DEVICE_ID ASCEND_RT_VISIBLE_DEVICES \
      PYTHONPATH NPU_DEVICE_ID 2>/dev/null
export LD_LIBRARY_PATH="$B/lib64:$B/lib64/plugin/opskernel:$B/lib64/plugin/nnengine:$B/lib64/plugin/vendors:$B/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64-linux:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/lib64:/usr/lib:/lib64:/lib"
export ASCEND_HOME_PATH="$B"
export ASCEND_TOOLKIT_HOME="$B"
export ASCEND_OPP_PATH="$B/opp"
export ASCEND_AICPU_PATH="$B"
export TOOLCHAIN_HOME="$B/toolkit"
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export PATH="/opt/dvmvenv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
exec "$@"
