# 公共环境变量（HCCL / 网卡绑定）
#
# ⚠️ 下面 NIC_NAME / LOCAL_IP 必须改成**你自己机器上 UP 的那个物理网卡和它的 IPv4**，
#    否则 HCCL 初始化会失败。查法：
#      ip -o link show up | grep -v lo     # 找 UP 的物理网卡名
#      ip -4 addr show <网卡名>            # 取它的 IPv4
#
# 本机实测的教训（所以留了这个自检）：文档里给的网卡名/IP 在实机上都不存在
# —— 一个网卡存在但无载波（operstate=down），另一个地址不在任何网卡上。
# 照抄的直接后果是 HCCL 卡死，且现象很难定位。故下面留了一个开机自检。
export NIC_NAME="${NIC_NAME:-<your-nic>}"
export LOCAL_IP="${LOCAL_IP:-<your-ip>}"
# 自检：IP 不在本机就立刻报错，别等到 HCCL 卡死
if ! hostname -I 2>/dev/null | grep -qw "$LOCAL_IP"; then
  echo "[env] 警告: $LOCAL_IP 不在本机网卡上，HCCL 会失败" >&2
fi
export HCCL_IF_IP=$LOCAL_IP
export GLOO_SOCKET_IFNAME=$NIC_NAME
export TP_SOCKET_IFNAME=$NIC_NAME
export HCCL_SOCKET_IFNAME=$NIC_NAME
export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=100
export TASK_QUEUE_ENABLE=1
export VLLM_RPC_TIMEOUT=300000
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
# sysctl 那几行需要 root，运行时单独执行
