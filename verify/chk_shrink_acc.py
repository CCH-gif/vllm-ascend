"""bgmv_shrink 新内核（宽归约）精度校验：vs fp64 与 vs AscendC。

覆盖 H 能被/不能被 BW 整除、R 非 16、以及 idx=-1 行。
"""
import torch, torch_npu  # noqa
import vllm_ascend.vllm_ascend_C  # noqa
from vllm_ascend.lora import lora_ops_triton as T

DEV, I64 = "npu", torch.int64
torch.manual_seed(0)

# 生产 H：q/k/v/o=4096、gate/up=12288、down=17408（TP4 时 4352）；故意加 3 个非整除值
CASES = [(4096, 16, 2), (12288, 16, 2), (4352, 16, 2), (17408, 16, 2),
         (2112, 16, 2), (320, 16, 2), (64, 16, 2), (2048, 8, 2), (512, 32, 3)]

print(f"{'H':>7} {'R':>3} {'B':>3} | {'vs fp64':>11} {'vs AscendC':>11} {'AscC vs fp64':>13}")
for H, R, L in CASES:
    B = 24
    x = (torch.randn(B, H, device=DEV) * 0.05).to(torch.bfloat16)
    w = (torch.randn(L, 1, R, H, device=DEV) * 0.05).to(torch.bfloat16)
    idx = torch.arange(B, dtype=I64, device=DEV) % L
    idx[3] = -1  # 无 LoRA 行

    base = torch.randn(B, R, dtype=torch.float32, device=DEV)

    y_t = base.clone()
    T.bgmv_shrink(x, w, y_t, idx, 1.0)
    y_a = base.clone()
    torch.ops._C_ascend.bgmv_shrink(x, w, idx, y_a, 1.0)
    torch.npu.synchronize()

    # fp64 精确参考（对同一 bf16 输入）
    xd = x.double()
    wd = w.reshape(L, R, H).double()
    ref = torch.empty(B, R, dtype=torch.float64, device=DEV)
    for b in range(B):
        i = int(idx[b])
        ref[b] = wd[i] @ xd[b] if i >= 0 else base[b].double()

    d_t = (y_t.double() - ref).abs().max().item()
    d_a = (y_a.double() - ref).abs().max().item()
    print(f"{H:7d} {R:3d} {B:3d} | {d_t:11.3e} {d_a:11.3e} {d_a:13.3e}"
          f"{'  <- triton 更准' if d_t < d_a else ''}", flush=True)
