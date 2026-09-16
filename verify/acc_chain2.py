"""公平三方对比：一层 LoRA，生产 dtype 精确复刻。
x(bf16) -> shrink A(bf16) -> buffer(fp32) -> expand B(bf16) -> y(bf16)
裁判 = CPU fp64（x/A/B 都是 bf16，gold 用 fp64 算"这些 bf16 输入下的精确值"）。
"""
import torch, torch_npu  # noqa
import vllm_ascend.vllm_ascend_C  # noqa
from vllm_ascend.lora import lora_ops_triton as T

DEV, I64 = "npu", torch.int64
torch.manual_seed(0)
H = Ho = 4096
R = 16
L = 2

print(f"{'B':>4} {'triton链 max|Δ|':>16} {'AscendC链 max|Δ|':>18} {'比值 Asc/tri':>12}  谁更准")
for B in (1, 8, 32):
    x = (torch.randn(B, H, device=DEV) * 0.08).to(torch.bfloat16)
    A = (torch.randn(L, R, H, device=DEV) * 0.08).to(torch.bfloat16)
    Bw = (torch.randn(L, Ho, R, device=DEV) * 0.08).to(torch.bfloat16)
    start = torch.zeros(1, dtype=I64, device=DEV)
    idx = torch.zeros(1, dtype=I64, device=DEV)

    # fp64 gold
    gold = ((x.double() @ A[0].double().T) @ Bw[0].double().T).to(torch.bfloat16)

    # triton 链
    buf_t = torch.zeros(B, R, dtype=torch.float32, device=DEV)
    T.sgmv_shrink(x, A.reshape(L, 1, R, H), buf_t, start, idx, 1.0)
    y_t = torch.zeros(B, Ho, device=DEV).to(torch.bfloat16)
    T.sgmv_expand(buf_t, Bw.reshape(L, 1, Ho, R), y_t, start, idx)
    et = (y_t.float() - gold.float()).abs().max().item()

    # AscendC 链
    buf_a = torch.zeros(B, R, dtype=torch.float32, device=DEV)
    torch.ops._C_ascend.sgmv_shrink(x, A.reshape(L, 1, R, H), idx, start, buf_a, 1.0)
    y_a = torch.zeros(B, Ho, device=DEV).to(torch.bfloat16)
    torch.ops._C_ascend.sgmv_expand(buf_a, Bw.reshape(L, 1, Ho, R), idx,
                                    torch.tensor([B], dtype=I64, device=DEV), y_a, 0, Ho)
    ea = (y_a.float() - gold.float()).abs().max().item()

    print(f"{B:4d} {et:16.3e} {ea:18.3e} {ea/max(et,1e-12):11.1f}x   "
          f"{'triton' if et <= ea else 'AscendC'}", flush=True)
