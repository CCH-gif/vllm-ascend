"""MoE 形状下 Triton LoRA 算子 vs AscendC 的算子级 A/B。

覆盖 dense A/B 没碰过的区间（Qwen3-30B-A3B, TP=4）：
  * L = max_loras * num_experts = 2 * 128 = 256 （dense 只有 2）
  * shrink 的 H = 192 = moe_intermediate/TP，非 2 幂且 BW=128 留 64 的尾块
    （现有测试只有 H ∈ {512, 2048}）
  * expand 的 Ho = 192 非 2 幂，带尾块
  * combined_idx = lora_id * E + expert_id 的交错取值

调用约定逐字照抄 punica_npu.py::add_lora_fused_moe：
  bgmv_shrink(x2d[B,H], a_flat[L,R,H], shrink_out[B,R] fp32, combined_idx, 1.0)
  bgmv_expand_slice(delta[B,R], b_flat[L,Ho,R], y2d[B,Y_HO], combined_idx, off, Ho)
"""
import importlib.machinery
import importlib.util
import sys

import torch
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401  registers torch.ops._C_ascend
from vllm_ascend.lora import lora_ops_triton  # noqa: F401

# 按路径加载 triton shim，绝不能通过 vllm_ascend.lora.lora_ops 导入 ——
# serve.sh 会在 .stock/.triton 之间换那个文件，导入错了就成了 AscendC 自比。
_l = importlib.machinery.SourceFileLoader(
    'lora_ops_shim',
    '/vllm-workspace/vllm-ascend/vllm_ascend/lora/lora_ops.py.triton')
_spec = importlib.util.spec_from_loader('lora_ops_shim', _l)
L = importlib.util.module_from_spec(_spec)
_l.exec_module(L)
assert 'vllm_ascend_triton' in str(L.sgmv_expand_slice.__code__.co_names), \
    'lora_ops shim is not the triton one -- the A/B would be vacuous'

torch.manual_seed(0)
DEV = "npu"
I64 = torch.int64

# Qwen3-30B-A3B, TP=4
H_HID = 2048          # hidden_size
H_MOE = 768 // 4      # moe_intermediate_size / TP  = 192
NEXP = 128
MAX_LORAS = 2
L_N = MAX_LORAS * NEXP   # 256
R = 16

FAILS = []


def mk(shape, dtype=torch.bfloat16):
    return torch.randn(shape, dtype=dtype, device=DEV)


EPS32 = 1.2e-7


def compare(name, tri, asc, ref, K):
    """ref 是 fp64 精确参考，K 是归约深度。

    判据按算子实际特性分开设，因为两者的输出 dtype 不同：

    * shrink 输出 fp32 -> 可以直接与 fp64 比。用**相对**误差，容差取 fp32 累加
      K 项的随机游走上界 8*eps*sqrt(K)。绝对误差在这里没有意义：H=2048 的随机
      点积幅值 ~100，bf16 量化台阶就有 0.5，而实测误差 1e-4 比它低四个数量级。
      AscendC 的偏差只作参考打印——两边都是同一随机游走的不同实现，胜负会随 B
      翻转（实测 B=32 时 triton 更准，B=1/128 时 AscendC 略准），谁更接近 fp64
      是掷硬币，不能拿来当判据。

    * expand 输出 bf16（y 由 bf16 base 克隆而来）-> 与 fp64 比毫无意义，残差被
      输出量化主导。判据是**与 AscendC 逐 bit 一致**。

    K 为 None 表示按 bf16 输出处理（走逐 bit 判据）。
    """
    d_ta = (tri.float() - asc.float()).abs().max().item()
    if K is None:
        # bf16 输出：判据是「差不超过 2 个 bf16 ulp」。expand 实测恒为 0（逐 bit），
        # 链式因为 shrink 的 fp32 重排（<1e-6 相对）会传导成 1 ulp，这是应有结果。
        ulp = max(tri.float().abs().max().item(),
                  asc.float().abs().max().item()) * torch.finfo(torch.bfloat16).eps
        ok = d_ta <= 2 * ulp
        print(f"  {'OK   ' if ok else 'MISMATCH':9s} {name}: "
              f"triton-vs-ascendc={d_ta:.3e} 2ulp={2*ulp:.3e} (bf16 输出，判据=2 ulp)")
        if not ok:
            FAILS.append(name)
        return ok

    d_t = (tri.float().double() - ref).abs().max().item()
    d_a = (asc.float().double() - ref).abs().max().item()
    scale = ref.abs().max().item()
    rel_t = d_t / max(scale, 1e-12)
    rel_a = d_a / max(scale, 1e-12)
    tol = 8 * EPS32 * (K ** 0.5)
    ok = rel_t <= tol
    print(f"  {'OK   ' if ok else 'MISMATCH':9s} {name}: "
          f"|ref|={scale:.1f} rel_triton={rel_t:.2e} rel_ascendc={rel_a:.2e} "
          f"tol={tol:.1e}" + ("" if ok else "  <<< 超容差"))
    if not ok:
        FAILS.append(name)
    return ok


def make_idx(B, n_neg=3):
    """combined_idx = lora_id * E + expert_id，外加几行 -1（无 adapter）。"""
    lora_id = torch.randint(0, MAX_LORAS, (B,), dtype=I64, device=DEV)
    expert_id = torch.randint(0, NEXP, (B,), dtype=I64, device=DEV)
    idx = lora_id * NEXP + expert_id
    if n_neg and B > n_neg:
        neg_rows = torch.randperm(B, device=DEV)[:n_neg]
        idx[neg_rows] = -1
    return idx


def test_shrink(B, H, tag):
    print(f"bgmv_shrink B={B} H={H} L={L_N} {tag}")
    x = mk((B, H))
    w = mk((L_N, 1, R, H))
    idx = make_idx(B)
    y_t = torch.zeros(B, R, dtype=torch.float32, device=DEV)
    y_a = torch.zeros(B, R, dtype=torch.float32, device=DEV)
    L.bgmv_shrink(x, w, y_t, idx, 0.7)
    torch.ops._C_ascend.bgmv_shrink(x, w, idx, y_a, 0.7)
    torch.npu.synchronize()

    wd = w.reshape(L_N, R, H).double()
    xd = x.double()
    ref = torch.zeros(B, R, dtype=torch.float64, device=DEV)
    for b in range(B):
        i = int(idx[b])
        if i >= 0:
            ref[b] = (xd[b] @ wd[i].T) * 0.7
    compare(f"bgmv_shrink B={B} H={H} {tag}", y_t, y_a, ref, H)


def test_expand(B, Ho, Y_HO, off, tag):
    print(f"bgmv_expand_slice B={B} Ho={Ho} Y_HO={Y_HO} off={off} {tag}")
    delta = torch.randn(B, R, dtype=torch.float32, device=DEV)
    w = mk((L_N, 1, Ho, R))
    idx = make_idx(B)
    base = mk((B, Y_HO))
    y_t = base.clone()
    y_a = base.clone()
    L.bgmv_expand_slice(delta, w, y_t, idx, off, Ho)
    torch.ops._C_ascend.bgmv_expand(delta, w, idx, y_a, off, Ho)
    torch.npu.synchronize()

    wd = w.reshape(L_N, Ho, R).double()
    ref = base.double().clone()
    for b in range(B):
        i = int(idx[b])
        if i >= 0:
            ref[b, off:off + Ho] += delta[b].double() @ wd[i].T
    compare(f"bgmv_expand_slice B={B} Ho={Ho} off={off} {tag}", y_t, y_a, ref, None)

    # -1 行必须原样保留
    for b in range(B):
        if int(idx[b]) < 0:
            assert torch.equal(y_t[b], base[b]), f"-1 行 {b} 被 expand 改写"
    print("    -1 行保持不动")


def test_w13_chain(B):
    """完整 w13：两片（gate, up）依次累加进同一个 y —— 复刻 add_lora_fused_moe。"""
    print(f"--- w13 两片链式累加 B={B} ---")
    x = mk((B, H_HID))
    a0, a1 = mk((L_N, 1, R, H_HID)), mk((L_N, 1, R, H_HID))
    b0, b1 = mk((L_N, 1, H_MOE, R)), mk((L_N, 1, H_MOE, R))
    idx = make_idx(B)
    Y_HO = 2 * H_MOE

    base = mk((B, Y_HO))
    y_t = base.clone()
    y_a = base.clone()
    for impl, y in ((L, y_t), (None, y_a)):
        shrink = torch.zeros(B, R, dtype=torch.float32, device=DEV)
        for s, (a, b, off) in enumerate(((a0, b0, 0), (a1, b1, H_MOE))):
            if impl is not None:
                impl.bgmv_shrink(x, a, shrink, idx, 1.0)
                impl.bgmv_expand_slice(shrink, b, y, idx, off, H_MOE)
        if impl is None:
            torch.ops._C_ascend.bgmv_shrink(x, a0, idx, shrink, 1.0)
            torch.ops._C_ascend.bgmv_expand(shrink, b0, idx, y, 0, H_MOE)
            torch.ops._C_ascend.bgmv_shrink(x, a1, idx, shrink, 1.0)
            torch.ops._C_ascend.bgmv_expand(shrink, b1, idx, y, H_MOE, H_MOE)
    torch.npu.synchronize()

    ref = base.double().clone()
    for j, (a, b, off) in enumerate(((a0, b0, 0), (a1, b1, H_MOE))):
        wd = b.reshape(L_N, H_MOE, R).double()
        ad = a.reshape(L_N, R, H_HID).double()
        for row in range(B):
            i = int(idx[row])
            if i >= 0:
                d = x[row].double() @ ad[i].T
                ref[row, off:off + H_MOE] += d @ wd[i].T
    compare(f"w13 链式 B={B}", y_t, y_a, ref, None)


print("=" * 72)
print(f"MoE 形状 A/B  L={L_N} (max_loras={MAX_LORAS} x experts={NEXP})  R={R}")
print(f"hidden={H_HID}  moe_inter/TP={H_MOE}")
print("=" * 72)

print()
print("A. w13 shrink (gate/up, H=2048)")
for B in (1, 8, 32, 128, 512):
    test_shrink(B, H_HID, "w13")

print()
print("B. w2 shrink (down_proj, H=192 非 2 幂, BW=128 留 64 尾块)")
for B in (1, 8, 32, 128, 512):
    test_shrink(B, H_MOE, "w2")

print()
print("C. w13 expand_slice (Ho=192 非 2 幂, 尾块)")
for B in (1, 8, 32, 128, 512):
    test_expand(B, H_MOE, 2 * H_MOE, 0, "gate off=0")
    test_expand(B, H_MOE, 2 * H_MOE, H_MOE, "up off=192")

print()
print("D. w2 expand_slice (Ho=2048)")
for B in (1, 8, 32, 128, 512):
    test_expand(B, H_HID, H_HID, 0, "down")

print()
print("E. w13 链式（两片累加进同一 y）")
for B in (1, 32, 512):
    test_w13_chain(B)

print()
if FAILS:
    print(f"FAILURES ({len(FAILS)}):")
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print("ALL PASSED")
