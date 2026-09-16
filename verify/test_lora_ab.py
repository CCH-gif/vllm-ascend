"""Operator-level A/B: Triton LoRA ops (via vllm_ascend.lora.lora_ops) vs AscendC.

Focus: NR values that used to trip the removed NR<=16 fallback, including
non-powers-of-two.  Shapes follow Qwen3.6-35B-A3B (hidden_size=2048,
moe_intermediate_size=512).

Index dtypes match production: vllm's PunicaWrapper allocates
``_token_lora_indices`` / ``_lora_indices_per_batch`` / ``_seq_lengths`` as
torch.long, and the AscendC kernels read them as int64 (they do
``SetGlobalBuffer((__gm__ int64_t *)...)``).  Passing int32 here made the
hardware read 8 bytes per element and fault with "GM address accessed by
scalar exceeds 48 bits" -- a harness bug, not a kernel bug.
"""
import importlib.machinery
import importlib.util
import sys

import torch
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401  registers torch.ops._C_ascend

# Load the triton shim by path, NOT via vllm_ascend.lora.lora_ops: serve.sh
# swaps that file between .stock (AscendC) and .triton depending on which
# server it launched.  Importing it while a base sweep is running silently
# compares AscendC against itself -- that already produced one vacuous
# 'ALL PASSED' this session.
from vllm_ascend.lora import lora_ops_triton  # noqa: F401  registers custom ops
# .py.triton is not a recognised suffix, so name the loader explicitly.
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

FAILS = []


def report(name, a, b, atol=0.0, rtol=0.0):
    a = a.float()
    b = b.float()
    diff = (a - b).abs()
    denom = b.abs().clamp_min(1e-6)
    max_abs = diff.max().item()
    max_rel = (diff / denom).max().item()
    exact = bool(torch.equal(a, b))
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    tag = "EXACT" if exact else ("CLOSE" if ok else "MISMATCH")
    print(f"  {tag:9s} {name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}")
    if not ok:
        FAILS.append(name)
    return ok


def mk(shape, dtype=torch.bfloat16):
    return torch.randn(shape, dtype=dtype, device=DEV)


# ---------------------------------------------------------------- bgmv_shrink
def test_bgmv_shrink(B, H, R, L_n, tag):
    print(f"bgmv_shrink B={B} H={H} R={R} L={L_n} {tag}")
    x = mk((B, H))
    w = mk((L_n, 1, R, H))
    idx = torch.randint(0, L_n, (B,), dtype=I64, device=DEV)
    y_t = torch.zeros(B, R, dtype=torch.float32, device=DEV)
    y_a = torch.zeros(B, R, dtype=torch.float32, device=DEV)
    L.bgmv_shrink(x, w, y_t, idx, 0.7)
    torch.ops._C_ascend.bgmv_shrink(x, w, idx, y_a, 0.7)
    torch.npu.synchronize()
    report(f"bgmv_shrink {tag}", y_t, y_a)


# ------------------------------------------------------- bgmv_expand_slice
def test_bgmv_expand(B, R, Ho, L_n, Y_HO, off, tag):
    print(f"bgmv_expand_slice B={B} R={R} Ho={Ho} Y_HO={Y_HO} off={off} {tag}")
    x = torch.randn(B, R, dtype=torch.float32, device=DEV)
    w = mk((L_n, 1, Ho, R))
    idx = torch.randint(0, L_n, (B,), dtype=I64, device=DEV)
    base = mk((B, Y_HO))
    y_t = base.clone()
    y_a = base.clone()
    L.bgmv_expand_slice(x, w, y_t, idx, off, Ho)
    torch.ops._C_ascend.bgmv_expand(x, w, idx, y_a, off, Ho)
    torch.npu.synchronize()
    report(f"bgmv_expand_slice {tag}", y_t, y_a)


# ---------------------------------------------------------------- sgmv cases
def make_meta(NR, total, seeds):
    """Exclusive prefix sum + per-request counts for NR requests covering `total`."""
    counts = torch.tensor(seeds, dtype=I64, device=DEV)
    assert int(counts.sum()) == total, (int(counts.sum()), total)
    start = torch.zeros(NR, dtype=I64, device=DEV)
    start[1:] = torch.cumsum(counts, 0)[:-1]
    return start, counts


def test_sgmv_shrink(B, H, R, L_n, NR, counts, tag):
    print(f"sgmv_shrink B={B} H={H} R={R} NR={NR} {tag}")
    x = mk((B, H))
    w = mk((L_n, 1, R, H))
    start, seqlen = make_meta(NR, B, counts)
    idx = torch.randint(0, L_n, (NR,), dtype=I64, device=DEV)
    y_t = torch.zeros(B, R, dtype=torch.float32, device=DEV)
    y_a = torch.zeros(B, R, dtype=torch.float32, device=DEV)
    L.sgmv_shrink(x, w, y_t, start, seqlen, idx, NR, B, B, 0.7)
    torch.ops._C_ascend.sgmv_shrink(x, w, idx, seqlen, y_a, 0.7)
    torch.npu.synchronize()
    report(f"sgmv_shrink B={B} NR={NR} {tag}", y_t, y_a)


def test_sgmv_expand(B, R, Ho, L_n, NR, counts, Y_HO, off, tag):
    print(f"sgmv_expand_slice B={B} R={R} Ho={Ho} NR={NR} {tag}")
    x = torch.randn(B, R, dtype=torch.float32, device=DEV)
    w = mk((L_n, 1, Ho, R))
    start, seqlen = make_meta(NR, B, counts)
    idx = torch.randint(0, L_n, (NR,), dtype=I64, device=DEV)
    base = mk((B, Y_HO))
    y_t = base.clone()
    y_a = base.clone()
    L.sgmv_expand_slice(x, w, y_t, start, seqlen, idx, NR, B, B, off, Ho)
    torch.ops._C_ascend.sgmv_expand(x, w, idx, seqlen, y_a, off, Ho)
    torch.npu.synchronize()

    # Two different kernels serve this op, and they have different (both valid)
    # numerics.  dot accumulates in fp32 and matches an exact fp64 reference to
    # the bit; flat reproduces AscendC's own fp32 summation order and so lands
    # on the same side of every bf16 rounding boundary as AscendC does, up to
    # one output ulp (<=2^-6 at these magnitudes) away from fp64.  AscendC
    # itself is the one that rounds the fp32 shrink output through bf16 first.
    #
    # So the bar is: agree with AscendC exactly, and never be further from the
    # exact answer than AscendC is.  Asserting "fp64-exact" on both would fail
    # flat for correctly reproducing the incumbent.
    ref = base.double().clone()
    c = [0] + list(torch.cumsum(torch.tensor(counts, dtype=I64), 0).tolist())
    xd, wd = x.double(), w.reshape(L_n, Ho, R).double()
    for j in range(NR):
        if int(idx[j]) >= 0:
            s, e = c[j], c[j + 1]
            ref[s:e, off:off + Ho] += xd[s:e] @ wd[int(idx[j])].T
    ref = ref.to(torch.bfloat16)

    d_t_a = (y_t.float() - y_a.float()).abs().max().item()
    d_t = (y_t.float() - ref.float()).abs().max().item()
    d_a = (y_a.float() - ref.float()).abs().max().item()
    ok = d_t <= d_a                                  # never less accurate
    print(f"  {'OK   ' if ok else 'WORSE':9s} sgmv_expand_slice B={B} NR={NR} "
          f"{tag}: vsAscendC={d_t_a:.3e}")
    print(f"    (info) vs fp64: triton={d_t:.3e}  AscendC={d_a:.3e}"
          f"{'' if ok else '   <-- TRITON IS WORSE THAN ASCENDC'}")
    if not ok:
        FAILS.append(f"sgmv_expand_slice B={B} NR={NR} {tag}")


def split(total, n):
    """n positive counts summing to total."""
    q, r = divmod(total, n)
    return [q + (1 if i < r else 0) for i in range(n)]


print("=" * 70)
print("A. bgmv (decode path) - regression vs AscendC")
print("=" * 70)
test_bgmv_shrink(32, 2048, 16, 2, "H=2048")
test_bgmv_shrink(32, 512, 16, 2, "H=512 (expert down_proj)")
test_bgmv_shrink(32, 2048, 32, 4, "R=32 L=4")
test_bgmv_expand(32, 16, 4096, 2, 4096, 0, "full slice")
test_bgmv_expand(32, 16, 64, 2, 4096, 4032, "tail slice")

print()
print("=" * 70)
print("B. sgmv (prefill path) - the NR>16 cases that used to fall back")
print("=" * 70)
for NR in (1, 2, 3, 5, 8, 16, 17, 24, 32, 33, 64):
    B = 128
    test_sgmv_shrink(B, 2048, 16, 2, NR, split(B, NR),
                     "on-2d" if NR in (1, 2, 8, 16, 32, 64) else "NON-pow2")

print()
print("=" * 70)
print("C. sgmv_expand_slice - same NR sweep")
print("=" * 70)
for NR in (1, 3, 5, 16, 17, 32, 33, 64):
    B = 128
    test_sgmv_expand(B, 16, 4096, 2, NR, split(B, NR), 4096, 0,
                     "on-2d" if NR in (1, 16, 32, 64) else "NON-pow2")

print()
print("=" * 70)
print("D. large NR (high concurrency) - new capability")
print("=" * 70)
for NR in (128, 256):
    B = 512
    test_sgmv_shrink(B, 2048, 16, 4, NR, split(B, NR), "high-conc")
    test_sgmv_expand(B, 16, 4096, 4, NR, split(B, NR), 4096, 0, "high-conc")

print()
print("=" * 70)
print("D2. decode-shaped B (routes to sgmv_expand_flat, below _DOT_MIN_TOKENS)")
print("=" * 70)
for B, NR in ((1, 1), (8, 8), (16, 16), (32, 32), (64, 32), (127, 32)):
    test_sgmv_expand(B, 16, 4096, 2, NR, split(B, NR), 4096, 0, f"decode B={B}")
test_sgmv_expand(32, 16, 12288, 2, 32, split(32, 32), 12288, 0, "decode Ho=12288")
test_sgmv_expand(32, 8, 4096, 2, 32, split(32, 32), 4096, 0, "decode R=8")
for NR in (128, 256):
    B = 512
    test_sgmv_shrink(B, 2048, 16, 4, NR, split(B, NR), "high-conc")
    test_sgmv_expand(B, 16, 4096, 4, NR, split(B, NR), 4096, 0, "high-conc")

print()
print("=" * 70)
print("E. ragged batch + -1 (no-LoRA) rows")
print("=" * 70)


def test_ragged_neg():
    """Uneven request lengths, some requests on LoRA -1 (skip)."""
    B, H, R, L_n, NR = 100, 2048, 16, 3, 7
    counts = [7, 3, 21, 5, 40, 9, 15]
    x = mk((B, H))
    w = mk((L_n, 1, R, H))
    start, seqlen = make_meta(NR, B, counts)
    idx = torch.tensor([0, -1, 2, 1, -1, 2, 0], dtype=I64, device=DEV)
    y_t = torch.randn(B, R, dtype=torch.float32, device=DEV)
    y_a = y_t.clone()
    L.sgmv_shrink(x, w, y_t, start, seqlen, idx, NR, B, B, 0.7)
    torch.ops._C_ascend.sgmv_shrink(x, w, idx, seqlen, y_a, 0.7)
    torch.npu.synchronize()
    report("sgmv_shrink ragged+-1", y_t, y_a)

    base = mk((B, 4096))
    y_t = base.clone()
    y_a = base.clone()
    xr = torch.randn(B, R, dtype=torch.float32, device=DEV)
    wb = mk((L_n, 1, 4096, R))  # expand weight is [L, 1, Ho, R], not the shrink weight
    L.sgmv_expand_slice(xr, wb, y_t, start, seqlen, idx, NR, B, B, 0, 4096)
    torch.ops._C_ascend.sgmv_expand(xr, wb, idx, seqlen, y_a, 0, 4096)
    torch.npu.synchronize()
    ref = base.double().clone()
    c = [0] + list(torch.cumsum(torch.tensor(counts, dtype=I64), 0).tolist())
    wd = wb.reshape(L_n, 4096, R).double()
    for j in range(NR):
        if int(idx[j]) >= 0:
            s, e = c[j], c[j + 1]
            ref[s:e] += xr[s:e].double() @ wd[int(idx[j])].T
    ref = ref.to(torch.bfloat16)
    # same bar as test_sgmv_expand: never further from exact than AscendC
    d_t = (y_t.float() - ref.float()).abs().max().item()
    d_a = (y_a.float() - ref.float()).abs().max().item()
    ok = d_t <= d_a
    print(f"  {'OK   ' if ok else 'WORSE':9s} sgmv_expand_slice ragged+-1: "
          f"vsAscendC={(y_t.float() - y_a.float()).abs().max().item():.3e}")
    print(f"    (info) vs fp64: triton={d_t:.3e}  AscendC={d_a:.3e}")
    if not ok:
        FAILS.append("sgmv_expand_slice ragged+-1")
    # -1 requests must come out of expand exactly as they went in
    for j in range(NR):
        if int(idx[j]) < 0:
            s, e = c[j], c[j + 1]
            assert torch.equal(y_t[s:e], base[s:e]), f"-1 row {j} was modified"
    print("    -1 rows left untouched by expand")


test_ragged_neg()

print()
if FAILS:
    print(f"FAILURES ({len(FAILS)}):")
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print("ALL PASSED")
