# Probe: option-2 carrier for fusion A (gate_up grouped-GEMM + SwiGLU).
# FUSED   (single launch, stock CANN): npu_grouped_matmul_swiglu_quant[_v2]
# UNFUSED (two launches, moe_mlp fallback): npu_grouped_matmul(int32) -> npu_dequant_swiglu_quant
# All native torch_npu ops, W8A8. Real Qwen3.6-35B-A3B shapes: K=2048, N=1024 (gate|up), d_ff=512.
# Numeric target: fused ≈ unfused (moe treats them as interchangeable), and both ≈ fp32 manual ref.
import os, sys, faulthandler
faulthandler.enable()
import torch, torch_npu
torch_npu.npu.set_device(0)
torch.manual_seed(0)

M   = 512
E   = 256          # total experts in weight
K   = 2048
N   = 1024         # 2 * d_ff (gate|up stacked)
USED = 256         # experts actually holding tokens (uniform: rows_per_exp each)

assert M % USED == 0
RPE = M // USED
counts_cpu = torch.full((E,), 0, dtype=torch.int64)
counts_cpu[:USED] = RPE
cum_cpu = torch.cumsum(counts_cpu, 0)       # [E] int64 cumsum, last == M
cum = cum_cpu.to("npu:0")                   # device ops need group_list on NPU
counts = counts_cpu.to("npu:0")
print("rows/expert =", RPE, "cum[-1] =", cum_cpu[-1].item(), flush=True)

# --- build W8A8 quantized inputs (per-token x, per-channel w, symmetric) ---
x = (torch.randn(M, K) * 1.5).npu().to(torch.bfloat16)
w = (torch.randn(E, K, N) * 0.02).npu().to(torch.bfloat16)

xq, xs = torch_npu.npu_dynamic_quant(x)    # int8 [M,K], fp32 [M]
amax = w.float().abs().amax(dim=1)          # [E,N]
ws = amax / 127.0
ws = ws.clamp_min(1e-8)
wq = (w.float() / ws.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
wq_nz = torch_npu.npu_format_cast(wq, 29)   # FRACTAL_NZ
del x, w, amax
print("dtypes xq", xq.dtype, xq.shape, "ws", ws.shape, "wq_nz", wq_nz.shape, wq_nz.dtype, flush=True)

# --- manual fp32 reference (per-expert slice) for numeric sanity ---
def manual_ref(xq, wq_nd, ws, xs, cum, E):
    # fp32 gate_up per expert then silu(gate)*up then per-token requant
    prev = 0
    outs = []
    for e in range(E):
        c = cum[e].item()
        if c == prev: continue
        g = xq[prev:c].float() @ wq_nd[e].float()          # [r,K]@[K,N]=[r,N]
        g = g * ws[e].unsqueeze(0) * xs[prev:c].unsqueeze(1)
        gate, up = g[:, :N // 2], g[:, N // 2:]
        h = torch.nn.functional.silu(gate) * up            # [r,512]
        s = h.abs().amax(dim=1).clamp_min(1e-8) / 127.0
        q = (h / s.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
        outs.append((q, s))
        prev = c
    return torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])

try:
    ref_q, ref_s = manual_ref(xq, wq, ws, xs, cum, E)
    print("manual ref done", ref_q.shape, flush=True)
except Exception as ex:
    print(f"[manual ref] FAIL {type(ex).__name__}: {ex}", flush=True)
    ref_q = None

# --- FUSED single-launch carriers (try v2 first, then non-v2) ---
def run_fused_v2():
    yq, ys = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
        x=xq, weight=[wq_nz], weight_scale=[ws], x_scale=xs, group_list=cum)
    return yq, ys

def run_fused_v1():
    yq, ys, _ = torch_npu.npu_grouped_matmul_swiglu_quant(
        x=xq, weight=wq_nz, weight_scale=ws, x_scale=xs, group_list=cum)
    return yq, ys

fused = None
for name, fn in [("v2", run_fused_v2), ("v1", run_fused_v1)]:
    try:
        yq, ys = fn()
        torch.npu.synchronize()
        if ref_q is not None:
            dq = (yq.float() - ref_q.float()).abs().max().item()
            print(f"[fused {name}] OK  yq {yq.shape} vs ref max_abs={dq:.4f}", flush=True)
        else:
            print(f"[fused {name}] OK  yq {yq.shape}", flush=True)
        if fused is None:
            fused = (name, yq, ys)
    except Exception as ex:
        print(f"[fused {name}] FAIL {type(ex).__name__}: {ex}", flush=True)

# --- UNFUSED two-step (moe_mlp fallback) ---
def run_unfused(group_index_cumsum=True):
    z = torch_npu.npu_grouped_matmul(
        x=[xq], weight=[wq_nz], split_item=3, group_list_type=0, group_type=0,
        group_list=cum, output_dtype=torch.int32)[0]      # int32 [M,N]
    gi = cum if group_index_cumsum else counts_cpu.to("npu:0")
    yq, ys = torch_npu.npu_dequant_swiglu_quant(
        x=z, weight_scale=ws, activation_scale=xs, bias=None,
        quant_scale=None, quant_offset=None, group_index=gi,
        activate_left=True, quant_mode=1)
    return yq, ys

for cumsum_gi in (True, False):
    try:
        yq, ys = run_unfused(cumsum_gi)
        torch.npu.synchronize()
        if ref_q is not None:
            dq = (yq.float() - ref_q.float()).abs().max().item()
            print(f"[unfused gi_cumsum={cumsum_gi}] OK yq {yq.shape} vs ref max_abs={dq:.4f}", flush=True)
        else:
            print(f"[unfused gi_cumsum={cumsum_gi}] OK yq {yq.shape}", flush=True)
    except Exception as ex:
        print(f"[unfused gi_cumsum={cumsum_gi}] FAIL {type(ex).__name__}: {ex}", flush=True)

print("DONE", flush=True)
