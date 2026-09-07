# bench_one <dev> <M> <U>  -- fused-v1 vs unfused-two-step, real gate_up shape K=2048 N=1024 E=256.
# W8A8. Input pool rotated per rep to dodge torch_npu same-input result-cache trap.
# Prints one JSON line. Exit non-zero on crash/hang.
import sys, json, faulthandler, time
faulthandler.enable()
import torch, torch_npu
dev, M, U = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
torch_npu.npu.set_device(dev); torch.manual_seed(0)
E, K, N = 256, 2048, 1024
assert M % U == 0
R = M // U
P = 4  # input pool size
x = [(torch.randn(M, K) * 1.5).to(f"npu:{dev}").to(torch.bfloat16) for _ in range(P)]
w = (torch.randn(E, K, N) * 0.02).to(f"npu:{dev}").to(torch.bfloat16)
xq, xs = [], []
for xi in x:
    a, b = torch_npu.npu_dynamic_quant(xi); xq.append(a); xs.append(b)
ws = w.float().abs().amax(dim=1).clamp_min(1e-8) / 127.0
wq = (w.float() / ws.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
wq_nz = torch_npu.npu_format_cast(wq, 29)
del x, w
# group boundaries: first U experts carry R rows each, rest inactive
cum = torch.cat([torch.arange(1, U + 1, dtype=torch.int64) * R,
                 torch.full((E - U,), M, dtype=torch.int64)]).to(f"npu:{dev}")
print(f"[{M},{U}] r={R} prep done", flush=True)

def fused(i):
    return torch_npu.npu_grouped_matmul_swiglu_quant(
        x=xq[i], weight=wq_nz, weight_scale=ws, x_scale=xs[i], group_list=cum)[0:2]

def unfused(i):
    z = torch_npu.npu_grouped_matmul(x=[xq[i]], weight=[wq_nz], split_item=3,
                                     group_list_type=0, group_type=0, group_list=cum,
                                     output_dtype=torch.int32)[0]
    return torch_npu.npu_dequant_swiglu_quant(x=z, weight_scale=ws, activation_scale=xs[i],
                                              group_index=cum, activate_left=True, quant_mode=1)

def timeit(fn, reps=25):
    fn(0); torch.npu.synchronize()
    ts = []
    for k in range(reps):
        torch.npu.synchronize(); t0 = time.perf_counter()
        fn(k % P)
        torch.npu.synchronize(); ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]

# numeric: fused vs unfused (whole M, dequant fp32)
yq_f, sq_f = fused(0); torch.npu.synchronize()
yq_u, sq_u = unfused(0); torch.npu.synchronize()
fp_f = yq_f.float() * sq_f.unsqueeze(1)
fp_u = yq_u.float() * sq_u.unsqueeze(1)
finite = bool(torch.isfinite(fp_f).all().item())
mae = float((fp_f - fp_u).abs().mean().item())
mae_scale = float(fp_u.abs().mean().item())

tf = timeit(fused); tu = timeit(unfused)
out = {"M": M, "U": U, "rows": R, "finite": finite, "fused_vs_unfused_mae": mae,
       "ref_absmean": mae_scale, "fused_ms": round(tf * 1e3, 4), "unfused_ms": round(tu * 1e3, 4),
       "ratio": round(tf / tu, 4)}
print("RESULT " + json.dumps(out), flush=True)
