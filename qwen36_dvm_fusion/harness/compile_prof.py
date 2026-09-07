"""Profile one real layer: eager op/kernel inventory vs compiled (inductor) kernel count + time.
Backend 'default'=triton-inductor, 'dvm'=DVM. Run under proper stack python + env.
Usage: python -u compile_prof.py --idx 3 --dev 0 --seq 64 [--compile 1 --backend default|dvm]
"""
import os, sys, time, json, argparse
import torch, torch_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_model import LayerLoader, make_layer, make_rotary, causal_mask_for, get_text_config

p = argparse.ArgumentParser()
p.add_argument("--idx", type=int, default=3)
p.add_argument("--dev", type=int, default=0)
p.add_argument("--seq", type=int, default=64)
p.add_argument("--compile", type=int, default=1)
p.add_argument("--backend", default="default")   # default=triton | dvm
p.add_argument("--iters", type=int, default=5)
a = p.parse_args()
torch.npu.set_device(a.dev)
if a.backend == "dvm":
    os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "dvm"   # must be set before import in real runs
tc = get_text_config()
loader = LayerLoader()
layer = make_layer(a.idx, tc, loader=loader, device=a.dev)
layer.eval()
H = tc.hidden_size
x = torch.randn(1, a.seq, H, dtype=torch.bfloat16, device=a.dev) * 0.02
pos = torch.arange(a.seq, device=a.dev)[None].expand(1, a.seq)
rotary = make_rotary(tc, device=a.dev, loader=loader)
pe = rotary(x, pos)
mask = None if tc.layer_types[a.idx] == "linear_attention" else causal_mask_for(tc, x, pos)

def run():
    with torch.no_grad():
        return layer(x, position_embeddings=pe, attention_mask=mask,
                     position_ids=pos, past_key_values=None, use_cache=False)

NOISE = ("_record_function", "Optimizer", "profile", "Memcpy", "Memset", "cudaLaunch")
def cpu_keys(prof):
    ks = {}
    for e in prof.key_averages():
        k = e.key
        if any(n in k for n in NOISE):
            continue
        ks[k] = ks.get(k, 0) + e.count
    return ks

def npu_keys(prof):
    ks = {}
    for e in prof.key_averages():
        if e.device_type == torch.autograd.DeviceType.NPU:
            k = e.key
            ks[k] = ks.get(k, 0) + e.count
    return ks

from torch.profiler import profile, ProfilerActivity
def timed_fn(fn, n=5):
    fn(); torch.npu.synchronize()
    ts = []
    for _ in range(n):
        torch.npu.synchronize(); t0 = time.time()
        fn(); torch.npu.synchronize(); ts.append((time.time() - t0) * 1000)
    ts.sort(); return round(ts[len(ts) // 2], 2)

def aclnn_of(ks):
    return sum(v for k, v in ks.items() if k.startswith("aclnn"))

# eager: time + profile (CPU only; device kernels proxy = aclnn* dispatch)
em = timed_fn(run, a.iters)
torch.npu.synchronize()
with profile(activities=[ProfilerActivity.CPU]) as prof:
    run(); torch.npu.synchronize()
cpu = cpu_keys(prof)
total_ops = sum(cpu.values()); acl = aclnn_of(cpu)
print("eager: median %.2f ms | CPU dispatch %d | aclnn(dvk) %d (%d types)" %
      (em, total_ops, acl, sum(1 for k in cpu if k.startswith('aclnn'))), flush=True)

out = dict(idx=a.idx, layer_type=tc.layer_types[a.idx], seq=a.seq, backend=a.backend,
           eager_median_ms=em,
           eager_cpu_dispatch=total_ops,
           eager_aclnn_ops=acl,
           eager_cpu_samples=[k for k in sorted(cpu, key=cpu.get, reverse=True)][:30])

if a.compile:
    torch._inductor.config.debug = False
    try:
        cf = torch.compile(layer, dynamic=False)
        t0 = time.time(); cf(x, position_embeddings=pe, attention_mask=mask,
                             position_ids=pos, past_key_values=None, use_cache=False)
        torch.npu.synchronize(); print("compile time %.1fs" % (time.time() - t0), flush=True)
    except Exception as e:
        print("COMPILE FAILED:", repr(e)[:500], flush=True)
        out["compile_error"] = repr(e)[:300]
    else:
        # warm + timed iters + profile kernels
        cf(x, position_embeddings=pe, attention_mask=mask, position_ids=pos, past_key_values=None, use_cache=False)
        torch.npu.synchronize()
        ts = []
        for _ in range(a.iters):
            torch.npu.synchronize(); t0 = time.time()
            cf(x, position_embeddings=pe, attention_mask=mask, position_ids=pos, past_key_values=None, use_cache=False)
            torch.npu.synchronize(); ts.append((time.time() - t0) * 1000)
        ts.sort()
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            cf(x, position_embeddings=pe, attention_mask=mask, position_ids=pos, past_key_values=None, use_cache=False)
            torch.npu.synchronize()
        ck = cpu_keys(prof)
        total_k = sum(ck.values()); cal = aclnn_of(ck)
        print("compiled: median %.2f ms | CPU dispatch %d | aclnn(dvk) %d" %
              (ts[len(ts)//2], total_k, cal), flush=True)
        out.update(compiled_median_ms=round(ts[len(ts) // 2], 2),
                   compiled_cpu_dispatch=total_k, compiled_aclnn_ops=cal,
                   compiled_cpu_samples=[k for k in sorted(ck, key=ck.get, reverse=True)][:30])
print(json.dumps({k: v for k, v in out.items() if not isinstance(v, list)}, indent=1), flush=True)
with open(f"/workspace/ascend-operator-agent-v2-gitcode6.18/qwen36_dvm_fusion/results/prof_{a.backend}_idx{a.idx}_s{a.seq}.json", "w") as f:
    json.dump(out, f, indent=1)
print("PROF DONE", flush=True)
