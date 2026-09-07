"""Profile a submodule (dense straight-line chain) of a real decoder layer: eager vs compiled(DVM).
Usage: source dvm_env; TORCHINDUCTOR_NPU_BACKEND=dvm python -u sub_prof.py --idx 3 --sub mlp.shared_expert --seq 512 --dev 0
"""
import os, sys, time, json, argparse
import torch, torch_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_model import LayerLoader, make_layer, get_text_config

p = argparse.ArgumentParser()
p.add_argument("--idx", type=int, default=3)
p.add_argument("--sub", default="mlp.shared_expert")
p.add_argument("--seq", type=int, default=512)
p.add_argument("--compile", type=int, default=1)
p.add_argument("--backend", default="dvm")
p.add_argument("--dev", type=int, default=0)
p.add_argument("--iters", type=int, default=7)
a = p.parse_args()
torch.npu.set_device(a.dev)
tc = get_text_config()
loader = LayerLoader()
layer = make_layer(a.idx, tc, loader=loader, device=a.dev)
mod = layer
for s in a.sub.split("."):
    mod = getattr(mod, s)
mod = mod.eval()
x = torch.randn(a.seq, tc.hidden_size, dtype=torch.bfloat16, device=a.dev) * 0.02

def timed(fn, n):
    fn(); torch.npu.synchronize(); ts = []
    for _ in range(n):
        torch.npu.synchronize(); t0 = time.time()
        fn(); torch.npu.synchronize(); ts.append((time.time()-t0)*1000)
    ts.sort(); return ts[len(ts)//2]

from torch.profiler import profile, ProfilerActivity
def count_aclnn(fn):
    with profile(activities=[ProfilerActivity.CPU]) as pr:
        fn(); torch.npu.synchronize()
    c = 0
    for e in pr.key_averages():
        if e.key.startswith("aclnn"):
            c += e.count
    return c

with torch.no_grad():
    med_e = timed(lambda: mod(x), a.iters)
aclnn_e = count_aclnn(lambda: mod(x))
print(f"eager median {med_e:.3f} ms | aclnn ops {aclnn_e}", flush=True)
res = dict(idx=a.idx, sub=a.sub, seq=a.seq, eager_median_ms=round(med_e, 3),
           eager_aclnn_ops=aclnn_e)
if a.compile:
    cf = torch.compile(mod, dynamic=False)
    t0 = time.time()
    with torch.no_grad():
        cf(x)
    torch.npu.synchronize()
    print("compile first-call %.1fs" % (time.time()-t0), flush=True)
    with torch.no_grad():
        med_c = timed(lambda: cf(x), a.iters)
    aclnn_c = count_aclnn(lambda: cf(x))
    print(f"compiled median {med_c:.3f} ms | aclnn ops {aclnn_c} | ratio(e/c) {med_e/med_c:.2f}", flush=True)
    res.update(compiled_median_ms=round(med_c, 3), speedup_x=round(med_e/med_c, 3),
               compiled_aclnn_ops=aclnn_c)
print(json.dumps(res), flush=True)
with open(f"/workspace/ascend-operator-agent-v2-gitcode6.18/qwen36_dvm_fusion/results/sub_{a.backend}_idx{a.idx}_{a.sub.replace('.','_')}_s{a.seq}.json", "w") as f:
    json.dump(res, f, indent=1)
print("SUB DONE", flush=True)
