"""Backend measurement for one or more real layers: eager | inductor-default | inductor-dvm.
For 'dvm', launch with TORCHINDUCTOR_NPU_BACKEND=dvm already exported (before import torch).
Also mode 'full' streams all 40 layers eager (whole-net) and reports totals.
Usage: python -u backend_measure.py --mode layer|full --idx 0,3 --seq 64 --backend triton --dev 0
"""
import os, sys, time, json, argparse
import torch, torch_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_model import LayerLoader, make_layer, make_rotary, causal_mask_for, get_text_config

p = argparse.ArgumentParser()
p.add_argument("--mode", default="layer")      # layer | full
p.add_argument("--idx", default="0,3")
p.add_argument("--seq", type=int, default=64)
p.add_argument("--backend", default="eager")    # eager | triton | dvm
p.add_argument("--dev", type=int, default=0)
p.add_argument("--iters", type=int, default=7)
p.add_argument("--out", default=None)
a = p.parse_args()
torch.npu.set_device(a.dev)
print("backend =", a.backend, "env =", os.environ.get("TORCHINDUCTOR_NPU_BACKEND", ""), "torch_npu =", torch_npu.__version__, flush=True)
tc = get_text_config()
loader = LayerLoader()
H, NL = tc.hidden_size, tc.num_hidden_layers
rotary = make_rotary(tc, device=a.dev, loader=loader)

def timed(fn, n):
    fn(); torch.npu.synchronize()
    ts = []
    for _ in range(n):
        torch.npu.synchronize(); t0 = time.time()
        fn(); torch.npu.synchronize(); ts.append((time.time() - t0) * 1000)
    ts.sort(); return ts[len(ts) // 2], ts

def layer_runner(idx, seq, compile_it, backend):
    layer = make_layer(idx, tc, loader=loader, device=a.dev); layer.eval()
    x = torch.randn(1, seq, H, dtype=torch.bfloat16, device=a.dev) * 0.02
    pos = torch.arange(seq, device=a.dev)[None].expand(1, seq)
    pe = rotary(x, pos)
    mask = None if tc.layer_types[idx] == "linear_attention" else causal_mask_for(tc, x, pos)
    kw = dict(position_embeddings=pe, attention_mask=mask, position_ids=pos,
              past_key_values=None, use_cache=False)
    if not compile_it:
        fn = lambda: layer(x, **kw)
        return layer, fn
    import torch._inductor.config as ic
    if backend == "dvm":
        ic.npu_backend = "dvm"
    else:
        ic.npu_backend = "default"
    cf = torch.compile(layer, dynamic=False)
    return layer, lambda: cf(x, **kw)

res = dict(backend=a.backend, seq=a.seq, torch_npu=torch_npu.__version__)

if a.mode == "full":
    # whole-net eager streaming
    from eager_stream import run_pass_ctx  # not available; inline minimal via per-layer build
    # reuse simpler: run each layer eager sequentially
    x = torch.randn(1, a.seq, H, dtype=torch.bfloat16, device=a.dev) * 0.02
    pos = torch.arange(a.seq, device=a.dev)[None].expand(1, a.seq)
    pe = rotary(x, pos)
    per = {}
    hidden = x
    import copy
    for i in range(NL):
        if tc.layer_types[i] == "full_attention":
            mask = causal_mask_for(tc, hidden, pos)
        else:
            mask = None
        layer = make_layer(i, tc, loader=loader, device=a.dev)
        with torch.no_grad():
            torch.npu.synchronize(); t0 = time.time()
            hidden = layer(hidden, position_embeddings=pe, attention_mask=mask,
                           position_ids=pos, past_key_values=None, use_cache=False)
            torch.npu.synchronize()
        per[str(i)] = round((time.time() - t0) * 1000, 2)
        del layer
        torch.npu.empty_cache()
    tot = sum(per.values())
    res.update(total_ms=round(tot, 1), per_layer_ms=per)
    print(json.dumps({k: v for k, v in res.items() if k != "per_layer_ms"}), flush=True)

else:
    for idx in [int(x) for x in a.idx.split(",")]:
        compile_it = a.backend in ("triton", "dvm")
        layer, fn = layer_runner(idx, a.seq, compile_it, a.backend)
        # eager reference once (for error check)
        e = layer if a.backend == "eager" else None
        med, ts = timed(fn, a.iters)
        d = dict(idx=idx, layer_type=tc.layer_types[idx], median_ms=round(med, 2),
                 runs_ms=[round(v, 2) for v in ts])
        res[f"layer_{idx}"] = d
        print(f"  layer {idx} {tc.layer_types[idx]}: median {med:.1f} ms", flush=True)
        del layer
        torch.npu.empty_cache()

print(json.dumps(res, indent=1), flush=True)
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
    print("saved", a.out, flush=True)
print("BM DONE", flush=True)
