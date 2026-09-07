"""Whole-net (40 real layers) eager streaming prefill on NPU, weights streamed per layer.
Loads each decoder layer from safetensors (bf16), runs on NPU, frees. Per-layer times,
saves real hidden states at chosen layers for later compiled-layer runs. Output json.
Usage: python -u eager_stream.py --seq 256 --dev 3 --out ../results/eager_p256.json --reps 0,3
"""
import os, sys, time, json, argparse
import torch, torch_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_model import LayerLoader, make_layer, make_rotary, causal_mask_for, get_text_config

p = argparse.ArgumentParser()
p.add_argument("--seq", type=int, default=256)
p.add_argument("--dev", type=int, default=3)
p.add_argument("--out", default=None)
p.add_argument("--states", default=None)
p.add_argument("--reps", default="0,3")
p.add_argument("--passes", type=int, default=2)
a = p.parse_args()
torch.npu.set_device(a.dev)
tc = get_text_config()
H, NL = tc.hidden_size, tc.num_hidden_layers
reps = [int(x) for x in a.reps.split(",")]

loader = LayerLoader()
emb_w = loader.get("model.language_model.embed_tokens.weight")  # bf16 cpu
print("embed_tokens", tuple(emb_w.shape), flush=True)
emb_w = emb_w.to(a.dev)
rotary = make_rotary(tc, device=a.dev, loader=loader)
tokens = torch.randint(0, emb_w.shape[0], (1, a.seq), device=a.dev)
hidden0 = torch.nn.functional.embedding(tokens, emb_w).to(torch.bfloat16)  # [1,S,H]
pos = torch.arange(a.seq, device=a.dev)[None].expand(1, a.seq)
pe = rotary(hidden0, pos)
am = torch.ones(1, a.seq, dtype=torch.bfloat16, device=a.dev)
full_causal = {i: causal_mask_for(tc, hidden0, pos) for i in range(NL) if tc.layer_types[i] == "full_attention"}

def run_pass(save_states=False):
    hidden = hidden0.clone()
    per = {}
    states = {}
    for i in range(NL):
        if save_states and i in reps:
            states[str(i)] = hidden.detach().float().cpu()  # real input to layer i
        layer = make_layer(i, tc, loader=loader, device=a.dev)
        mask = full_causal[i] if tc.layer_types[i] == "full_attention" else None
        torch.npu.synchronize()
        t0 = time.time()
        with torch.no_grad():
            hidden = layer(hidden, position_embeddings=pe, attention_mask=mask,
                           position_ids=pos, past_key_values=None, use_cache=False)
        torch.npu.synchronize()
        per[str(i)] = round((time.time() - t0) * 1000, 2)
        del layer
        torch.npu.empty_cache()
    return per, states, hidden

# warm (single quick pass at same seq to settle allocator), keep states off warmup
print("warm pass ...", flush=True)
per_w, _, _ = run_pass(save_states=False)
print("warm total %.1fs" % (sum(per_w.values()) / 1000), flush=True)
per_best = {k: float("inf") for k in per_w}
states = {}
for r in range(a.passes):
    per, st, _ = run_pass(save_states=(r == a.passes - 1))
    for k, v in per.items():
        per_best[k] = min(per_best[k], v)
    if r == a.passes - 1:
        states = st
    print(f"pass {r} total {sum(per.values())/1000:.1f}s", flush=True)

tot = sum(per_best.values())
lin = [per_best[str(i)] for i in range(NL) if tc.layer_types[i] == "linear_attention"]
ful = [per_best[str(i)] for i in range(NL) if tc.layer_types[i] == "full_attention"]
res = dict(seq=a.seq, device=a.dev, per_layer_ms=per_best, total_ms=round(tot,1),
           linear_ms=round(sum(lin),1), linear_n=len(lin), full_ms=round(sum(ful),1), full_n=len(ful),
           stack=os.environ.get("STACK", "85"), wall_passes=a.passes)
print(json.dumps({k: v for k, v in res.items() if k != "per_layer_ms"}, indent=1), flush=True)
if a.out:
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("saved", a.out, flush=True)
if a.states and states:
    os.makedirs(os.path.dirname(a.states) or ".", exist_ok=True)
    torch.save(states, a.states)
    print("saved states", a.states, [k for k in states], flush=True)
print("EAGER DONE", flush=True)
