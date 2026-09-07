"""Build real layer (linear=0, full=3) from safetensors via meta loader, forward on NPU.
Validates npu execution + loader speed for both layer types. [8.5 stack py3.11 / DVM venv both]
Usage: python -u 02_layer_check.py [seq]
"""
import os, sys, time, torch, torch_npu
torch.npu.set_device(int(os.environ.get("NPU_DEV", "0")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_model import LayerLoader, make_layer, make_rotary, causal_mask_for, get_text_config

seq = int(sys.argv[1]) if len(sys.argv) > 1 else 256
tc = get_text_config()
loader = LayerLoader()
H = tc.hidden_size

for idx, want in [(0, "linear_attention"), (3, "full_attention")]:
    t0 = time.time()
    layer = make_layer(idx, tc, loader=loader, device="npu")   # includes weight fill
    torch.npu.synchronize()
    print(f"layer {idx} ({want}) built+filled in {time.time()-t0:.1f}s", flush=True)
    hidden = torch.randn(1, seq, H, dtype=torch.bfloat16, device="npu") * 0.02
    pos = torch.arange(seq, device="npu")[None].expand(1, seq)
    emb = make_rotary(tc, device="npu", loader=loader)
    pe = emb(hidden, pos)
    mask = None if want == "linear_attention" else causal_mask_for(tc, hidden, pos)
    t0 = time.time()
    with torch.no_grad():
        out = layer(hidden, position_embeddings=pe, attention_mask=mask,
                    position_ids=pos, past_key_values=None, use_cache=False)
    torch.npu.synchronize()
    print(f"  fwd out {tuple(out.shape)} sum~{out.float().sum().item():.4f}  {(time.time()-t0)*1000:.0f} ms  [seq {seq}]", flush=True)
    del layer, out, hidden
    torch.npu.empty_cache()
print("CHECK OK", flush=True)
