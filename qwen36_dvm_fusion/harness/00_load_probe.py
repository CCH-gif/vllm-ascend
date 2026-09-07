"""WP1 probe: load real Qwen3.6-35B-A3B on CPU, stream a couple of real layers to NPU, run a forward.
Validates: model loads, layer modules run on NPU under torch_npu 2.9/torch 2.9.
Usage: LD_LIBRARY_PATH=... python -u 00_load_probe.py [layer_idx...]  (device 0)
"""
import os, sys, time, json
import torch
import torch_npu

MODEL = "/data/models/Qwen3.6-35B-A3B"
DEV = int(os.environ.get("NPU_DEV", "0"))
torch.npu.set_device(DEV)

from transformers import AutoConfig, AutoModel

cfg = AutoConfig.from_pretrained(MODEL, local_files_only=True)
print("arch", cfg.architectures, "model_type", cfg.model_type, flush=True)
tc = cfg.text_config
print("text layers", tc.num_hidden_layers, "layer_types len", len(tc.layer_types),
      "num_experts", tc.num_experts, "topk", tc.num_experts_per_tok,
      "moe_inter", tc.moe_intermediate_size, "head_dim", tc.head_dim, flush=True)

t0 = time.time()
print("loading (cpu, bf16, low_cpu_mem)...", flush=True)
model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                  low_cpu_mem_usage=True, local_files_only=True)
print("loaded in %.1fs" % (time.time() - t0), flush=True)

text = model.language_model
print("text module keys sample:", [n for n, _ in text.named_parameters()][:3], flush=True)
# free vision tower (not needed)
for m in (model.visual,):
    m.cpu(); del m
torch.cuda.empty_cache() if hasattr(torch, "cuda") else None

def run_one_layer(lidx, seq=8):
    layer = text.layers[lidx]
    n_par = sum(p.numel() for p in layer.parameters())
    print(f"layer {lidx} type={layer.layer_type} params={n_par/1e6:.1f}M", flush=True)
    layer = layer.to(DEV)
    hidden = torch.randn(1, seq, tc.hidden_size, dtype=torch.bfloat16, device=DEV) * 0.02
    # rotary embeddings (depend only on seq len & device)
    pos_ids = torch.arange(seq, device=DEV)[None, :].expand(tc.num_attention_heads, -1) if False else None
    text_position_ids = torch.arange(seq, device=DEV)[None].expand(1, seq)
    position_embeddings = text.rotary_emb(hidden, text_position_ids)
    attn_mask = None
    if layer.layer_type == "full_attention":
        # causal mask identical to TextModel.create_causal_mask with no cache / full attend
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import create_causal_mask
        am = torch.ones(1, seq, dtype=torch.bfloat16, device=DEV)
        attn_mask = create_causal_mask(config=tc, inputs_embeds=hidden,
                                       attention_mask=am, past_key_values=None,
                                       position_ids=text_position_ids)
    t = time.time()
    out = layer(hidden, position_embeddings=position_embeddings,
                attention_mask=attn_mask, position_ids=text_position_ids,
                past_key_values=None, use_cache=False)
    torch.npu.synchronize()
    dt = (time.time() - t) * 1000
    print(f"  layer {lidx} fwd out {tuple(out.shape)} sum~{out.float().sum().item():.3f}  {dt:.1f} ms", flush=True)
    del out, layer
    torch.npu.empty_cache()

idxs = [int(a) for a in sys.argv[1:]] or [0, 3, 7]
for i in idxs:
    run_one_layer(i)
print("PROBE OK", flush=True)
