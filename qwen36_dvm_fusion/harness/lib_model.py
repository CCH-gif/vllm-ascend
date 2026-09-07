"""Fast per-layer loader for Qwen3.6-35B-A3B (transformers qwen3_5_moe).

Builds any decoder layer (or shared params) on meta device then fills weights from
safetensors on demand. Avoids materializing the full 72GB model. Works for both the
8.5 stack (torch2.9) and the DVM venv (torch2.12).
"""
import os, json, torch
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer, Qwen3_5MoeTextRotaryEmbedding, create_causal_mask,
)

MODEL = "/data/models/Qwen3.6-35B-A3B"


def get_text_config():
    cfg = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    return cfg.text_config


def load_weight_map():
    idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))
    return idx.get("weight_map", {})


class LayerLoader:
    """Loads named params of one module from safetensors by full key prefix."""
    def __init__(self):
        self.wmap = load_weight_map()
        self._shards = {}

    def _shard(self, path):
        if path not in self._shards:
            from safetensors import safe_open
            self._shards[path] = safe_open(os.path.join(MODEL, path), framework="pt", device="cpu")
        return self._shards[path]

    def get(self, full_key, dtype=torch.bfloat16):
        path = self.wmap.get(full_key)
        if path is None:
            return None
        return self._shard(path).get_tensor(full_key).to(dtype)

    def fill(self, module, prefix="model.language_model.", dtype=torch.bfloat16):
        """Fill all parameters/buffers of `module` from safetensors. Returns missing keys."""
        missing = []
        for name, p in list(module.named_parameters()) + list(module.named_buffers(recurse=True)):
            key = prefix + name
            t = self.get(key, dtype)
            if t is None:
                missing.append(key)
                continue
            # p may be meta: create data tensor of matching shape/dtype/stridedness
            p.data = t.to(p.device).to(p.dtype) if t.dtype != p.dtype else t.to(p.device)
        return missing


def make_layer(layer_idx, config=None, loader=None, device="cpu", dtype=torch.bfloat16):
    """Construct decoder layer under bf16 default dtype, fill real weights if loader,
    then move to `device`. Construction on CPU is a few seconds/layer (random init) and
    fill() overwrites with the stored (bf16) weights so init values don't matter."""
    if config is None:
        config = get_text_config()
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        layer = Qwen3_5MoeDecoderLayer(config, layer_idx)   # on CPU
    finally:
        torch.set_default_dtype(prev)
    if loader is not None:
        loader.fill(layer)
    if device != "cpu":
        layer = layer.to(device)
    return layer


def make_rotary(config=None, device="cpu", dtype=torch.bfloat16, loader=None, layer_types_len=40):
    """Build the TextModel rotary embedding (weights from safetensors if stored)."""
    if config is None:
        config = get_text_config()
    with torch.device("cpu"):
        emb = Qwen3_5MoeTextRotaryEmbedding(config=config)
    if loader is not None:
        loader.fill(emb, prefix="model.language_model.rotary_emb.")
    emb = emb.to(dtype).to(device)
    return emb


def causal_mask_for(config, hidden, position_ids):
    am = torch.ones(1, hidden.shape[1], dtype=torch.bfloat16, device=hidden.device)
    return create_causal_mask(config=config, inputs_embeds=hidden,
                              attention_mask=am, past_key_values=None,
                              position_ids=position_ids)


def layer_io(idx, config):
    return dict(layer_type=config.layer_types[idx], hidden=config.hidden_size)
