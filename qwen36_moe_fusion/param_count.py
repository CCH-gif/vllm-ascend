# -*- coding: utf-8 -*-
"""Scan Qwen3.6-35B-A3B safetensors headers -> real per-component param counts.
No NPU / no weight load; reads only the 8-byte len + json header of each shard.
Fast and deterministic. Grounds the network-level extrapolation.
"""
import glob, json, os, sys

MODEL_DIR = "/data/models/Qwen3.6-35B-A3B"
TAGS = [  # (substring, tag)
    ("experts", "expert"),            # routed MoE expert weights (gate_up/down)
    ("shared_expert", "shared_exp"),
    ("router", "router"),
    ("q_proj", "attn"), ("k_proj", "attn"), ("v_proj", "attn"), ("o_proj", "attn"),
    ("key_proj", "lattn"), ("value_proj", "lattn"), ("in_proj", "lattn"),
    ("conv1d", "lattn"), ("dt_proj", "lattn"), ("ssm", "lattn"),
    ("A_log", "lattn"), ("D", "lattn"),
    ("norm", "norm"), ("gate", "gate"),
    ("embed_tokens", "embed"), ("lm_head", "lm_head"),
    ("visual", "vision"),
]

def classify(name):
    for sub, tag in TAGS:
        if sub in name:
            return tag
    return "other"

def main():
    total = 0
    counts = {}
    files = sorted(glob.glob(os.path.join(MODEL_DIR, "model-*.safetensors")))
    print(f"shards: {len(files)}")
    by_tag = {}
    examples = {}
    for fp in files:
        with open(fp, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            hdr = json.loads(f.read(n))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            nel = 1
            for d in v.get("shape", []):
                nel *= d
            total += nel
            tag = classify(k)
            by_tag[tag] = by_tag.get(tag, 0) + nel
            if tag not in examples:
                examples[tag] = k
    print(f"\nTOTAL params: {total/1e9:.4f} G   ({total*2/1e9:.1f} GB bf16)")
    for tag in sorted(by_tag, key=lambda t: -by_tag[t]):
        print(f"  {tag:12s} {by_tag[tag]/1e9:8.3f} G   {by_tag[tag]/total*100:5.1f}%   e.g. {examples[tag]}")
    # routed-expert per-expert split
    with open(files[0], "rb") as f:
        n = int.from_bytes(f.read(8), "little"); hdr0 = json.loads(f.read(n))
    exp_keys = [k for k in hdr0 if "experts" in k]
    print(f"\nfirst-shard expert-like keys sample:")
    for k in exp_keys[:6]:
        nel = 1
        for d in hdr0[k]["shape"]: nel *= d
        print(f"   {k}  {hdr0[k]['shape']}  {nel/1e6:.1f}M")

if __name__ == "__main__":
    main()
