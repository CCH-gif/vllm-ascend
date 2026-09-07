#!/usr/bin/env python3
"""Collect a version-independent fingerprint of the built-in ChunkGatedDeltaRule /
RecurrentGatedDeltaRule op assets under a CANN install root.

Usage: collect_op_fingerprint.py <cann_root> <op> [out.json]
  cann_root : e.g. /tmp/cann91/root/usr/local/Ascend/cann-9.1.0  (a dir whose
              layout has opp/built-in/... and include/aclnnop/... )
  op        : ChunkGatedDeltaRule | RecurrentGatedDeltaRule
  out.json  : optional output file (default prints to stdout)

Produces a manifest of sha256 of every located asset + the aclnn entry decl text
+ kernel json summary. Run on two CANN roots and diff the manifests.
Stdlib only. Be careful running with root the toolkit root itself.
"""
import hashlib
import json
import os
import re
import sys

OP = sys.argv[2] if len(sys.argv) > 2 else "ChunkGatedDeltaRule"
ROOT = sys.argv[1]
OUT = sys.argv[3] if len(sys.argv) > 3 else None
# candidate names (camel + snake) for both file & dir basenames
SNAKE = re.sub(r'(?<=[a-z])(?=[A-Z])', '_', OP).lower()          # ChunkGatedDeltaRule->chunk_gated_delta_rule
LEAF = OP  # dirs like kernel/ascend910b/ops_transformer/<OpName or snake>


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def exists(p):
    return p and os.path.exists(p)


def find_first(base_rel_dirs, name):
    for rel in base_rel_dirs:
        p = os.path.join(ROOT, rel)
        if exists(p):
            return p
    return None


files = {}

# ---- 1. aclnn host headers ----
for rel in [
    "aarch64-linux/include/aclnnop/aclnn_{s}.h".format(s=SNAKE),
    "include/aclnnop/aclnn_{s}.h".format(s=SNAKE),
    "aarch64-linux/include/aclnnop/level2/aclnn_{s}.h".format(s=SNAKE),
]:
    p = find_first([rel], None)
    if p is None:
        # fallback: scan include dirs for the basename
        for base in ["aarch64-linux/include", "include"]:
            for dirpath, _, fn in os.walk(os.path.join(ROOT, base)):
                for f in fn:
                    if f == "aclnn_{s}.h".format(s=SNAKE):
                        p = os.path.join(dirpath, f)
                        break
                if p:
                    break
            if p:
                break
    if p:
        key = "aclnn_header"
        files.setdefault(key, []).append(p)

# ---- 2. impl dirs (ascendc + dynamic) ----
for rel in [
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/{l}".format(l=LEAF),
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/{s}".format(s=SNAKE),
]:
    d = os.path.join(ROOT, rel)
    if os.path.isdir(d):
        for dirpath, _, fn in os.walk(d):
            for f in sorted(fn):
                files.setdefault("ascendc_src", []).append(os.path.join(dirpath, f))
        break

dyn_py = find_first([
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/dynamic/{s}.py".format(s=SNAKE),
], None)
if dyn_py:
    files.setdefault("dynamic_py", []).append(dyn_py)

# ---- 3. kernel config + per-op compiled kernel binaries ----
cfg = find_first([
    "opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend910b/ops_transformer/{l}.json".format(l=LEAF),
    "opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend910b/ops_transformer/{s}.json".format(s=SNAKE),
    "opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend910_93/ops_transformer/{l}.json".format(l=LEAF),
], None)
if cfg:
    files.setdefault("kernel_config_json", []).append(cfg)

# walk kernel arch dirs for the op binaries (.o/.json)
for arch in os.listdir(os.path.join(ROOT, "opp/built-in/op_impl/ai_core/tbe/kernel")):
    opsdir = os.path.join(ROOT, "opp/built-in/op_impl/ai_core/tbe/kernel", arch, "ops_transformer")
    for cand in (LEAF, SNAKE):
        d = os.path.join(opsdir, cand)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.startswith(OP) or f.lower().startswith(SNAKE):
                    files.setdefault("kernel_bin_" + arch, []).append(os.path.join(d, f))

# ---- 4. op registration entry in ops-info json (attributes/dtypes) ----
opsinfo = find_first([
    "opp/built-in/op_impl/ai_core/tbe/config/ascend910b/aic-ascend910b-ops-info-transformer.json",
    "opp/built-in/op_impl/ai_core/tbe/config/ascend910b/aic-ascend910b-ops-info.json",
], None)
# also allow recursive discovery
if not opsinfo:
    for dirpath, _, fn in os.walk(os.path.join(ROOT, "opp/built-in/op_impl/ai_core/tbe/config")):
        for f in fn:
            if "ops-info" in f and f.endswith(".json") and (LEAF in f):
                opsinfo = os.path.join(dirpath, f)
                break
        if opsinfo:
            break

manifest = {
    "op": OP,
    "root": ROOT,
    "version_hint": None,
    "files": {},
    "aclnn_decl": None,
    "kernel_summary": {},
}
# try version hint
for p in (os.path.join(ROOT, "version.cfg"),
          os.path.join(os.path.dirname(ROOT.rstrip("/")), "version.cfg")):
    if os.path.exists(p):
        try:
            with open(p) as f:
                for line in f:
                    if line.startswith("toolkit_running_version=") or line.startswith("opp_running_version="):
                        manifest["version_hint"] = line.strip()
                        break
        except Exception:
            pass
    if manifest.get("version_hint"):
        break

for kind, paths in files.items():
    manifest["files"][kind] = []
    for p in paths:
        manifest["files"][kind].append({
            "rel": os.path.relpath(p, ROOT),
            "sha256": sha(p),
            "size": os.path.getsize(p),
        })

# aclnn decl text (first/second-stage signatures) from header
hdr_paths = files.get("aclnn_header", [])
if hdr_paths:
    try:
        with open(hdr_paths[0], encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        decls = re.findall(
            r'(ACLNN_API|__attribute__\(\(visibility\("default"\)\)\))\s+aclnnStatus\s+'
            r'(aclnn{op}GetWorkspaceSize|aclnn{op})\s*\([^;]*?\)\s*;'.format(op=OP), txt)
        sigs = []
        for pre, name in decls:
            m = re.search(name + r'\s*\(', txt)
            # re-grab the whole statement body
            body = re.search(name + r'\([^;]*?\);', txt, re.S)
            if body:
                sigs.append(re.sub(r'\s+', ' ', body.group(0)).strip())
        manifest["aclnn_decl"] = sigs
    except Exception as e:
        manifest["aclnn_decl"] = "ERR " + str(e)

# kernel json summary: capture bin hash, sha256, staticKey, coreType, kernelList, inputs/outputs count
for kind, arr in manifest["files"].items():
    if not kind.startswith("kernel_bin_"):
        continue
    for e in arr:
        if e["rel"].endswith(".json"):
            try:
                with open(os.path.join(ROOT, e["rel"])) as f:
                    j = json.load(f)
                summary = {
                    "bin": j.get("binFileName"),
                    "o_sha256": j.get("sha256"),
                    "staticKey": j.get("supportInfo", {}).get("staticKey"),
                    "coreType": j.get("coreType"),
                    "magic": j.get("magic"),
                    "implMode": j.get("supportInfo", {}).get("implMode"),
                    "kernelList": j.get("kernelList"),
                    "n_inputs": len(j.get("supportInfo", {}).get("inputs", [])),
                    "n_outputs": len(j.get("supportInfo", {}).get("outputs", [])),
                    "attrs": [a.get("name") for a in j.get("supportInfo", {}).get("attrs", [])],
                    "input_sig": [(i.get("name"), i.get("dtype"), i.get("paramType"))
                                  for i in j.get("supportInfo", {}).get("inputs", [])],
                }
                manifest["kernel_summary"][e["rel"]] = summary
            except Exception as ex:
                manifest["kernel_summary"][e["rel"]] = "ERR " + str(ex)

if OUT:
    with open(OUT, "w") as f:
        json.dump(manifest, f, indent=1)
    print("wrote", OUT)
else:
    print(json.dumps(manifest, indent=1))
