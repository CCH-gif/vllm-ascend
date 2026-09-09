"""Concurrency sweep: throughput + latency at C=1/4/8/16/32, single serve.

Fixed-length template prompts (identical token count across requests -> one
prefill shape).  Each request streams up to MAX_TOKENS (no natural EOS on the
template, so every request emits exactly MAX_TOKENS tokens).  Output per point:
aggregate tok/s (reqs*MAX_TOKENS / wall), TTFT median/p90, per-request e2e
median, TPOT est.  Run points sequentially, clients in parallel.

Usage: python3 bench_concurrency.py <port> <out.json>
"""
import json
import sys
import threading
import time
import urllib.request

MAX_TOKENS = 128
CONCS = [1, 4, 8, 16, 32]
REQS_PER_WORKER = 6

# 8 fixed-length template prompts with single-digit ids -> same token count.
_BODY = ("Explain the underlying mechanisms and the main factors that drive the "
         "process of %s. Give a step-by-step account of how it works and which "
         "conditions matter most. Be concrete and avoid generic statements. "
         "Task number %d.")
_TOPICS = ["photosynthesis in green plants on land", "rainfall formation inside a cloud",
           "urban heat island intensity at night", "muscle contraction during exercise",
           "rust formation on exposed steel surfaces", "seed germination under drought",
           "the water cycle in a closed basin", "sound travel through layered air"]
PROMPTS = [_BODY % (t, i + 1) for i, t in enumerate(_TOPICS)]

lock = threading.Lock()


def stream(prompt):
    """Return (text_len, ttft_s, e2e_s) for one streaming request."""
    body = {"model": "openscad", "prompt": prompt, "max_tokens": MAX_TOKENS,
            "temperature": 0, "seed": 42, "stream": True}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    ttft = None
    n = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            s = line.decode(errors="replace").strip()
            if not s.startswith("data: "):
                continue
            if s[6:] == "[DONE]":
                break
            try:
                txt = json.loads(s[6:])["choices"][0].get("text", "")
            except Exception:
                continue
            now = time.monotonic()
            if ttft is None:
                ttft = now - t0
            n += len(txt)
    return n, ttft, time.monotonic() - t0


def worker(c, results, start):
    for k in range(REQS_PER_WORKER):
        p = PROMPTS[(c * REQS_PER_WORKER + k) % len(PROMPTS)]
        try:
            ln, ttft, e2e = stream(p)
            with lock:
                results.append({"worker": c, "ttft_s": ttft, "e2e_s": e2e,
                                "ch": ln})
        except Exception as e:
            with lock:
                results.append({"worker": c, "err": str(e)[:120]})


def sweep(port, out_path):
    global BASE
    BASE = f"http://127.0.0.1:{port}/v1/completions"
    report = {"port": port, "max_tokens": MAX_TOKENS,
              "reqs_per_worker": REQS_PER_WORKER, "points": []}
    for C in CONCS:
        results = []
        t0 = time.monotonic()
        threads = [threading.Thread(target=worker, args=(c, results, t0))
                   for c in range(C)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.monotonic() - t0
        ok = [r for r in results if "err" not in r]
        ntok = len(ok) * MAX_TOKENS
        toks = sorted(r["ttft_s"] for r in ok)
        e2e = sorted(r["e2e_s"] for r in ok)
        med = lambda xs: xs[len(xs) // 2] if xs else None
        p90 = lambda xs: xs[int(len(xs) * 0.9)] if xs else None
        pt = {"conc": C, "reqs": len(results), "ok": len(ok), "wall_s": round(wall, 1),
              "tok_s": round(ntok / wall, 1) if ok else 0,
              "ttft_med_ms": round(med(toks) * 1e3, 1) if toks else None,
              "ttft_p90_ms": round(p90(toks) * 1e3, 1) if toks else None,
              "e2e_med_ms": round(med(e2e) * 1e3, 1) if e2e else None}
        if ok:
            mid = e2e[len(e2e) // 2]
            tt = toks[len(toks) // 2]
            pt["tpot_est_ms"] = round((mid - tt) / MAX_TOKENS * 1e3, 1)
            pt["per_seq_tok_s"] = round(MAX_TOKENS / (mid - tt), 1) if mid > tt else None
        report["points"].append(pt)
        print(f"  C={C}: {pt}", flush=True)
        json.dump(report, open(out_path, "w"), indent=1)
    print(f"DONE -> {out_path}")


if __name__ == "__main__":
    sweep(sys.argv[1], sys.argv[2])
