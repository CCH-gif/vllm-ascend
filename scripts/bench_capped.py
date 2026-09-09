"""Oversubscription fix demo: same total load as a collapse point, but in-flight
capped by a semaphore so vLLM never sees offered > resident capacity.

Usage: python3 bench_capped.py <port> <out.json> <workers> <cap> [max_tokens]
WORKERS logical clients each send REQS_PER_WORKER sequentially; a global
semaphore of CAP gates how many are in-flight concurrently.
"""
import json
import sys
import threading
import time
import urllib.request

REQS_PER_WORKER = 6

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
SEM = None
MAX_TOKENS = 128


def stream(prompt):
    body = {"model": "openscad", "prompt": prompt, "max_tokens": MAX_TOKENS,
            "temperature": 0, "seed": 42, "stream": True}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    ttft = None
    n = 0
    with urllib.request.urlopen(req, timeout=900) as r:
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


def worker(c, results):
    for k in range(REQS_PER_WORKER):
        p = PROMPTS[(c * REQS_PER_WORKER + k) % len(PROMPTS)]
        with SEM:
            try:
                ln, ttft, e2e = stream(p)
                with lock:
                    results.append({"worker": c, "ttft_s": ttft, "e2e_s": e2e, "ch": ln})
            except Exception as e:
                with lock:
                    results.append({"worker": c, "err": str(e)[:120]})


def run(port, out_path, workers, cap, max_tokens):
    global BASE, SEM, MAX_TOKENS
    BASE = f"http://127.0.0.1:{port}/v1/completions"
    SEM = threading.BoundedSemaphore(cap)
    MAX_TOKENS = max_tokens
    results = []
    t0 = time.monotonic()
    threads = [threading.Thread(target=worker, args=(c, results)) for c in range(workers)]
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
    pt = {"workers": workers, "cap": cap, "reqs": len(results), "ok": len(ok),
          "wall_s": round(wall, 1), "tok_s": round(ntok / wall, 1) if ok else 0,
          "ttft_med_ms": round(med(toks) * 1e3, 1) if toks else None,
          "ttft_p90_ms": round(toks[int(len(toks) * 0.9)] * 1e3, 1) if toks else None,
          "e2e_med_ms": round(med(e2e) * 1e3, 1) if e2e else None}
    if ok:
        mid = e2e[len(e2e) // 2]
        tt = toks[len(toks) // 2]
        pt["tpot_est_ms"] = round((mid - tt) / MAX_TOKENS * 1e3, 1)
        pt["per_seq_tok_s"] = round(MAX_TOKENS / (mid - tt), 1) if mid > tt else None
    json.dump(pt, open(out_path, "w"), indent=1)
    print(json.dumps(pt, indent=1))


if __name__ == "__main__":
    port = sys.argv[1]
    out = sys.argv[2]
    workers = int(sys.argv[3])
    cap = int(sys.argv[4])
    mt = int(sys.argv[5]) if len(sys.argv) > 5 else 128
    run(port, out, workers, cap, mt)
