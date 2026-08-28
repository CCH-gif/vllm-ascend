"""3-dataset LoRA A/B eval: TTFT/e2e + outputs via /v1/completions.

Usage: python3 eval_lora.py <gsm8k|humaneval|mmlu> <out.json> [max_tokens]
"""
import csv
import json
import sys
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:7519/v1/completions"
K = 4


def load_gsm8k():
    out = []
    for line in open("/tmp/evals/gsm8k_test.jsonl"):
        d = json.loads(line)
        out.append({"q": f"Question: {d['question']}\nAnswer:", "ref": d["answer"]})
    return out


def load_humaneval():
    out = []
    for line in open("/tmp/evals/humaneval_test.jsonl"):
        d = json.loads(line)
        out.append({"q": d["prompt"], "ref": None})
    return out


def load_mmlu():
    subjects = ["abstract_algebra", "college_physics", "computer_security",
                "high_school_geography", "professional_psychology"]
    out = []
    for s in subjects:
        rows = list(csv.DictReader(open(f"/tmp/evals/mmlu/test/{s}_test.csv")))
        for r in rows[:100]:
            out.append({"q": (f"Question: {r['Question']}\n"
                              f"A. {r['A']}\nB. {r['B']}\nC. {r['C']}\nD. {r['D']}\n"
                              f"Answer:"),
                        "ref": r["Answer"]})
    return out


def complete(prompt, max_tokens):
    body = {"model": "openscad", "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0, "seed": 42, "stream": True}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=600)
    ttft = None
    last = None
    text = ""
    for raw in resp:
        line = raw.decode(errors="replace").strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        now = time.perf_counter()
        if ttft is None:
            ttft = now - t0
        try:
            text += json.loads(payload)["choices"][0].get("text", "")
        except Exception:
            pass
        last = now
    return text, ttft, last - t0 if last else None


def worker(items, results, start, max_tokens):
    for i in range(start, len(items), K):
        try:
            text, ttft, e2e = complete(items[i]["q"], max_tokens)
            results[i] = {"ok": True, "text": text,
                          "ttft_ms": round(ttft * 1e3, 1),
                          "e2e_ms": round(e2e * 1e3, 1)}
        except Exception as e:
            results[i] = {"ok": False, "err": str(e)[:200]}
        done = sum(1 for r in results if r)
        if done % 20 == 0 or done == len(items):
            print(f"  {done}/{len(items)}", flush=True)


def main():
    dataset, out_path = sys.argv[1], sys.argv[2]
    max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    items = {"gsm8k": load_gsm8k, "humaneval": load_humaneval,
             "mmlu": load_mmlu}[dataset]()
    results = [None] * len(items)
    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(items, results, k, max_tokens))
               for k in range(K)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    report = {"dataset": dataset, "n": len(items), "max_tokens": max_tokens,
              "wall_s": round(wall, 1),
              "requests": [{"id": i, "ref": items[i]["ref"], **results[i]}
                           for i in range(len(items))]}
    json.dump(report, open(out_path, "w"), indent=1, ensure_ascii=False)
    ok = sum(1 for r in results if r and r["ok"])
    print(f"DONE {dataset}: {ok}/{len(items)} ok, {wall:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
