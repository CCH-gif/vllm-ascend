"""聚合 vLLM torch profiler 的 chrome trace，按内核名统计次数/总时长/平均时长。

用法: python3 trace_agg.py /tmp/e2e/profiling_base [更多目录...]

回答的核心问题：
  1. LoRA 内核（bgmv*/sgmv*）在 trace 里出现多少次 —— 是「每 decode step 288 次」
     （→ 没被 aclgraph 捕获，eager 重发），还是「固定很少的次数」（→ 被捕获了）。
  2. 它们在设备上的平均时长，与 AscendC 版本对比。
"""
import json
import os
import sys
from collections import defaultdict

LORA_HINT = ("bgmv", "sgmv", "lora", "Lora", "LoRA")


def load_events(path):
    """一个目录下可能有多个 .pt.trace.json；全并进来。"""
    evs = []
    for root, _dirs, files in os.walk(path):
        for f in files:
            if not f.endswith(".json"):
                continue
            p = os.path.join(root, f)
            try:
                with open(p) as fh:
                    data = json.load(fh)
            except Exception as e:  # noqa: BLE001
                print(f"  !! 读不动 {p}: {e}")
                continue
            got = data.get("traceEvents", data if isinstance(data, list) else [])
            print(f"  {p}: {len(got)} events")
            evs.extend(got)
    return evs


def summarize(path):
    print(f"\n{'='*78}\n### {path}\n{'='*78}")
    evs = load_events(path)
    if not evs:
        print("  (没有事件)")
        return None

    # 按 cat 统计
    by_cat = defaultdict(int)
    for e in evs:
        by_cat[e.get("cat", "?")] += 1
    print("  --- 事件按 cat ---")
    for c, n in sorted(by_cat.items(), key=lambda kv: -kv[1])[:12]:
        print(f"    {c:24s} {n}")

    # 内核事件：有 dur 的 X 阶段事件
    kern = defaultdict(lambda: [0, 0.0])  # name -> [count, total_us]
    for e in evs:
        if e.get("ph") != "X":
            continue
        d = e.get("dur")
        if d is None:
            continue
        cat = (e.get("cat") or "").lower()
        if "kernel" not in cat and "npu" not in cat and "ai" not in cat:
            continue
        k = kern[e.get("name", "?")]
        k[0] += 1
        k[1] += d

    lora = {n: v for n, v in kern.items() if any(h in n for h in LORA_HINT)}
    print(f"  --- LoRA 相关内核（共 {len(lora)} 种）---")
    if not lora:
        print("    !!! 一个都没有 —— profiler 没采到内核，或内核名不含 lora 关键字")
    tot_lora = 0
    for n, (c, t) in sorted(lora.items(), key=lambda kv: -kv[1][1]):
        print(f"    {n[:64]:64s} n={c:<7d} total={t/1000:9.2f}ms mean={t/c:8.2f}us")
        tot_lora += c
    print(f"    LoRA 内核事件总数 = {tot_lora}")

    top = sorted(kern.items(), key=lambda kv: -kv[1][1])[:15]
    print(f"  --- 设备时间 top15（共 {len(kern)} 种内核）---")
    tot_all = sum(t for _c, t in kern.values())
    for n, (c, t) in top:
        print(f"    {n[:56]:56s} n={c:<7d} total={t/1000:9.2f}ms ({t/tot_all*100:5.1f}%) mean={t/c:8.2f}us")
    print(f"    内核总设备时间 = {tot_all/1000:.2f}ms")
    return kern


if __name__ == "__main__":
    dirs = sys.argv[1:] or ["/tmp/e2e/profiling_base", "/tmp/e2e/profiling_triton"]
    res = {}
    for d in dirs:
        res[d] = summarize(d)

    if len(res) == 2 and all(res.values()):
        (da, ka), (db, kb) = res.items()
        print(f"\n{'='*78}\n### 对比\n{'='*78}")
        names = set(ka) | set(kb)
        rows = []
        for n in names:
            a = ka.get(n, [0, 0.0])
            b = kb.get(n, [0, 0.0])
            rows.append((n, a[0], b[0], a[0] and a[1] / a[0] or 0, b[0] and b[1] / b[0] or 0, a[1], b[1]))
        rows.sort(key=lambda r: -abs(r[5] - r[6]))
        print(f"{'内核':52s} {'n_a':>7s} {'n_b':>7s} {'mean_a':>9s} {'mean_b':>9s} {'tot_a':>10s} {'tot_b':>10s}")
        for n, na, nb, ma, mb, ta, tb in rows[:25]:
            print(f"{n[:52]:52s} {na:7d} {nb:7d} {ma:8.1f}u {mb:8.1f}u {ta/1000:9.2f}ms {tb/1000:9.2f}ms")
