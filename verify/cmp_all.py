"""全量对比：4k c1-c32 + 6k c1-c32，共 10 点。判据 ±5%。"""
import re, pathlib, sys

CASES = ["c1_4k", "c4_4k", "c8_4k", "c16_4k", "c32_4k",
         "c1_6k", "c4_6k", "c8_6k", "c16_6k", "c32_6k"]
F = [("out", r"Output token throughput \(tok/s\)"),
     ("req", r"Request throughput \(req/s\)"),
     ("dur", r"Benchmark duration \(s\)"),
     ("ttft", r"Mean TTFT \(ms\)"),
     ("tpot", r"Mean TPOT \(ms\)"),
     ("pttft", r"P99 TTFT \(ms\)"),
     ("ptpot", r"P99 TPOT \(ms\)")]
ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/e2e/results")
BASE_DIR = sys.argv[2] if len(sys.argv) > 2 else "full160_base"
TRITON_DIR = sys.argv[3] if len(sys.argv) > 3 else "full160_triton"

def parse(p):
    if not p.exists():
        return None
    t = p.read_text()
    d = {}
    for k, pat in F:
        m = re.findall(pat + r":\s*([\d.]+)", t)
        d[k] = float(m[-1]) if m else None
    return d

B = {c: parse(ROOT / f"{BASE_DIR}/{c}.txt") for c in CASES}
T = {c: parse(ROOT / f"{TRITON_DIR}/{c}.txt") for c in CASES}

# 用户最终口径（2026-09-14）：
#   高并发（c>=8）不回退 —— 可以有优化、变快；低并发（c1/c4）稍微差一点可以。
# 早期文档里的 ±5% 已作废。
HIGH = {c for c in CASES if not c.startswith(("c1_", "c4_"))}

print("=" * 92)
print(f"10 点 A/B：{BASE_DIR} (AscendC) vs {TRITON_DIR} (Triton)")
print("判据：高并发 c>=8 不回退（Δ>=0）；低并发 c1/c4 允许略降")
print("=" * 92)
print(f"{'case':8s} {'AscendC':>11s} {'Triton':>11s} {'Δ%':>9s} {'TPOT Δ%':>9s} {'TTFT Δ%':>9s} {'判定':>8s}")
print("-" * 92)
for c in CASES:
    b, t = B[c], T[c]
    if b is None or t is None or b["out"] is None or t["out"] is None:
        print(f"{c:8s} 数据不全 (base={b is not None} triton={t is not None})")
        continue
    dp = (t["out"] - b["out"]) / b["out"] * 100
    dtp = (t["tpot"] - b["tpot"]) / b["tpot"] * 100 if b["tpot"] and t["tpot"] else 0
    dtt = (t["ttft"] - b["ttft"]) / b["ttft"] * 100 if b["ttft"] and t["ttft"] else 0
    if c in HIGH:
        mark = "✅ 无回退" if dp >= 0 else "❌ 回退"
    else:
        mark = "✅" if dp >= 0 else "➖ 允许"
    print(f"{c:8s} {b['out']:11.2f} {t['out']:11.2f} {dp:+8.1f}% {dtp:+8.1f}% {dtt:+8.1f}% {mark:>8s}")

print("-" * 92)
hi = [(c, (T[c]["out"] - B[c]["out"]) / B[c]["out"] * 100) for c in sorted(HIGH)
      if B[c] and T[c] and B[c]["out"] and T[c]["out"]]
lo = [(c, (T[c]["out"] - B[c]["out"]) / B[c]["out"] * 100) for c in CASES
      if c not in HIGH and B[c] and T[c] and B[c]["out"] and T[c]["out"]]
if hi:
    reg = [c for c, d in hi if d < 0]
    print(f"高并发 {len(hi)-len(reg)}/{len(hi)} 无回退   "
          f"平均 {sum(d for _, d in hi)/len(hi):+.1f}%  "
          f"区间 [{min(d for _, d in hi):+.1f}%, {max(d for _, d in hi):+.1f}%]")
    if reg:
        print(f"  回退点: {', '.join(reg)}")
if lo:
    print(f"低并发 平均 {sum(d for _, d in lo)/len(lo):+.1f}%  "
          f"（{'、'.join(f'{c} {d:+.1f}%' for c, d in lo)}）")
