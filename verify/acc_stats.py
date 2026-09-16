"""把 token 一致率拆成两件事：
   1) 数值噪声 —— 前缀还相同的位置上的 logprob 差（跨实现可比，干净）
   2) 级联     —— 首次分叉位置的分布（一次近 tie 翻转后，后面全部不可比）
   整体一致率被 2) 主导，单看它会误判：base 自己跑两遍也只有 ~90%。
"""
import json, sys, statistics

def load(p):
    d = json.load(open(p)); return d, d["results"]

def report(pa, pb):
    A, ra = load(pa); B, rb = load(pb)
    assert set(ra) == set(rb)
    same = tot = 0
    same_all = tot_all = 0
    first_div = []           # 每个请求的首次分叉位置
    dlp_clean = []           # 分叉前（含分叉点之前）的 |Δlogprob|
    dlp_all = []
    dlp_pos0 = []
    for k in sorted(ra, key=int):
        ta, tb = ra[k]["tokens"], rb[k]["tokens"]
        la, lb = ra[k]["logprobs"], rb[k]["logprobs"]
        n = min(len(ta), len(tb))
        fd = n
        for j in range(n):
            tot_all += 1
            if ta[j] == tb[j]:
                same_all += 1
            elif fd == n:
                fd = j
        for j in range(fd):
            tot += 1
            if ta[j] == tb[j]:
                same += 1
        first_div.append(fd)
        for j in range(min(fd + 1, len(la), len(lb))):
            if la[j] is None or lb[j] is None:
                continue
            d = abs(la[j] - lb[j])
            dlp_clean.append(d)
            if j == 0:
                dlp_pos0.append(d)
        for j in range(min(len(la), len(lb))):
            if la[j] is None or lb[j] is None:
                continue
            dlp_all.append(abs(la[j] - lb[j]))

    print(f"A = {A['label']:8s} {pa}")
    print(f"B = {B['label']:8s} {pb}")
    print(f"  整体 token 一致率     : {same_all}/{tot_all} = {same_all/tot_all*100:.2f}%")
    print(f"  首次分叉位置 中位数   : {statistics.median(first_div):.1f}  "
          f"(位置0就分叉的请求 {sum(1 for x in first_div if x==0)}/{len(first_div)}, "
          f"8 个 token 全同的 {sum(1 for x in first_div if x==8)})")
    if dlp_pos0:
        print(f"  位置0 |Δlogprob|      : max {max(dlp_pos0):.3e}  "
              f"中位 {statistics.median(dlp_pos0):.3e}   ← 纯数值噪声，无级联")
    if dlp_clean:
        print(f"  分叉前 |Δlogprob|     : max {max(dlp_clean):.3e}  "
              f"中位 {statistics.median(dlp_clean):.3e}  n={len(dlp_clean)}")
    print(f"  全位置 |Δlogprob| max : {max(dlp_all):.3e}   ← 含级联，不可比")
    print()

for i in range(1, len(sys.argv) - 1):
    report(sys.argv[i], sys.argv[i + 1])
