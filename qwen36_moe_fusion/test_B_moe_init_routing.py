#!/usr/bin/env python3
"""Task B: moe_init_routing_custom (copy+concat dispatch) precision + perf vs torch_npu.npu_moe_init_routing_v2."""
import json, time
import torch, torch_npu
import vllm_ascend
import vllm_ascend.vllm_ascend_C

torch_npu.npu.set_device(0)
DEV = "npu:0"
DT = torch.bfloat16


def custom(x, ids, active_num, expert_num, rng, quant_mode):
    ids = ids.to(torch.int32)  # compiled bins expect int32 expert_idx
    return torch.ops._C_ascend.npu_moe_init_routing_custom(
        x, ids, scale=None, active_num=active_num, expert_num=expert_num,
        expert_tokens_num_type=1, expert_tokens_num_flag=True,
        active_expert_range=rng, quant_mode=quant_mode)


def ref(x, ids, active_num, expert_num, rng, quant_mode):
    return torch_npu.npu_moe_init_routing_v2(
        x, ids, scale=None, active_num=active_num, expert_num=expert_num,
        expert_tokens_num_type=1, expert_tokens_num_flag=True,
        active_expert_range=rng, quant_mode=quant_mode)


def drain(r):
    # force real completion: slice-read on every non-None output
    s=0.0
    for o in r:
        if o is None: continue
        s += o[..., :1].float().sum().item()
    return s
def t_ms(fn, warmup=10, reps=50):
    for _ in range(warmup):
        drain(fn())
    acc = 0.0
    for _ in range(reps):
        t0 = time.perf_counter()
        drain(fn())
        acc += time.perf_counter() - t0
    return acc * 1e3 / reps


def main():
    H, EXP, TOPK = 2048, 256, 8
    out = {"cases": []}
    for T in [256, 1024, 4096]:
        x = torch.randn(T, H, device=DEV, dtype=DT)
        ids = torch.randint(0, EXP, (T, TOPK), device=DEV, dtype=torch.int32)
        active_num = T * TOPK
        rng = [0, EXP]
        a = custom(x, ids, active_num, EXP, rng, -1)
        b = ref(x, ids, active_num, EXP, rng, -1)
        # compare the 4 outputs (sh, row_idx, expert_tokens, scale)
        diffs = []
        for i, (ai, bi) in enumerate(zip(a, b)):
            if ai is None and bi is None:
                diffs.append(0.0)
                continue
            if ai is None or bi is None:
                diffs.append(float("nan"))
                continue
            aa = ai.float() if ai.is_floating_point() else ai
            bb = bi.float() if bi.is_floating_point() else bi
            diffs.append((aa - bb).abs().max().item() if aa.numel() else 0.0)
        tc = t_ms(lambda: custom(x, ids, active_num, EXP, rng, -1))
        tr = t_ms(lambda: ref(x, ids, active_num, EXP, rng, -1))
        row = {"T": T, "custom_us": round(tc * 1e3, 2), "torch_npu_us": round(tr * 1e3, 2),
               "speedup": round(tr / tc, 3), "out_diffs": [round(d, 6) for d in diffs]}
        out["cases"].append(row)
        print(f"T={T:5d} custom={tc*1e3:8.2f}us torch_npu={tr*1e3:8.2f}us speedup={tr/tc:.3f}x diffs={[round(d,4) for d in diffs]}")
    json.dump(out, open("test_B_results.json", "w"), indent=2)
    print("wrote test_B_results.json")


if __name__ == "__main__":
    main()
