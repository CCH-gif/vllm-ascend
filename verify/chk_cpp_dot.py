"""Does the cpp launch path handle the new dot kernel (extra runtime scalar)?  """
import os
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "5")
os.environ["TRITON_LORA_TIME"] = "1"

import torch, torch_npu
import vllm_ascend.vllm_ascend_C
from vllm_ascend.lora import lora_ops_triton as L
from vllm_ascend.lora import lora_ops_triton_kernels as K

torch.manual_seed(0)
DEV, I64, BF = "npu", torch.int64, torch.bfloat16

def run(B, Ho, R, Ln, NR, off=0, Y_HO=None):
    Y_HO = Y_HO or Ho
    x = torch.randn(B, R, dtype=torch.float32, device=DEV)
    w = (torch.randn(Ln, 1, Ho, R, dtype=torch.float32, device=DEV) * 0.05).to(BF)
    counts = [B // NR + (1 if i < B % NR else 0) for i in range(NR)]
    st = torch.zeros(NR, dtype=I64, device=DEV)
    if NR > 1:
        st[1:] = torch.cumsum(torch.tensor(counts, dtype=I64, device=DEV), 0)[:-1]
    idx = torch.randint(0, Ln, (NR,), dtype=I64, device=DEV)
    seqlen = torch.tensor(counts, dtype=I64, device=DEV)
    base = torch.randn(B, Y_HO, dtype=BF, device=DEV) * 0.1

    n_before = len(L._CPP_CASES)
    y_c = base.clone()
    L.sgmv_expand_slice(x, w, y_c, st, idx, off, Ho)
    n_mid = len(L._CPP_CASES)

    # python path, same inputs
    os.environ["TRITON_LORA_CPP"] = "0"
    y_p = base.clone()
    L.sgmv_expand_slice(x, w, y_p, st, idx, off, Ho)
    os.environ["TRITON_LORA_CPP"] = "1"
    torch.npu.synchronize()

    same = bool(torch.equal(y_c, y_p))
    if not same:
        d = (y_c.float() - y_p.float()).abs()
        print(f"    max_abs={d.max().item():.3e} 不等={int((d>0).sum())}")
    print(f"  B={B} Ho={Ho} R={R} NR={NR} off={off} -> cpp==python: {same}  "
          f"cases {n_before}->{n_mid}")
    return same

ok = True
for args in [(4096, 4096, 16, 2, 1), (8192, 4096, 16, 2, 32),
             (4096, 12288, 16, 2, 1), (2048, 4096, 16, 2, 8),
             (512, 512, 16, 2, 1), (1024, 1024, 32, 4, 4)]:
    ok &= run(*args)
# sliced (off != 0) and a rank-8 fallback (R<16 -> old kernel)
ok &= run(1024, 1024, 16, 2, 4, off=1024, Y_HO=4096)
ok &= run(1024, 2048, 8, 2, 4)
print("ALL OK" if ok else "MISMATCH")
