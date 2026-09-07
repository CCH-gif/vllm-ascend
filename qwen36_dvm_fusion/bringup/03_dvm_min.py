"""DVM bring-up gate: torch.compile with TORCHINDUCTOR_NPU_BACKEND=dvm on a tiny model.
Env (dvm_env.sh) must be sourced and TORCHINDUCTOR_NPU_BACKEND set BEFORE import.
Prints kernel count & runs. [DVM venv]
"""
import os, sys, time
os.environ["TORCHINDUCTOR_NPU_BACKEND"] = os.environ.get("NPU_BACKEND", "dvm")
print("TORCHINDUCTOR_NPU_BACKEND =", os.environ["TORCHINDUCTOR_NPU_BACKEND"], flush=True)
import torch, torch_npu
torch.npu.set_device(int(os.environ.get("NPU_DEV", "0")))

import torch_npu._inductor as ii
print("loader map has dvm:", hasattr(ii, "_load_dvm_backend"), flush=True)
# find config knobs
try:
    import torch._inductor.config as ic
    print("ic.npu_backend =", getattr(ic, "npu_backend", "<none>"), flush=True)
except Exception as e:
    print("cfg err", e, flush=True)

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(256, 256, bias=False)
        self.g = torch.nn.Linear(256, 256, bias=False)
    def forward(self, x):
        a = self.fc(x)
        y = torch.nn.functional.silu(a) * self.g(x)   # epilogue vertical into mm
        return (y + 1).relu() + y.sum(dim=-1, keepdim=True)

m = Tiny().to(torch.bfloat16).to("npu")
x = torch.randn(8, 256, dtype=torch.bfloat16, device="npu")
with torch.no_grad():
    ref = m(x).float().cpu()

cf = torch.compile(m, dynamic=False)
try:
    t0 = time.time()
    with torch.no_grad():
        out = cf(x)
    torch.npu.synchronize()
    print("compile first-call %.1fs" % (time.time() - t0), flush=True)
    for _ in range(3):
        with torch.no_grad():
            cf(x)
    torch.npu.synchronize()
    err = (out.float().cpu() - ref).abs().max().item()
    print("DVM OK  out", tuple(out.shape), "maxerr_vs_eager %.4f" % err, flush=True)
except Exception as e:
    print("DVM FAIL:", repr(e)[:600], flush=True)
    raise
