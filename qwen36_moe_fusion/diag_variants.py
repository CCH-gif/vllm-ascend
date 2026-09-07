# Variant sweep for the FUSED single-launch op: find the call form that matches ground truth.
import sys, faulthandler; faulthandler.enable()
import torch, torch_npu
import torch.nn.functional as Fn
dev=int(sys.argv[1]); torch_npu.npu.set_device(dev); torch.manual_seed(0)
M,E,K,N = 32, 8, 256, 64
RPE=M//E
x = (torch.randn(M,K)*1.5).to(f"npu:{dev}").to(torch.bfloat16)
w = (torch.randn(E,K,N)*0.02).to(f"npu:{dev}").to(torch.bfloat16)
xq,xs = torch_npu.npu_dynamic_quant(x)
ws = w.float().abs().amax(dim=1).clamp_min(1e-8)/127.0
wq = (w.float()/ws.unsqueeze(1)).round().clamp(-128,127).to(torch.int8)
wq_nz = torch_npu.npu_format_cast(wq, 29)
cum = (torch.arange(1,E+1,dtype=torch.int64)*RPE).to(f"npu:{dev}")
cnt = (torch.full((E,),RPE,dtype=torch.int64)).to(f"npu:{dev}")

gt=[]
prev=0
for e in range(E):
    c=cum[e].item()
    g = xq[prev:c].float() @ wq[e].float()
    g = g*ws[e].unsqueeze(0)*xs[prev:c].unsqueeze(1)
    h = Fn.silu(g[:,:N//2])*g[:,N//2:]
    gt.append(h); prev=c
gt=torch.cat(gt)

def eval(name, fn):
    try:
        r = fn(); yq,ys = r[0], r[1]; torch.npu.synchronize()
        fp = yq.float()*ys.unsqueeze(1)
        if not torch.isfinite(fp).all():
            print(f"{name}: NONFINITE ({torch.isinf(fp).sum().item()} inf, {torch.isnan(fp).sum().item()} nan)")
            return
        mae=(fp-gt).abs().mean().item(); rel=(fp-gt).abs().max().item()/gt.abs().max().item()
        print(f"{name}: mae={mae:.4f} relmax={rel:.4f}")
    except Exception as ex:
        print(f"{name}: EXC {type(ex).__name__}: {str(ex)[:120]}")

# v2 variants
eval("[v2 scale-list]", lambda: torch_npu.npu_grouped_matmul_swiglu_quant_v2(x=xq, weight=[wq_nz], weight_scale=[ws], x_scale=xs, group_list=cum, group_list_type=0))
eval("[v2 cnt+type1]",   lambda: torch_npu.npu_grouped_matmul_swiglu_quant_v2(x=xq, weight=[wq_nz], weight_scale=[ws], x_scale=xs, group_list=cnt, group_list_type=1))
# non-v2 (moe_mlp-wired op)
eval("[v1 nz3d]",        lambda: torch_npu.npu_grouped_matmul_swiglu_quant(x=xq, weight=wq_nz, weight_scale=ws, x_scale=xs, group_list=cum))
eval("[v1 nd wq]",       lambda: torch_npu.npu_grouped_matmul_swiglu_quant(x=xq, weight=wq,   weight_scale=ws, x_scale=xs, group_list=cum))
eval("[v1 cnt]",         lambda: torch_npu.npu_grouped_matmul_swiglu_quant(x=xq, weight=wq_nz, weight_scale=ws, x_scale=xs, group_list=cnt))
print("DONE")
