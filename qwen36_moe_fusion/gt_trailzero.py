# Which path (fused-v1 vs two-step) is correct when some experts are inactive (trailing zero-count groups)?
# Validate each against manual fp32 GT over the first G active experts.
import sys, faulthandler; faulthandler.enable()
import torch, torch_npu
import torch.nn.functional as Fn
dev=int(sys.argv[1]); M=int(sys.argv[2]); U=int(sys.argv[3])
torch_npu.npu.set_device(dev); torch.manual_seed(0)
E,K,N=256,2048,1024; R=M//U
x=(torch.randn(M,K)*1.5).to(f"npu:{dev}").to(torch.bfloat16)
w=(torch.randn(E,K,N)*0.02).to(f"npu:{dev}").to(torch.bfloat16)
xq,xs=torch_npu.npu_dynamic_quant(x)
ws=w.float().abs().amax(dim=1).clamp_min(1e-8)/127.0
wq=(w.float()/ws.unsqueeze(1)).round().clamp(-128,127).to(torch.int8)
wq_nz=torch_npu.npu_format_cast(wq,29)
cum=torch.cat([torch.arange(1,U+1,dtype=torch.int64)*R, torch.full((E-U,),M,dtype=torch.int64)]).to(f"npu:{dev}")
print(f"M{M} U{U} r{R} prep", flush=True)
# GT over first G active experts
G=min(U,32); gt=[]; prev=0
for e in range(G):
    c=(e+1)*R
    g=xq[prev:c].float()@wq[e].float()
    g=g*ws[e].unsqueeze(0)*xs[prev:c].unsqueeze(1)
    gt.append(Fn.silu(g[:,:N//2])*g[:,N//2:]); prev=c
gt=torch.cat(gt); gtabs=gt.abs().mean().item()+1e-6
def report(name,yq,sq):
    fp=yq.float()*sq.unsqueeze(1)
    fg=fp[:gt.shape[0]]
    mae=(fg-gt).abs().mean().item()
    print(f"  {name}: finite={torch.isfinite(fg).all().item()} mae-vs-GT={mae:.4f} rel={mae/gtabs:.3f} first3={fg[:3,0].tolist()}")
try:
    y,s=torch_npu.npu_grouped_matmul_swiglu_quant(x=xq,weight=wq_nz,weight_scale=ws,x_scale=xs,group_list=cum)[0:2]; torch.npu.synchronize()
    report("[fused-v1]",y,s)
except Exception as ex: print("  [fused-v1] EXC",str(ex)[:80],flush=True)
try:
    z=torch_npu.npu_grouped_matmul(x=[xq],weight=[wq_nz],split_item=3,group_list_type=0,group_type=0,group_list=cum,output_dtype=torch.int32)[0]
    y,s=torch_npu.npu_dequant_swiglu_quant(x=z,weight_scale=ws,activation_scale=xs,group_index=cum,activate_left=True,quant_mode=1); torch.npu.synchronize()
    report("[two-step]",y,s)
except Exception as ex: print("  [two-step] EXC",str(ex)[:80],flush=True)
print("DONE",flush=True)
