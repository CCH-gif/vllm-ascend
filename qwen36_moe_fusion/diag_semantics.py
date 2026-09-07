# Diagnose fused/unfused semantics vs manual fp32 ground truth (tiny, E=8, M=32).
import sys, faulthandler; faulthandler.enable()
import torch, torch_npu
dev=int(sys.argv[1]); torch_npu.npu.set_device(dev); torch.manual_seed(0)
M,E,K,N = 32, 8, 256, 64
RPE=M//E
import torch.nn.functional as Fn
x = (torch.randn(M,K)*1.5).to(f"npu:{dev}").to(torch.bfloat16)
w = (torch.randn(E,K,N)*0.02).to(f"npu:{dev}").to(torch.bfloat16)
xq,xs = torch_npu.npu_dynamic_quant(x)
ws = w.float().abs().amax(dim=1).clamp_min(1e-8)/127.0
wq = (w.float()/ws.unsqueeze(1)).round().clamp(-128,127).to(torch.int8)
wq_nz = torch_npu.npu_format_cast(wq, 29)
cum = (torch.arange(1,E+1,dtype=torch.int64)*RPE).to(f"npu:{dev}")

# manual fp32 ground truth (per expert slice)
gt=[]; gts=[]
prev=0
for e in range(E):
    c=cum[e].item()
    g = xq[prev:c].float() @ wq[e].float()      # [r,N]
    g = g*ws[e].unsqueeze(0)*xs[prev:c].unsqueeze(1)
    gate,up = g[:,:N//2], g[:,N//2:]
    h = Fn.silu(gate)*up
    gt.append(h); gts.append(g)
    prev=c
gt = torch.cat(gt); gts=torch.cat(gts)

def cmp(name, yq, ys):
    fp = yq.float()*ys.unsqueeze(1)
    # compare dequantized swiglu activations
    mae = (fp-gt).abs().mean().item(); rel=(fp-gt).abs().max().item()/gt.abs().max().item()
    print(f"{name}: dequant-fp32 vs manualGT  mae={mae:.4f} relmax={rel:.4f}")

# fused
yq,ys = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x=xq, weight=[wq_nz], weight_scale=[ws], x_scale=xs, group_list=cum)
torch.npu.synchronize(); cmp("[fused-v2]", yq, ys)

# unfused: try group_index = cumsum and = counts, quant_mode 1
for gi_name, gi in [("cum", cum), ("counts", cum-RPE)]:
    z = torch_npu.npu_grouped_matmul(x=[xq], weight=[wq_nz], split_item=3, group_list_type=0, group_type=0, group_list=cum, output_dtype=torch.int32)[0]
    y2,s2 = torch_npu.npu_dequant_swiglu_quant(x=z, weight_scale=ws, activation_scale=xs, group_index=gi, activate_left=True, quant_mode=1)
    torch.npu.synchronize()
    fp = y2.float()*s2.unsqueeze(1)
    print(f"[unfused gi={gi_name}] mae={ (fp-gt).abs().mean().item():.4f} relmax={(fp-gt).abs().max().item()/gt.abs().max().item():.4f}")
print("DONE")
