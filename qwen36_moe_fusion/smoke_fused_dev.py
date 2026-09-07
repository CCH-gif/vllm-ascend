# Tiny smoke: does the native fused op + two-step run on a given device?
# Usage: python smoke_fused_dev.py <dev>
import sys, faulthandler; faulthandler.enable()
import torch, torch_npu
dev = int(sys.argv[1]); torch_npu.npu.set_device(dev)
torch.manual_seed(0)
M,E,K,N = 16, 8, 256, 64
RPE = M//E
cum = (torch.arange(1,E+1,dtype=torch.int64)*RPE).to(f"npu:{dev}")
x = (torch.randn(M,K)*1.5).to(f"npu:{dev}").to(torch.bfloat16)
w = (torch.randn(E,K,N)*0.02).to(f"npu:{dev}").to(torch.bfloat16)
xq,xs = torch_npu.npu_dynamic_quant(x)
ws = w.float().abs().amax(dim=1).clamp_min(1e-8)/127.0
wq = (w.float()/ws.unsqueeze(1)).round().clamp(-128,127).to(torch.int8)
wq_nz = torch_npu.npu_format_cast(wq, 29)
print(f"[dev{dev}] prep done", flush=True)
yq,ys = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x=xq, weight=[wq_nz], weight_scale=[ws], x_scale=xs, group_list=cum)
torch.npu.synchronize()
print(f"[dev{dev}] FUSED-v2 OK yq={tuple(yq.shape)} val_ok={yq.float().abs().max().item()<128}", flush=True)
z = torch_npu.npu_grouped_matmul(x=[xq], weight=[wq_nz], split_item=3, group_list_type=0, group_type=0, group_list=cum, output_dtype=torch.int32)[0]
y2,s2 = torch_npu.npu_dequant_swiglu_quant(x=z, weight_scale=ws, activation_scale=xs, group_index=cum, activate_left=True, quant_mode=1)
torch.npu.synchronize()
print(f"[dev{dev}] UNFUSED OK yq={tuple(y2.shape)}", flush=True)
d = (yq.float()-y2.float()).abs().max().item()
print(f"[dev{dev}] fused-vs-unfused max_abs={d:.3f}", flush=True)
print(f"[dev{dev}] ALLDONE", flush=True)
