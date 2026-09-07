#!/usr/bin/env python3
"""C re-validate: correctness vs torch_npu.npu_rms_norm (same bf16 add), perf under ASCEND_LAUNCH_BLOCKING."""
import os, json, time, torch, torch_npu, torch.distributed as dist
import vllm_ascend, vllm_ascend.vllm_ascend_C
os.environ.setdefault("MASTER_ADDR","127.0.0.1"); os.environ.setdefault("MASTER_PORT","29502")
os.environ.setdefault("RANK","0"); os.environ.setdefault("WORLD_SIZE","1")
torch_npu.npu.set_device(0)
if not dist.is_initialized(): dist.init_process_group(backend="hccl",world_size=1,rank=0)
DEV="npu:0"; DT=torch.bfloat16
comm=dist.distributed_c10d._get_default_group()._get_backend(torch.device("npu")).get_hccl_comm_name(0)

def fused(x,w,res,gam,eps):
    y,add=torch.ops._C_ascend.matmul_allreduce_add_rmsnorm(x,w,res,gam,comm,1,0,eps,True,False); return y,add

def main():
    H,eps=2048,1e-6; out={"op":"matmul_allreduce_add_rmsnorm","mode":"launch_blocking","tp":1,"cases":[]}
    for S in [512,2048,8192]:
        x=torch.randn(1,S,H,device=DEV,dtype=DT); w=torch.randn(H,H,device=DEV,dtype=DT)
        res=torch.randn(1,S,H,device=DEV,dtype=DT); gam=torch.randn(H,device=DEV,dtype=DT)
        y,add = fused(x,w,res,gam,eps)
        rn = torch_npu.npu_rms_norm(add, gam)
        ref = rn[0] if isinstance(rn, tuple) else rn   # rmsnorm of exact bf16 add
        dy=(y.float()-ref.float()).abs().max().item(); rel=dy/(ref.float().abs().max().item()+1e-6)
        # unfused baseline: matmul + add (in-place) + rms_norm = 3 npu ops
        def unf():
            a = torch.matmul(x, w.t()); a.add_(res)
            rr = torch_npu.npu_rms_norm(a, gam); return rr[0] if isinstance(rr, tuple) else rr
        # warmup
        # timing: drain the hccl+compute streams via a process-group barrier each iter
        fa=0.0; ua=0.0
        for _ in range(15):
            t0=time.perf_counter(); fused(x,w,res,gam,eps); dist.barrier(); fa+=time.perf_counter()-t0
            t0=time.perf_counter(); unf(); dist.barrier(); ua+=time.perf_counter()-t0
        tf=fa*1e3/15; tu=ua*1e3/15
        print(f"S={S:5d} fused={tf:8.1f}us unfused3op={tu:8.1f}us speedup={tu/tf:.3f}x  y_max_abs={dy:.5f} y_rel={rel:.6f}", flush=True)
        out["cases"].append({"S":S,"fused_us":round(tf,1),"unfused_us":round(tu,1),"speedup":round(tu/tf,3),"y_max_abs":dy,"y_rel":rel})
    json.dump(out,open("test_C_results.json","w"),indent=2); print("wrote test_C_results.json",flush=True)
if __name__=="__main__": main()
