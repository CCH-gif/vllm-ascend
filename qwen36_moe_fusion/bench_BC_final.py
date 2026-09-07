#!/usr/bin/env python3
"""Final B+C perf: pool of DISTINCT inputs per rep (defeats async stale-read), drain read each rep.
B: custom moe_init_routing vs torch_npu.npu_moe_init_routing_v2  (ids int32)
C: matmul_allreduce_add_rmsnorm (TP=1) vs matmul+add+rms_norm     (hccl barrier drain)
"""
import os, json, torch, torch_npu, torch.distributed as dist, time
import vllm_ascend, vllm_ascend.vllm_ascend_C
os.environ.setdefault("MASTER_ADDR","127.0.0.1"); os.environ.setdefault("MASTER_PORT","29504")
os.environ.setdefault("RANK","0"); os.environ.setdefault("WORLD_SIZE","1")
torch_npu.npu.set_device(0)
if not dist.is_initialized(): dist.init_process_group(backend="hccl",world_size=1,rank=0)
DEV="npu:0"; DT=torch.bfloat16
comm=dist.distributed_c10d._get_default_group()._get_backend(torch.device("npu")).get_hccl_comm_name(0)
R=20  # distinct input sets
def dr(o):
    for t in o:
        if t is not None: t[...,:1].float().sum().item()
def bench(fn_sets, do_barrier=False):
    # fn_sets[i]() builds+returns result of distinct input set i; warmup on set 0..2
    for i in range(3): dr(fn_sets(i))
    acc=0.0
    for i in range(R):
        t0=time.perf_counter(); r=fn_sets(i)
        if do_barrier: dist.barrier()
        else: dr(r)
        acc+=time.perf_counter()-t0
    return acc*1e3/R

out={}
# ---------- B ----------
HE,EX,TOP=2048,256,8; bres={"cases":[]}
for T in [256,1024,4096]:
    xs=[torch.randn(T,HE,device=DEV,dtype=DT) for _ in range(R)]
    ids=[torch.randint(0,EX,(T,TOP),device=DEV,dtype=torch.int32) for _ in range(R)]
    def cfn(i):
        return torch.ops._C_ascend.npu_moe_init_routing_custom(xs[i],ids[i],scale=None,active_num=T*TOP,
            expert_num=EX,expert_tokens_num_type=1,expert_tokens_num_flag=True,active_expert_range=[0,EX],quant_mode=-1)
    def rfn(i):
        return torch_npu.npu_moe_init_routing_v2(xs[i],ids[i],scale=None,active_num=T*TOP,
            expert_num=EX,expert_tokens_num_type=1,expert_tokens_num_flag=True,active_expert_range=[0,EX],quant_mode=-1)
    tc=bench(cfn); tr=bench(rfn)
    # exact diff check on set0
    a=cfn(0); b=rfn(0)
    dif=[0 if (x is None and y is None) else float('nan') if (x is None or y is None) else (x.float()-y.float()).abs().max().item() for x,y in zip(a,b)]
    print(f"B T={T:5d} custom={tc:7.1f}us npu_v2={tr:7.1f}us speedup={tr/tc:.3f}x diffs={[round(d,4) for d in dif]}", flush=True)
    bres["cases"].append({"T":T,"custom_us":round(tc,1),"npu_v2_us":round(tr,1),"speedup":round(tr/tc,3),"diffs":dif})
out["B"]=bres
# ---------- C ----------
H,eps=2048,1e-6; cres={"cases":[]}
def fused(x,w,res,gam):
    y,add=torch.ops._C_ascend.matmul_allreduce_add_rmsnorm(x,w,res,gam,comm,1,0,eps,True,False); return (y,add)
def unf(x,w,res,gam):
    a=torch.matmul(x,w.t()); a=a.add(res)
    rr=torch_npu.npu_rms_norm(a,gam); return (rr[0] if isinstance(rr,tuple) else rr,)
for S in [512,2048,8192]:
    xs=[torch.randn(1,S,H,device=DEV,dtype=DT) for _ in range(R)]
    ws=[torch.randn(H,H,device=DEV,dtype=DT) for _ in range(R)]
    rs=[torch.randn(1,S,H,device=DEV,dtype=DT) for _ in range(R)]
    gs=[torch.randn(H,device=DEV,dtype=DT) for _ in range(R)]
    def cfn(i): return fused(xs[i],ws[i],rs[i],gs[i])
    def rfn(i): return unf(xs[i],ws[i],rs[i],gs[i])
    tc=bench(cfn,do_barrier=True); tr=bench(rfn,do_barrier=True)
    # correctness on set0 vs npu_rms_norm
    y,add=fused(xs[0],ws[0],rs[0],gs[0]); rn=torch_npu.npu_rms_norm(add,gs[0]); ref=rn[0] if isinstance(rn,tuple) else rn
    dy=(y.float()-ref.float()).abs().max().item(); rel=dy/(ref.float().abs().max().item()+1e-6)
    print(f"C S={S:5d} fused={tc:7.1f}us unfused3op={tr:7.1f}us speedup={tr/tc:.3f}x y_rel={rel:.5f}", flush=True)
    cres["cases"].append({"S":S,"fused_us":round(tc,1),"unfused_us":round(tr,1),"speedup":round(tr/tc,3),"y_rel":rel})
out["C"]=cres
json.dump(out,open("bench_BC_final_results.json","w"),indent=2); print("wrote bench_BC_final_results.json",flush=True)
