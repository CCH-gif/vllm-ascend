# vllm-ascend LoRA 算子 Triton 替换 —— MoE 端到端验证材料

把 vllm-ascend 的 AscendC LoRA 算子（`torch.ops._C_ascend.*` 的 bgmv / sgmv）
替换为 **Triton 内核**，在 **Qwen3-30B-A3B（MoE）+ LoRA、昇腾 910B4 × 4（TP4）**
上做的完整验证材料：算子代码、服务端参数、压测与精度脚本、测试报告。

- **结论以报告为准**：[`report/测试报告_2026-09-16.md`](report/测试报告_2026-09-16.md)
- 环境：vLLM 0.26.0 + vllm-ascend，Ascend 910B4，卡 0–3，TP4

## 结论摘要

| 维度 | 结果 |
|---|---|
| **吞吐** | 10 点 **6 提高 / 4 持平 / 0 劣化**（±5% 口径），最高 **+15.1%**；并发越高收益越大 |
| **延迟** | TTFT 8/10 改善（−3.4% ~ −23.3%）；高并发 c8/c16/c32 的 TPOT 6/6 改善 |
| **精度** | **一致**：以 fp64 精确参考为裁判无劣化，且比 AscendC 精确 **572~16064 倍**；MoE 生产形状 expand 侧逐 bit 相同 |

机制一句话：**Triton 赢在 prefill（shrink 权重驻留 + `tl.dot`），输在低并发的 decode**
（256 个输出 token 把 TPOT 的差放大 256 倍），交叉点在 c1 与 c4 之间。

## 目录

```
lora_ops.py                       算子入口 shim —— 把 bgmv/sgmv 路由到 Triton
lora_ops_triton.py                派发层（tile/split 选择、cpp launcher、workspace 判定）
lora_ops_triton_kernels.py        Triton 内核本体
lora_cpp_launcher.cpp             最小 launcher（[ffts][lock][workspace] 三槽）
lora_cpp_launcher...-linux-gnu.so 预编译 .so（python3.12 + aarch64）
lora_native_ops.cpp               历史遗留，**不在运行路径上**（见下）
lora_ops.py.stock                 AscendC 原版（仅用于 A/B 对照，不参与 triton 路径）

scripts/                          服务端参数 + 压测/profiler
  serve.sh                        ★ 服务端启动 + 全部 vLLM 参数（base|triton 切换）
  env.sh                          HCCL / 网卡环境变量（⚠️ 需改成你本机网卡）
  model.sh                        基座模型路径与 served-model-name
  moe_rest_sweep.sh               ★ 产出本报告的 20 点全量压测
  c1_ab.sh                        c1_4k 单点 A/B（改内核后判断方向用）
  profile_cmp.sh                  真服务 profiler A/B（判断算子是否被 aclgraph 捕获）
  run_full160.sh / run_6k_extra.sh  早期全量压测
  moe_acc_report.sh / moe_acc_fill.sh  MoE 精度配对报告 / 补跑
  acc_floor.sh / acc_full.sh      精度本底噪声对照 / 完整精度对照

verify/                           算子级与端到端验证
  test_lora_ab.py                 算子级 A/B vs AscendC（逐 bit 判据）
  test_lora_ab_cur.py             同上的**可用版**：路径指向当前代码 + 跳过 R=32 溢出用例
  test_lora_moe_shapes.py         MoE 生产形状算子级 A/B
  chk_shrink_acc.py               bgmv_shrink 逐形状 vs fp64
  acc_chain2.py                   完整 LoRA 链 fp64 三方对比
  chk_cpp_dot.py                  cpp launcher 路径 vs python 路径
  acc_stats.py / accuracy_probe.py / cmp_all.py   端到端 token/logprob 统计
  trace_agg.py                    profiler trace 聚合

report/测试报告_2026-09-16.md       ★ 测试报告
```

`lora_native_ops.cpp` 是早期基于一个**已证伪**的假设（「Triton 注册在 Python dispatch key
上，aclgraph 捕获不到」）写的，实测 ACL 捕获是**流级**的、与 dispatch key 无关，
该路径已不参与运行。保留在此仅为记录，**可以删**。

## 一、装算子代码（drop-in，不改上游文件）

```bash
VLLM_ASCEND=<你的 vllm-ascend 源码树>
L=$VLLM_ASCEND/vllm_ascend/lora

cp lora_ops.py lora_ops_triton.py lora_ops_triton_kernels.py "$L"/
cp lora_cpp_launcher.cpp lora_cpp_launcher.cpython-312-aarch64-linux-gnu.so "$L"/
rm -rf "$L/__pycache__"        # 换实现必须清字节码，否则跑的还是上一版
```

`.so` 是 python3.12 + aarch64 预编译的；若版本不符，按 `lora_ops_triton.py` 里
`_native_build()` 的同一组 g++ 参数重编 `lora_cpp_launcher.cpp`。

`lora_ops.py.stock` 是 AscendC 原版，**只**给 A/B 对照用，不要覆盖进去。

## 二、冻结压测场景（不可变）

```
模型      /models/Qwen3-30B-A3B（MoE，128 experts，30B total / 3B active）
adapter   /tmp/e2e/lora_moe（自造，r=16，lora_alpha=32，权重随机初始化）
LoRA 目标 q_proj k_proj v_proj o_proj experts
硬件      昇腾 910B4 × 4（卡 0–3，TP4）
矩阵      输入 {4096, 6144} × 并发 {1,4,8,16,32} = 10 点，每点 160 prompts
```

`vllm bench serve` 参数（`--random-range-ratio 0` 是定长关键）：

```bash
vllm bench serve --backend openai-chat --model /models/Qwen3-30B-A3B \
  --base-url http://localhost:8001 --endpoint /v1/chat/completions \
  --num-prompts 160 --trust-remote-code \
  --dataset-name random --ignore-eos --seed 1234 \
  --served-model-name Qwen3-30B \
  --random-input-len {4096|6144} --random-output-len 256 --random-range-ratio 0 \
  --max-concurrency {1|4|8|16|32} \
  --lora-modules mylora --lora-assignment round-robin
```

服务端 LoRA 侧：`--enable-lora --max-loras 2 --max-lora-rank 16`。

## 三、精确命令

```bash
# 0) 前置：把本分支的算子代码装进源码树（见上），并准备好 ref/ 目录给 serve.sh 做 A/B 切换：
#       mkdir -p /tmp/e2e/ref
#       cp lora_ops.py.stock /tmp/e2e/ref/lora_ops.py.stock     # AscendC 原版
#       cp lora_ops.py       /tmp/e2e/ref/lora_ops.py.triton    # Triton 版
#    （serve.sh 从 $REF_DIR 覆盖到源码树，实现同一份代码上跑 A/B）

# 1) 改 env.sh 的 NIC_NAME / LOCAL_IP 为本机实际网卡
$EDITOR scripts/env.sh

# 2) 压测：10 点全量（每点先起服务、跑完收服务；两台各起一次）
scripts/moe_rest_sweep.sh
#    结果落在 /tmp/e2e/results/rest_{base,triton}/，日志 /tmp/e2e/moe_rest_sweep.log

# 3) 单点 A/B（改内核后快速判断方向，约 45 分钟）
scripts/c1_ab.sh triton

# 4) profiler A/B（判断 LoRA 算子是否被 aclgraph 捕获）
scripts/profile_cmp.sh
python3 verify/trace_agg.py /tmp/e2e/profiling_base /tmp/e2e/profiling_triton
```

精度：

```bash
# 算子级（最强判据）
python3 verify/test_lora_ab_cur.py        # 算子级 A/B（vs AscendC）
python3 verify/test_lora_moe_shapes.py    # MoE 生产形状
python3 verify/chk_shrink_acc.py          # vs fp64
python3 verify/acc_chain2.py              # 完整 LoRA 链 fp64 三方对比

# 端到端（MoE 上本底即 0%，仅作记录，不作判据）
scripts/moe_acc_report.sh
```

## 四、路径假设（脚本是当时实跑的原文，未改路径）

脚本里写死了本机路径，移植时按需替换：

| 路径 | 用途 |
|---|---|
| `/vllm-workspace/vllm-ascend/vllm_ascend/lora` | vllm-ascend 源码树（`serve.sh` 覆盖算子处） |
| `/tmp/e2e/ref/lora_ops.py.{stock,triton}` | A/B 切换源（见上面第 0 步） |
| `/tmp/e2e/{env.sh,model.sh,serve.sh}` | `env.sh`/`model.sh`/`serve.sh` 的互相 source |
| `/tmp/e2e/lora_moe` | 自造 MoE LoRA adapter |
| `/models/Qwen3-30B-A3B` | 基座模型 |
| `/tmp/e2e/results/`、`/tmp/e2e/*.log` | 结果与日志落盘处 |

`verify/test_lora_ab.py` 里的路径指向 `/vllm-workspace/.../lora_ops.py.triton`，
该文件后来被移到 `/tmp/e2e/ref/`；**请用 `test_lora_ab_cur.py`**（路径已修正，
并跳过了 `R=32, BW=1024, B=32` 这个编译期 UB 溢出的用例 —— 那是非 16 rank 的
已知限制，不在 rank-16 生产路径上）。

## 五、脱敏说明

`scripts/env.sh` 里的 **网卡名、IP、主机名、RDMA 设备名已替换为占位符**
`<your-nic>` / `<your-ip>`，并改写为通用查法。原文件含本机的公网 IP 与主机名，
本仓库是 public，故未原样上传。其余文件不含机器标识。

## 六、收服务的纪律（重要）

这台机器是多卡共享的，**卡 4 以上是别人的作业**。所有脚本都：

```bash
setsid nohup ./serve.sh ... &      # 自建进程组
SPID=$!
kill -TERM -- "-$SPID"             # 只按自己建的进程组收
kill -KILL -- "-$SPID"
```

**绝不按进程名 pkill** —— 会打到别人的作业上。移植到共享机器时请保留这条。
