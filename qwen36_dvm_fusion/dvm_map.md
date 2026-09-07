# DVM 可融合空间 —— 候选映射表（WP3，草稿）

模型：Qwen3.6-35B-A3B（Qwen3_5Moe 文本层：40 层 = 30 linear_attn + 10 full_attn，每层全含 MoE FFN）
每层结构（transformers qwen3_5_moe，`/data/.../modeling_qwen3_5_moe.py`）：
- decoder layer：input_layernorm(rms) → token mixer → +residual → post_attention_layernorm(rms) → MoE FFN → +residual
  - full_attn = Qwen3_5MoeAttention（q/k/v 投影 + RoPE + softmax attn + o 投影 + gate，seq≥…）
  - linear_attn = Qwen3_5MoeGatedDeltaNet（in_proj_qkv/z/b/a + conv1d + chunk/recurrent gated-delta-rule + rmsnorm-gated + out_proj）
  - MoE FFN = SparseMoeBlock：router(topk) + shared_expert(MLP gate/up/down + sigmoid 门) + experts(256×[gate_up 3D + down 3D] + topk 加权 index_add)

## 静态候选（按模块结构 + eager 实测派发事件数，载体=torch_npu 2.9/8.5 栈）

eager 派发事件（CPU dispatch, seq=64, 单层，torch.profiler CPU）：
- full_attn 层 idx3：**1084 ops**（85 去重）；linear_attn 层 idx0：**3393 ops**（117 去重）
- 全 40 层 ≈ 10×1084 + 30×3393 ≈ **112k ops/prefill**（“未融合 op” 分子）

纵向 pointwise-epilogue 入前驱 GEMM 的候选链（DVM matmul-template eligible 判定在最后）：
| 链 | 位置/层型 | 组成 | 归属 GEMM | 判定（待 WP2 填实测） |
|---|---|---|---|---|
| C1 swiglu epilogue | MoE dense FFN（shared expert 与全模型 MLP 通路） | gate_mm → silu ⊙ up_mm（两 mm 并行 = 水平融合 → **不 eligible**）；group 化单 mm(gate|up concat N) → silu⊙ → down | gate/up mm | 水平不融；纵向下游 swiglu⊙ 若 gate/up 已 concat 成 1 mm 则 epilogue eligible |
| C2 o_proj→gate→residual | attn(full/linear) out | o_mm → attn_output_gate pointwise → residual add | o_proj | epilogue eligible（须 broadcast 合法、无中间 GM 往返干扰） |
| C3 router epilogue | MoE | router mm → softmax→topk→归一 | router gate | topk 非 pointwise → **不 eligible**（graph_fusion 候选，需判） |
| C4 rmsnorm→mm（prologue） | 各 mm 前 rmsnorm | prologue 归一 → mm | 下游 mm | prologue **不 eligible**（DVM matmul-template 文档明确） |
| C5 residual add 链 | 层末 | 2×mm 输出相加 + … | — | 水平，不 eligible |
| C6 conv1d 后 split/pointwise（linear-attn） | linear | depthwise conv1d(groups) → activation → split → reshape | conv1d(depthwise) | 偏“conv+depthwise”，非 matmul-template 范围 |
| C7 gated-delta-rule 内部 | linear | 大量小 mm/pointwise（chunk 扫描） | 多个小 mm | 依赖融合模式（graph_fusion）；逐 chain 判 |
| C8 experts 3D 行 GEMM 通路 | MoE routed | index_select 行 → gate_up 3D mm → silu⊙up → down 3D mm → ×w → index_add_ | 3D 行 mm | 数据相关（index）；索引/归约非 pointwise → 主要不 eligible；index_add_ 归约不可融 |

## DVM 判定规则（语义依据，外推部分用）
来自公开文档/源码（TorchNPU 26.1 inductor 指南；torch_npu/_inductor DVM matmul-template 代码）：
- `TORCHINDUCTOR_NPU_BACKEND` ∈ {default(triton), mlir, dvm}，须 import torch 前设置；dvm 加载 `_load_dvm_backend`。
- DVM = inductor 的算子编译器（替代 triton codegen），复用 inductor-MLIR 融合结果（mlir_fusion）或自定义图融合（graph_fusion）。
- opt-in `enable_matmul_fusion`（默认关；`INDUCTOR_DVM_ENABLE_MATMUL_FUSION=1` 开启）→ 为 aten.mm/bmm/addmm/baddbmm 注册 template：
  - **仅支持合法 pointwise-epilogue 纵向融合**；
  - **prologue / horizontal fusion / reduction / template-to-template / group 或 numel 不一致 / 不支持 broadcast → 不融合**。
- 硬件注意：Ascend 950DT 仅 triton+dvm（无 mlir）；mlir 支持范围另有限制。

## 实测结果（WP2，同 torch_npu 2.13.0rc1/CANN9.1 栈，真权重）
- full_attn idx3：eager 13.8/14.3/16.8ms(aclnn 180) → DVM 16.1/21.7/21.2ms(aclnn 107)；开 `INDUCTOR_DVM_ENABLE_MATMUL_FUSION=1`(seq512) → 17.8ms(aclnn 96)。**DVM 设备算子 −40%，但时延仍 ≥ eager（0.65–0.8×）**。
- linear_attn idx0：eager 33.2ms(aclnn 607)；**DVM 编译失败** `aclnnArange …161002`（30/40 层无法成图）。
- dense FFN（shared_expert，每层有）：M512 eager 0.295ms(5 ops) → DVM 0.508ms(29 ops)；M8192 eager 0.341ms(5) → DVM 0.497ms(**3 ops**, 真融成 3 kernel)。融合生效但 native aclnn 更快。
- `default`=triton inductor：本机所有 torch_npu 均 `0 active drivers`（不可编译）；**DVM 是本机唯一能跑通的 inductor 后端**。

## 判定汇总
- C1/C2（纵向 pointwise-epilogue）→ 【实测·26】DVM 确能融（full 180→107/96；dense 5→3），但生成 kernel 未快过 native aclnn → **结构可融、当前不产生时延收益**。
- C3/C8（router topk、experts 3D 行 mm + index_add、nonzero/index/one_hot）→ 数据相关，DVM matmul-template 不融；实测留在 eager（compiled CPU 事件里仍见 aclnnIndex/Nonzero/item/_local_scalar_dense）。
- C4/C5（prologue、horizontal）→ 文档明确不融，本机无冲突。
- C6/C7（conv1d→split；gated-delta chunk）→ 非 matmul-template 范围；C7 所在 linear 层整层成图失败（aclnnArange）。
- 【外推】真实部署若先把 MoE 专家 python 循环/linear-attn 回退换成少数 fused 大算子，DVM 在其 epilogue 的融合才可能转化为收益；本报告只在 transformers 回退形态下测量（见 REPORT §4 限制）。
