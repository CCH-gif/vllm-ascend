# Qwen3.6-35B × DVM 可融合空间分析 —— 交付报告

日期 2026-09-07 · 昇腾 **910B**(Ascend910B4-1) 单卡逐层 · host CANN 8.5.0.alpha001；DVM 用容器镜像提取 CANN9.1
交付目录：`/workspace/ascend-operator-agent-v2-gitcode6.18/qwen36_dvm_fusion/`
标注：**【实测·85】**= torch_npu2.9/CANN8.5 真跑；**【实测·26】**= torch_npu2.13.0rc1(**DVM**)/CANN9.1 真跑；其余为方法/说明。

> 范围（用户拍板 v2）：**真跑 DVM 为主体** + **贴整网**（真权重/真结构，40 层逐层流式）+ **并列真实后端**。
> 载体：DVM 在 torch_npu≥26（要求 CANN≥9.x），host 仅 CANN8.5 → 经 quay `ascend/cann:9.1.0-910b-ubuntu22.04-py3.11` **镜像提取 CANN9.1 到独立前缀**（不扰动 host），配 torch2.13.0+cpu + torch_npu 2.13.0rc1（compat，`_BACKEND_LOADERS` 含 dvm）+ triton-ascend 覆盖。host 驱动 25.5.1 直连可用。

---

## 0. 结论速览

**DVM（torch_npu 2.13.0rc1 / CANN9.1）在本机 910B 真跑成功**：`TORCHINDUCTOR_NPU_BACKEND=dvm` 可编译并运行。但**对本模型，实测 DVM 编译不构成时延收益**：

| 结论 | 证据 |
|---|---|
| 整网 eager 基线（40 层真权重逐层流式, seq256） | 【实测·85】≈**0.93s**；【实测·26】≈**1.19s**（26 栈 eager 比 8.5 慢 ~27%，跨栈不直接比） |
| `default`(triton) inductor | **本机不可编译**（`0 active drivers`，需 CUDA 驱动探测）→ DVM 是本机**唯一可跑**的 inductor 后端 |
| DVM 编译 vs eager（同 26 栈、同层同时延） | **full 层 0.65–0.8×（更慢）**；dense FFN 0.58–0.69×（更慢）；linear 层编译**失败**（aclnnArange 161002）|
| DVM 设备算子数（融合程度代理） | full 层 180→107(开 matmul 融合→96)；dense FFN 5→3 —— **结构上确有融合**，但 <b>native aclnn 已高度紧凑、DVM 生成 kernel 未快过它</b>，融合数下降未转为时延 |
| 瓶颈不在可融合范围 | MoE 每专家 python 循环(index/数据相关)与 linear-attn torch 回退(chunk 扫描)主导 40 层 eager 时延，且 seq64→2048 时延几乎不增 ⇒ **launch/数据相关主导，非 pointwise-epilogue 融合可救** |

→ 一句话：**DVM 的“可融合空间”在本模型内是真实的（pointwise-epilogue 纵向链，实测把 full 层设备算子压掉 ~40%），但以本机 CANN/torch_npu 的调度与 kernel 质量，这些融合不产生时延收益；且模型的时延主项（MoE 路由索引循环、linear-attn 回退）在 DVM 融合范围之外，linear 层还因 aclnnArange 无法成图。** 判定与边界见 §3/§4。

---

## 1. 方法与载体

- 模型：`Qwen3_5MoeForConditionalGeneration`（transformers 5.5.3 原生），文本 40 层 = 30 `linear_attn`(GatedDeltaNet) + 10 `full_attn`，全含 MoE FFN(256 exp/topk8/shared)。权重 `/data/models/Qwen3.6-35B-A3B`(26 shard, bf16)。
- **贴整网**：bf16/meta 逐层 safetensors loader（单层构造 ~0.5s，不物化 72GB）→ 40 层真权重逐层上卡前向 → per-layer min(2 passes)、per-iter sync、输入随机但固定 shape。层输入为上一层真输出/embed 真权重。
- 算子/内核计数：`torch.profiler` CPU activity；`aclnn*` CPU 派发事件数 ≈ 设备算子/内核数代理（NPU activity 在本组合下计 0，故用此代理，已在方法注明）。
- 后端：eager / `TORCHINDUCTOR_NPU_BACKEND=dvm`（默认配置）/ dvm+`INDUCTOR_DVM_ENABLE_MATMUL_FUSION=1` / `default`=triton。
- DVM 栈 bring-up（WP0）：见 `bringup/` 与 PROGRESS；go 判据（最小编译跑通）已过。

## 2. 实测结果

### 2.1 整网 eager 基线（40 层真权重逐层流式，不含 embed/final-norm/lm_head 权重读）
| seq | 栈 | total | linear(30) | full(10) | per-linear | per-full |
|---|---|---|---|---|---|---|
| 64 | 【实测·85】 | 1130 ms | 1006 ms | 124 ms | ≈33.5 ms | ≈12.4 ms |
| 256 | 【实测·85】 | 931 ms | 806 ms | 125 ms | ≈26.9 ms | ≈12.5 ms |
| 256 | 【实测·26】 | 1187 ms | 1046 ms | 141 ms | ≈34.9 ms | ≈14.1 ms |
- seq64 反而慢于 seq256：小 M 下 MoE/linear 小批 launch+低利用率主导。seq64→2048 full 层 eager 时延 13.8→16.8ms（近乎不随 M 增长）⇒ **该 eager 前向被 MoE 每专家 python 循环与 linear-attn 回退主导，非计算（GEMM）主导**。

### 2.2 单层 eager 派发（seq64, 单次前向）
| 层 | 栈 | CPU 派发 | aclnn(设备算子) | 中位时延 |
|---|---|---|---|---|
| idx3 full_attn | 【实测·26】 | 1121 | 180 (26 类) | ~13.8 ms |
| idx0 linear_attn | 【实测·26】 | 3559 | 607 (37 类) | ~33.2 ms |
| 全 40 层估算 | — | ~118k | ~10k+ | —（eager 见 2.1）|

### 2.3 DVM vs eager（同 26 栈、真权重、同一层）
full_attn idx3：
| seq | eager ms | DVM ms | DVM+mmfus ms | aclnn e/DVM/DVM+mmf | CPU 派发 DVM |
|---|---|---|---|---|---|
| 64 | 13.8 | 16.1 | — | 180/107/— | 1018 |
| 512 | 14.3 | 21.7 | 17.8 | 180/107/96 | 1018/1044 |
| 2048 | 16.8 | 21.2 | — | 180/107/— | 1018 |

dense FFN（`mlp.shared_expert` 真权重，每层都有）：
| M | eager ms(aclnn) | DVM ms(aclnn) | e/c |
|---|---|---|---|
| 512 | 0.295 (5) | 0.508 (29) | 0.58× |
| 8192 | 0.341 (5) | 0.497 (3) | 0.69× |

linear_attn idx0：DVM 编译**失败** `aclnnArange … error code 161002`（编译期 arange 下放 device 报错；数据/说明见 `results/prof_dvm_idx0_s64.json`）。

读数：
1. DVM 确能**减少设备算子派发**（full 层 180→107，开 matmul 融合再→96；dense FFN 8192 时 5→3 真融合成 3 个 kernel）→ **结构上 fusion 生效**。
2. 但**所有测得 DVM 编译时延 ≥ eager**（0.55–0.8×）。dense FFN 在 M512 反而 29 个 aclnn > eager 5 个（编译把 native 已紧凑的 aclnnMatmul 拆成更细 kernel + copy/format 转换）；M8192 虽融成 3 个仍慢（0.69×）→ **CANN native aclnn kernel 质量 ≥ DVM 生成 kernel**（本组合）。
3. 开 matmul 融合（`INDUCTOR_DVM_ENABLE_MATMUL_FUSION=1`）在 full 层把 DVM 时延 21.7→17.8ms（仍比 eager 慢 24%）、设备算子 107→96 → 该开关确有正效果但不足以翻盘。
4. MoE/linear-attn 主计算路径**数据相关**（每专家 index/loop、chunk 扫描）→ DVM pointwise-epilogue 融合不覆盖；linear 层整层成图还失败。40 层 eager 时延几乎不随 M 涨 ⇒ 当前时延瓶颈不在 DVM 的可融范围内。
5. `default`=triton inductor 在本机任何 torch_npu 下均 `0 active drivers`（triton-ascend 的 CUDA 驱动探测），故 DVM 是唯一能跑通 compile 的后端（这本身是 DVM 存在意义的实测佐证）。

## 3. DVM 可融合空间判定（WP3，`dvm_map.md` 详表）

判定依据（来源见 PROGRESS 底部）：HiAscend TORCHINDUCTOR_NPU_BACKEND 文档；torch_npu `_inductor` loader(`_load_dvm_backend`/`from .dvm import mlir_fusion`，2.13.0rc1 实测);matmul-template 限制（仅合法 pointwise-epilogue **纵向**融合;prologue/horizontal/reduction/data 相关 不融）。

| 候选链 | 位置 | 判定（DVM 语义） | 实测/说明 |
|---|---|---|---|
| C1 dense FFN gate/up/down (swiglu) | shared_expert & 各 dense | 两并行 mm(水平)不融；若 gate|up concat 成单 mm 则可融 epilogue | 【实测·26】dense FFN M8192 编译 5→3 个算子（确融）；仍慢于 eager（native 5 个更优）|
| C2 o_proj→gate→residual | attn out | 纵向 epilogue eligible | full 层部分融入（full aclnn 180→107 的一环）|
| C3 router softmax/topk | MoE | topk 非 pointwise → 不 eligible | 实测 full 层仍现 aclnnIndex/Nonzero/item 等（数据相关留在 eager）|
| C4 rmsnorm→mm prologue | 各 mm | prologue 不融 | — |
| C5 residual/逐点合并 | 多处 | 水平，不融 | — |
| C6 conv1d(depthwise)→split(pointwise) | linear | 非 matmul-template 范围 | — |
| C7 gated-delta chunk 核心 | linear | 数据相关/扫描，compile 失败 | linear 层 DVM 编译失败(arange 161002) |
| C8 experts 3D 行 mm + index_add | MoE routed | 数据相关 index/归约，不融 | 每层每 batch python 循环；eager 主瓶颈，非 DVM 可救 |

**可融合空间结论（结构层面，实测支撑）**：模型内 DVM matmul-template 的合法融合面 = 各 GEMM 后紧跟的 pointwise-epilogue 纵向链（o_proj→gate/residual、FFN 的 epilogue、norm 后单 mm 前不加 prelude 的链）。它**不包括**本模型时延主项：MoE 每专家索引循环（数据相关）与 linear-attn 回退 chunk 核心。即便在合法面内，实测 DVM 生成 kernel 也慢于 CANN native aclnn。⇒ **“可融但当前不划算/不落地”**。

## 4. 限制与诚实边界
1. **跨栈不可直比**：8.5(CANN8.5/torch2.9) 与 26(CANN9.1/torch2.13) 的 eager 本就不等（26 慢 ~27%）。DVM 收益一律按**同 26 栈** eager 对照。
2. 模型运行形态 = transformers eager + **torch 回退**（无 fla/causal-conv1d/vllm fused experts）：MoE 专家是 python 循环、linear-attn 是 torch 实现 → 时延形态与真实部署（vllm fused MoE/量化 + 专用 attention kernel）不同。在此形态下 DVM/inductor 融合能触及的面更小；真实部署的融合收益可能不同（需在 fused-op 服务图上复测）。
3. NPU kernel 计数用 `aclnn*` CPU 派发代理（torch.profiler NPU activity 在本组合计 0）；数值是相对、同方法自洽，不是 msprof 级精确。
4. linear 层 DVM 编译失败(aclnnArange 161002) 未深挖（应为 PTA/CANN op 级 bug，非融合逻辑）；可用 workaround 后复测。
5. 未做真实 serve/长文 decode（35B>64GB，vllm 基础缺失），decode 端到端为单层形状外推；MoE 专家权重整读带宽项（~50MB/token topk8）依旧存在，DVM 不影响。
6. DVM 语义来自公开文档 + 本机 2.13.0rc1 loader 源码；matmul-fusion 默认关、开后有改善但仍未反超 eager（§2.3）。

## 5. 交付物与复现
| 文件 | 内容 |
|---|---|
| `REPORT.md` | 本报告 |
| `PROGRESS.md` | 全程日志（bring-up/实测/测量陷阱）|
| `dvm_map.md` | DVM 候选-判定表 |
| `bringup/03_dvm_min.py` + `dvm_min.log` | DVM go 判定（最小编译 OK）|
| `harness/lib_model.py` | 逐层 safetensors loader（两栈通用）|
| `harness/eager_stream.py` + `results/eager85_p64|p256.json`、`eager26_p256.json` | 40 层整网 eager（85/26 栈）|
| `harness/compile_prof.py` + `results/prof_{dvm,dvmmf}_idx3_s*.json`、`prof_eager26_idx3/idx0*.json` | 层级 DVM vs eager 计数/时延 |
| `harness/sub_prof.py` + `results/sub_*.json` | dense FFN（shared_expert）DVM vs eager |
| `harness/backend_measure.py` | 通用后端测量入口 |

复现（26 栈，需已提取 CANN9.1 + venv `/opt/dvmvenv`）：
```bash
source /tmp/cann91/dvm_env.sh
cd harness && export TORCHINDUCTOR_NPU_BACKEND=dvm
timeout 1800 /opt/dvmvenv/bin/python -u compile_prof.py --idx 3 --dev 0 --seq 512 --compile 1 --backend dvm
```
8.5 栈（host）：见 `PROGRESS.md` LD_LIBRARY_PATH，py3.11 torch2.9/torch_npu2.9。

## 6. 结论（写回用户关注点）
- DVM 本机**真跑通过**（唯一可用 inductor 后端；triton-default 因驱动探测不可用）。
- 对本模型（真权重、逐层、bf16、torch 回退形态）：**DVM 融合空间是结构性的（full 层设备算子 −40%，dense FFN 可融成 3 个 kernel），但未转成时延收益（全档 DVM ≤ eager，0.55–0.8×）**；linear 层编译失败阻断 30/40 层；模型时延主项（MoE 路由循环 + linear-attn 回退）不在 DVM 可融范围。
- 建议（若目标是把 DVM 变成收益）：① 先替换 MoE 专家 python 循环/linear-attn 回退为少数 fused 大算子（如 vllm `npu_grouped_matmul*`/FA/量产线性注意），DVM 才可能在其 epilogue 上体现；② 修 aclnnArange(161002) 使 linear 层可成图；③ 在真实 fused-op 服务图上（非 transformers 回退形态）重新评估。
