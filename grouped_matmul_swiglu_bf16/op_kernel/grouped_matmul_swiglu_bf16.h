/*
 * bf16 grouped_matmul + SwiGLU fused op (Qwen3.6-35B-A3B MoE FFN).
 *
 * Merged from two on-disk CANN references (see /tmp/vlla):
 *   * moe_grouped_matmul (mgm/)       — bf16 ND grouped GEMM, gives MatmulConfig + types
 *   * grouped_matmul_swiglu_quant_* (gmsq_ws/) — AIC/AIV split + SwiGLU epilogue
 *
 * Fusion semantics (from Qwen3_5MoeMLP.forward):  down(silu(gate(x)) * up(x))
 *   g = x @ W          // [m_e, 2048] @ [2048, 1024] -> g [m_e, 1024]  (gate|up stacked)
 *   h = silu(g[:,:512]) * g[:,512:]                                      // fused epilogue
 *   y = h @ Wd         // NOT in this op (separate down GEMM)
 *
 * AIC: grouped cube GEMM -> g written to bf16 workspace (mmOutGM).
 * AIV: read g, cast bf16->fp32, SwiGLU(silu(gate)*up), cast fp32->bf16, write h [m,512].
 */
#ifndef ASCENDC_GMM_SWIGLU_BF16_H
#define ASCENDC_GMM_SWIGLU_BF16_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace GMM_SWIGLU_BF16 {

constexpr uint32_t UB_BLOCK_UNIT_SIZE = 32;
constexpr uint32_t THRESHOLD_BLOCK_NUM = 8;
constexpr uint32_t SINGLE_CORE_M = 128;
constexpr uint32_t SINGLE_CORE_N = 512;
constexpr uint32_t SINGLE_CORE_K = 7168;
constexpr uint32_t BASIC_M = 128;
constexpr uint32_t BASIC_N = 256;
constexpr uint32_t BASIC_K = 128;
constexpr uint32_t STEP_M = 1;
constexpr uint32_t STEP_N = 1;
constexpr uint32_t STEP_Ka = 4;
constexpr uint32_t STEP_Kb = 4;
constexpr uint32_t DEPTH_A1 = 8;
constexpr uint32_t DEPTH_B1 = 8;
constexpr uint32_t VEC_LEN_ONCE_REPEAT_ELE = 64;
constexpr uint32_t VEC_LEN_ONCE_REPEAT_BLOCK = 8;
constexpr uint32_t BISECT = 2;              // stacked gate|up width = 2 * d_ff
constexpr uint32_t MOD_32_MASK = 0x1F;
constexpr uint32_t MOD_16_MASK = 0x0F;
constexpr uint32_t ALIGN_8_ELE = 8;
constexpr uint32_t ALIGN_16_ELE = 16;

// bf16 ND matmul config, taken verbatim from moe_grouped_matmul (mgm/kernel.h).
constexpr MatmulConfig matmulCFGUnitFlag{false, false, true, 0, 0, 0, false, false,
                                         false, false, false, 0, 0, 0, 0, 0, 0, 0, true};

struct MNConfig {
  int64_t m = 0, k = 0, n = 0;
  int64_t baseM = 0, baseN = 0;
  int64_t mIdx = 0, nIdx = 0;
  int64_t blockDimM = 0, blockDimN = 0;
  int64_t singleM = 0, singleN = 0;
  int64_t wBaseOffset = 0;
  int64_t xBaseOffset = 0, yBaseOffset = 0;
  int64_t workSpaceOffset = 0;
};

struct VecConfig {
  int64_t M = 0, usedCoreNum = 0;
  int64_t startIdx = 0, curIdx = 0;
  int64_t taskNum = 0;
  int64_t curGroupIdx = 0;
  int64_t outLoopNum = 0, innerLoopNum = 0, tailLoopNum = 0;
  int64_t nextUpadteInterVal = 0;
};

template <typename T>
__aicore__ inline T AlignUp(T a, T base) {
  return (a + base - 1) / base * base;
}

__aicore__ inline uint32_t Ceil(uint32_t a, uint32_t b) {
  return b == 0 ? a : (a + b - 1) / b;
}

// ---------------------------------------------------------------------------
// Compute class: AIC does grouped GEMM, AIV does SwiGLU epilogue.
// ---------------------------------------------------------------------------
template <class mmType>
class GMMSwigluBF16Compute {
public:
  __aicore__ inline GMMSwigluBF16Compute(mmType& mm_) : mm(mm_) {}

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR groupList, GM_ADDR y,
                              GM_ADDR workspace,
                              const GMMSwigluBF16BaseParams* baseParamsIn,
                              const TCubeTiling* mmTilingIn,
                              const GMMSwigluBF16List* listIn, TPipe* pipeIn);

  __aicore__ inline void Process();

private:
  __aicore__ inline void MMCompute(uint32_t groupIdx, MNConfig& mnConfig);
  __aicore__ inline void SetMNConfig(const int32_t splitValue, MNConfig& mnConfig);
  __aicore__ inline void MNBlockIdxCompute(MNConfig& mnConfig, const uint32_t curBlock,
                                           const uint32_t count);
  __aicore__ inline void UpdateVecConfig(uint32_t blockIdx, VecConfig& vecConfig);
  __aicore__ inline void customDataCopyIn(uint32_t outLoopIdx);
  __aicore__ inline void Swiglu(uint32_t loopIdx);
  __aicore__ inline void customDataCopyOut();

private:
  mmType& mm;
  const GMMSwigluBF16BaseParams* baseParams;
  const GMMSwigluBF16List* list;
  const TCubeTiling* mmTiling;
  TPipe* pipe;
  VecConfig vecConfig;
  GlobalTensor<bfloat16_t> xGM, weightGM;
  GlobalTensor<int64_t> groupListGM;
  GlobalTensor<bfloat16_t> yGM;
  GlobalTensor<bfloat16_t> mmOutGM;   // AIC writes g [m,1024] here (bf16 workspace)
  TQue<QuePosition::VECIN, 1> mmOutQueue;
  TBuf<TPosition::VECCALC> reduceWorkspace;
  uint64_t xTensorPtr, weightTensorPtr;
  AscendC::ListTensorDesc weightListDesc;   // weight is a DYNAMIC tensor list; parse to elem-0 data base
  uint32_t aicCoreNum, aivCoreNum;
};

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
template <class mmType>
__aicore__ inline void GMMSwigluBF16Compute<mmType>::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR groupList, GM_ADDR y, GM_ADDR workspace,
    const GMMSwigluBF16BaseParams* baseParamsIn, const TCubeTiling* mmTilingIn,
    const GMMSwigluBF16List* listIn, TPipe* pipeIn) {
  aicCoreNum = GetBlockNum();
  aivCoreNum = aicCoreNum * 2;
  mmTiling = mmTilingIn;
  baseParams = baseParamsIn;
  list = listIn;
  pipe = pipeIn;
  xTensorPtr = reinterpret_cast<uint64_t>(x);
  // weight is a DYNAMIC tensor list -> the runtime hands the kernel a *list
  // descriptor table*, not the raw weight data. Parse elem 0 (the single
  // [E,K,N] bf16 tensor) for its data base, then slice per-group by K*N.
  // Root cause found 09-06 vs two working references: quant op uses
  // GetTensorAddr(groupIdx,...), moe_grouped_matmul uses ListTensorDesc +
  // GetDataPtr(0); our previous raw-pointer read treated the descriptor table
  // as bf16 data -> cube reads garbage -> device never completes.
  weightListDesc.Init((__gm__ void*)weight);
  weightTensorPtr = reinterpret_cast<uint64_t>(
      (__gm__ uint8_t*)weightListDesc.GetDataPtr<__gm__ uint8_t>(0));
  groupListGM.SetGlobalBuffer((__gm__ int64_t*)groupList, list->groupListLen);
  yGM.SetGlobalBuffer((__gm__ bfloat16_t*)y, baseParams->M * list->tokenLen / BISECT);
  mmOutGM.SetGlobalBuffer((__gm__ bfloat16_t*)workspace,
                          baseParams->M * list->tokenLen);
}

// ---------------------------------------------------------------------------
// Process
// ---------------------------------------------------------------------------
template <class mmType>
__aicore__ inline void GMMSwigluBF16Compute<mmType>::Process() {
  MNConfig mnConfig;
  if ASCEND_IS_AIC {
    int32_t prevSplitValue = 0;
    for (uint32_t groupIdx = 0, count = 0; groupIdx < list->groupListLen; ++groupIdx) {
      int32_t currSplitValue = static_cast<int32_t>(groupListGM.GetValue(groupIdx));
      int32_t splitValue = currSplitValue - prevSplitValue;
      prevSplitValue = currSplitValue;
      SetMNConfig(splitValue, mnConfig);
      if (mnConfig.m <= 0 || mnConfig.k <= 0 || mnConfig.n <= 0) {
        continue;
      }
      mnConfig.blockDimM = Ceil((uint32_t)mnConfig.m, mnConfig.singleM);
      mnConfig.blockDimN = Ceil((uint32_t)mnConfig.n, mnConfig.singleN);

      uint32_t curCount = count + mnConfig.blockDimM * mnConfig.blockDimN;
      uint32_t blockIdx = GetBlockIdx();
      uint32_t curBlock = blockIdx >= count ? blockIdx : blockIdx + baseParams->coreNum;
      uint32_t thresholdM_dimN = THRESHOLD_BLOCK_NUM * mnConfig.blockDimN;

      while (curBlock < curCount) {
        MNBlockIdxCompute(mnConfig, curBlock, count);
        MMCompute(groupIdx, mnConfig);
        curBlock += aicCoreNum;
      }
      count = curCount % baseParams->coreNum;
      mnConfig.xBaseOffset += mnConfig.m * mnConfig.k;
      mnConfig.yBaseOffset += mnConfig.m * mnConfig.n;
    }
    SyncAll<false>();
  }

  if ASCEND_IS_AIV {
    uint32_t blockIdx = GetBlockIdx();
    UpdateVecConfig(blockIdx, vecConfig);
    if (blockIdx < vecConfig.usedCoreNum) {
      LocalTensor<bfloat16_t> mmLocal = mmOutQueue.AllocTensor<bfloat16_t>();
      mmOutQueue.EnQue(mmLocal);
    }
    SyncAll<false>();
    if (blockIdx < vecConfig.usedCoreNum) {
      for (uint32_t outLoopIdx = 0; outLoopIdx < vecConfig.outLoopNum; outLoopIdx++) {
        vecConfig.innerLoopNum = outLoopIdx == (vecConfig.outLoopNum - 1)
                                     ? vecConfig.tailLoopNum
                                     : list->maxProcessRowNum;
        customDataCopyIn(outLoopIdx);
        for (uint32_t innerLoopIdx = 0; innerLoopIdx < vecConfig.innerLoopNum; innerLoopIdx++) {
          Swiglu(innerLoopIdx);
        }
        customDataCopyOut();
      }
      LocalTensor<bfloat16_t> mmLocal = mmOutQueue.DeQue<bfloat16_t>();
      mmOutQueue.FreeTensor(mmLocal);
    }
  }
}

// ---------------------------------------------------------------------------
// AIC: per-block cube matmul -> g (bf16) workspace
// ---------------------------------------------------------------------------
template <class mmType>
__aicore__ inline void GMMSwigluBF16Compute<mmType>::SetMNConfig(const int32_t splitValue,
                                                                 MNConfig& mnConfig) {
  mnConfig.m = static_cast<uint32_t>(splitValue);
  mnConfig.k = baseParams->K;
  mnConfig.n = baseParams->N;
  mnConfig.baseM = BASIC_M;
  mnConfig.baseN = BASIC_N;
  mnConfig.singleM = SINGLE_CORE_M;
  mnConfig.singleN = SINGLE_CORE_N;
}

template <class mmType>
__aicore__ inline void GMMSwigluBF16Compute<mmType>::MNBlockIdxCompute(MNConfig& mnConfig,
                                                                      const uint32_t curBlock,
                                                                      const uint32_t count) {
  mnConfig.mIdx = (curBlock - count) / mnConfig.blockDimN;
  mnConfig.nIdx = (curBlock - count) % mnConfig.blockDimN;
}

template <class mmType>
__aicore__ inline void GMMSwigluBF16Compute<mmType>::MMCompute(uint32_t groupIdx,
                                                               MNConfig& mnConfig) {
  uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;
  uint32_t curSingleN = mnConfig.nIdx < mnConfig.blockDimN - 1 ? mnConfig.singleN
                                                               : mnConfig.n - tailN;
  uint32_t curSingleM = mnConfig.mIdx < mnConfig.blockDimM - 1
                            ? mnConfig.singleM
                            : mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
  uint64_t xOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.k;
  uint64_t outOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.n + tailN;

  xGM.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(xTensorPtr) + mnConfig.xBaseOffset);
  // weight: [groupIdx][k][n], ND contiguous per group
  weightGM.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(weightTensorPtr) +
                           (uint64_t)groupIdx * baseParams->K * baseParams->N + tailN);
  if (mnConfig.blockDimM == 1) {
    weightGM.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  }
  mnConfig.workSpaceOffset = outOffset + mnConfig.yBaseOffset;
  mm.SetOrgShape(mnConfig.m, mnConfig.n, mnConfig.k);
  mm.SetSingleShape(curSingleM, curSingleN, mnConfig.k);
  mm.SetTensorA(xGM[xOffset], false);
  mm.SetTensorB(weightGM, false);
  mm.template IterateAll<false>(mmOutGM[mnConfig.workSpaceOffset], 0);
}

// ---------------------------------------------------------------------------
// AIV: SwiGLU epilogue
// ---------------------------------------------------------------------------
template <class mmType>
__aicore__ inline void GMMSwigluBF16Compute<mmType>::UpdateVecConfig(uint32_t blockIdx,
                                                                     VecConfig& vecConfig) {
  int64_t prevM = 0;
  for (uint32_t groupIdx = 0; groupIdx < list->groupListLen; groupIdx++) {
    int64_t currM = groupListGM.GetValue(groupIdx);
    vecConfig.M += (currM - prevM);
    prevM = currM;
  }
  uint32_t eachCoreTaskNum = (vecConfig.M + aivCoreNum - 1) / aivCoreNum;
  vecConfig.usedCoreNum = vecConfig.M >= aivCoreNum ? aivCoreNum : vecConfig.M;
  uint32_t tailCoreIdx = vecConfig.M - (eachCoreTaskNum - 1) * vecConfig.usedCoreNum;
  vecConfig.taskNum = blockIdx < tailCoreIdx ? eachCoreTaskNum : eachCoreTaskNum - 1;
  vecConfig.startIdx = blockIdx < tailCoreIdx ? eachCoreTaskNum * blockIdx
                                              : ((eachCoreTaskNum - 1) * blockIdx + tailCoreIdx);
  vecConfig.curIdx = vecConfig.startIdx;
  int64_t curStartIdx = vecConfig.startIdx;
  prevM = 0;
  for (uint32_t groupIdx = 0; groupIdx < list->groupListLen; groupIdx++) {
    int64_t currM = groupListGM.GetValue(groupIdx);
    int64_t tempM = currM - prevM;
    prevM = currM;
    if (curStartIdx >= 0 && curStartIdx - tempM < 0) {
      vecConfig.curGroupIdx = groupIdx;
      vecConfig.nextUpadteInterVal = tempM - curStartIdx;
    }
    curStartIdx -= tempM;
  }
  vecConfig.outLoopNum = (vecConfig.taskNum + list->maxProcessRowNum - 1) / list->maxProcessRowNum;
  vecConfig.tailLoopNum = vecConfig.taskNum % list->maxProcessRowNum
                              ? vecConfig.taskNum % list->maxProcessRowNum
                              : list->maxProcessRowNum;
  pipe->Reset();
  pipe->InitBuffer(mmOutQueue, 1, list->maxProcessRowNum * list->tokenLen * sizeof(bfloat16_t));
  pipe->InitBuffer(reduceWorkspace, 1024 * sizeof(float));
}

template <class mmType>
__aicore__ inline void GMMSwigluBF16Compute<mmType>::customDataCopyIn(uint32_t outLoopIdx) {
  LocalTensor<bfloat16_t> inMM = mmOutQueue.DeQue<bfloat16_t>();
  DataCopyExtParams copyParams{1,
                               static_cast<uint32_t>(vecConfig.innerLoopNum * list->tokenLen *
                                                     sizeof(bfloat16_t)),
                               0, 0, 0};
  DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};
  DataCopyPad(inMM, mmOutGM[vecConfig.curIdx * list->tokenLen], copyParams, padParams);
  mmOutQueue.EnQue(inMM);
}

template <class mmType>
__aicore__ inline void GMMSwigluBF16Compute<mmType>::Swiglu(uint32_t loopIdx) {
  LocalTensor<bfloat16_t> inMM = mmOutQueue.DeQue<bfloat16_t>();
  // cast bf16 -> fp32 in place (uses half of UB; inMM reinterpreted)
  LocalTensor<float> inF = inMM.ReinterpretCast<float>();
  uint32_t half = list->tokenLen / BISECT;   // d_ff = 512
  Cast(inF, inMM, RoundMode::CAST_NONE, list->tokenLen);
  PipeBarrier<PIPE_V>();

  LocalTensor<float> workspaceLocal = reduceWorkspace.Get<float>();
  // gate = left half (offset 0), up = right half (offset half)
  LocalTensor<float> src0Local = inF[loopIdx * list->tokenLen + half];  // up
  LocalTensor<float> src1Local = inF[loopIdx * list->tokenLen];         // gate
  SwiGLU<float, false>(workspaceLocal, src0Local, src1Local, 1.0f, half);
  PipeBarrier<PIPE_ALL>();

  // result silu(gate)*up is in workspaceLocal [0..half); cast back to bf16 into inMM
  Cast(inMM[loopIdx * half], workspaceLocal, RoundMode::CAST_NONE, half);
  mmOutQueue.EnQue(inMM);
}

template <class mmType>
__aicore__ inline void GMMSwigluBF16Compute<mmType>::customDataCopyOut() {
  LocalTensor<bfloat16_t> inMM = mmOutQueue.DeQue<bfloat16_t>();
  uint32_t half = list->tokenLen / BISECT;
  DataCopyParams copyParams{1, static_cast<uint16_t>(vecConfig.innerLoopNum * half * sizeof(bfloat16_t) / 32), 0, 0};
  DataCopy(yGM[vecConfig.startIdx * half], inMM, copyParams);
  vecConfig.startIdx += vecConfig.innerLoopNum;
  mmOutQueue.EnQue(inMM);
}

}  // namespace GMM_SWIGLU_BF16

#endif  // ASCENDC_GMM_SWIGLU_BF16_H
