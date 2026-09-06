/*
 * Entry kernel: bf16 grouped_matmul + SwiGLU fused op.
 * AIC runs grouped cube GEMM -> g (bf16 workspace); AIV runs SwiGLU epilogue -> h (bf16).
 */
#include "grouped_matmul_swiglu_bf16.h"

using namespace AscendC;
using namespace GMM_SWIGLU_BF16;

using xType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
using weightType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
using yType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using biasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using mmT = matmul::MatmulImpl<xType, weightType, yType, biasType, matmulCFGUnitFlag>;

extern "C" __global__ __aicore__ void grouped_matmul_swiglu_bf16(GM_ADDR x, GM_ADDR weight,
                                                                 GM_ADDR groupList, GM_ADDR y,
                                                                 GM_ADDR workspace, GM_ADDR tiling) {
  TPipe tPipe;
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  AscendCUtils::SetOverflow(1);
  GM_ADDR user1 = GetUserWorkspace(workspace);
  if (TILING_KEY_IS(0)) {
    KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);
    mmT mm;
    GET_TILING_DATA_MEMBER(GMMSwigluBF16TilingData, baseParams, baseParams_, tiling);
    GET_TILING_DATA_MEMBER(GMMSwigluBF16TilingData, mmTilingData, mmTilingData_, tiling);
    GET_TILING_DATA_MEMBER(GMMSwigluBF16TilingData, list, list_, tiling);
    if ASCEND_IS_AIC {
      mm.Init(&mmTilingData_, &tPipe);
    }
    GMMSwigluBF16Compute<mmT> op(mm);
    op.Init(x, weight, groupList, y, user1, &baseParams_, &mmTilingData_, &list_, &tPipe);
    op.Process();
  }
}
