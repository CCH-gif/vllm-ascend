/*
 * Host tiling for bf16 grouped_matmul + SwiGLU fused op.
 * Adapted from moe_grouped_matmul (mgm) host tiling (plain printf, direct
 * PlatformAscendC — no log/ops_log.h, error/ops_error.h, tiling/tiling_base.h),
 * with the fused-op SwiGLU epilogue params taken from grouped_matmul_swiglu_quant.
 *
 *   x [M, K] @ weight[E][K, N] -> g [M, N];  y = silu(g[:, :N/2]) * g[:, N/2:].
 */
#include <climits>
#include <cstdio>
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "grouped_matmul_swiglu_bf16_tiling.h"

#define OP_LOGE(fmt, ...)            \
    do {                             \
        printf(fmt "\n", ##__VA_ARGS__); \
    } while (0)

using namespace ge;
using namespace AscendC;
using namespace GroupedMatmulSwigluBF16Tiling;

namespace optiling {

constexpr uint32_t BASE_M = 128;
constexpr uint32_t BASE_N = 256;
constexpr uint64_t REDUCE_WS_SIZE = 4UL * 1024UL;  // fp32 reduce workspace

static uint32_t CalRows(const uint64_t ubSize, const uint64_t n) {
    // AIV per-row UB budget: bf16 in (n*2) + fp32 swiglu tmp (n/2*4) + bf16 out (n/2*2).
    const uint64_t perRow = n * 2UL + (n / 2UL) * 4UL + (n / 2UL) * 2UL;
    uint64_t usable = ubSize > REDUCE_WS_SIZE ? ubSize - REDUCE_WS_SIZE : ubSize;
    uint64_t rows = usable / perRow;
    if (rows == 0) rows = 1;
    if (rows > 128) rows = 128;
    return static_cast<uint32_t>(rows);
}

static ge::graphStatus TilingGMMSwigluBF16(gert::TilingContext* context) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    auto xTensor = context->GetInputTensor(X_INDEX);
    if (xTensor == nullptr) {
        OP_LOGE("grouped_matmul_swiglu_bf16 tiling: x is null.");
        return GRAPH_FAILED;
    }
    const int64_t m = xTensor->GetStorageShape().GetDim(0);
    const int64_t k = xTensor->GetStorageShape().GetDim(1);

    auto wTensor = context->GetDynamicInputTensor(WEIGHT_INDEX, 0);
    if (wTensor == nullptr) {
        OP_LOGE("grouped_matmul_swiglu_bf16 tiling: weight is null.");
        return GRAPH_FAILED;
    }
    const int64_t n = wTensor->GetStorageShape().GetDim(2);  // weight[0] [E, K, N]

    auto groupListTensor = context->GetInputTensor(GROUPLIST_INDEX);
    if (groupListTensor == nullptr) {
        OP_LOGE("grouped_matmul_swiglu_bf16 tiling: groupList is null.");
        return GRAPH_FAILED;
    }
    const int64_t groupNum = groupListTensor->GetStorageShape().GetDim(0);

    GMMSwigluBF16TilingData tilingData;
    tilingData.baseParams.set_groupNum(static_cast<uint32_t>(groupNum));
    tilingData.baseParams.set_coreNum(aicNum);
    tilingData.baseParams.set_K(static_cast<uint32_t>(k));
    tilingData.baseParams.set_N(static_cast<uint32_t>(n));
    tilingData.baseParams.set_M(static_cast<uint32_t>(m));
    tilingData.list.set_maxProcessRowNum(CalRows(ubSize, static_cast<uint64_t>(n)));
    tilingData.list.set_groupListLen(static_cast<uint32_t>(groupNum));
    tilingData.list.set_tokenLen(static_cast<uint32_t>(n));

    using namespace matmul_tiling;
    MatmulApiTiling tiling(ascendcPlatform);
    tiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_BF16);
    tiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_BF16);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_BF16);
    tiling.SetBias(false);
    tiling.SetShape(BASE_M, BASE_N, k);
    tiling.SetOrgShape(m, n, k);
    tiling.SetBufferSpace(-1, -1, -1);
    if (tiling.GetTiling(tilingData.mmTilingData) == -1) {
        OP_LOGE("grouped_matmul_swiglu_bf16 tiling: get tiling failed.");
        return GRAPH_FAILED;
    }

    // workspace: [sys workspace][intermediate g buffer (m * n * bf16)].
    auto workspaceSizes = context->GetWorkspaceSizes(1);
    workspaceSizes[0] = SYS_WORKSPACE_SIZE + static_cast<uint64_t>(m) * static_cast<uint64_t>(n) * 2UL;

    context->SetTilingKey(0);
    context->SetScheduleMode(BATCH_MODE_SCHEDULE);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    context->SetBlockDim(aicNum);
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return GRAPH_SUCCESS;
}

struct GMMSwigluBF16CompileInfo {};

static ge::graphStatus TilingPrepareGMMSwigluBF16(gert::TilingParseContext* context) {
    (void)context;
    return GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupedMatmulSwigluBF16)
    .Tiling(TilingGMMSwigluBF16)
    .TilingParse<GMMSwigluBF16CompileInfo>(TilingPrepareGMMSwigluBF16);

}  // namespace optiling
