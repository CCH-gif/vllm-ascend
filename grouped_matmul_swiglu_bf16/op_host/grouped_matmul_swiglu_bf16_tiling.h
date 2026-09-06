/*
 * Tiling data for bf16 grouped_matmul + SwiGLU fused op.
 */
#ifndef ASCENDC_GROUPED_MATMUL_SWIGLU_BF16_TILING_H
#define ASCENDC_GROUPED_MATMUL_SWIGLU_BF16_TILING_H

#include <set>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GMMSwigluBF16BaseParams)
  TILING_DATA_FIELD_DEF(uint32_t, groupNum);
  TILING_DATA_FIELD_DEF(uint32_t, coreNum);
  TILING_DATA_FIELD_DEF(uint32_t, K);
  TILING_DATA_FIELD_DEF(uint32_t, N);
  TILING_DATA_FIELD_DEF(uint32_t, M);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GMMSwigluBF16BaseParamsOp, GMMSwigluBF16BaseParams)

BEGIN_TILING_DATA_DEF(GMMSwigluBF16List)
  TILING_DATA_FIELD_DEF(uint32_t, maxProcessRowNum);
  TILING_DATA_FIELD_DEF(uint32_t, groupListLen);
  TILING_DATA_FIELD_DEF(uint32_t, tokenLen);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GMMSwigluBF16ListOp, GMMSwigluBF16List)

BEGIN_TILING_DATA_DEF(GMMSwigluBF16TilingData)
  TILING_DATA_FIELD_DEF_STRUCT(GMMSwigluBF16BaseParams, baseParams);
  TILING_DATA_FIELD_DEF_STRUCT(GMMSwigluBF16List, list);
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GroupedMatmulSwigluBF16, GMMSwigluBF16TilingData)

}  // namespace optiling

namespace GroupedMatmulSwigluBF16Tiling {
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t WEIGHT_INDEX = 1;
constexpr uint32_t GROUPLIST_INDEX = 2;
constexpr uint32_t BATCH_MODE_SCHEDULE = 1;
constexpr uint32_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr uint32_t BF16_DTYPE_SIZE = 2;
}  // namespace GroupedMatmulSwigluBF16Tiling

#endif  // ASCENDC_GROUPED_MATMUL_SWIGLU_BF16_TILING_H
