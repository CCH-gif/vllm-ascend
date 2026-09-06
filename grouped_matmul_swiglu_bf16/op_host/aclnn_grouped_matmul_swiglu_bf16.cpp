/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <dlfcn.h>
#include <new>
#include "aclnn_kernels/contiguous.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "grouped_matmul_swiglu_bf16.h"
#include "aclnn_grouped_matmul_swiglu_bf16.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static constexpr int64_t SPLIT = 2;
static constexpr size_t X_DIM_LIMIT = 2;
static constexpr size_t WEIGHT_DIM_LIMIT = 3;   // [E, K, N]
static constexpr size_t GROUP_LIST_DIM_LIMIT = 1;
static constexpr size_t OUT_DIM_LIMIT = 2;

static const std::initializer_list<DataType> X_DTYPE_SUPPORT_LIST = {DataType::DT_BF16, DataType::DT_FLOAT16};
static const std::initializer_list<DataType> WEIGHT_DTYPE_SUPPORT_LIST = {DataType::DT_BF16, DataType::DT_FLOAT16};
static const std::initializer_list<DataType> GROUP_LIST_DTYPE_SUPPORT_LIST = {DataType::DT_INT64, DataType::DT_INT32};
static const std::initializer_list<DataType> OUT_DTYPE_SUPPORT_LIST = {DataType::DT_BF16, DataType::DT_FLOAT16};

static bool CheckNotNull(const aclTensor* x, const aclTensorList* weight, const aclTensor* groupList, const aclTensor* y) {
  OP_CHECK_NULL(x, return false);
  OP_CHECK_NULL(weight, return false);
  OP_CHECK_NULL(groupList, return false);
  OP_CHECK_NULL(y, return false);
  return true;
}

static bool CheckInputOutDims(const aclTensor* x, const aclTensorList* weight, const aclTensor* groupList, const aclTensor* y) {
  OP_CHECK_WRONG_DIMENSION(x, X_DIM_LIMIT, return false);
  OP_CHECK_WRONG_DIMENSION((*weight)[0], WEIGHT_DIM_LIMIT, return false);
  OP_CHECK_WRONG_DIMENSION(groupList, GROUP_LIST_DIM_LIMIT, return false);
  OP_CHECK_WRONG_DIMENSION(y, OUT_DIM_LIMIT, return false);
  return true;
}

static bool CheckInputOutShape(const aclTensor* x, const aclTensorList* weight, const aclTensor* groupList, const aclTensor* y) {
  int64_t m = x->GetViewShape().GetDim(0);
  int64_t k = x->GetViewShape().GetDim(1);
  int64_t e = (*weight)[0]->GetViewShape().GetDim(0);
  int64_t n = (*weight)[0]->GetViewShape().GetDim(2);
  if (n % SPLIT != 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnGroupedMatmulSwigluBf16, N is %ld, not an even number.", n);
    return false;
  }
  int64_t nAfterHalve = n / SPLIT;
  op::Shape xExpectShape = {m, k};
  op::Shape weightExpectShape = {e, k, n};
  op::Shape yExpectShape = {m, nAfterHalve};
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(x, xExpectShape, return false);
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*weight)[0], weightExpectShape, return false);
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(y, yExpectShape, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensorList* weight, const aclTensor* groupList, const aclTensor* y) {
  OP_CHECK_DTYPE_NOT_SUPPORT(x, X_DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT((*weight)[0], WEIGHT_DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(groupList, GROUP_LIST_DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(y, OUT_DTYPE_SUPPORT_LIST, return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensorList* weight, const aclTensor* groupList, const aclTensor* y) {
  CHECK_RET(CheckNotNull(x, weight, groupList, y), ACLNN_ERR_PARAM_NULLPTR);
  CHECK_RET(CheckInputOutDims(x, weight, groupList, y), ACLNN_ERR_PARAM_INVALID);
  CHECK_RET(CheckInputOutShape(x, weight, groupList, y), ACLNN_ERR_PARAM_INVALID);
  CHECK_RET(CheckDtypeValid(x, weight, groupList, y), ACLNN_ERR_PARAM_INVALID);
  return ACLNN_SUCCESS;
}

static aclnnStatus aclnnGroupedMatmulSwigluBf16GetWorkspaceSizeCommon(const aclTensor* x, const aclTensorList* weight,
                                                                      const aclTensor* groupList, aclTensor* y,
                                                                      uint64_t* workspaceSize, aclOpExecutor** executor) {
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
  auto ret = CheckParams(x, weight, groupList, y);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  if (y->IsEmpty() || groupList->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }
  x = l0op::Contiguous(x, uniqueExecutor.get());
  CHECK_RET(x != nullptr, ACLNN_ERR_INNER_NULLPTR);
  groupList = l0op::Contiguous(groupList, uniqueExecutor.get());
  CHECK_RET(groupList != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto ret_0 = l0op::GroupedMatmulSwigluBF16(x, weight, groupList, uniqueExecutor.get());
  CHECK_RET(ret_0 != std::tuple<aclTensor*>(nullptr), ACLNN_ERR_INNER_NULLPTR);
  auto out0 = std::get<0>(ret_0);
  auto ret_1 = l0op::ViewCopy(out0, y, uniqueExecutor.get());
  CHECK_RET(ret_1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedMatmulSwigluBf16GetWorkspaceSize(const aclTensor* x, const aclTensorList* weight,
                                                         const aclTensor* groupList, aclTensor* y,
                                                         uint64_t* workspaceSize, aclOpExecutor** executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  L2_DFX_PHASE_1(aclnnGroupedMatmulSwigluBf16,
                 DFX_IN(x, weight, groupList),
                 DFX_OUT(y));
  return aclnnGroupedMatmulSwigluBf16GetWorkspaceSizeCommon(x, weight, groupList, y, workspaceSize, executor);
}

aclnnStatus aclnnGroupedMatmulSwigluBf16(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnGroupedMatmulSwigluBf16);
  CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
             "This is an error in GroupedMatmulSwigluBF16 launch aicore");
  return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
