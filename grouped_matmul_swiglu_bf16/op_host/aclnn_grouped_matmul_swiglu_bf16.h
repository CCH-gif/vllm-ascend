#ifndef OP_API_INC_GROUPED_MATMUL_SWIGLU_BF16_H
#define OP_API_INC_GROUPED_MATMUL_SWIGLU_BF16_H
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnGroupedMatmulSwigluBf16GetWorkspaceSize(
    const aclTensor* x, const aclTensorList* weight, const aclTensor* groupList,
    aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor);

__attribute__((visibility("default"))) aclnnStatus aclnnGroupedMatmulSwigluBf16(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif
