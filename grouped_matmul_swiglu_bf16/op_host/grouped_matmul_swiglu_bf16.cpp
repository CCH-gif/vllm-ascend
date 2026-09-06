#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "grouped_matmul_swiglu_bf16.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(GroupedMatmulSwigluBF16);

const std::tuple<aclTensor*> GroupedMatmulSwigluBF16(const aclTensor* x,
                                                      const aclTensorList* weight,
                                                      const aclTensor* groupList,
                                                      aclOpExecutor* executor) {
    L0_DFX(GroupedMatmulSwigluBF16, x, weight, groupList);
    if (x == nullptr || weight == nullptr || weight->Size() == 0 || groupList == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "nullptr input.");
        return std::tuple<aclTensor*>(nullptr);
    }
    int64_t m = x->GetViewShape().GetDim(0);
    int64_t n = (*weight)[0]->GetViewShape().GetDim(2);  // weight[0] [E, K, N]
    int64_t nHalf = n / 2;
    gert::Shape outShape({m, nHalf});
    auto out = executor->AllocTensor(outShape, DataType::DT_BF16, ge::FORMAT_ND);
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out tensor failed.");
        return std::tuple<aclTensor*>(nullptr);
    }
    // Note: shape is decided on host from x[0] and weight[0][2]; do NOT call
    // INFER_SHAPE here -- it overrides the output with an unshaped [M, 0]
    // placeholder (weight is a DYNAMIC list and its element shape is not visible
    // to the InferShape context). B (moe_init_routing_custom) uses the same
    // pattern (AllocTensor + ADD_TO_LAUNCHER_LIST_AICORE only) and runs fine.
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(GroupedMatmulSwigluBF16, OP_INPUT(x, weight, groupList),
                                           OP_OUTPUT(out));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return std::tuple<aclTensor*>(nullptr);
    }
    return std::tie(out);
}

}  // namespace l0op
