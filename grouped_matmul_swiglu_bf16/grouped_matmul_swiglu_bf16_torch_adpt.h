/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef GROUPED_MATMUL_SWIGLU_BF16_TORCH_ADPT_H
#define GROUPED_MATMUL_SWIGLU_BF16_TORCH_ADPT_H
namespace vllm_ascend {
at::Tensor grouped_matmul_swiglu_bf16(const at::Tensor &x, const at::TensorList &weight, const at::Tensor &group_list)
{
    int m = x.sizes()[0];
    int n = weight[0].sizes()[2];  // weight[0] is [E, K, N]
    at::Tensor output = at::empty({m, n/2}, x.options());
    EXEC_NPU_CMD(aclnnGroupedMatmulSwigluBf16, x, weight, group_list, output);
    return output;
}
}
#endif
