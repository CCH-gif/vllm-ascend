#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LoRA bgmv/sgmv ops backed by Triton kernels.

``lora_ops_triton`` registers the kernels in the ``vllm_ascend_triton``
namespace (``torch.library.custom_op``) so ``torch._dynamo`` treats them like
the stock ``torch.ops._C_ascend.*`` ops, and the wrappers below keep the stock
``vllm_ascend.lora.lora_ops`` API.

Each wrapper takes the custom op when a compilation is in flight and the bare
implementation otherwise.  The custom op is an opaque boundary -- that is what
makes it traceable, and it costs nothing under ``torch.compile``.

Registering at the Python dispatch key is enough for aclgraph: ACL capture is
stream-level, not dispatch-key-level, so the kernel launches are recorded into
the captured graph and replay with no Python on the replay path (measured on
910B4: 1 / 10 / 50 ops in a graph all replay in 8.8-8.9us, i.e. +0.0us per op).
"""
import torch

from vllm_ascend.lora import lora_ops_triton as _triton  # registers the custom ops

# Bound locally: this is on the per-call path of every LoRA op.
_is_compiling = torch.compiler.is_compiling


def bgmv_shrink(inputs, lora_a_weights, output_tensor, lora_indices_tensor, scaling=1.0):
    if _is_compiling():
        torch.ops.vllm_ascend_triton.bgmv_shrink(
            inputs, lora_a_weights, output_tensor, lora_indices_tensor, scaling)
    else:
        _triton.bgmv_shrink(
            inputs, lora_a_weights, output_tensor, lora_indices_tensor, scaling)
    return output_tensor


def bgmv_expand(inputs, lora_b_weights, output_tensor, lora_indices_tensor, add_inputs=True):
    return bgmv_expand_slice(inputs, lora_b_weights, output_tensor, lora_indices_tensor,
                             0, output_tensor.size(1), add_inputs)


def bgmv_expand_slice(inputs, lora_b_weights, output_tensor, lora_indices_tensor,
                      slice_offset, slice_size, add_inputs=True):
    if _is_compiling():
        torch.ops.vllm_ascend_triton.bgmv_expand_slice(
            inputs, lora_b_weights, output_tensor, lora_indices_tensor,
            slice_offset, slice_size)
    else:
        _triton.bgmv_expand_slice(
            inputs, lora_b_weights, output_tensor, lora_indices_tensor,
            slice_offset, slice_size)
    return output_tensor


def sgmv_shrink(inputs, lora_a_weights, output_tensor, b_seq_start_loc, seq_len_tensor,
                lora_indices_tensor, batches, max_seq_length, token_nums, scaling):
    if _is_compiling():
        torch.ops.vllm_ascend_triton.sgmv_shrink(
            inputs, lora_a_weights, output_tensor, b_seq_start_loc,
            lora_indices_tensor, scaling)
    else:
        _triton.sgmv_shrink(
            inputs, lora_a_weights, output_tensor, b_seq_start_loc,
            lora_indices_tensor, scaling)
    return output_tensor


def sgmv_expand(inputs, lora_b_weights, output_tensor, b_seq_start_loc, seq_len_tensor,
                lora_indices_tensor, batches, max_seq_length, token_nums, add_inputs=False):
    return sgmv_expand_slice(inputs, lora_b_weights, output_tensor, b_seq_start_loc, seq_len_tensor,
                             lora_indices_tensor, batches, max_seq_length, token_nums,
                             0, output_tensor.size(1), add_inputs)


def sgmv_expand_slice(inputs, lora_b_weights, output_tensor, b_seq_start_loc, seq_len_tensor,
                      lora_indices_tensor, batches, max_seq_length, token_nums,
                      slice_offset, slice_size, add_inputs=False):
    if _is_compiling():
        torch.ops.vllm_ascend_triton.sgmv_expand_slice(
            inputs, lora_b_weights, output_tensor, b_seq_start_loc,
            lora_indices_tensor, slice_offset, slice_size)
    else:
        _triton.sgmv_expand_slice(
            inputs, lora_b_weights, output_tensor, b_seq_start_loc,
            lora_indices_tensor, slice_offset, slice_size)
    return output_tensor
