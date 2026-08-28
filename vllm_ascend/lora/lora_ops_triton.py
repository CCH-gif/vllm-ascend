"""Triton bgmv/sgmv ops exposed as torch custom ops (torch.library).

The kernels are registered in the ``vllm_ascend_triton`` namespace so that
torch._dynamo treats them as opaque, allowed-in-graph nodes -- the same
mechanism the stock ``torch.ops._C_ascend.*`` ops use.  All runtime checks
(dtype / NR / token-count consistency) and the AscendC fallbacks live inside
the eager impls, which execute outside the traced graph (during aclgraph
capture recording the impls run eagerly and their kernel launches get
recorded into the graph).

Kernel launches go through a C++ launcher (rtKernelLaunch on a flat-packed
arg buffer, ~14us/launch instead of the triton python path's ~66us+):
- per-(kernel, constexpr, dtype) case: compile once via warmup, register
  once, verify-retry warmup until launches actually land.
- CANN quirk: the first launches of a freshly registered binary are silently
  dropped until the device-side load settles (~tens of ms); the verify-retry
  loop launches on a dummy output until it changes.  The syncs in that loop
  are only on case creation -- vllm warmup populates all cases before
  aclgraph capture, so no sync happens during capture.  If that assumption
  ever breaks (EE1016 during capture), set TRITON_LORA_CPP_VERIFY=0.
- TRITON_LORA_CPP=0 restores the plain triton launch path.
- Indices are converted to int32 in the wrappers (same as the plain triton
  path; int64 indices are a separate follow-up optimization).

The module also keeps plain-Triton wrappers (``bgmv_shrink``, ...) that the
custom op impls call and that test_compare.py uses directly.
"""
import ctypes
import os
import struct
import time

import torch
from torch.library import custom_op, register_fake

from vllm_ascend.lora import lora_ops_triton_kernels as K

_announced = False
_NR_MAX = 16  # prefix-sum request mapping is O(NR^2); beyond this use AscendC


def _announce():
    global _announced
    if not _announced:
        mode = "C++ rtKernelLaunch" if _cpp_enabled() else "triton python path"
        print(f"[triton-lora] dispatch ACTIVE ({mode})", flush=True)
        _announced = True


def _triton_dtype_ok(t: torch.Tensor) -> bool:
    return t.dtype in (torch.float16, torch.bfloat16)


def _ascend_bgmv_shrink(inputs, lora_a_weights, output_tensor, lora_indices_tensor, scaling):
    torch.ops._C_ascend.bgmv_shrink(inputs, lora_a_weights, lora_indices_tensor, output_tensor, scaling)


def _ascend_bgmv_expand(inputs, lora_b_weights, output_tensor, lora_indices_tensor,
                        slice_offset, slice_size):
    torch.ops._C_ascend.bgmv_expand(inputs, lora_b_weights, lora_indices_tensor,
                                    output_tensor, slice_offset, slice_size)


def _ascend_sgmv_shrink(inputs, lora_a_weights, output_tensor, lora_indices_tensor,
                        seq_len_tensor, scaling):
    torch.ops._C_ascend.sgmv_shrink(inputs, lora_a_weights, lora_indices_tensor,
                                    seq_len_tensor, output_tensor, scaling)


def _ascend_sgmv_expand(inputs, lora_b_weights, output_tensor, lora_indices_tensor,
                        seq_len_tensor, slice_offset, slice_size):
    torch.ops._C_ascend.sgmv_expand(inputs, lora_b_weights, lora_indices_tensor,
                                    seq_len_tensor, output_tensor, slice_offset, slice_size)


def _sgmv_triton_ok(seq_len_tensor: torch.Tensor) -> bool:
    # NOTE: no device syncs allowed here -- this runs during aclgraph capture
    # (EE1016: synchronizing a captured stream is not supported).  The kernel
    # maps token rows via prefix sum and skips rows beyond the total, so a
    # sum-vs-batches check is not needed.
    return int(seq_len_tensor.numel()) <= _NR_MAX


# ---- C++ launcher (rtKernelLaunch direct) ----

_CPP_DIR = os.path.dirname(os.path.abspath(__file__))
_CPP_SRC = os.path.join(_CPP_DIR, "lora_cpp_launcher.cpp")
_CPP_SO = os.path.join(_CPP_DIR, "lora_cpp_launcher.cpython-312-aarch64-linux-gnu.so")

_CPP_STATE = None          # (CDLL, ffts_addr) or (None, None) on failure
_CPP_CASES = {}            # key -> case dict
_CPP_FAILED = set()        # keys that must use the triton fallback


def _cpp_enabled() -> bool:
    return os.environ.get("TRITON_LORA_CPP", "1") != "0"


def _cpp_setup():
    global _CPP_STATE
    if _CPP_STATE is not None:
        return _CPP_STATE
    try:
        so = _CPP_SO
        if not os.path.exists(so):
            from triton.backends.ascend.utils import _build_npu_ext
            so = _build_npu_ext("lora_cpp_launcher", _CPP_SRC,
                                kernel_launcher="torch")
        CL = ctypes.CDLL(so)
        CL.lora_register_kernel.restype = ctypes.c_uint64
        CL.lora_register_kernel.argtypes = [ctypes.c_char_p, ctypes.c_void_p,
                                            ctypes.c_uint64, ctypes.c_char_p,
                                            ctypes.c_int]
        CL.lora_launch_flat.restype = ctypes.c_uint64
        CL.lora_launch_flat.argtypes = [ctypes.c_uint64, ctypes.c_uint64,
                                        ctypes.c_int32, ctypes.c_void_p,
                                        ctypes.c_uint64]
        CL.lora_get_ffts_addr.restype = ctypes.c_uint64
        CL.lora_get_ffts_addr.argtypes = [ctypes.c_int]
        CL.lora_peek_stub.restype = ctypes.c_uint64
        CL.lora_peek_stub.argtypes = [ctypes.c_uint64]
        _CPP_STATE = (CL, CL.lora_get_ffts_addr(0))
    except Exception as e:
        print(f"[triton-lora] C++ launcher unavailable, triton fallback: {e}",
              flush=True)
        _CPP_STATE = (None, None)
    return _CPP_STATE


def _cpp_case_key(kernel_fn, kwargs, tensors):
    return (kernel_fn.__name__,
            tuple(sorted((k, str(v)) for k, v in kwargs.items())),
            tuple(str(t.dtype) for t in tensors))


def _cpp_launch(case, grid_x, ptrs, floats=()):
    CL, _ = _cpp_setup()
    b = case["buf"]
    off = 24  # [ffts][syncBlockLock][workspace]
    for p in ptrs:
        struct.pack_into("<Q", b, off, p)
        off += 8
    for f in floats:
        struct.pack_into("<f", b, off, f)
        off += 4
    off = (off + 3) & ~3
    struct.pack_into("<iii", b, off, grid_x, 1, 1)
    return CL.lora_launch_flat(case["func"],
                               torch.npu.current_stream().npu_stream,
                               grid_x, b, off + 12)


def _cpp_peek_stub(func):
    CL, _ = _cpp_setup()
    if CL is None:
        return 0
    return CL.lora_peek_stub(func)


def _cpp_make_case(kernel_fn, kwargs, example_args, grid):
    """Compile + register one (kernel, constexpr, dtype) case and verify the
    launches land.  Returns case dict or None (caller falls back to triton).

    example_args: kernel positional args with the real tensors of the first
    call; tensors[2] is indices and tensors[-1] is the output for all 4
    kernels.  Compile/warmup and the verify-retry run on dummy output + dummy
    indices so the real output_tensor is never written here.
    """
    CL, ffts = _cpp_setup()
    if CL is None:
        return None

    tensors = [t for t in example_args if isinstance(t, torch.Tensor)]
    floats = [f for f in example_args if not isinstance(f, torch.Tensor)]
    ti = [i for i, a in enumerate(example_args)
          if isinstance(a, torch.Tensor)]

    # dummy args for compile+verify, DATA-INDEPENDENT: x and w -> ones so the
    # kernel output is deterministically nonzero regardless of what the serve
    # passes (vllm's LoRA warmup may use zeroed dummy weights -> real-data
    # verify would read a legitimate all-zero result and false-negative).
    # indices -> zeros (all -> LoRA 0); every y-position arg -> zero dummies
    # (the two-y kernels write the FIRST y and read the second, so both must
    # be dummies, and the real output tensors are never written here).
    dummy_args = list(example_args)
    dummy_args[ti[0]] = torch.ones_like(example_args[ti[0]])
    dummy_args[ti[1]] = torch.ones_like(example_args[ti[1]])
    dummy_args[ti[2]] = torch.zeros_like(example_args[ti[2]])
    y_positions = [ti[-1]]
    if len(ti) >= 2 and tuple(tensors[-2].shape) == tuple(tensors[-1].shape):
        y_positions = [ti[-2], ti[-1]]
    for p in y_positions:
        dummy_args[p] = torch.zeros_like(example_args[p])

    compiled = kernel_fn.warmup(*dummy_args, **kwargs, grid=grid)
    data = bytes(compiled.kernel)
    buf = ctypes.create_string_buffer(data)
    mode = getattr(compiled.metadata, "mix_mode", "aiv")
    func = CL.lora_register_kernel(compiled.name.encode(), buf, len(data),
                                   mode.encode(), 0)
    if not func:
        return None

    ptrs_d = [t.data_ptr() for t in dummy_args if isinstance(t, torch.Tensor)]
    n = 24 + 8 * len(ptrs_d) + 4 * len(floats) + 16
    cbuf = ctypes.create_string_buffer(n)
    struct.pack_into("<QQQ", cbuf, 0, ffts, 0, 0)
    case = dict(func=func, buf=cbuf, nptrs=len(ptrs_d), nfloats=len(floats))

    if os.environ.get("TRITON_LORA_CPP_VERIFY", "1") == "0":
        return case

    # verify-retry: fresh registrations drop launches until the device-side
    # load settles; retry until any dummy y changes from zero.
    ctx = []
    try:
        if torch.compiler.is_compiling():
            ctx.append("is_compiling")
    except Exception:
        pass
    try:
        if torch.npu.is_current_stream_capturing():
            ctx.append("stream_capturing")
    except Exception:
        pass
    t0 = time.perf_counter()
    tries = 0
    while True:
        for p in y_positions:
            dummy_args[p].zero_()
        ret = _cpp_launch(case, grid[0], ptrs_d, floats)
        torch.npu.synchronize()
        tries += 1
        y_sum = sum(float(dummy_args[p].abs().sum()) for p in y_positions)
        if y_sum != 0.0:
            print(f"[triton-lora] case OK {kernel_fn.__name__} "
                  f"kw={sorted(kwargs.items())} tries={tries} "
                  f"ctx={ctx}", flush=True)
            return case
        if time.perf_counter() - t0 > 10.0:
            # cross-check: does the triton path write on the same dummy args?
            try:
                for p in y_positions:
                    dummy_args[p].zero_()
                kernel_fn[(grid[0],)](*dummy_args, **kwargs)
                torch.npu.synchronize()
                tri_sum = sum(float(dummy_args[p].abs().sum())
                              for p in y_positions)
            except Exception as e:
                tri_sum = f"ERR {e}"
            try:
                stub = _cpp_peek_stub(func)
            except Exception:
                stub = 0
            print(f"[triton-lora] WARN: {kernel_fn.__name__} warmup launches "
                  f"never landed, triton fallback. kw={sorted(kwargs.items())} "
                  f"shapes={[tuple(t.shape) for t in tensors]} "
                  f"x_sum={float(tensors[0].abs().sum()):.1f} "
                  f"w_sum={float(tensors[1].abs().sum()):.1f} "
                  f"idx[:8]={[int(v) for v in tensors[2].flatten().tolist()[:8]]} "
                  f"seq[:8]={[int(v) for v in tensors[3].flatten().tolist()[:8]] if len(tensors) > 4 else '-'} "
                  f"ret={ret:#x} ctx={ctx} tri_sum={tri_sum} "
                  f"stub={stub:#x}", flush=True)
            return None
        time.sleep(0.05)


def _cpp_get_case(kernel_fn, kwargs, example_args, grid):
    key = _cpp_case_key(kernel_fn, kwargs, example_args)
    case = _CPP_CASES.get(key)
    if case is not None:
        return case
    if key in _CPP_FAILED:
        return None
    try:
        case = _cpp_make_case(kernel_fn, kwargs, example_args, grid)
    except Exception as e:
        print(f"[triton-lora] WARN: case setup failed for "
              f"{kernel_fn.__name__}: {e}", flush=True)
        case = None
    if case is None:
        _CPP_FAILED.add(key)
    else:
        _CPP_CASES[key] = case
    return case


# ---- plain wrappers (used by the custom op impls and tests) ----

def bgmv_shrink(inputs, lora_a_weights, output_tensor, lora_indices_tensor, scaling=1.0):
    B, H = inputs.shape
    # vllm packs linear lora_a as [L, 1, R, H]; drop the middle 1 dim
    w = lora_a_weights.reshape(lora_a_weights.shape[0], -1, lora_a_weights.shape[-1])
    L, R, _ = w.shape
    idx32 = lora_indices_tensor.to(torch.int32)
    kwargs = dict(B=B, H=H, R=R, L=L)
    if _cpp_enabled():
        case = _cpp_get_case(K.bgmv_shrink, kwargs,
                             (inputs, w, idx32, output_tensor, scaling), (B,))
        if case is not None:
            _cpp_launch(case, B, [inputs.data_ptr(), w.data_ptr(),
                                  idx32.data_ptr(),
                                  output_tensor.data_ptr()], [float(scaling)])
            return output_tensor
    K.bgmv_shrink[(B,)](inputs, w, idx32, output_tensor, scaling,
                        B=B, H=H, R=R, L=L)
    return output_tensor


def _expand_blk_ho(R: int) -> int:
    # fp32 accumulator tile R x BLOCK_HO must fit 64KB UB
    return 128 if R * 256 * 4 > 64 * 1024 else 256


def bgmv_expand(inputs, lora_b_weights, output_tensor, lora_indices_tensor,
                add_inputs=True):
    return bgmv_expand_slice(inputs, lora_b_weights, output_tensor,
                             lora_indices_tensor, 0, output_tensor.size(1), add_inputs)


def bgmv_expand_slice(inputs, lora_b_weights, output_tensor, lora_indices_tensor,
                      slice_offset, slice_size, add_inputs=True):
    B, R = inputs.shape
    # vllm packs linear lora_b as [L, 1, Ho, R]; drop the middle 1 dim
    w = lora_b_weights.reshape(lora_b_weights.shape[0], -1, lora_b_weights.shape[-1])
    L, Ho, _ = w.shape
    idx32 = lora_indices_tensor.to(torch.int32)
    kwargs = dict(B=B, R=R, Ho=Ho, L=L, BLOCK_HO=_expand_blk_ho(R),
                  Y_HO=output_tensor.size(1), SLICE_OFF=slice_offset)
    if _cpp_enabled():
        case = _cpp_get_case(K.bgmv_expand, kwargs,
                             (inputs, w, idx32,
                              output_tensor, output_tensor), (B,))
        if case is not None:
            _cpp_launch(case, B, [inputs.data_ptr(), w.data_ptr(),
                                  idx32.data_ptr(),
                                  output_tensor.data_ptr(),
                                  output_tensor.data_ptr()])
            return output_tensor
    K.bgmv_expand[(B,)](inputs, w, idx32,
                        output_tensor, output_tensor, **kwargs)
    return output_tensor


def sgmv_shrink(inputs, lora_a_weights, output_tensor, b_seq_start_loc,
                seq_len_tensor, lora_indices_tensor, batches, max_seq_length,
                token_nums, scaling):
    B, H = inputs.shape
    # vllm packs linear lora_a as [L, 1, R, H]; drop the middle 1 dim
    w = lora_a_weights.reshape(lora_a_weights.shape[0], -1, lora_a_weights.shape[-1])
    L, R, _ = w.shape
    idx32 = lora_indices_tensor.to(torch.int32)
    seq32 = seq_len_tensor.to(torch.int32)
    kwargs = dict(B=B, H=H, R=R, L=L, NR=seq_len_tensor.numel(), scale=scaling)
    if _cpp_enabled():
        case = _cpp_get_case(K.sgmv_shrink_kernel, kwargs,
                             (inputs, w, idx32, seq32,
                              output_tensor), (B,))
        if case is not None:
            _cpp_launch(case, B, [inputs.data_ptr(), w.data_ptr(),
                                  idx32.data_ptr(),
                                  seq32.data_ptr(),
                                  output_tensor.data_ptr()])
            return output_tensor
    K.sgmv_shrink_kernel[(B,)](inputs, w, idx32, seq32,
                               output_tensor, **kwargs)
    return output_tensor


def sgmv_expand(inputs, lora_b_weights, output_tensor, b_seq_start_loc,
                seq_len_tensor, lora_indices_tensor, batches, max_seq_length,
                token_nums, add_inputs=False):
    return sgmv_expand_slice(inputs, lora_b_weights, output_tensor, b_seq_start_loc,
                             seq_len_tensor, lora_indices_tensor, batches, max_seq_length,
                             token_nums, 0, output_tensor.size(1), add_inputs)


def sgmv_expand_slice(inputs, lora_b_weights, output_tensor, b_seq_start_loc,
                      seq_len_tensor, lora_indices_tensor, batches, max_seq_length,
                      token_nums, slice_offset, slice_size, add_inputs=False):
    B, R = inputs.shape
    # vllm packs linear lora_b as [L, 1, Ho, R]; drop the middle 1 dim
    w = lora_b_weights.reshape(lora_b_weights.shape[0], -1, lora_b_weights.shape[-1])
    L, Ho, _ = w.shape
    idx32 = lora_indices_tensor.to(torch.int32)
    seq32 = seq_len_tensor.to(torch.int32)
    kwargs = dict(B=B, R=R, Ho=Ho, L=L, NR=seq_len_tensor.numel(),
                  BLOCK_HO=_expand_blk_ho(R), Y_HO=output_tensor.size(1),
                  SLICE_OFF=slice_offset)
    if _cpp_enabled():
        case = _cpp_get_case(K.sgmv_expand, kwargs,
                             (inputs, w, idx32, seq32,
                              output_tensor, output_tensor), (B,))
        if case is not None:
            _cpp_launch(case, B, [inputs.data_ptr(), w.data_ptr(),
                                  idx32.data_ptr(),
                                  seq32.data_ptr(),
                                  output_tensor.data_ptr(),
                                  output_tensor.data_ptr()])
            return output_tensor
    K.sgmv_expand[(B,)](inputs, w, idx32, seq32,
                        output_tensor, output_tensor, **kwargs)
    return output_tensor


# ---- torch custom ops (dynamo-opaque, eager impls run during graph capture) ----


@custom_op("vllm_ascend_triton::bgmv_shrink", mutates_args=("output_tensor",))
def _op_bgmv_shrink(inputs: torch.Tensor, lora_a_weights: torch.Tensor,
                    output_tensor: torch.Tensor, lora_indices_tensor: torch.Tensor,
                    scaling: float) -> None:
    _announce()
    if (_triton_dtype_ok(inputs) and _triton_dtype_ok(lora_a_weights)
            and lora_a_weights.shape[-1] == inputs.shape[-1]):
        bgmv_shrink(inputs, lora_a_weights, output_tensor, lora_indices_tensor, scaling)
    else:
        _ascend_bgmv_shrink(inputs, lora_a_weights, output_tensor, lora_indices_tensor, scaling)


@register_fake("vllm_ascend_triton::bgmv_shrink")
def _fake_bgmv_shrink(inputs: torch.Tensor, lora_a_weights: torch.Tensor,
                      output_tensor: torch.Tensor, lora_indices_tensor: torch.Tensor,
                      scaling: float) -> None:
    return None


@custom_op("vllm_ascend_triton::bgmv_expand_slice", mutates_args=("output_tensor",))
def _op_bgmv_expand_slice(inputs: torch.Tensor, lora_b_weights: torch.Tensor,
                          output_tensor: torch.Tensor, lora_indices_tensor: torch.Tensor,
                          slice_offset: int, slice_size: int) -> None:
    _announce()
    if (_triton_dtype_ok(lora_b_weights)
            and lora_b_weights.shape[-1] == inputs.shape[-1]):  # linear [L,1,Ho,R], not embedding
        bgmv_expand_slice(inputs, lora_b_weights, output_tensor, lora_indices_tensor,
                          slice_offset, slice_size, True)
    else:
        _ascend_bgmv_expand(inputs, lora_b_weights, output_tensor, lora_indices_tensor,
                            slice_offset, slice_size)


@register_fake("vllm_ascend_triton::bgmv_expand_slice")
def _fake_bgmv_expand_slice(inputs: torch.Tensor, lora_b_weights: torch.Tensor,
                            output_tensor: torch.Tensor, lora_indices_tensor: torch.Tensor,
                            slice_offset: int, slice_size: int) -> None:
    return None


@custom_op("vllm_ascend_triton::sgmv_shrink", mutates_args=("output_tensor",))
def _op_sgmv_shrink(inputs: torch.Tensor, lora_a_weights: torch.Tensor,
                    output_tensor: torch.Tensor, seq_len_tensor: torch.Tensor,
                    lora_indices_tensor: torch.Tensor, scaling: float) -> None:
    _announce()
    if (_triton_dtype_ok(inputs) and _triton_dtype_ok(lora_a_weights)
            and lora_a_weights.shape[-1] == inputs.shape[-1]
            and _sgmv_triton_ok(seq_len_tensor)):
        sgmv_shrink(inputs, lora_a_weights, output_tensor, None, seq_len_tensor,
                    lora_indices_tensor, int(inputs.shape[0]), int(inputs.shape[1]),
                    int(seq_len_tensor.numel()), scaling)
    else:
        _ascend_sgmv_shrink(inputs, lora_a_weights, output_tensor, lora_indices_tensor,
                            seq_len_tensor, scaling)


@register_fake("vllm_ascend_triton::sgmv_shrink")
def _fake_sgmv_shrink(inputs: torch.Tensor, lora_a_weights: torch.Tensor,
                      output_tensor: torch.Tensor, seq_len_tensor: torch.Tensor,
                      lora_indices_tensor: torch.Tensor, scaling: float) -> None:
    return None


@custom_op("vllm_ascend_triton::sgmv_expand_slice", mutates_args=("output_tensor",))
def _op_sgmv_expand_slice(inputs: torch.Tensor, lora_b_weights: torch.Tensor,
                          output_tensor: torch.Tensor, seq_len_tensor: torch.Tensor,
                          lora_indices_tensor: torch.Tensor,
                          slice_offset: int, slice_size: int) -> None:
    _announce()
    if (_triton_dtype_ok(lora_b_weights)
            and lora_b_weights.shape[-1] == inputs.shape[-1]  # linear [L,1,Ho,R], not embedding
            and _sgmv_triton_ok(seq_len_tensor)):
        sgmv_expand_slice(inputs, lora_b_weights, output_tensor, None, seq_len_tensor,
                          lora_indices_tensor, int(inputs.shape[0]), int(inputs.shape[1]),
                          int(seq_len_tensor.numel()), slice_offset, slice_size, True)
    else:
        _ascend_sgmv_expand(inputs, lora_b_weights, output_tensor, lora_indices_tensor,
                            seq_len_tensor, slice_offset, slice_size)


@register_fake("vllm_ascend_triton::sgmv_expand_slice")
def _fake_sgmv_expand_slice(inputs: torch.Tensor, lora_b_weights: torch.Tensor,
                            output_tensor: torch.Tensor, seq_len_tensor: torch.Tensor,
                            lora_indices_tensor: torch.Tensor,
                            slice_offset: int, slice_size: int) -> None:
    return None
