"""Triton bgmv/sgmv ops exposed as torch custom ops (torch.library).

The kernels are registered in the ``vllm_ascend_triton`` namespace so that
torch._dynamo treats them as opaque, allowed-in-graph nodes -- the same
mechanism the stock ``torch.ops._C_ascend.*`` ops use.  The runtime checks
(dtype / weight layout) live inside the eager impls, which execute outside the
traced graph (during aclgraph capture recording the impls run eagerly and their
kernel launches get recorded into the graph).  There is no AscendC fallback:
an unsupported call raises instead of taking a second code path.

Kernel launches go through a C++ launcher (rtKernelLaunch on a flat-packed
arg buffer, ~14us/launch instead of the triton python path's ~66us+); set
TRITON_LORA_CPP=0 to fall back to the plain triton launch path.
- blockDim is clamped to the physical AIV count with the true grid left in
  the arg buffer as the device loop bound -- the same clamp triton's own
  launcher applies.  See _cpp_launch.
- per-(kernel, constexpr, dtype) case: compile once via warmup, register
  once, verify-retry warmup until launches actually land.
- CANN quirk: the first launches of a freshly registered binary are silently
  dropped until the device-side load settles (~tens of ms); the verify-retry
  loop launches on a dummy output until it changes.  The syncs in that loop
  are only on case creation -- vllm warmup populates all cases before
  aclgraph capture, so no sync happens during capture.  If that assumption
  ever breaks (EE1016 during capture), set TRITON_LORA_CPP_VERIFY=0.
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

_TIMING = os.environ.get("TRITON_LORA_TIME", "") != ""
_TIMER = {}  # name -> [count, total_s]

# Opt-in host-call sampler (TRITON_LORA_SAMPLE=1).  Answers "does the decode
# step call these ops on the host, or is it graph-replayed?" without a
# profiler: prefill bursts show up as spikes, a graph-replayed decode shows a
# flat zero between bursts, an eager decode shows a steady rate near
# calls_per_step / step_time.
_SAMPLE = os.environ.get("TRITON_LORA_SAMPLE", "0") == "1"
_CALLS = {}  # name -> count
_SAMPLER_STARTED = False


def _sample_loop():
    last = {}
    t_last = time.time()
    while True:
        time.sleep(1.0)
        now = time.time()
        dt = now - t_last
        t_last = now
        cur = dict(_CALLS)
        delta = {k: v - last.get(k, 0) for k, v in cur.items()}
        delta = {k: v for k, v in delta.items() if v}
        last = cur
        if delta:
            rate = sum(delta.values()) / dt
            print(f"[triton-lora] SAMPLE {rate:8.0f}/s "
                  f"{' '.join(f'{k}={v}' for k, v in sorted(delta.items()))}",
                  flush=True)


def _probe(name):
    """SAMPLE-mode path counter: records which launch path and whether the call
    happened while an aclgraph capture was in flight.

    ``torch.compiler.is_compiling()`` is True while dynamo traces this call
    site, so the guarded branch folds away and never becomes part of a traced
    graph -- without the guard dynamo inlines the dict mutation and the serve
    dies with "Unsupported function call".
    """
    if _SAMPLE and not torch.compiler.is_compiling():
        _CALLS[name] = _CALLS.get(name, 0) + 1


def _timing_start(name):
    global _SAMPLER_STARTED
    if _SAMPLE:
        if not _SAMPLER_STARTED:
            _SAMPLER_STARTED = True
            import threading
            threading.Thread(target=_sample_loop, daemon=True).start()
    return time.perf_counter() if _TIMING else None


def _timing_end(name, t0):
    if _SAMPLE:
        _CALLS[name] = _CALLS.get(name, 0) + 1
    if t0 is None:
        return
    c, s = _TIMER.get(name, (0, 0.0))
    total = s + time.perf_counter() - t0
    _TIMER[name] = (c + 1, total)
    if (c + 1) % 200 == 0:
        print(f"[triton-lora] TIMING {name}: {total * 1e6 / (c + 1):.1f} "
              f"us/call ({c + 1} calls)", flush=True)


def _idx(t: torch.Tensor) -> torch.Tensor:
    """Pass the caller's index tensor through as-is.

    vllm allocates these as torch.long and the AscendC kernels read them as
    int64, so int64 is the native dtype on this path -- the kernel loads it
    and narrows in-register.  That matters more than it looks: the int64->int32
    conversion this replaced was a *device copy* (a kernel launch) executed 2x
    per LoRA op, so 512 launches per decode step, on top of the Python and
    NPUCachingAllocator cost.  Only a non-int64 caller pays anything now, and
    ``.to`` is a no-op returning the same tensor when the dtype already
    matches.
    """
    return t if t.dtype == torch.int64 else t.to(torch.int64)


_RAW_STREAM_FN = None


def _raw_stream() -> int:
    """Current NPU stream handle, without building a python ``Stream`` object.

    ``torch.npu.current_stream()`` runs ``_lazy_init``, calls into C++ for
    ``(id, device, type)`` and then constructs a ``Stream`` -- measured 21.9us
    on 910B4, against 0.5us for the raw handle that ``Stream.npu_stream``
    simply hands back.  Both return the same value (verified equal on 910B4),
    so this is a drop-in, and it runs once per LoRA op -- ~288x per decode
    step -- where it was pure overhead.

    It only costs anything off-graph: inside an aclgraph replay nothing on
    this path runs at all.  That is exactly why the regression it causes lands
    on low concurrency, where the device is idle enough for host dispatch to
    sit on the critical path instead of overlapping.
    """
    global _RAW_STREAM_FN
    if _RAW_STREAM_FN is None:
        try:
            import torch_npu._C as _C

            dev = torch.npu.current_device()
            raw = _C._npu_getCurrentRawStream
            raw(dev)  # fail here, at setup, not in the hot path
            _RAW_STREAM_FN = lambda: int(raw(dev))  # noqa: E731
        except Exception:  # noqa: BLE001 - any failure falls back to the slow but correct path
            _RAW_STREAM_FN = lambda: int(torch.npu.current_stream().npu_stream)  # noqa: E731
    return _RAW_STREAM_FN()


def _announce():
    global _announced
    if not _announced:
        mode = ("C++ rtKernelLaunch" if _cpp_enabled() else "triton python path")
        if _TIMING:
            mode += " + TIMING"
        print(f"[triton-lora] dispatch ACTIVE ({mode})", flush=True)
        _announced = True


def _triton_dtype_ok(t: torch.Tensor) -> bool:
    return t.dtype in (torch.float16, torch.bfloat16)


# The AscendC fallbacks (_ascend_bgmv_shrink / _ascend_bgmv_expand /
# _ascend_sgmv_shrink / _ascend_sgmv_expand) are gone: every path that used to
# fall back now runs the Triton kernel unconditionally, so AscendC can be
# dropped from the build.  The three former fallback conditions are handled as
# follows:
#   * NR > 16 (sgmv): gone outright -- the kernels locate each token row with an
#     O(NR) scan over b_seq_start_loc instead of an O(NR^2) triangular prefix
#     sum, so there is no longer any NR-dependent cost cliff.
#   * non-fp16/bf16 dtype: AscendC only ever accepted half/bf16
#     (csrc/torch_binding.cpp TORCH_CHECK), so this never was a working
#     fallback -- it only turned one error into another.  Now it raises here.
#   * weight/input trailing-dim mismatch: likewise a silent reroute of a shape
#     that should never reach this op.  Now it raises here.


# ---- C++ launcher (rtKernelLaunch direct) ----

_CPP_DIR = os.path.dirname(os.path.abspath(__file__))
_CPP_SRC = os.path.join(_CPP_DIR, "lora_cpp_launcher.cpp")
_CPP_SO = os.path.join(_CPP_DIR, "lora_cpp_launcher.cpython-312-aarch64-linux-gnu.so")

_CPP_STATE = None          # (CDLL, ffts_addr) or (None, None) on failure
_CPP_CUR = False           # launcher resolves the stream in C++ (fast path)
_CPP_CASES = {}            # key -> case dict
_CPP_FAILED = set()        # keys that must use the triton fallback
_CPP_NEEDS_WS = set()      # kernel names the minimal launcher cannot run at all


# ---- native kPrivateUse1 impls (the serve path, gated by TRITON_LORA_NATIVE) ----
#
# lora_native_ops.cpp registers these schemas on the PrivateUse1 dispatch key
# ONLY -- the same shape as the AscendC path (csrc/torch_binding.cpp:2493) --
# so torch_npu's aclgraph records the launches as pure device nodes and replays
# them with no python on the step path.
#
# Registering a Python-key impl as well (which this file did until 2026-09-15)
# is strictly worse: the Python key outranks PrivateUse1, the op stops being a
# pure device node, and torch_npu re-dispatches it through the python dispatcher
# on EVERY replay.  Measured with a probe at the punica binding site: ~496 host
# calls/s sustained through a c=1 decode plateau, against 0/s for both the
# AscendC path and the PrivateUse1-only path.  Dropping the Python-key impl is
# the whole of the low-concurrency fix (c1_4k: -21.7% -> +38.9%).
#
# Superseded claims, recorded so they are not re-derived:
#   * "aclgraph dispatches these ops through the python dispatcher with wrapper
#     tensors, and C++ kernels receive them raw" -- false.  The C++ side has
#     uw() (lora_native_ops.cpp:260), which unwraps.
#   * "LoRA was silently NEVER applied (probe 09-02: outputs byte-identical to
#     the base model)" -- does not reproduce on the PrivateUse1-only path.  The
#     lora-vs-nolora self-check against one server shows the outputs differ.

_NATIVE_SO = os.path.join(_CPP_DIR, "lora_native_ops.so")
_NATIVE_SRC = os.path.join(_CPP_DIR, "lora_native_ops.cpp")
_NATIVE = None  # ctypes.CDLL of lora_native_ops.so, or None on failure

_DT_MAP = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.int32": torch.int32,
    "torch.float32": torch.float32,
}

_KEY_KERNELS = {
    "bgmv_shrink": K.bgmv_shrink,
    "bgmv_expand": K.bgmv_expand,
    "sgmv_shrink_kernel": K.sgmv_shrink_kernel,
    "sgmv_expand": K.sgmv_expand,
}


def _native_build():
    """g++ the native impls .so (links libtorch + CANN runtime)."""
    if os.path.exists(_NATIVE_SO):
        return _NATIVE_SO
    import subprocess
    import sysconfig
    from torch.utils import cpp_extension as ce
    cann = os.environ.get("ASCEND_HOME_PATH") or os.environ.get(
        "ASCEND_TOOLKIT_HOME") or "/usr/local/Ascend/ascend-toolkit/latest"
    cann_inc = os.path.join(cann, "aarch64-linux", "pkg_inc")
    cann_lib = os.path.join(cann, "aarch64-linux", "lib64")
    if not os.path.isdir(cann_inc):  # older layout: <cann>/include + lib64
        cann_inc, cann_lib = os.path.join(cann, "include"), os.path.join(cann, "lib64")
    flags = ["g++", "-std=c++17", "-fPIC", "-O2", "-shared"]
    for i in ce.include_paths(device_type=None) + [
            sysconfig.get_paths()["include"], cann_inc]:
        flags += ["-I", i]
    torch_lib = ce.library_paths()
    for l in torch_lib + [cann_lib]:
        flags += ["-L", l, "-Wl,-rpath," + l]
    flags += ["-o", _NATIVE_SO, _NATIVE_SRC, "-ltorch", "-lc10",
              "-lruntime", "-lascendcl"]
    r = subprocess.run(flags, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("g++ failed:\n" + r.stderr[-3000:])
    return _NATIVE_SO


def _cpp_stream_getter():
    return int(torch.npu.current_stream().npu_stream)


def _cpp_parse_key(key_s):
    name, kvs_s, dts_s = key_s.split("|")
    kvs = {}
    for kv in kvs_s.split(","):
        k, v = kv.split("=", 1)
        kvs[k] = float(v) if "." in v else int(v)
    return name, kvs, dts_s.split(",")


def _cpp_dummy_args(name, kwargs, dts):
    """Dummy example args for case compilation (shapes only need to be
    consistent with the constexprs; B is the grid, fixed at 8).  Indices
    come first-call-shape-free; the C++ key carries everything else."""
    i32 = torch.int32
    if name == "bgmv_shrink":
        H, R, L = kwargs["H"], kwargs["R"], kwargs["L"]
        x = torch.empty(8, H, dtype=_DT_MAP[dts[0]], device="npu")
        w = torch.empty(L, R, H, dtype=_DT_MAP[dts[1]], device="npu")
        idx = torch.zeros(8, dtype=i32, device="npu")
        y = torch.zeros(8, R, dtype=_DT_MAP[dts[3]], device="npu")
        return x, w, idx, y, 1.0
    if name == "bgmv_expand":
        R, Ho, L = kwargs["R"], kwargs["Ho"], kwargs["L"]
        ydt = _DT_MAP[dts[3]]
        x = torch.empty(8, R, dtype=_DT_MAP[dts[0]], device="npu")
        w = torch.empty(L, Ho, R, dtype=_DT_MAP[dts[1]], device="npu")
        idx = torch.zeros(8, dtype=i32, device="npu")
        y = torch.zeros(8, kwargs["Y_HO"], dtype=ydt, device="npu")
        return x, w, idx, y, y
    if name == "sgmv_shrink_kernel":
        H, R, L, NR = kwargs["H"], kwargs["R"], kwargs["L"], kwargs["NR"]
        x = torch.empty(8, H, dtype=_DT_MAP[dts[0]], device="npu")
        w = torch.empty(L, R, H, dtype=_DT_MAP[dts[1]], device="npu")
        idx = torch.zeros(NR, dtype=i32, device="npu")
        # b_seq_start_loc: exclusive prefix sum, sorted, starting at 0, so every
        # row resolves to some request and the verify-retry sees a nonzero write
        start = torch.arange(NR, dtype=i32, device="npu") * (8 // NR)
        y = torch.zeros(8, R, dtype=_DT_MAP[dts[4]], device="npu")
        return x, w, idx, start, y
    # sgmv_expand
    R, Ho, L, NR = kwargs["R"], kwargs["Ho"], kwargs["L"], kwargs["NR"]
    ydt = _DT_MAP[dts[4]]
    x = torch.empty(8, R, dtype=_DT_MAP[dts[0]], device="npu")
    w = torch.empty(L, Ho, R, dtype=_DT_MAP[dts[1]], device="npu")
    idx = torch.zeros(NR, dtype=i32, device="npu")
    start = torch.arange(NR, dtype=i32, device="npu") * (8 // NR)
    y = torch.zeros(8, kwargs["Y_HO"], dtype=ydt, device="npu")
    return x, w, idx, start, y, y


def _cpp_miss_handler(key_s):
    """Called by the native C++ impls when a case key is not yet bound.
    Compiles + verifies the case and binds key -> rtFunction handle.
    Returns False when the case cannot be built here (C++ impl then falls
    back to the AscendC native op so results stay correct)."""
    try:
        if torch.npu.is_current_stream_capturing():
            return False  # no compile/sync during capture
        name, kwargs, dts = _cpp_parse_key(key_s)
        kernel_fn = _KEY_KERNELS[name]
        args = _cpp_dummy_args(name, kwargs, dts)
        case = _CPP_CASES.get(key_s)
        if case is None:
            case = _cpp_make_case(kernel_fn, kwargs, args, (8,))
            if case is None:
                _CPP_FAILED.add(key_s)
                return False
            _CPP_CASES[key_s] = case
        _NATIVE.lora_native_bind_case(key_s.encode(), case["func"])
        return True
    except Exception as e:
        print(f"[triton-lora] native miss-handler failed {key_s}: {e}",
              flush=True)
        _CPP_FAILED.add(key_s)
        return False


def _native_setup():
    global _NATIVE
    if _NATIVE is not None:
        return _NATIVE
    try:
        so = _native_build()
        CL = ctypes.CDLL(so)
        CL.lora_native_bind_case.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
        CL.lora_native_set_handler.argtypes = [ctypes.py_object]
        CL.lora_native_set_stream_getter.argtypes = [ctypes.py_object]
        CL.lora_native_set_handler(_cpp_miss_handler)
        CL.lora_native_set_stream_getter(_cpp_stream_getter)
        _NATIVE = CL
        print("[triton-lora] native kPrivateUse1 impls ACTIVE", flush=True)
    except Exception as e:
        print(f"[triton-lora] native impls unavailable: {e}", flush=True)
        _NATIVE = None
    return _NATIVE


# Read once: this is consulted on every op call (~107x per decode step) and
# os.environ.get alone is ~4.2us of that.  TRITON_LORA_CPP is a process-wide
# debugging switch, so a module-level constant is the right lifetime.
_CPP_ENABLED = os.environ.get("TRITON_LORA_CPP", "1") != "0"


def _cpp_enabled() -> bool:
    return _CPP_ENABLED


def _capturing() -> bool:
    """True while an aclgraph capture is in flight (False if unknown)."""
    try:
        return bool(torch.npu.is_current_stream_capturing())
    except Exception:
        return False


def _cpp_build():
    """g++ the launcher .so.

    Unlike triton's `_build_npu_ext`, this adds torch_npu's include path and
    links libtorch_npu -- the launcher uses `c10_npu::getCurrentNPUStream()` to
    fetch the current stream, which is what makes the per-call cost acceptable
    (see the comment in lora_cpp_launcher.cpp).  The source is `__has_include`-
    guarded, so if torch_npu headers are missing the build still succeeds and
    the launcher just exposes one fewer entry point.  Mirrors `_native_build`.
    """
    import subprocess
    import sysconfig
    from torch.utils import cpp_extension as ce
    cann = os.environ.get("ASCEND_HOME_PATH") or os.environ.get(
        "ASCEND_TOOLKIT_HOME") or "/usr/local/Ascend/ascend-toolkit/latest"
    cann_inc = os.path.join(cann, "aarch64-linux", "pkg_inc")
    cann_lib = os.path.join(cann, "aarch64-linux", "lib64")
    if not os.path.isdir(cann_inc):  # older layout: <cann>/include + lib64
        cann_inc, cann_lib = os.path.join(cann, "include"), os.path.join(cann, "lib64")
    flags = ["g++", "-std=c++17", "-fPIC", "-O2", "-shared"]
    for i in ce.include_paths(device_type=None) + [
            sysconfig.get_paths()["include"], cann_inc]:
        flags += ["-I", i]
    libs = ["-ltorch", "-lc10", "-lruntime", "-lascendcl"]
    try:  # optional: enables lora_launch_flat_cur
        import torch_npu
        tn = os.path.dirname(os.path.abspath(torch_npu.__file__))
        flags += ["-I", tn, "-I", os.path.join(tn, "include"),
                  "-I", os.path.join(tn, "include", "third_party", "acl", "inc")]
        flags += ["-L", os.path.join(tn, "lib"), "-Wl,-rpath," + os.path.join(tn, "lib")]
        libs.append("-ltorch_npu")
    except Exception:
        pass
    for l in ce.library_paths() + [cann_lib]:
        flags += ["-L", l, "-Wl,-rpath," + l]
    # libs last: GNU ld resolves symbols left to right, so -ltorch_npu before
    # the source file would leave getCurrentNPUStream undefined.
    r = subprocess.run(flags + ["-o", _CPP_SO, _CPP_SRC] + libs,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("g++ failed:\n" + r.stderr[-3000:])
    return _CPP_SO


def _cpp_setup():
    global _CPP_STATE, _CPP_CUR
    if _CPP_STATE is not None:
        return _CPP_STATE
    try:
        so = _CPP_SO
        if not os.path.exists(so):
            try:
                so = _cpp_build()
            except Exception as e:
                print(f"[triton-lora] g++ launcher build failed ({e}); "
                      f"falling back to triton's _build_npu_ext", flush=True)
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
        try:
            CL.lora_launch_flat_cur.restype = ctypes.c_uint64
            CL.lora_launch_flat_cur.argtypes = [ctypes.c_uint64, ctypes.c_int32,
                                                ctypes.c_void_p, ctypes.c_uint64]
            CL.lora_current_stream.restype = ctypes.c_uint64
            # c10_npu::getCurrentNPUStream().stream(false) does NOT return the
            # same handle torch_npu's Python `Stream.npu_stream` property does
            # (observed: 0x3aed8008 vs 0xfffd3aed8008 -- same low 32 bits, so the
            # same stream, but a different encoding).  Launching on a handle we
            # cannot corroborate faults the device, so this stays opt-in until
            # the encoding is understood.
            _CPP_CUR = os.environ.get("TRITON_LORA_CPP_STREAM", "0") == "1"
            if _CPP_CUR and CL.lora_current_stream() != torch.npu.current_stream().npu_stream:
                print("[triton-lora] WARN: TRITON_LORA_CPP_STREAM=1 but the C++ "
                      "and Python stream handles disagree; expect faults", flush=True)
        except AttributeError:
            _CPP_CUR = False  # built without torch_npu; pass the stream in
        CL.lora_get_ffts_addr.restype = ctypes.c_uint64
        CL.lora_get_ffts_addr.argtypes = [ctypes.c_int]
        CL.lora_peek_stub.restype = ctypes.c_uint64
        CL.lora_peek_stub.argtypes = [ctypes.c_uint64]
        _CPP_STATE = (CL, CL.lora_get_ffts_addr(torch.npu.current_device()))
    except Exception as e:
        print(f"[triton-lora] C++ launcher unavailable, triton fallback: {e}",
              flush=True)
        _CPP_STATE = (None, None)
    return _CPP_STATE


def _cpp_case_key(kernel_fn, kwargs, tensors):
    """Cache key for one compiled + registered launch case.

    This runs on every op call (~107x per decode step), so it stays as cheap as
    it can: dtypes are already hashable (stringifying them was ~2us of pure
    waste), and kwargs keep the insertion order their call sites give them --
    each call site builds the same dict literal, so the order is fixed for the
    life of the process.  example_args may mix tensors and python scalars
    (bgmv_shrink passes its float `scaling`); only the tensor dtypes belong in
    the key.
    """
    return (kernel_fn.__name__, tuple(kwargs.items()),
            tuple(t.dtype for t in tensors if isinstance(t, torch.Tensor)))


_CORE_NUM = {}


def _core_num(mode: str) -> int:
    """Physical block count for `mode` -- the same clamp source triton's own
    launcher uses (driver.py:524: ``num_physical_blocks = get_aivector_core_num()
    if mix_mode == "aiv" else get_aicore_num()``).  A `tl.dot` kernel compiles as
    mix_mode="mix" and therefore runs one block per AI *cube* core, which is half
    the vector count on 910B4 -- clamping those to the vector count oversubscribes
    the cubes and the launch faults (MTE invalid GM address).
    0 if unavailable, in which case the caller launches the raw grid."""
    if mode not in _CORE_NUM:
        n = 0
        try:
            from triton.backends.ascend.driver import NPUUtils
            u = NPUUtils()
            n = int(u.get_aivector_core_num() if mode == "aiv"
                    else u.get_aicore_num())
        except Exception:
            n = 0
        _CORE_NUM[mode] = n
    return _CORE_NUM[mode]


def _cpp_launch(case, grid_x, ptrs, scalars=()):
    """Fire one pre-registered kernel.

    All pointer args come first, then the runtime scalars in kernel-signature
    order.  Scalar width follows triton's ABI: a plain python int is compiled
    as i32, a float as fp32 -- both 4 bytes, so the flat buffer size is the
    same either way and only the pack format differs.
    """
    CL, _ = _cpp_setup()
    b = case["buf"]
    off = 24  # [ffts][syncBlockLock][workspace]
    for p in ptrs:
        struct.pack_into("<Q", b, off, p)
        off += 8
    for v in scalars:
        struct.pack_into("<i" if isinstance(v, int) else "<f", b, off, v)
        off += 4
    off = (off + 3) & ~3
    # The true grid goes in the arg buffer: the device program-id loop is
    # bounded by it.  blockDim must NOT exceed the physical core count --
    # triton clamps it for the same reason.  Passing the raw grid made CANN
    # clamp the launch to `cores` blocks while the device loop still strode by
    # the value we passed, so every pid >= cores was silently skipped (a
    # 128-token prefill only ever ran pids 0..39).
    struct.pack_into("<iii", b, off, grid_x, 1, 1)
    n = _core_num(case["mode"])
    block = grid_x if (n <= 0 or grid_x <= n) else n
    if _CPP_CUR:
        return CL.lora_launch_flat_cur(case["func"], block, b, off + 12)
    return CL.lora_launch_flat(case["func"], _raw_stream(),
                               block, b, off + 12)


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

    # Triton's binder may DROP a runtime argument from the launch buffer:
    # `equal_to_1` folds any python int equal to 1 into a compile-time
    # constant, and the argument then occupies no bytes at all.  This launcher
    # writes arguments at fixed offsets, so a dropped one shifts the grid 4
    # bytes and the device reads gridX from the wrong slot -- measured at
    # B=1 as 1 program running instead of NR*NBLK, writing only the first Ho
    # block (256 of 4096 columns) and returning silently wrong results.
    # `do_not_specialize` on the kernel is the fix; this guard is what stops a
    # future kernel from reintroducing it without anyone noticing.
    dns = set(getattr(kernel_fn, "do_not_specialize", ()) or ())
    arg_names = list(getattr(kernel_fn, "arg_names", ()) or ())
    for i, a in enumerate(example_args):
        if isinstance(a, bool) or not isinstance(a, int) or a != 1:
            continue
        name = arg_names[i] if i < len(arg_names) else f"#{i}"
        if name not in dns:
            print(f"[triton-lora] {kernel_fn.__name__}: arg {name}={a} is 1 and "
                  "is not marked do_not_specialize, so triton folds it away and "
                  "the fixed-offset launcher would misplace the grid; using the "
                  "triton launcher", flush=True)
            return None

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

    # The minimal launcher passes NULL for the workspace slot, which is right
    # for the aiv-only kernels (no workspace declared) but faults with "MTE
    # accesses an invalid GM address" for any kernel that does declare one --
    # a tl.dot kernel compiles as mix_mode="mix" and needs
    # workspace_size * gridX*gridY*gridZ bytes (544KB/block, 139MB at grid 256).
    # triton's own launcher allocates that; rather than replicate an allocator
    # the launch path has no measured use for, send these kernels back through
    # the plain triton launcher.
    if getattr(compiled.metadata, "workspace_size", -1) > 0:
        # Remember it per kernel, not per case: the answer is a property of the
        # compiled artifact's shape class, and re-deriving it for every new
        # (NR, Ho, BLOCK_*) key meant a wasted triton compile each time --
        # 116 of them, all inside aclgraph capture, during one serve startup.
        _CPP_NEEDS_WS.add(kernel_fn.__name__)
        print(f"[triton-lora] {kernel_fn.__name__} needs a device workspace "
              f"({compiled.metadata.workspace_size}B/block, mix_mode="
              f'"{compiled.metadata.mix_mode}"), using the triton launcher',
              flush=True)
        return None

    data = bytes(compiled.kernel)
    buf = ctypes.create_string_buffer(data)
    mode = getattr(compiled.metadata, "mix_mode", "aiv")

    func = CL.lora_register_kernel(compiled.name.encode(), buf, len(data),
                                   mode.encode(),
                                   torch.npu.current_device())
    if not func:
        return None

    ptrs_d = [t.data_ptr() for t in dummy_args if isinstance(t, torch.Tensor)]
    n = 24 + 8 * len(ptrs_d) + 4 * len(floats) + 16
    cbuf = ctypes.create_string_buffer(n)
    struct.pack_into("<QQQ", cbuf, 0, ffts, 0, 0)
    case = dict(func=func, buf=cbuf, nptrs=len(ptrs_d), nfloats=len(floats),
                mode=mode)

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
                  f"start[:8]={[int(v) for v in tensors[3].flatten().tolist()[:8]] if len(tensors) > 4 else '-'} "
                  f"ret={ret:#x} ctx={ctx} tri_sum={tri_sum} "
                  f"stub={stub:#x}", flush=True)
            return None
        time.sleep(0.05)


def _cpp_get_case(kernel_fn, kwargs, example_args, grid):
    if kernel_fn.__name__ in _CPP_NEEDS_WS:
        return None
    key = _cpp_case_key(kernel_fn, kwargs, example_args)
    case = _CPP_CASES.get(key)
    if case is not None:
        return case
    if key in _CPP_FAILED:
        return None
    # A missing case means compile + verify, and the verify loop has to
    # synchronize.  That is illegal inside an aclgraph capture, and the decode
    # graph in serve is captured with cudagraph_mode=FULL_DECODE_ONLY -- so a
    # key first seen mid-capture must fall back to the triton launcher (which
    # is what the captured graph then contains, consistently, for its lifetime)
    # rather than abort the capture.
    if _capturing():
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
        _probe("p:case_made")
        _CPP_CASES[key] = case
    return case


# ---- plain wrappers (used by the custom op impls and tests) ----

def _flat_w(t):
    """Fold vllm's packing dim away: return ``(view, shape)`` with shape ``(L, K, N)``.

    vllm packs linear LoRA weights as ``[L, 1, K, N]``, and every kernel here
    indexes them as flat ``[L, K, N]`` -- they consume ``data_ptr()`` only, so
    the logical shape never reaches the device.  With a middle dim of 1 the
    reshape is a pure view with an identical pointer, i.e. ~5.4us of Tensor
    dispatch per call for nothing, and these run ~107x per decode step.

    Anything else keeps the real reshape: a middle dim > 1 genuinely needs the
    fold, and a non-contiguous input needs the (copying) reshape for the flat
    indexing above to be valid.
    """
    s = t.shape
    if t.is_contiguous():
        if len(s) == 4 and s[1] == 1:
            return t, (s[0], s[2], s[3])
        if len(s) == 3:
            return t, s
    w = t.reshape(s[0], -1, s[-1])
    return w, w.shape


def _bgmv_shrink_bw(H: int) -> int:
    """Reduction width for bgmv_shrink: the widest power of two <= H, capped at
    1024 (measured best at both ends of the batch range: B=1 -> 9.5us,
    B=32 -> 17.7us, both ahead of AscendC)."""
    bw = 64
    while bw * 2 <= min(H, 1024):
        bw *= 2
    return bw


def bgmv_shrink(inputs, lora_a_weights, output_tensor, lora_indices_tensor, scaling=1.0):
    t0 = _timing_start("bgmv_shrink")
    B, H = inputs.shape
    w, (L, R, _) = _flat_w(lora_a_weights)
    idx_l = _idx(lora_indices_tensor)
    kwargs = dict(H=H, R=R, L=L, BW=_bgmv_shrink_bw(H))
    if _cpp_enabled():
        case = _cpp_get_case(K.bgmv_shrink, kwargs,
                             (inputs, w, idx_l, output_tensor, scaling), (B,))
        if case is not None:
            _probe("p:shrink_cpp")
            _cpp_launch(case, B, [inputs.data_ptr(), w.data_ptr(),
                                  idx_l.data_ptr(),
                                  output_tensor.data_ptr()], [float(scaling)])
            _timing_end("bgmv_shrink", t0)
            return output_tensor
    K.bgmv_shrink[(B,)](inputs, w, idx_l, output_tensor, scaling,
                        **kwargs)
    _timing_end("bgmv_shrink", t0)
    return output_tensor


def _expand_blk_ho(R: int) -> int:
    # fp32 accumulator tile R x BLOCK_HO must fit 64KB UB
    return 128 if R * 256 * 4 > 64 * 1024 else 256


_DOT_BH = 512          # widest [R, BLOCK_HO] weight tile that still compiles
_DOT_TILE = 32768      # BLOCK_T * BLOCK_HO cap imposed by the UB budget


def _dot_ok(R: int) -> bool:
    """tl.dot needs a power-of-two K >= 16; else fall back to the old kernel."""
    return R >= 16 and not (R & (R - 1))


def _expand_tile(Ho: int):
    """Pick (BLOCK_HO, BLOCK_T) for the dot kernel.

    The fp32 accumulator [BLOCK_T, BLOCK_HO] has to fit UB, which caps the
    product at ~32768 elements ([64, 512] and [128, 256] both compile,
    [128, 512] does not).  Within that budget prefer a wide BLOCK_HO: it is
    what keeps the weight resident for the most tokens.
    """
    bh = 16
    while bh < min(_DOT_BH, Ho):
        bh *= 2
    bt = max(16, min(256, _DOT_TILE // bh))
    return bh, bt


def _expand_split(NR: int, nblk: int, ntok: int) -> int:
    """Split each (request, Ho-block) across tokens until the grid is big enough.

    Grid size matters more than it looks: this kernel measures 0.27x AscendC
    at grid 512 and 7.4x at grid 32768, because per-program setup (weight
    load + index math) dominates once the tiles get small.  Under ~256
    programs there is not enough parallelism, so grow towards that, but never
    split a request so finely that a program has no token left to do.
    """
    grid = NR * nblk
    ns = 1
    while grid * ns < 256 and ns < ntok and grid * ns * 2 <= 1024:
        ns *= 2
    return ns


# Where flat and dot cross over.  Both were measured against AscendC on
# 910B4 (Ho=4096, NR=32, L=2): flat is ~0.56us per token row and dot is ~67us
# flat in B, so they meet near B=120.  Below that dot pays for a 16-row
# tl.dot tile it cannot fill; above it flat pays to re-read the weight once
# per request.  The gap is wide on both sides -- at B=32 flat is 2.4x and dot
# 7.0x, at B=4096 flat is 5.1x and dot 0.21x -- so the exact threshold is not
# delicate.
_DOT_MIN_TOKENS = 128


def _bgmv_blk_ho(R: int, Ho: int, B: int) -> int:
    """BLOCK_HO for the grid-split bgmv_expand.

    Same UB-derived cap as the old in-program loop used, but shrunk when the
    grid would be too small to fill the cores: bgmv's grid is B * ceil(Ho /
    BLOCK_HO), so a B=1 decode step on Ho=4096 would otherwise spread over only
    16 programs.  256 still measures best wherever the grid is already wide
    enough (B >= 2), so the shrink only fires at the small end.
    """
    bh = _expand_blk_ho(R)
    while bh > 64 and B * ((Ho + bh - 1) // bh) < 32:
        bh //= 2
    return bh


def _flat_blk_ho(Ho: int) -> int:
    """BLOCK_HO for the flat kernel.  256 measured best across B = 1..32 on
    both Ho=4096 and Ho=12288; wider blocks lose more to the tail mask than
    they gain in per-program weight reuse (a program's weight is read once
    regardless, and decode has one token to apply it to)."""
    bh = 32
    while bh < min(256, Ho):
        bh *= 2
    return bh


def bgmv_expand(inputs, lora_b_weights, output_tensor, lora_indices_tensor,
                add_inputs=True):
    return bgmv_expand_slice(inputs, lora_b_weights, output_tensor,
                             lora_indices_tensor, 0, output_tensor.size(1), add_inputs)


def bgmv_expand_slice(inputs, lora_b_weights, output_tensor, lora_indices_tensor,
                      slice_offset, slice_size, add_inputs=True):
    t0 = _timing_start("bgmv_expand_slice")
    B, R = inputs.shape
    w, (L, Ho, _) = _flat_w(lora_b_weights)
    idx_l = _idx(lora_indices_tensor)
    blk_ho = _bgmv_blk_ho(R, Ho, B)
    nblk = (Ho + blk_ho - 1) // blk_ho
    kwargs = dict(R=R, Ho=Ho, L=L, BLOCK_HO=blk_ho, NBLK=nblk,
                  Y_HO=output_tensor.size(1), SLICE_OFF=slice_offset)
    grid = (B * nblk,)
    if _cpp_enabled():
        case = _cpp_get_case(K.bgmv_expand, kwargs,
                             (inputs, w, idx_l,
                              output_tensor, output_tensor), grid)
        if case is not None:
            _cpp_launch(case, grid[0], [inputs.data_ptr(), w.data_ptr(),
                                        idx_l.data_ptr(),
                                        output_tensor.data_ptr(),
                                        output_tensor.data_ptr()])
            _timing_end("bgmv_expand_slice", t0)
            return output_tensor
    K.bgmv_expand[grid](inputs, w, idx_l,
                        output_tensor, output_tensor, **kwargs)
    _timing_end("bgmv_expand_slice", t0)
    return output_tensor


_SHRINK_BB = 256        # widest BLOCK_B measured to win; the fp32 accumulator
                        # [BLOCK_B, R] and the x tile [BLOCK_B, BLOCK_H] both
                        # scale with it, so this is the UB-bound knob
_SHRINK_BH = 128        # measured best at both ends of the grid range (32 and
                        # 1024 programs): 54.8us and 185.0us at B=8192/H=4096
# Below this the dot kernel loses to the per-token one: tl.dot needs a 16-row
# tile, and decode (B = 32) cannot fill it.  Measured on 910B4, H=4096:
#   B=128  dot 143.8us  flat 81.3us      B=512  dot 127.6us  flat 231.3us
# so the crossover sits between the two and the gap is wide on both sides.
_SHRINK_DOT_MIN_TOKENS = 256


def _shrink_block_b(B: int, R: int) -> int:
    """BLOCK_B, capped so the kernel's two live tiles fit UB.

    At BLOCK_H=128 the x tile is BLOCK_B*256 bytes and the accumulator is
    BLOCK_B*R*4; together they get ~128KB of the 192KB budget.
    """
    limit = _SHRINK_BB
    while limit > 16 and limit * (R * 4 + _SHRINK_BH * 2) > 128 * 1024:
        limit //= 2
    bb = 16
    while bb * 2 <= B and bb < limit:
        bb *= 2
    return bb


def _shrink_block_h(H: int) -> int:
    """BLOCK_H is the tl.dot K dimension, so it must be a power of two >= 16."""
    bh = 16
    while bh * 2 <= min(H, _SHRINK_BH):
        bh *= 2
    return bh


def sgmv_shrink(inputs, lora_a_weights, output_tensor, b_seq_start_loc,
                lora_indices_tensor, scaling):
    t0 = _timing_start("sgmv_shrink")
    m1 = _timing_start("sgmv_shrink|prep")
    _probe("p:cap" if _capturing() else "p:eager")
    B, H = inputs.shape
    w, (L, R, _) = _flat_w(lora_a_weights)
    idx_l = _idx(lora_indices_tensor)
    start_l = _idx(b_seq_start_loc)
    NR = b_seq_start_loc.numel()
    kwargs = dict(H=H, R=R, L=L, NR=NR, scale=scaling)
    m2 = _timing_start("sgmv_shrink|lookup")
    m3 = _timing_start("sgmv_shrink|launch")
    if _dot_ok(R) and NR > 0 and B >= _SHRINK_DOT_MIN_TOKENS:
        # NSPLIT is an upper bound -- no request can hold more than B tokens --
        # so requests shorter than that just leave programs idle.  This kernel
        # is mix mode and declares a device workspace, so it is not eligible
        # for the minimal C++ launcher; go straight to triton's.
        bb = _shrink_block_b(B, R)
        ns = (B + bb - 1) // bb
        _timing_end("sgmv_shrink", t0)
        _timing_end("sgmv_shrink|prep", m1)
        _timing_end("sgmv_shrink|lookup", m2)
        _timing_end("sgmv_shrink|launch", m3)
        _probe("p:shrink_dot")
        K.sgmv_shrink_dot[(NR * ns,)](
            # float() keeps this a runtime fp32 argument: an int here would be
            # specialized into a constant, which is what breaks the kernel.
            inputs, w, idx_l, start_l, output_tensor, B, float(scaling),
            H=H, R=R, L=L, NR=NR,
            BLOCK_B=bb, BLOCK_H=_shrink_block_h(H), NSPLIT=ns)
        _timing_end("sgmv_shrink", t0)
        return output_tensor
    if _cpp_enabled():
        case = _cpp_get_case(K.sgmv_shrink_kernel, kwargs,
                             (inputs, w, idx_l, start_l,
                              output_tensor), (B,))
        if case is not None:
            _cpp_launch(case, B, [inputs.data_ptr(), w.data_ptr(),
                                  idx_l.data_ptr(),
                                  start_l.data_ptr(),
                                  output_tensor.data_ptr()])
            _timing_end("sgmv_shrink", t0)
            _timing_end("sgmv_shrink|prep", m1)
            _timing_end("sgmv_shrink|lookup", m2)
            _timing_end("sgmv_shrink|launch", m3)
            return output_tensor
    _timing_end("sgmv_shrink", t0)
    _timing_end("sgmv_shrink|prep", m1)
    _timing_end("sgmv_shrink|lookup", m2)
    _timing_end("sgmv_shrink|launch", m3)
    _probe("p:shrink_tri")
    K.sgmv_shrink_kernel[(B,)](inputs, w, idx_l, start_l,
                               output_tensor, **kwargs)
    _timing_end("sgmv_shrink", t0)
    return output_tensor


def sgmv_expand(inputs, lora_b_weights, output_tensor, b_seq_start_loc,
                lora_indices_tensor, add_inputs=False):
    return sgmv_expand_slice(inputs, lora_b_weights, output_tensor, b_seq_start_loc,
                             lora_indices_tensor, 0, output_tensor.size(1), add_inputs)


def sgmv_expand_slice(inputs, lora_b_weights, output_tensor, b_seq_start_loc,
                      lora_indices_tensor, slice_offset, slice_size, add_inputs=False):
    t0 = _timing_start("sgmv_expand_slice")
    m1 = _timing_start("sgmv_expand_slice|prep")
    _probe("p:cap" if _capturing() else "p:eager")
    B, R = inputs.shape
    w, (L, Ho, _) = _flat_w(lora_b_weights)
    idx_l = _idx(lora_indices_tensor)
    start_l = _idx(b_seq_start_loc)
    NR = b_seq_start_loc.numel()
    if _dot_ok(R) and B >= _DOT_MIN_TOKENS:
        blk_ho, blk_t = _expand_tile(Ho)
        nblk = (Ho + blk_ho - 1) // blk_ho
        nsplit = _expand_split(NR, nblk, B)
        kern = K.sgmv_expand_dot
        kwargs = dict(R=R, Ho=Ho, L=L, NR=NR, BLOCK_HO=blk_ho, BLOCK_T=blk_t,
                      Y_HO=output_tensor.size(1), SLICE_OFF=slice_offset,
                      NBLK=nblk, NSPLIT=nsplit)
        args = (inputs, w, idx_l, start_l, output_tensor, output_tensor, B)
        grid_x = NR * nblk * nsplit
        scalars = (B,)
    elif not (R & (R - 1)):
        # flat: decode-shaped.  Unlike the dot kernel it has no tl.dot and so
        # no power-of-two-K >= 16 restriction -- only tl.arange(0, R) needs R
        # itself to be a power of two, which is the same constraint the old
        # per-token kernel below has.  Same signature minus R/L/NBLK/NSPLIT;
        # see sgmv_expand_flat for why it beats both of the others here.
        blk_ho = _flat_blk_ho(Ho)
        nblk = (Ho + blk_ho - 1) // blk_ho
        kern = K.sgmv_expand_flat
        kwargs = dict(R=R, Ho=Ho, NR=NR, BLOCK_HO=blk_ho,
                      Y_HO=output_tensor.size(1), SLICE_OFF=slice_offset)
        args = (inputs, w, idx_l, start_l, output_tensor, output_tensor, B)
        grid_x = NR * nblk
        scalars = (B,)
    else:
        kern = K.sgmv_expand
        kwargs = dict(R=R, Ho=Ho, L=L, NR=NR, BLOCK_HO=_expand_blk_ho(R),
                      Y_HO=output_tensor.size(1), SLICE_OFF=slice_offset)
        args = (inputs, w, idx_l, start_l, output_tensor, output_tensor)
        grid_x = B
        scalars = ()
    m2 = _timing_start("sgmv_expand_slice|lookup")
    m3 = _timing_start("sgmv_expand_slice|launch")
    if _cpp_enabled():
        case = _cpp_get_case(kern, kwargs, args, (grid_x,))
        if case is not None:
            _probe("p:expand_cpp")
            _cpp_launch(case, grid_x,
                        [a.data_ptr() for a in args
                         if isinstance(a, torch.Tensor)], scalars)
            _timing_end("sgmv_expand_slice", t0)
            _timing_end("sgmv_expand_slice|prep", m1)
            _timing_end("sgmv_expand_slice|lookup", m2)
            _timing_end("sgmv_expand_slice|launch", m3)
            return output_tensor
    _timing_end("sgmv_expand_slice", t0)
    _timing_end("sgmv_expand_slice|prep", m1)
    _timing_end("sgmv_expand_slice|lookup", m2)
    _timing_end("sgmv_expand_slice|launch", m3)
    _probe("p:expand_tri")
    kern[(grid_x,)](*args, **kwargs)
    _timing_end("sgmv_expand_slice", t0)
    return output_tensor


# ---- torch custom ops (dynamo-opaque, eager impls run during graph capture) ----
#
# Registration happens at the bottom of the file.  With TRITON_LORA_NATIVE=1
# (what serve sets) the C++ .so is loaded and the ops dispatch at PrivateUse1,
# which the aclgraph captures as device nodes.  Without it the python impls
# below are registered at the python dispatch key instead: still numerically
# correct, but every graph replay re-enters the python dispatcher (~496 host
# calls/s in decode) -- see the header note above.


def _unsupported(op: str, detail: str) -> None:
    """Hard error where the AscendC fallback used to be.

    AscendC accepted no dtype these kernels do not (see the note above), so the
    reroute never rescued a call -- it only replaced the error message.  Now
    that AscendC is going away, an unsupported call must fail here instead of
    silently taking a second code path.
    """
    raise RuntimeError(
        f"{op}: {detail}. The Triton kernels are the only LoRA implementation "
        f"(AscendC fallback removed); fix the caller or extend the kernel.")


def _op_bgmv_shrink(inputs: torch.Tensor, lora_a_weights: torch.Tensor,
                    output_tensor: torch.Tensor, lora_indices_tensor: torch.Tensor,
                    scaling: float) -> None:
    _announce()
    if not (_triton_dtype_ok(inputs) and _triton_dtype_ok(lora_a_weights)):
        _unsupported("bgmv_shrink",
                     f"dtype {inputs.dtype}/{lora_a_weights.dtype} is not fp16/bf16")
    if lora_a_weights.shape[-1] != inputs.shape[-1]:
        _unsupported("bgmv_shrink",
                     f"weights.shape[-1]={lora_a_weights.shape[-1]} != "
                     f"inputs.shape[-1]={inputs.shape[-1]}")
    bgmv_shrink(inputs, lora_a_weights, output_tensor, lora_indices_tensor, scaling)


def _fake_bgmv_shrink(inputs: torch.Tensor, lora_a_weights: torch.Tensor,
                      output_tensor: torch.Tensor, lora_indices_tensor: torch.Tensor,
                      scaling: float) -> None:
    return None


def _op_bgmv_expand_slice(inputs: torch.Tensor, lora_b_weights: torch.Tensor,
                          output_tensor: torch.Tensor, lora_indices_tensor: torch.Tensor,
                          slice_offset: int, slice_size: int) -> None:
    _announce()
    # inputs is the fp32 shrink buffer here, so only the weights are dtype-checked.
    if not _triton_dtype_ok(lora_b_weights):
        _unsupported("bgmv_expand_slice", f"dtype {lora_b_weights.dtype} is not fp16/bf16")
    if lora_b_weights.shape[-1] != inputs.shape[-1]:  # linear [L,1,Ho,R], not embedding
        _unsupported("bgmv_expand_slice",
                     f"weights.shape[-1]={lora_b_weights.shape[-1]} != "
                     f"inputs.shape[-1]={inputs.shape[-1]}")
    bgmv_expand_slice(inputs, lora_b_weights, output_tensor, lora_indices_tensor,
                      slice_offset, slice_size, True)


def _fake_bgmv_expand_slice(inputs: torch.Tensor, lora_b_weights: torch.Tensor,
                            output_tensor: torch.Tensor, lora_indices_tensor: torch.Tensor,
                            slice_offset: int, slice_size: int) -> None:
    return None


def _op_sgmv_shrink(inputs: torch.Tensor, lora_a_weights: torch.Tensor,
                    output_tensor: torch.Tensor, seq_start: torch.Tensor,
                    lora_indices_tensor: torch.Tensor, scaling: float) -> None:
    _announce()
    if not (_triton_dtype_ok(inputs) and _triton_dtype_ok(lora_a_weights)):
        _unsupported("sgmv_shrink",
                     f"dtype {inputs.dtype}/{lora_a_weights.dtype} is not fp16/bf16")
    if lora_a_weights.shape[-1] != inputs.shape[-1]:
        _unsupported("sgmv_shrink",
                     f"weights.shape[-1]={lora_a_weights.shape[-1]} != "
                     f"inputs.shape[-1]={inputs.shape[-1]}")
    sgmv_shrink(inputs, lora_a_weights, output_tensor, seq_start,
                lora_indices_tensor, scaling)


def _fake_sgmv_shrink(inputs: torch.Tensor, lora_a_weights: torch.Tensor,
                      output_tensor: torch.Tensor, seq_len_tensor: torch.Tensor,
                      lora_indices_tensor: torch.Tensor, scaling: float) -> None:
    return None


def _op_sgmv_expand_slice(inputs: torch.Tensor, lora_b_weights: torch.Tensor,
                          output_tensor: torch.Tensor, seq_start: torch.Tensor,
                          lora_indices_tensor: torch.Tensor,
                          slice_offset: int, slice_size: int) -> None:
    _announce()
    if not _triton_dtype_ok(lora_b_weights):
        _unsupported("sgmv_expand_slice", f"dtype {lora_b_weights.dtype} is not fp16/bf16")
    if lora_b_weights.shape[-1] != inputs.shape[-1]:  # linear [L,1,Ho,R], not embedding
        _unsupported("sgmv_expand_slice",
                     f"weights.shape[-1]={lora_b_weights.shape[-1]} != "
                     f"inputs.shape[-1]={inputs.shape[-1]}")
    sgmv_expand_slice(inputs, lora_b_weights, output_tensor, seq_start,
                      lora_indices_tensor, slice_offset, slice_size, True)


def _fake_sgmv_expand_slice(inputs: torch.Tensor, lora_b_weights: torch.Tensor,
                            output_tensor: torch.Tensor, seq_len_tensor: torch.Tensor,
                            lora_indices_tensor: torch.Tensor,
                            slice_offset: int, slice_size: int) -> None:
    return None


# ---- op registration (python impls by default; native C++ only opt-in) ----

_OPS = (
    ("vllm_ascend_triton::bgmv_shrink", _op_bgmv_shrink, _fake_bgmv_shrink,
     ("output_tensor",)),
    ("vllm_ascend_triton::bgmv_expand_slice", _op_bgmv_expand_slice,
     _fake_bgmv_expand_slice, ("output_tensor",)),
    ("vllm_ascend_triton::sgmv_shrink", _op_sgmv_shrink, _fake_sgmv_shrink,
     ("output_tensor",)),
    ("vllm_ascend_triton::sgmv_expand_slice", _op_sgmv_expand_slice,
     _fake_sgmv_expand_slice, ("output_tensor",)),
)


_SCHEMA_LIB = None  # keep the DEF Library alive (namespace dies with the handle)


def _define_schemas():
    """Schema-only registration; NPU dispatch then falls through to the C++
    TORCH_LIBRARY_IMPL(..., PrivateUse1) kernels so torch_npu's aclgraph
    capture records them exactly like the AscendC ops.

    NOTE: the DEF Library object MUST be kept alive for the whole process --
    when the handle is garbage-collected torch destroys the namespace with
    everything defined in it (custom_op works around this by leaking).
    """
    global _SCHEMA_LIB
    try:
        lib = torch.library.Library("vllm_ascend_triton", "DEF")
        for s in (
            "bgmv_shrink(Tensor inputs, Tensor lora_a_weights, Tensor output_tensor, Tensor lora_indices_tensor, float scaling) -> ()",
            "bgmv_expand_slice(Tensor inputs, Tensor lora_b_weights, Tensor output_tensor, Tensor lora_indices_tensor, int slice_offset, int slice_size) -> ()",
            "sgmv_shrink(Tensor inputs, Tensor lora_a_weights, Tensor output_tensor, Tensor seq_start, Tensor lora_indices_tensor, float scaling) -> ()",
            "sgmv_expand_slice(Tensor inputs, Tensor lora_b_weights, Tensor output_tensor, Tensor seq_start, Tensor lora_indices_tensor, int slice_offset, int slice_size) -> ()",
        ):
            try:
                lib.define(s)
            except RuntimeError:
                pass  # already defined
        _SCHEMA_LIB = lib
    except RuntimeError:
        pass  # namespace already defined


if os.environ.get("TRITON_LORA_NATIVE", "0") != "0":
    # PrivateUse1-only registration -- the shape aclgraph captures as device
    # nodes.  serve.sh sets this; see the header note for why it matters.
    _define_schemas()
    _native_setup()

for _opname, _opfn, _fakfn, _mut in _OPS:
    if _NATIVE is not None:
        # native .so loaded (TRITON_LORA_NATIVE=1): schema + fake only,
        # dispatch goes to the C++ PrivateUse1 kernels.
        register_fake(_opname, _fakfn)
    else:
        # Fallback when the .so is unavailable: python-key impl.  Correct, but
        # every graph replay re-enters the python dispatcher.  FakeTensor
        # tracing is served by the Fake key via register_fake either way.
        try:
            custom_op(_opname, mutates_args=_mut)(_opfn)
        except RuntimeError:
            # schema already defined: register the python impl directly
            torch.library.impl(_opname, "Python", _opfn)
        register_fake(_opname, _fakfn)


