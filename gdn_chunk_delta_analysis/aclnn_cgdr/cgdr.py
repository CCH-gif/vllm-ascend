#!/usr/bin/env python3
"""Thin ctypes wrapper to call the CANN9.1 aclnnChunkGatedDeltaRule (built-in op)
directly on host 910B via libopapi/libnnopbase/libascendcl from the extracted 9.1 root.

Run under run9.sh (clean env). Import by probe/timing scripts.
"""
import ctypes as C
import numpy as np

B = "/tmp/cann91/root/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64"
DT = dict(FLOAT=0, FLOAT16=1, INT32=3, BF16=27)
FMT_ND = 2
H2D, D2H = 1, 2

class Acl:
    def __init__(self, device=0):
        acl = C.CDLL(B + "/libascendcl.so", mode=C.RTLD_GLOBAL)
        nnop = C.CDLL(B + "/libnnopbase.so", mode=C.RTLD_GLOBAL)
        opapi = C.CDLL(B + "/libopapi.so", mode=C.RTLD_GLOBAL)
        self.acl, self.nnop, self.opapi = acl, nnop, opapi

        i32 = C.c_int32
        for f, argt in [
            ("aclInit", [C.c_char_p]), ("aclrtSetDevice", [i32]),
            ("aclrtSynchronizeDevice", []),
            ("aclrtCreateEvent", [C.POINTER(C.c_void_p)]),
            ("aclrtDestroyEvent", [C.c_void_p]),
            ("aclrtRecordEvent", [C.c_void_p, C.c_void_p]),
            ("aclrtSynchronizeEvent", [C.c_void_p]),
            ("aclrtSynchronizeStream", [C.c_void_p]),
            ("aclrtFree", [C.c_void_p]),
        ]:
            fn = getattr(acl, f); fn.restype = i32; fn.argtypes = argt
        acl.aclrtCreateStream.restype = i32
        acl.aclrtCreateStream.argtypes = [C.POINTER(C.c_void_p)]
        acl.aclrtMalloc.restype = i32
        acl.aclrtMalloc.argtypes = [C.POINTER(C.c_void_p), C.c_size_t, i32]
        acl.aclrtMemcpy.restype = i32
        acl.aclrtMemcpy.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t, i32]
        acl.aclrtGetSocName.restype = C.c_char_p
        acl.aclrtGetSocName.argtypes = []
        acl.aclGetRecentErrMsg.restype = C.c_char_p
        acl.aclGetRecentErrMsg.argtypes = []

        rc = acl.aclInit(None)
        if rc != 0:
            raise RuntimeError("aclInit rc=%d %s" % (rc, self.err()))
        rc = acl.aclrtSetDevice(device)
        if rc != 0:
            raise RuntimeError("aclrtSetDevice rc=%d" % rc)
        st = C.c_void_p()
        rc = acl.aclrtCreateStream(C.byref(st))
        if rc != 0:
            raise RuntimeError("create stream rc=%d" % rc)
        self.stream = st
        self.soc = (acl.aclrtGetSocName() or b"?").decode()
        self._keep = []  # keep dims arrays alive for tensor lifetime

    def err(self):
        try:
            return (self.acl.aclGetRecentErrMsg() or b"").decode(errors="replace")
        except Exception:
            return ""

    def malloc(self, nbytes):
        p = C.c_void_p()
        rc = self.acl.aclrtMalloc(C.byref(p), nbytes, 0)
        if rc != 0:
            raise RuntimeError("aclrtMalloc(%d) rc=%d %s" % (nbytes, rc, self.err()))
        return p.value

    def free(self, ptr):
        if ptr:
            self.acl.aclrtFree(C.c_void_p(ptr))

    def h2d(self, raw: bytes, dst_ptr: int):
        if not raw:
            return
        src = C.create_string_buffer(raw, len(raw))
        rc = self.acl.aclrtMemcpy(C.c_void_p(dst_ptr), len(raw), src, len(raw), H2D)
        if rc != 0:
            raise RuntimeError("h2d rc=%d %s" % (rc, self.err()))

    def d2h(self, dst_ptr: int, nbytes: int) -> bytes:
        buf = C.create_string_buffer(nbytes)
        rc = self.acl.aclrtMemcpy(buf, nbytes, C.c_void_p(dst_ptr), nbytes, D2H)
        if rc != 0:
            raise RuntimeError("d2h rc=%d %s" % (rc, self.err()))
        return buf.raw

    def tensor(self, shape, dtype_str, addr):
        dims = (C.c_int64 * len(shape))(*[int(x) for x in shape])
        self.nnop.aclCreateTensor.restype = C.c_void_p
        self.nnop.aclCreateTensor.argtypes = [
            C.POINTER(C.c_int64), C.c_uint64, C.c_int32, C.POINTER(C.c_int64),
            C.c_int64, C.c_int32, C.POINTER(C.c_int64), C.c_uint64, C.c_void_p]
        t = self.nnop.aclCreateTensor(dims, len(shape), DT[dtype_str], None, 0,
                                      FMT_ND, dims, len(shape), C.c_void_p(addr))
        self.nnop.aclDestroyTensor.restype = C.c_int32
        self.nnop.aclDestroyTensor.argtypes = [C.c_void_p]
        self._keep.append(dims)  # hold reference: aclTensor points into dims
        return t, dims

    def destroy_tensor(self, t):
        if t:
            self.nnop.aclDestroyTensor(C.c_void_p(t))

    def load_op(self, name):
        opapi = self.opapi
        gw = getattr(opapi, "aclnn%sGetWorkspaceSize" % name)
        gw.restype = C.c_int32
        ex = getattr(opapi, "aclnn%s" % name)
        ex.restype = C.c_int32
        self._gw, self._ex = gw, ex

    def gw(self, tensors, scale, extra=None):
        """GetWorkspaceSize probe. tensors: list of aclTensor* (ints) or 0 for absent.
        Returns (rc, ws, executor, errmsg)."""
        ws = C.c_uint64(0)
        executor = C.c_void_p()
        args = [C.c_void_p(t) if t else C.c_void_p(0) for t in tensors]
        args.append(C.c_float(scale))
        if extra:  # optional trailing byref pairs handled by caller via wrapper below
            pass
        args += [C.byref(ws), C.byref(executor)]
        rc = self._gw(*args)
        return rc, ws.value, int(executor.value or 0), self.err()

    def launch(self, ws_ptr, ws_size, executor):
        rc = self._ex(C.c_void_p(ws_ptr), C.c_uint64(ws_size),
                      C.c_void_p(executor), C.c_void_p(self.stream.value))
        return rc


def to_bf16(f32):
    """f32 -> uint16 array of bf16 bit patterns (truncation)."""
    u = f32.astype(np.float32).view(np.uint32)
    return (u >> 16).astype(np.uint16)


def event_timer(acl):
    """returns (start,stop) event pair created on current stream context."""
    e1 = C.c_void_p(); e2 = C.c_void_p()
    acl.acl.aclrtCreateEvent(C.byref(e1))
    acl.acl.aclrtCreateEvent(C.byref(e2))
    return e1.value, e2.value
