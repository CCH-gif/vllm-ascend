#!/usr/bin/env python3
"""Run aclnnChunkGatedDeltaRule (9.1, 910B) with chosen model dims + sanity + timing.
Run: run9.sh /opt/dvmvenv/bin/python -u run_cgdr.py --mode zero|rand|time --t 512 --b 1 --nk 16 --nv 32 --dk 128 --dv 128 [--g 0|1] [--n 50] [--dev 0]
"""
import sys, time, numpy as np, ctypes as C, argparse, faulthandler
faulthandler.enable()
import cgdr

def bf16_to_f32(raw):
    u = np.frombuffer(raw, np.uint16).astype(np.uint32)
    f = (u << 16).view(np.float32)
    return f

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="rand")
    ap.add_argument("--t", type=int, default=512)
    ap.add_argument("--b", type=int, default=1)
    ap.add_argument("--nk", type=int, default=16)
    ap.add_argument("--nv", type=int, default=32)
    ap.add_argument("--dk", type=int, default=128)
    ap.add_argument("--dv", type=int, default=128)
    ap.add_argument("--g", type=int, default=1, help="pass gOptional tensor (1) or omit (0)")
    ap.add_argument("--n", type=int, default=50, help="timing iters")
    ap.add_argument("--dev", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()

    acl = cgdr.Acl(device=a.dev)
    acl.load_op("ChunkGatedDeltaRule")
    t, b, nk, nv, dk, dv = a.t, a.b, a.nk, a.nv, a.dk, a.dv
    T = a.t
    lens = [T] * b
    qshape = [T, nk, dk]; vshape = [T, nv, dv]; bshape = [T, nv]
    stshape = [b, nv, dv, dk]

    def nel(sh):
        n = 1
        for x in sh: n *= x
        return n

    rng = np.random.default_rng(7)
    def bf16_bytes(sh, lo=-0.2, hi=0.2):
        return cgdr.to_bf16(rng.uniform(lo, hi, size=sh)).tobytes()

    qaddr = acl.malloc(nel(qshape) * 2); kaddr = acl.malloc(nel(qshape) * 2)
    vaddr = acl.malloc(nel(vshape) * 2)
    baddr = acl.malloc(nel(bshape) * 2); sadd = acl.malloc(nel(stshape) * 2)
    fadd = acl.malloc(nel(stshape) * 2); oaddr = acl.malloc(nel(vshape) * 2)
    seq = acl.malloc(b * 4)
    acl.h2d(np.asarray(lens, np.int32).tobytes(), seq)
    gaddr = acl.malloc(nel(bshape) * 4) if a.g else 0

    # ---- zero test: beta=0, init=0, g chosen; expect out/final == 0 exactly ----
    if a.mode == "zero":
        acl.h2d(bf16_bytes([T, nk, dk], -0.2, 0.2), qaddr)
        acl.h2d(bf16_bytes([T, nk, dk], -0.2, 0.2), kaddr)
        acl.h2d(bf16_bytes(vshape, -0.2, 0.2), vaddr)
        acl.h2d(np.zeros(nel(bshape), np.uint16).tobytes(), baddr)   # beta=0
        acl.h2d(np.zeros(nel(stshape), np.uint16).tobytes(), sadd)   # init=0
        if a.g:
            acl.h2d(rng.normal(0, .05, size=nel(bshape)).astype(np.float32).tobytes(), gaddr)
        garbage = (np.full(nel(vshape), 0xCD, np.uint16)).tobytes()
        acl.h2d(garbage, oaddr)
        acl.h2d(garbage, fadd)

        tq, dq = acl.tensor(qshape, 'BF16', qaddr); tk, _ = acl.tensor(qshape, 'BF16', kaddr)
        tv, _ = acl.tensor(vshape, 'BF16', vaddr); tb, _ = acl.tensor(bshape, 'BF16', baddr)
        ti, _ = acl.tensor(stshape, 'BF16', sadd); ts, _ = acl.tensor([b], 'INT32', seq)
        tg, _ = acl.tensor(bshape, 'FLOAT', gaddr) if a.g else (0, None)
        to, _ = acl.tensor(vshape, 'BF16', oaddr); tf, _ = acl.tensor(stshape, 'BF16', fadd)
        rc, ws, ex, em = acl.gw([tq, tk, tv, tb, ti, ts, tg, to, tf], a.scale)
        if rc != 0:
            print("gw rc=%d %s" % (rc, em[:200])); sys.exit(1)
        wsp = acl.malloc(ws)
        rc = acl.launch(wsp, ws, ex)
        acl.acl.aclrtSynchronizeStream(acl.stream)
        print("zero-launch rc=%d (0=ok)" % rc)
        o = bf16_to_f32(acl.d2h(oaddr, nel(vshape) * 2))
        f = bf16_to_f32(acl.d2h(fadd, nel(stshape) * 2))
        print("out  nonzero=%d min=%g max=%g" % (np.count_nonzero(o), o.min(), o.max()))
        print("final nonzero=%d min=%g max=%g" % (np.count_nonzero(f), f.min(), f.max()))
        acl.free(wsp)
        sys.exit(0)

    # ---- random sanity / timing ----
    acl.h2d(bf16_bytes([T, nk, dk], -0.2, 0.2), qaddr)
    acl.h2d(bf16_bytes([T, nk, dk], -0.2, 0.2), kaddr)
    acl.h2d(bf16_bytes(vshape, -0.2, 0.2), vaddr)
    acl.h2d(bf16_bytes(bshape, -0.05, 0.05), baddr)
    acl.h2d(bf16_bytes(stshape, -0.05, 0.05), sadd)
    if a.g:
        acl.h2d(rng.uniform(-0.01, 0.01, size=nel(bshape)).astype(np.float32).tobytes(), gaddr)

    tq, dq = acl.tensor(qshape, 'BF16', qaddr); tk, _ = acl.tensor(qshape, 'BF16', kaddr)
    tv, _ = acl.tensor(vshape, 'BF16', vaddr); tb, _ = acl.tensor(bshape, 'BF16', baddr)
    ti, _ = acl.tensor(stshape, 'BF16', sadd); ts, _ = acl.tensor([b], 'INT32', seq)
    tg, _ = acl.tensor(bshape, 'FLOAT', gaddr) if a.g else (0, None)
    to, _ = acl.tensor(vshape, 'BF16', oaddr); tf, _ = acl.tensor(stshape, 'BF16', fadd)
    rc, ws, ex, em = acl.gw([tq, tk, tv, tb, ti, ts, tg, to, tf], a.scale)
    print("gw rc=%d ws=%.1fMB err=%s" % (rc, ws / 1e6, em[:120]))
    if rc != 0:
        sys.exit(1)
    wsp = acl.malloc(ws)
    print("dbg: ws alloc ok", flush=True)

    def sync():
        return acl.acl.aclrtSynchronizeStream(acl.stream)

    tensors = [tq, tk, tv, tb, ti, ts, tg, to, tf]
    def fresh():
        rc, ws2, ex2, em2 = acl.gw(tensors, a.scale)
        return rc, ex2
    sync()
    print("dbg: warmup begin", flush=True)
    # warmup (fresh executor each)
    for w in range(3):
        rc, exw = fresh()
        print("dbg: warmup iter %d gw_rc=%d" % (w, rc), flush=True)
        rcl = acl.launch(wsp, ws, exw)
        rcs = sync()
        print("dbg: warmup iter %d launch_rc=%d sync_rc=%d" % (w, rcl, rcs), flush=True)
    print("dbg: warmup ok", flush=True)
    # perturb q every iter to defeat any caching (rewrite 8 floats)
    pert = cgdr.to_bf16(rng.uniform(-.2, .2, size=8))
    if a.mode == "rand":
        print("dbg: rand launch", flush=True)
        t0 = time.time(); acl.launch(wsp, ws, ex); sync(); dt = time.time() - t0
        print("dbg: rand sync ok", flush=True)
        o = bf16_to_f32(acl.d2h(oaddr, nel(vshape) * 2))
        f = bf16_to_f32(acl.d2h(fadd, nel(stshape) * 2))
        print("rand first: wall=%.4fs  out[finite=%d nonz=%d min=%g max=%g mean|o|=%g" %
              (dt, int(np.isfinite(o).all()), np.count_nonzero(o), o.min(), o.max(), np.abs(o).mean()))
        print("final: finite=%d nonz=%d min=%g max=%g" % (int(np.isfinite(f).all()), np.count_nonzero(f), f.min(), f.max()))
        return

    # ---- device-time timing loop ----
    n = a.n
    ev1 = C.c_void_p(); ev2 = C.c_void_p()
    acl.acl.aclrtCreateEvent(C.byref(ev1)); acl.acl.aclrtCreateEvent(C.byref(ev2))
    dev = []
    wall0 = time.time()
    for i in range(n):
        acl.h2d(pert.tobytes(), qaddr + (i % 17) * 16)  # perturb q
        rcg, exi = fresh()
        if rcg != 0:
            print("gw rc=%d at iter %d" % (rcg, i)); break
        acl.acl.aclrtRecordEvent(ev1, acl.stream)
        acl.launch(wsp, ws, exi)
        acl.acl.aclrtRecordEvent(ev2, acl.stream)
        acl.acl.aclrtSynchronizeEvent(ev1)  # ensure prior done (serialized)
        acl.acl.aclrtSynchronizeEvent(ev2)
        mt = C.c_float(0)
        # elapsed ms
        elapsed_ms = C.c_float(0)
        acl.acl.aclrtEventElapsedTime.restype = C.c_int32
        acl.acl.aclrtEventElapsedTime.argtypes = [C.POINTER(C.c_float), C.c_void_p, C.c_void_p]
        rc = acl.acl.aclrtEventElapsedTime(C.byref(elapsed_ms), ev1, ev2)
        if rc != 0:
            raise RuntimeError("elapsed rc=%d" % rc)
        dev.append(elapsed_ms.value)
    wall = time.time() - wall0
    dev = np.array(dev)
    print("DEV ms: n=%d mean=%.4f median=%.4f min=%.4f p90=%.4f (per-iter event, sync each)" %
          (n, dev.mean(), np.median(dev), dev.min(), np.percentile(dev, 90)))
    print("WALL total=%.4fs mean=%.4fms" % (wall, wall / n * 1e3))

if __name__ == "__main__":
    main()
