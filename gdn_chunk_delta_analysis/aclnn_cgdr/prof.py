#!/usr/bin/env python3
"""Short fixed-shape runner for profiling. 6 launches, then exit."""
import numpy as np, time
import cgdr

def nel(sh):
    n = 1
    for x in sh: n *= x
    return n

def main():
    import sys
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    nk, nv, dk, dv = 16, 32, 128, 128
    acl = cgdr.Acl(device=0)
    acl.load_op("ChunkGatedDeltaRule")
    rng = np.random.default_rng(5)
    qs = [T, nk, dk]; vs = [T, nv, dv]; bsh = [T, nv]; stsh = [1, nv, dv, dk]
    def mall(raw):
        p = acl.malloc(len(raw)); acl.h2d(bytes(raw), p); return p
    qp = mall(cgdr.to_bf16(rng.uniform(-.2, .2, (T, nk, dk))).tobytes())
    kp = mall(cgdr.to_bf16(rng.uniform(-.2, .2, (T, nk, dk))).tobytes())
    vp = mall(cgdr.to_bf16(rng.uniform(-.2, .2, (T, nv, dv))).tobytes())
    bp = mall(cgdr.to_bf16(rng.uniform(-.05, .05, (T, nv))).tobytes())
    sp = mall(cgdr.to_bf16(rng.uniform(-.05, .05, (1, nv, dv, dk))).tobytes())
    seq = mall(np.asarray([T], np.int32).tobytes())
    gp = mall(rng.uniform(-.01, .01, (T, nv)).astype(np.float32).tobytes())
    op = acl.malloc(nel(vs) * 2); fp = acl.malloc(nel(stsh) * 2)
    tq, _ = acl.tensor(qs, 'BF16', qp); tk, _ = acl.tensor(qs, 'BF16', kp)
    tv, _ = acl.tensor(vs, 'BF16', vp); tb, _ = acl.tensor(bsh, 'BF16', bp)
    ti, _ = acl.tensor(stsh, 'BF16', sp); ts, _ = acl.tensor([1], 'INT32', seq)
    tg, _ = acl.tensor(bsh, 'FLOAT', gp); to, _ = acl.tensor(vs, 'BF16', op)
    tf, _ = acl.tensor(stsh, 'BF16', fp)
    ten = [tq, tk, tv, tb, ti, ts, tg, to, tf]
    rc, ws, ex, em = acl.gw(ten, 1.0)
    assert rc == 0, em[:200]
    wsp = acl.malloc(ws)
    acl.acl.aclrtSynchronizeStream(acl.stream)
    for i in range(6):
        rc, _w, exi, _e = acl.gw(ten, 1.0)
        acl.launch(wsp, ws, exi)
        acl.acl.aclrtSynchronizeStream(acl.stream)
        if i == 0:
            acl.h2d(np.asarray([0xbeef, 0xbeef], np.uint16).tobytes(), qp)  # perturb once
    acl.acl.aclrtSynchronizeDevice()
    print("prof run done T=%d" % T)

if __name__ == "__main__":
    main()
