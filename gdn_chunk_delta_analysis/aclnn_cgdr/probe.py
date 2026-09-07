#!/usr/bin/env python3
"""Probe which ND layouts pass host validation (GetWorkspaceSize) for ChunkGatedDeltaRule.
Run: run9.sh /opt/dvmvenv/bin/python -u probe.py [device]
"""
import sys, numpy as np, ctypes as C
import cgdr

def elem(shape):
    n = 1
    for s in shape:
        n *= s
    return n

def try_shape(acl, tag, T, nk, dk, nv, dv, b, lens, qk_shape, v_shape, beta_shape,
              state_shape, g_opt=True, g_shape=None):
    t = int(sum(lens))
    shapes = {
        'q': qk_shape, 'k': qk_shape, 'v': v_shape,
        'beta': beta_shape, 'init': state_shape, 'final': state_shape,
        'g': g_shape or beta_shape if g_opt else None,
    }
    out_shape = v_shape
    # allocate device
    bufs = {}
    def alloc(name, shape, dtype, nbytes):
        addr = acl.malloc(nbytes)
        bufs[name] = (addr, nbytes)
        return addr
    try:
        qaddr = alloc('q', qk_shape, 'BF16', elem(qk_shape) * 2)
        kaddr = alloc('k', qk_shape, 'BF16', elem(qk_shape) * 2)
        vaddr = alloc('v', v_shape, 'BF16', elem(v_shape) * 2)
        betaddr = alloc('beta', beta_shape, 'BF16', elem(beta_shape) * 2)
        staddr = alloc('init', state_shape, 'BF16', elem(state_shape) * 2)
        fnaddr = alloc('final', state_shape, 'BF16', elem(state_shape) * 2)
        seql = acl.malloc(b * 4)
        acl.h2d(np.asarray(lens, np.int32).tobytes(), seql)
        gaddr = 0
        if g_opt:
            gaddr = alloc('g', shapes['g'], 'FLOAT', elem(shapes['g']) * 4)
        outaddr = alloc('out', out_shape, 'BF16', elem(out_shape) * 2)
        # aclTensor
        tq, dq = acl.tensor(qk_shape, 'BF16', qaddr)
        tk, dk_ = acl.tensor(qk_shape, 'BF16', kaddr)
        tv, dv_ = acl.tensor(v_shape, 'BF16', vaddr)
        tb, db = acl.tensor(beta_shape, 'BF16', betaddr)
        ti, di = acl.tensor(state_shape, 'BF16', staddr)
        ts, ds = acl.tensor([b], 'INT32', seql)
        tg = acl.tensor(shapes['g'], 'FLOAT', gaddr)[0] if g_opt else 0
        to, do = acl.tensor(out_shape, 'BF16', outaddr)
        tf, df = acl.tensor(state_shape, 'BF16', fnaddr)
        rc, ws, ex, errmsg = acl.gw([tq, tk, tv, tb, ti, ts, tg, to, tf], 1.0)
        print("[%s] rc=%d ws=%d soc=%s err=%s" % (tag, rc, ws, acl.soc,
              errmsg[:200] if errmsg else ""), flush=True)
        if rc != 0:
            return None
        return (tag, shapes, ws)
    finally:
        for n, (a, _) in bufs.items():
            acl.free(a)
        acl.free(seql)
        acl.destroy_tensor(tq) if 'tq' in dir() else None
        for name in ('tq','tk','tv','tb','ti','ts','to','tf'):
            pass
        # tensors leak a bit per iter; acceptable for a short probe

def main():
    dev = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    acl = cgdr.Acl(device=dev)
    acl.load_op("ChunkGatedDeltaRule")
    print("soc:", acl.soc, flush=True)
    # real model dims
    nk, dk, nv, dv, b = 16, 128, 32, 128, 1
    lens = [512]
    T = sum(lens)
    cands = [
        ("qk2D_v2D", [T, nk*dk], [T, nv*dv], [T, nv], [b, nv, dv, dk]),
        ("qk3D_v3D", [T, nk, dk], [T, nv, dv], [T, nv], [b, nv, dv, dk]),
        ("qk4D_v4D", [1, T, nk, dk], [1, T, nv, dv], [1, T, nv], [b, nv, dv, dk]),
        ("qk2D_v2D_st2", [T, nk*dk], [T, nv*dv], [T, nv], [b, nv, dv*dk]),
        ("qk3D_v3D_st3", [T, nk, dk], [T, nv, dv], [T, nv], [b, nv, dv, dk]),
        ("qk1D", [T*nk*dk], [T*nv*dv], [T*nv], [b*nv*dv*dk]),
    ]
    ok = None
    for tag, qks, vs, bs, ss in cands:
        r = try_shape(acl, tag, T, nk, dk, nv, dv, b, lens, qks, vs, bs, ss, True)
        if r:
            ok = r
            break
    print("RESULT_OK:", ok)

if __name__ == "__main__":
    main()
