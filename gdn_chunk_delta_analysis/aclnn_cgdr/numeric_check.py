#!/usr/bin/env python3
"""Self-consistency: one full ChunkGatedDeltaRule call == split into successive calls
feeding back final_state as initial_state. Also tests g (gate) interpretation across
segment boundaries. Run: run9.sh /opt/dvmvenv/bin/python -u numeric_check.py
"""
import numpy as np, ctypes as C
import cgdr

def nel(sh):
    n = 1
    for x in sh: n *= x
    return n

def b16(f32):
    return cgdr.to_bf16(f32)

class Runner:
    def __init__(self, acl):
        self.acl = acl
        acl.load_op("ChunkGatedDeltaRule")
    def run(self, q, k, v, beta, g, s0, scale=1.0, seed_pert=0):
        """q..: np arrays [m,nk,dk] etc. returns (out[m,nv,dv], final[b,nv,dv,dk]) as float32"""
        a = self.acl
        m = q.shape[0]; b, nv, dv, dk = s0.shape; nk = q.shape[1]
        qs = [m, nk, dk]; vs = [m, nv, dv]; bs_ = [m, nv]
        def alloc(raw):
            p = a.malloc(len(raw)); a.h2d(bytes(raw), p); return p
        qp = alloc(b16(q).tobytes()); kp = alloc(b16(k).tobytes()); vp = alloc(b16(v).tobytes())
        bp = alloc(b16(beta).tobytes()); sp = alloc(b16(s0).tobytes())
        seq = alloc(np.asarray([m], np.int32).tobytes())
        gp = alloc(g.astype(np.float32).tobytes()) if g is not None else 0
        op = alloc(bytes(np.full(nel(vs), 0xCD, np.uint16)))
        fp = alloc(bytes(np.full(nel(s0.shape), 0xCD, np.uint16)))
        tq, _ = a.tensor(qs, 'BF16', qp); tk, _ = a.tensor(qs, 'BF16', kp)
        tv, _ = a.tensor(vs, 'BF16', vp); tb, _ = a.tensor(bs_, 'BF16', bp)
        ti, _ = a.tensor(list(s0.shape), 'BF16', sp); ts, _ = a.tensor([b], 'INT32', seq)
        tg, _ = a.tensor(bs_, 'FLOAT', gp) if g is not None else (0, None)
        to, _ = a.tensor(vs, 'BF16', op); tf, _ = a.tensor(list(s0.shape), 'BF16', fp)
        rc, ws, ex, em = a.gw([tq, tk, tv, tb, ti, ts, tg, to, tf], scale)
        if rc != 0:
            raise RuntimeError("gw rc=%d %s" % (rc, em[:160]))
        wsp = a.malloc(ws)
        rc = a.launch(wsp, ws, ex)
        a.acl.aclrtSynchronizeStream(a.stream)
        out = np.frombuffer(a.d2h(op, nel(vs) * 2), np.uint16).astype(np.uint32).reshape(vs)
        fin = np.frombuffer(a.d2h(fp, nel(s0.shape) * 2), np.uint16).astype(np.uint32).reshape(list(s0.shape))
        for p in (qp, kp, vp, bp, sp, op, fp, seq):
            a.free(p)
        if g is not None: a.free(gp)
        a.free(wsp)
        return ((out << 16).view(np.float32), (fin << 16).view(np.float32))

def main():
    dev = 0
    acl = cgdr.Acl(device=dev)
    rng = np.random.default_rng(3)
    nk, nv, dk, dv = 16, 32, 128, 128
    T = 512
    b = 1
    q = rng.uniform(-0.2, 0.2, size=(T, nk, dk)).astype(np.float32)
    k = rng.uniform(-0.2, 0.2, size=(T, nk, dk)).astype(np.float32)
    v = rng.uniform(-0.2, 0.2, size=(T, nv, dv)).astype(np.float32)
    beta = rng.uniform(-0.05, 0.05, size=(T, nv)).astype(np.float32)
    s0 = rng.uniform(-0.05, 0.05, size=(b, nv, dv, dk)).astype(np.float32)
    S0 = s0
    runner = Runner(acl)

    def split_report(name, gfull, g_factory):
        """compare full call vs splits where segment j gets g_slice produced by g_factory."""
        gT = gfull
        outF, finF = runner.run(q, k, v, beta, gT, S0)
        for kcut in (128, 256, 384):
            ga = gT[:kcut]
            outA, finA = runner.run(q[:kcut], k[:kcut], v[:kcut], beta[:kcut], ga, S0)
            gb = g_factory(kcut)
            outB, finB = runner.run(q[kcut:], k[kcut:], v[kcut:], beta[kcut:], gb, finA)
            dA = np.abs(outA - outF[:kcut]).max()
            dB = np.abs(outB - outF[kcut:]).max()
            dS = np.abs(finB - finF).max()
            print("[%s] kcut=%d  |outA-fullA|max=%.3e  |outB-fullB|max=%.3e  |finalB-finalFull|max=%.3e"
                  % (name, kcut, dA, dB, dS), flush=True)

    import sys
    if "sweep" in sys.argv:
        # which cut positions give EXACT state handoff? (g=0 to isolate state composition)
        g0 = np.zeros((T, nv), np.float32)
        outF, finF = runner.run(q, k, v, beta, g0, S0)
        viol = []
        for kcut in range(1, T):
            outA, finA = runner.run(q[:kcut], k[:kcut], v[:kcut], beta[:kcut], g0[:kcut], S0)
            outB, finB = runner.run(q[kcut:], k[kcut:], v[kcut:], beta[kcut:], g0[kcut:], finA)
            d = max(np.abs(outA - outF[:kcut]).max(), np.abs(outB - outF[kcut:]).max(),
                    np.abs(finB - finF).max())
            if d > 1e-6:
                viol.append((kcut, d))
        print("sweep violators (kcut, diff):", viol[:80], "count=%d" % len(viol))
        return

    # case 1: g=0 constant -> no decay; expect splits equal if state handoff is correct
    g0 = np.zeros((T, nv), np.float32)
    split_report("g0", g0, lambda kcut: np.zeros((T - kcut, nv), np.float32))
    # case 2: constant +0.01 gate increments
    gc = np.full((T, nv), 0.01, np.float32)
    split_report("g+0.01", gc, lambda kcut: np.full((T - kcut, nv), 0.01, np.float32))
    # case 3: random per-token gate (approx log-decay), sliced raw
    gr = rng.uniform(-0.02, 0.02, size=(T, nv)).astype(np.float32)
    split_report("g_rand_slice", gr, lambda kcut: gr[kcut:].copy())

if __name__ == "__main__":
    main()
