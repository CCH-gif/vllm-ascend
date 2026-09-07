#!/usr/bin/env python3
"""Device + host timing of official ChunkGatedDeltaRule (9.1) across seq length T.
Run: run9.sh /opt/dvmvenv/bin/python -u timing.py --ts 512,2048,8192,32768 --g on,off --n 40
CSV to stdout: T,g,wsMB,hostgw_ms,dev_ms_mean,dev_ms_median,dev_ms_p90,dev_ms_min,wall_ms_mean,niter
"""
import argparse, time, numpy as np, ctypes as C
import cgdr

def nel(sh):
    n = 1
    for x in sh: n *= x
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default="512,2048,8192")
    ap.add_argument("--g", default="on,off")
    ap.add_argument("--nk", type=int, default=16)
    ap.add_argument("--nv", type=int, default=32)
    ap.add_argument("--dk", type=int, default=128)
    ap.add_argument("--dv", type=int, default=128)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--dev", type=int, default=0)
    a = ap.parse_args()
    nk, nv, dk, dv = a.nk, a.nv, a.dk, a.dv
    acl = cgdr.Acl(device=a.dev)
    acl.load_op("ChunkGatedDeltaRule")
    rng = np.random.default_rng(11)

    ev1 = C.c_void_p(); ev2 = C.c_void_p()
    acl.acl.aclrtCreateEvent(C.byref(ev1)); acl.acl.aclrtCreateEvent(C.byref(ev2))
    acl.acl.aclrtEventElapsedTime.restype = C.c_int32
    acl.acl.aclrtEventElapsedTime.argtypes = [C.POINTER(C.c_float), C.c_void_p, C.c_void_p]

    print("T,g,wsMB,hostgw_ms_mean,dev_mean_ms,dev_median_ms,dev_p90_ms,dev_min_ms,wall_ms_mean,n")
    for Ts in a.ts.split(","):
        T = int(Ts)
        for gmode in a.g.split(","):
            gon = gmode == "on"
            qs = [T, nk, dk]; vs = [T, nv, dv]; bsh = [T, nv]; stsh = [1, nv, dv, dk]
            q = cgdr.to_bf16(rng.uniform(-0.2, 0.2, size=(T, nk, dk))).tobytes()
            k = cgdr.to_bf16(rng.uniform(-0.2, 0.2, size=(T, nk, dk))).tobytes()
            v = cgdr.to_bf16(rng.uniform(-0.2, 0.2, size=(T, nv, dv))).tobytes()
            beta = cgdr.to_bf16(rng.uniform(-0.05, 0.05, size=(T, nv))).tobytes()
            s0 = cgdr.to_bf16(rng.uniform(-0.05, 0.05, size=(1, nv, dv, dk))).tobytes()
            gbytes = rng.uniform(-0.01, 0.01, size=(T, nv)).astype(np.float32).tobytes() if gon else None
            def mall(raw):
                p = acl.malloc(len(raw)); acl.h2d(bytes(raw), p); return p
            qp = mall(q); kp = mall(k); vp = mall(v); bp = mall(beta); sp = mall(s0)
            op = acl.malloc(nel(vs) * 2); fp = acl.malloc(nel(stsh) * 2)
            seq = mall(np.asarray([T], np.int32).tobytes())
            gp = mall(gbytes) if gon else 0
            tq, _ = acl.tensor(qs, 'BF16', qp); tk, _ = acl.tensor(qs, 'BF16', kp)
            tv, _ = acl.tensor(vs, 'BF16', vp); tb, _ = acl.tensor(bsh, 'BF16', bp)
            ti, _ = acl.tensor(stsh, 'BF16', sp); ts, _ = acl.tensor([1], 'INT32', seq)
            tg, _ = acl.tensor(bsh, 'FLOAT', gp) if gon else (0, None)
            to, _ = acl.tensor(vs, 'BF16', op); tf, _ = acl.tensor(stsh, 'BF16', fp)
            tensors = [tq, tk, tv, tb, ti, ts, tg, to, tf]
            # first gw
            rc, ws, ex, em = acl.gw(tensors, 1.0)
            if rc != 0:
                print("ERR T=%d g=%s gw rc=%d %s" % (T, gmode, rc, em[:120])); continue
            wsp = acl.malloc(ws)
            n = a.n
            host = []
            devm = []
            wall = []
            # measure host gw cost separately (per iter)
            for i in range(n + 6):
                t0 = time.time()
                rcg, _w2, exi, _em = acl.gw(tensors, 1.0)  # fresh executor each iter (reuse crashes)
                if rcg != 0:
                    print("gw rc=%d iter %d T=%d" % (rcg, i, T)); break
                hg = (time.time() - t0) * 1e3
                # perturb q (few bytes) to defeat any device read cache
                if i % 4 == 0:
                    pert = cgdr.to_bf16(rng.uniform(-.2, .2, size=8)).tobytes()
                    acl.h2d(pert, qp + ((i * 37) % max(1, T - 1)) * nk * dk * 2)
                t0 = time.time()
                acl.acl.aclrtRecordEvent(ev1, acl.stream)
                acl.launch(wsp, ws, exi)
                acl.acl.aclrtRecordEvent(ev2, acl.stream)
                acl.acl.aclrtSynchronizeEvent(ev2)
                ms = C.c_float(0)
                acl.acl.aclrtEventElapsedTime(C.byref(ms), ev1, ev2)
                w = (time.time() - t0) * 1e3
                if i >= 6:
                    host.append(hg); devm.append(ms.value); wall.append(w)
            devm = np.array(devm)
            host = np.array(host)
            print("%d,%s,%.1f,%.3f,%.4f,%.4f,%.4f,%.4f,%.3f,%d" % (
                T, gmode, ws / 1e6, host.mean(), devm.mean(), np.median(devm),
                np.percentile(devm, 90), devm.min(), np.mean(wall), n), flush=True)
            for p in (qp, kp, vp, bp, sp, op, fp, seq, wsp):
                acl.free(p)
            if gon: acl.free(gp)

if __name__ == "__main__":
    main()
