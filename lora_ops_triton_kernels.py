"""Triton bgmv/sgmv kernels (validated bit-exact vs AscendC on 910B4).

Adapted from /tmp/bgmv-triton + /tmp/sgmv_repo (tail-block mask fix).
expand kernels support slice semantics (y_out[:, offset:offset+size] += x @ w[:, offset:offset+size]).
"""
import triton
import triton.language as tl

# ---- bgmv ----

GRID_HINT_B = ["B"]

_WIN: tl.constexpr = 11776  # native TILE_LENGTH


@triton.jit
def bgmv_shrink(X, lora_a, indices, y, scale,
                H: tl.constexpr, R: tl.constexpr, L: tl.constexpr,
                BW: tl.constexpr):
    """Bgmv shrink (``x @ A^T``), one program per token row.

    The B axis is the only parallelism this kernel has, so a decode step at
    B=1 reduces the whole H axis on a single core.  Reducing in 64-wide chunks
    (AscendC's ``TILE_LENGTH`` structure) cost 64 dependent iterations there and
    measured 18.7us against AscendC's 14.2us.  A single ``tl.sum`` over a
    BW-wide tile is ~2x cheaper per element and comes in FASTER than AscendC at
    every batch size measured (B=1 9.5us, B=4 10.3us, B=8 11.0us, B=32 17.7us).

    That reorders the fp32 accumulation, so the result is no longer bit-identical
    to AscendC: measured 3e-8 .. 1.8e-7 absolute on bf16 inputs across B=1..32,
    i.e. far below what the bf16 output can represent, and within 1e-7 of a
    float64 reference.  ``sgmv_shrink_kernel`` keeps the old chunked order; only
    the decode path (this kernel) changed.
    """
    b = tl.program_id(0)
    # indices arrive as int64 (vllm's PunicaWrapper allocates torch.long, and
    # AscendC reads them as int64 too).  Load native and narrow in-register:
    # narrowing on the host was a device op per call and 2 per LoRA op, i.e.
    # 512 launches per decode step.
    idx = tl.load(indices + b).to(tl.int32)

    r = tl.arange(0, R)
    acc = tl.zeros([R], dtype=tl.float32)

    if idx >= 0:
        x_base = X + b * H
        a_base = lora_a + idx * (R * H)
        acc_w = tl.zeros([R], dtype=tl.float32)
        # H % BW == 0 is a constexpr test, so the mask-free body is the only one
        # compiled on the shapes that take it.
        if H % BW == 0:
            for h0 in range(0, H, BW):
                hs = h0 + tl.arange(0, BW)
                x = tl.load(x_base + hs).to(tl.float32)
                a = tl.load(a_base + r[:, None] * H + hs[None, :]).to(tl.float32)
                acc_w += tl.sum(a * x[None, :], axis=1)
        else:
            for h0 in range(0, H, BW):
                hs = h0 + tl.arange(0, BW)
                m = hs < H
                x = tl.load(x_base + hs, mask=m, other=0.0).to(tl.float32)
                a = tl.load(a_base + r[:, None] * H + hs[None, :],
                            mask=m[None, :], other=0.0).to(tl.float32)
                acc_w += tl.sum(a * x[None, :], axis=1)
        acc = acc_w * scale

    # AscendC: idx < 0 => skip row, y left unchanged
    old = tl.load(y + b * R + r)
    out = tl.where(idx >= 0, acc, old)
    tl.store(y + b * R + r, out)


@triton.jit
def bgmv_expand(
    y_ptr,
    lora_b_ptr,
    indices_ptr,
    y_in_ptr,
    y_out_ptr,
    R: tl.constexpr,
    Ho: tl.constexpr,        # weight slice dim (lora_b already narrowed by caller)
    L: tl.constexpr,
    BLOCK_HO: tl.constexpr,
    NBLK: tl.constexpr,      # ceil(Ho / BLOCK_HO); one program per (token, block)
    Y_HO: tl.constexpr,      # full output dim of y tensors
    SLICE_OFF: tl.constexpr, # column offset of this slice in y
):
    """Bgmv expand with the Ho axis spread over the grid.

    The Ho loop used to live *inside* one program per token, so grid was B and
    a decode step at B=1 ran the whole Ho=4096 axis on a single core: measured
    28.3us against AscendC's 9.9us.  Ho is embarrassingly parallel here -- every
    output column is an independent R-wide dot -- so giving each (token, Ho
    block) pair its own program cuts that to 11.9us and is a win at every batch
    size (B=8: 29.5 -> 17.2us, B=32: 36.8 -> 31.8us).  The R-axis tl.sum is
    untouched, so the result stays bit-identical to AscendC.

    Loading the weight block as one flat span and reshaping was tried and is
    slower at every B (B=4: 16.4 vs 13.9us): the block is already contiguous
    (offsets h0*R .. (h0+BLOCK_HO)*R), so the flat form buys no burst
    efficiency and only adds a layout conversion.
    """
    pid = tl.program_id(0)
    b = pid // NBLK
    blk = pid % NBLK

    idx = tl.load(indices_ptr + b).to(tl.int32)
    safe_idx = tl.minimum(tl.maximum(idx, 0), L - 1)

    r_offs = tl.arange(0, R)
    y_row = tl.load(y_ptr + b * R + r_offs)  # [R] fp32

    ho_offs = blk * BLOCK_HO + tl.arange(0, BLOCK_HO)
    ho_mask = ho_offs < Ho

    lora_offs = safe_idx * (Ho * R) + ho_offs[:, None] * R + r_offs[None, :]
    lora_vals = tl.load(lora_b_ptr + lora_offs, mask=ho_mask[:, None], other=0.0)

    prod = lora_vals.to(tl.float32) * y_row[None, :]
    acc = tl.sum(prod, axis=1)  # [BLOCK_HO] fp32
    acc = tl.where(idx >= 0, acc, 0.0)

    out_offs = b * Y_HO + SLICE_OFF + ho_offs
    y_in = tl.load(y_in_ptr + out_offs, mask=ho_mask, other=0.0)
    res = (y_in.to(tl.float32) + acc).to(y_in.dtype)

    tl.store(y_out_ptr + out_offs, res, mask=ho_mask)


# ---- sgmv ----

_BLOCK_H: tl.constexpr = 64


@triton.jit
def sgmv_shrink_kernel(
    X,            # [B, H]    fp16/bf16 activations, one row per token
    lora_a,       # [L, R, H] fp16/bf16 LoRA A weights (H contiguous)
    indices,      # [NR] int32, LoRA index per request (-1 => skip whole request)
    seq_start,    # [NR] int32, exclusive prefix sum of per-request token counts
    y,            # [B, R] fp32 output
    scale: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    L: tl.constexpr,
    NR: tl.constexpr,
):
    pid = tl.program_id(0)

    # ---- token row -> request: last j whose seq_start[j] <= pid ----
    # seq_start (== b_seq_start_loc) is the exclusive prefix sum of the per-request
    # token counts, so it is sorted and seq_start[0] == 0: every row matches at
    # least j == 0.  O(NR) per row instead of the O(NR^2) triangular prefix sum,
    # which is what let the former NR <= 16 admission cap (and its AscendC
    # fallback) go away.
    #
    # The reduce must yield the LAST matching j, and must do it by position, not
    # by value: lora indices are arbitrary and repeat, so reducing over `idxs`
    # (e.g. tl.max) would hand every row the largest adapter index in the batch
    # instead of its own.  Reduce over the positions to get `last`, then select
    # the index at that position.
    offs_nr = tl.arange(0, NR)
    starts = tl.load(seq_start + offs_nr).to(tl.int32)        # [NR], non-decreasing
    idxs = tl.load(indices + offs_nr).to(tl.int32)            # [NR]
    last = tl.max(tl.where(starts <= pid, offs_nr, -1), axis=0)
    idx = tl.sum(tl.where(offs_nr == last, idxs, 0), axis=0)  # scalar LoRA index (-1 => skip)

    keep = tl.where(idx >= 0, 1.0, 0.0)                       # scalar fp32 gate
    idx_safe = tl.maximum(idx, 0)                             # keep all loads in bounds

    offs_r = tl.arange(0, R)
    acc = tl.zeros([R], dtype=tl.float32)

    a_base = idx_safe * (R * H)
    x_base = pid * H

    # Same native-window reduction structure as bgmv_shrink (TILE_LENGTH=11776).
    for w in range(0, H // _WIN):
        acc_w = tl.zeros([R], dtype=tl.float32)
        for h0 in range(0, _WIN, _BLOCK_H):
            offs_h = w * _WIN + h0 + tl.arange(0, _BLOCK_H)
            x_blk = tl.load(X + x_base + offs_h).to(tl.float32)          # [BH]
            a_blk = tl.load(
                lora_a + a_base + offs_r[:, None] * H + offs_h[None, :],
            ).to(tl.float32)                                             # [R, BH]
            acc_w += tl.sum(a_blk * x_blk[None, :], axis=1)
        acc += acc_w
    tail = (H // _WIN) * _WIN
    acc_t = tl.zeros([R], dtype=tl.float32)
    for h0 in range(tail, H, _BLOCK_H):
        offs_h = h0 + tl.arange(0, _BLOCK_H)
        hmask = offs_h < H
        x_blk = tl.load(X + x_base + offs_h, mask=hmask, other=0.0).to(tl.float32)   # [BH]
        a_blk = tl.load(
            lora_a + a_base + offs_r[:, None] * H + offs_h[None, :],
            mask=hmask[None, :],
            other=0.0,
        ).to(tl.float32)                                                             # [R, BH]
        acc_t += tl.sum(a_blk * x_blk[None, :], axis=1)
    acc += acc_t

    # AscendC: idx < 0 => skip row, y left unchanged
    old = tl.load(y + pid * R + offs_r)
    out = acc * scale * keep + old * (1.0 - keep)
    tl.store(y + pid * R + offs_r, out)


@triton.jit(do_not_specialize=["B"])
def sgmv_shrink_dot(
    X,            # [B, H]    fp16/bf16 activations, one row per token
    lora_a,       # [L, R, H] fp16/bf16 LoRA A weights (H contiguous)
    indices,      # [NR] int64, LoRA index per request (-1 => skip whole request)
    seq_start,    # [NR] int64, exclusive prefix sum of per-request token counts
    y,            # [B, R] fp32 output
    B,            # runtime: number of token rows (kept runtime: it varies per
                  # step under chunked prefill, and specializing it would
                  # recompile on every distinct batch size)
    scale,        # runtime, NOT tl.constexpr -- see the note in the docstring
    H: tl.constexpr,
    R: tl.constexpr,
    L: tl.constexpr,
    NR: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NSPLIT: tl.constexpr,
):
    """shrink (``x @ A^T``) with ``tl.dot`` instead of a per-token ``tl.sum``.

    ``sgmv_shrink_kernel`` gives one program per token row and reduces along H
    with ``tl.sum``, which is why it measures 1.26-1.81x AscendC on the prefill:
    the reduction never touches the cube unit and each row re-reads its own
    ``[R, H]`` weight slice.  Here each program owns one (request, token-block)
    pair, keeps its ``[R, BLOCK_H]`` weight tile resident, and accumulates the
    whole H axis with a single ``tl.dot``.

    Anchoring each block on its request's start is what makes the dot legal:
    a block never crosses a request boundary, so its adapter is unique.  The
    number of blocks a request may need is ``ceil(B / BLOCK_B)`` -- an upper
    bound, since no request can exceed the batch -- so the grid is
    ``NR * NSPLIT`` and requests shorter than that simply leave programs idle.

    Three bishengir (CANN 8.x) limitations shaped this kernel; each was
    bisected to a minimal repro:

      * A ``tl.dot`` inside a **dynamic** ``range`` loop fails to compile
        (``[ConvertLinalgRToBinary] ... Unknown core type: llvm.func @malloc``).
        Hence the H loop is a static ``range`` and the token axis is handled by
        the grid rather than by a loop.
      * Two ``tl.sum`` reductions whose scalars both feed address arithmetic
        compile but fault at runtime with 507057 (SUSPECT REMOTE ERROR).  That
        is why ``indices``/``seq_start`` are fetched with a direct scalar load
        indexed by the program's own request, rather than by the O(NR)
        "find my request" reduction the per-token kernel needs.
      * ``scale`` must stay a **runtime** argument.  Declaring it
        ``tl.constexpr`` -- the obvious thing, and what ``sgmv_shrink_kernel``
        does -- makes the kernel silently compute every request but the first
        with the wrong adapter: measured 3.5e-01 max error vs 1.3e-07, on rows
        belonging entirely to request 1.  Body-for-body identical otherwise.

    The floating-point contract also differs from ``sgmv_shrink_kernel`` on
    purpose: that one reproduces AscendC's 64-wide ``tl.sum`` reduction order
    bit-for-bit, whereas ``tl.dot`` with ``input_precision="ieee"`` accumulates
    in a cube-shaped order.  Both land within 1.3e-07 of a float64 reference on
    bf16 inputs, i.e. the reordering costs nothing measurable, but the two are
    not bit-identical to each other.
    """
    pid = tl.program_id(0)
    i = pid // NSPLIT     # request
    s = pid % NSPLIT      # which token block of that request

    # Each program claims exactly one request, so the request number is a
    # scalar and its adapter / token range are direct indexed loads.
    t_lo = tl.load(seq_start + i).to(tl.int32)
    # vLLM's b_seq_start_loc carries a trailing total, so the last slot has
    # t_lo == B and owns no rows; the mask keeps that slot's index load in
    # bounds instead of reading one past the end of `indices`.
    idx = tl.load(indices + i, mask=t_lo < B, other=-1).to(tl.int32)
    nxt = tl.load(seq_start + tl.minimum(i + 1, NR - 1),
                  mask=(i + 1) < NR, other=0).to(tl.int32)
    t_hi = tl.where(i == NR - 1, B, nxt)

    t0 = t_lo + s * BLOCK_B
    # idx < 0 => skip the whole request; blocks past the request's end own no
    # rows.  Either way y keeps its previous value, matching AscendC.
    if (idx < 0) | (t0 >= t_hi):
        return

    offs_r = tl.arange(0, R)
    offs_t = t0 + tl.arange(0, BLOCK_B)
    tmask = offs_t < t_hi
    a_base = idx * (R * H)

    acc = tl.zeros([BLOCK_B, R], dtype=tl.float32)
    for h0 in range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        hmask = offs_h < H
        x_blk = tl.load(X + offs_t[:, None] * H + offs_h[None, :],
                        mask=tmask[:, None] & hmask[None, :], other=0.0)
        a_blk = tl.load(lora_a + a_base + offs_r[:, None] * H + offs_h[None, :],
                        mask=hmask[None, :], other=0.0)
        acc += tl.dot(x_blk, tl.trans(a_blk), input_precision="ieee")
    tl.store(y + offs_t[:, None] * R + offs_r[None, :], acc * scale,
             mask=tmask[:, None])


@triton.jit(do_not_specialize=["B"])
def sgmv_expand_dot(
    X_ptr,            # fp32 [B, R]
    lora_b_ptr,       # fp16/bf16 [L, Ho, R]  (Ho = slice dim, already narrowed by caller)
    indices_ptr,      # int64 [NR] (native; narrowed in-register)
    seq_start_ptr,    # int64 [NR], exclusive prefix sum of per-request token counts
    y_in_ptr,         # fp16/bf16 [B, Y_HO]
    y_out_ptr,        # fp16/bf16 [B, Y_HO]
    B,                # runtime: number of token rows (keep it a runtime value:
                      # B == 1 would otherwise be specialized into a constant
                      # and dropped from the arg buffer -- see _cpp_make_case)
    R: tl.constexpr,
    Ho: tl.constexpr,
    L: tl.constexpr,
    NR: tl.constexpr,
    BLOCK_HO: tl.constexpr,
    BLOCK_T: tl.constexpr,
    Y_HO: tl.constexpr,
    SLICE_OFF: tl.constexpr,
    NBLK: tl.constexpr,      # ceil(Ho / BLOCK_HO)
    NSPLIT: tl.constexpr,    # token splits per (request, ho-block)
):
    """expand = x @ B^T, tiled so the LoRA weight stays resident.

    The obvious mapping -- one program per token, loop over Ho -- re-reads the
    whole [Ho, R] weight for every token (B x Ho x R x 2 bytes of traffic for
    a weight that is L x Ho x R x 2).  Measured on 910B4 that is 4.4-7x slower
    than AscendC for a 4k prefill, because AscendC serves those re-reads out
    of cache and the per-token kernel does not.

    Instead each program claims one (request, ho-block, token-split) triple,
    keeps its [R, BLOCK_HO] weight tile resident, and walks its tokens in
    BLOCK_T chunks with a single tl.dot.  Grid is NR*NBLK*NSPLIT; that count
    matters -- the same kernel at grid 32768 measures 7x SLOWER than at 256
    (per-program setup dominates), so the wrapper picks NSPLIT to land the
    grid in the measured sweet spot rather than to maximise parallelism.
    """
    pid = tl.program_id(0)
    req = pid // (NBLK * NSPLIT)
    rem = pid % (NBLK * NSPLIT)
    blk = rem // NSPLIT
    sp = rem % NSPLIT

    lora_id = tl.load(indices_ptr + req).to(tl.int32)
    t_start = tl.load(seq_start_ptr + req).to(tl.int32)
    # seq_start is the exclusive prefix sum, so request `req` covers
    # [seq_start[req], seq_start[req+1]) and the last one runs to B.
    if req + 1 < NR:
        t_end = tl.load(seq_start_ptr + req + 1).to(tl.int32)
    else:
        t_end = B

    r_idx = tl.arange(0, R)
    ho = blk * BLOCK_HO + tl.arange(0, BLOCK_HO)
    ho_mask = ho < Ho

    # Safe index keeps every load in bounds; `gate` is what makes a -1 row
    # contribute nothing (AscendC leaves y untouched for those).
    base = tl.maximum(lora_id, 0) * (Ho * R)
    wT = tl.load(lora_b_ptr + base + ho[None, :] * R + r_idx[:, None],
                 mask=ho_mask[None, :], other=0.0).to(tl.float32)
    gate = tl.where(lora_id >= 0, 1.0, 0.0)

    per = (t_end - t_start + NSPLIT - 1) // NSPLIT
    my_start = t_start + sp * per
    my_end = tl.minimum(my_start + per, t_end)

    for t0 in range(my_start, my_end, BLOCK_T):
        ts = t0 + tl.arange(0, BLOCK_T)
        t_mask = ts < my_end
        x = tl.load(X_ptr + ts[:, None] * R + r_idx[None, :],
                    mask=t_mask[:, None], other=0.0)          # [BLOCK_T, R] fp32
        acc = tl.dot(x, wT, input_precision="ieee")           # [BLOCK_T, BLOCK_HO]
        offs = ts[:, None] * Y_HO + SLICE_OFF + ho[None, :]
        mask = t_mask[:, None] & ho_mask[None, :]
        yi = tl.load(y_in_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(y_out_ptr + offs,
                 (yi + acc * gate).to(y_out_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["B"])
def sgmv_expand_flat(
    X_ptr,            # fp32 [B, R]
    lora_b_ptr,       # fp16/bf16 [L, Ho, R]  (Ho = slice dim, already narrowed by caller)
    indices_ptr,      # int64 [NR] (native; narrowed in-register)
    seq_start_ptr,    # int64 [NR], exclusive prefix sum of per-request token counts
    y_in_ptr,         # fp16/bf16 [B, Y_HO]
    y_out_ptr,        # fp16/bf16 [B, Y_HO]
    B,                # runtime: number of token rows (keep it a runtime value:
                      # B == 1 would otherwise be specialized into a constant
                      # and dropped from the arg buffer -- see _cpp_make_case)
    R: tl.constexpr,
    Ho: tl.constexpr,
    NR: tl.constexpr,
    BLOCK_HO: tl.constexpr,
    Y_HO: tl.constexpr,
    SLICE_OFF: tl.constexpr,
):
    """expand for the decode-shaped case: few tokens, one program per (request, ho-block).

    ``sgmv_expand_dot`` wins the prefill but loses the decode: ``tl.dot`` needs a
    16-row tile, so at B <= 32 the kernel spends most of its work on padding --
    measured 2.4-7x AscendC at B = 32 with BLOCK_T fixed at 64.  Dropping
    ``tl.dot`` entirely and reducing with ``tl.sum`` lets BLOCK_T be 1, which is
    what decode actually has.

    The other half of the decode win is how the weight is fetched.  Indexing it
    as ``ho[:, None] * R + r[None, :]`` issues 2*R-byte runs for what is a
    contiguous [BLOCK_HO, R] block, and 32-byte bursts leave the memory system
    at ~145 GB/s.  Loading the block as one flat ``tl.arange(0, BLOCK_HO * R)``
    span and reshaping afterwards reaches ~215 GB/s, and is bit-identical to
    AscendC rather than merely close -- the R-axis ``tl.sum`` here reproduces
    AscendC's own fp32 reduction order.

    No ``tl.dot`` also means mix_mode stays "aiv" and the kernel declares no
    device workspace, so it remains eligible for the minimal C++ launcher
    (see ``_cpp_make_case``); the dot kernel is not.
    """
    pid = tl.program_id(0)
    nblk = (Ho + BLOCK_HO - 1) // BLOCK_HO
    req = pid // nblk
    blk = pid % nblk

    lora_id = tl.load(indices_ptr + req).to(tl.int32)
    t_start = tl.load(seq_start_ptr + req).to(tl.int32)
    # seq_start is the exclusive prefix sum, so request `req` covers
    # [seq_start[req], seq_start[req+1]) and the last one runs to B.
    if req + 1 < NR:
        t_end = tl.load(seq_start_ptr + req + 1).to(tl.int32)
    else:
        t_end = B

    h0 = blk * BLOCK_HO
    e = tl.arange(0, BLOCK_HO * R)
    # Safe index keeps every load in bounds; `gate` is what makes a -1 row
    # contribute nothing (AscendC leaves y untouched for those).
    w = tl.reshape(
        tl.load(lora_b_ptr + tl.maximum(lora_id, 0) * (Ho * R) + h0 * R + e,
                mask=(h0 + e // R) < Ho, other=0.0),
        (BLOCK_HO, R)).to(tl.float32)
    gate = tl.where(lora_id >= 0, 1.0, 0.0)

    r_idx = tl.arange(0, R)
    ho = h0 + tl.arange(0, BLOCK_HO)
    ho_mask = ho < Ho

    for t in range(t_start, t_end):
        x = tl.load(X_ptr + t * R + r_idx).to(tl.float32)      # [R]
        acc = tl.sum(w * x[None, :], axis=1)                   # [BLOCK_HO]
        offs = t * Y_HO + SLICE_OFF + ho
        yi = tl.load(y_in_ptr + offs, mask=ho_mask, other=0.0).to(tl.float32)
        tl.store(y_out_ptr + offs,
                 (yi + acc * gate).to(y_out_ptr.dtype.element_ty), mask=ho_mask)


@triton.jit
def sgmv_expand(
    X_ptr,            # fp32 [B, R]
    lora_b_ptr,       # fp16/bf16 [L, Ho, R]  (Ho = slice dim, already narrowed by caller)
    indices_ptr,      # int64 [NR] (native; narrowed in-register)
    seq_start_ptr,    # int64 [NR], exclusive prefix sum of per-request token counts
    y_in_ptr,         # fp16/bf16 [B, Y_HO]
    y_out_ptr,        # fp16/bf16 [B, Y_HO]
    R: tl.constexpr,
    Ho: tl.constexpr,
    L: tl.constexpr,
    NR: tl.constexpr,
    BLOCK_HO: tl.constexpr,
    Y_HO: tl.constexpr,
    SLICE_OFF: tl.constexpr,
):
    pid = tl.program_id(0)

    # ---- map token row -> request: last j whose seq_start[j] <= pid ----
    # Same O(NR) positional lookup as sgmv_shrink_kernel: seq_start is sorted
    # with seq_start[0] == 0, so at least j == 0 matches every row.
    r_off = tl.arange(0, NR)
    starts = tl.load(seq_start_ptr + r_off).to(tl.int32)       # [NR], non-decreasing
    idxs = tl.load(indices_ptr + r_off).to(tl.int32)           # [NR]
    last = tl.max(tl.where(starts <= pid, r_off, -1), axis=0)
    lora_id = tl.sum(tl.where(r_off == last, idxs, 0), axis=0)
    safe_id = tl.maximum(lora_id, 0)
    skip_scale = tl.where(lora_id >= 0, 1.0, 0.0)

    # ---- fp32 shrink-output row ----
    r_idx = tl.arange(0, R)
    x = tl.load(X_ptr + pid * R + r_idx)                       # fp32 [R]

    base = safe_id * (Ho * R)
    for h0 in range(0, Ho, BLOCK_HO):
        ho = h0 + tl.arange(0, BLOCK_HO)
        ho_mask = ho < Ho
        w = tl.load(lora_b_ptr + base + ho[:, None] * R + r_idx[None, :],
                    mask=ho_mask[:, None], other=0.0)
        acc = tl.sum(w.to(tl.float32) * x[None, :], axis=1)    # fp32 [BLOCK_HO]
        out_offs = pid * Y_HO + SLICE_OFF + ho
        yi = tl.load(y_in_ptr + out_offs, mask=ho_mask, other=0.0).to(tl.float32)
        out = yi + acc * skip_scale
        tl.store(y_out_ptr + out_offs, out.to(y_out_ptr.dtype.element_ty),
                 mask=ho_mask)
