#!/usr/bin/env python3
"""Generate 3D binary dispatch tables for AMR ghost fill.

Fully symmetric: all 26 directions handled by the same code path.
No axis-specific branches. Each ghost cell computed identically.

Table index: [cx+1][cy+1][cz+1][xp%2][yp%2][zp%2][status]
  3*3*3 * 2*2*2 * 16 = 3456 entries, each ENTRY_SIZE bytes.
"""

import struct

BS = 8
MAX_PRE = 400
MAX_POST = 64
MAX_OPS = MAX_PRE + MAX_POST
ENTRY_SIZE = 1 + 4 * 12 + 3 + 4 + 4 + MAX_OPS * 20

OP_COPY = 0
OP_AVG = 1
OP_INTERP27 = 2
OP_BC_SCALAR = 3
OP_BC_VECTOR = 4
OP_BC_FIXED = 5

ZERO_OP = b'\x00' * 20


def ghost_bounds(c, ss):
    if c < 0: return (-ss, 0)
    if c > 0: return (BS, BS + ss)
    return (0, BS)


def make_src(**kw):
    d = dict(level_delta=0, xi_mul=0, yi_mul=0, zi_mul=0,
             xi_add=0, yi_add=0, zi_add=0,
             xi_shift=0, yi_shift=0, zi_shift=0,
             is_self=0, self_idx=0)
    d.update(kw)
    return d


def pack_src(s):
    return struct.pack('<bbbbbbbbbbbb',
        s['level_delta'], s['xi_mul'], s['yi_mul'], s['zi_mul'],
        s['xi_add'], s['yi_add'], s['zi_add'],
        s['xi_shift'], s['yi_shift'], s['zi_shift'],
        s['is_self'], s['self_idx'])


def pack_op(op):
    return struct.pack('<bbbb iiii', *op)


def fine_offset(x, y, z, ss, nm, dim):
    """Offset into fine buffer m."""
    return dim * ((z + ss) * nm * nm + (y + ss) * nm + (x + ss))


def blk_offset(x, y, z, dim):
    """Offset into a block's data array."""
    return dim * ((z * BS + y) * BS + x)


def build_entry(cx, cy, cz, xp, yp, zp, s, ss, dim):
    nm = 2 * ss + BS
    dirs = [cx, cy, cz]
    parity = [xp, yp, zp]
    gs = [ghost_bounds(c, ss) for c in dirs]

    blk_src = []
    pre_ops = []
    post_ops = []

    # ==================================================================
    # s=0: same-level neighbor — copy row by row (symmetric across axes)
    # ==================================================================
    if s == 0:
        blk_src = [make_src(level_delta=0,
                   xi_mul=1, yi_mul=1, zi_mul=1,
                   xi_add=cx, yi_add=cy, zi_add=cz)]
        for iz in range(gs[2][0], gs[2][1]):
            for iy in range(gs[1][0], gs[1][1]):
                sx0 = gs[0][0] - cx * BS
                cols = gs[0][1] - gs[0][0]
                pre_ops.append((OP_COPY, 0, 0, 0,
                    blk_offset(sx0, iy - cy*BS, iz - cz*BS, dim),
                    fine_offset(gs[0][0], iy, iz, ss, nm, dim),
                    cols, 0))

    # ==================================================================
    # s=1: finer neighbor — per-cell OP_AVG, fully symmetric
    # ==================================================================
    elif s == 1:
        zero_axes = [d for d in range(3) if dirs[d] == 0]
        nch = 1 << len(zero_axes)
        nch = min(nch, 4)
        for b in range(nch):
            off = [0, 0, 0]
            for d in range(3):
                off[d] = 0 if dirs[d] == 0 else dirs[d] * 2
            for i, ax in enumerate(zero_axes):
                off[ax] = (b >> i) & 1
            blk_src.append(make_src(level_delta=1,
                xi_mul=2, yi_mul=2, zi_mul=2,
                xi_add=off[0], yi_add=off[1], zi_add=off[2]))

        # For each ghost cell, compute child index + local fine coords
        for iz in range(gs[2][0], gs[2][1]):
            for iy in range(gs[1][0], gs[1][1]):
                for ix in range(gs[0][0], gs[0][1]):
                    g = [ix, iy, iz]
                    # Per-axis: determine child bit and local fine coordinate
                    child_bits = {}
                    local_f = [0, 0, 0]
                    for d in range(3):
                        if dirs[d] == 0:
                            # Zero axis: children split here
                            fg = 2 * g[d]
                            child_bits[d] = 1 if fg >= BS else 0
                            local_f[d] = fg - child_bits[d] * BS
                        else:
                            # Nonzero axis: edge of child block
                            child_bits[d] = 0  # not used for child index
                            local_f[d] = BS - 2 * ss if dirs[d] < 0 else 0
                            # Offset within ghost: multiple cells for ss>1
                            local_f[d] += 2 * (g[d] - gs[d][0])

                    # Child index from bits at zero axes
                    b = 0
                    for i, ax in enumerate(zero_axes):
                        b |= child_bits[ax] << i

                    pre_ops.append((OP_AVG, b, 0, 1,
                        blk_offset(local_f[0], local_f[1], local_f[2], dim),
                        fine_offset(ix, iy, iz, ss, nm, dim),
                        BS, BS * BS))

    # ==================================================================
    # s=2: coarser neighbor — trilinear interpolation, fully symmetric
    # ==================================================================
    elif s == 2:
        blk_src = [make_src(level_delta=-1,
                   xi_add=cx, yi_add=cy, zi_add=cz,
                   xi_shift=1, yi_shift=1, zi_shift=1)]

        def coarse_local(d, g):
            return (parity[d] * BS + g) // 2 - ((parity[d] + dirs[d]) // 2) * BS

        for iz in range(gs[2][0], gs[2][1]):
            for iy in range(gs[1][0], gs[1][1]):
                for ix in range(gs[0][0], gs[0][1]):
                    g = [ix, iy, iz]
                    sc = [coarse_local(d, g[d]) for d in range(3)]
                    pc = [(parity[d] * BS + g[d]) % 2 for d in range(3)]
                    # Base of 2x2x2: shift left for parity 0
                    base = [sc[d] - (1 - pc[d]) for d in range(3)]
                    base = [max(0, min(BS - 2, b)) for b in base]
                    pf = (pc[0] & 1) | ((pc[1] & 1) << 1) | ((pc[2] & 1) << 2)
                    pre_ops.append((OP_INTERP27, 0, 0, pf,
                        blk_offset(base[0], base[1], base[2], dim),
                        fine_offset(ix, iy, iz, ss, nm, dim),
                        BS, BS * BS))

    # ==================================================================
    # s=3..9: wall/symmetry BC — reflect with negate mask
    # ==================================================================
    if 3 <= s <= 9:
        blk_src = blk_src or [make_src(is_self=1, self_idx=0)]
        bc_axes = []
        if s in (3, 6, 7, 9): bc_axes.append(0)
        if s in (4, 6, 8, 9): bc_axes.append(1)
        if s in (5, 7, 8, 9): bc_axes.append(2)
        negate_mask = sum(1 << ax for ax in bc_axes)
        sides = [1 if c > 0 else 0 for c in dirs]

        for iz in range(gs[2][0], gs[2][1]):
            for iy in range(gs[1][0], gs[1][1]):
                for ix in range(gs[0][0], gs[0][1]):
                    g = [ix, iy, iz]
                    mirror = list(g)
                    for ax in bc_axes:
                        mirror[ax] = -g[ax] - 1 if sides[ax] == 0 else 2*BS - 1 - g[ax]
                    src = fine_offset(mirror[0], mirror[1], mirror[2], ss, nm, dim)
                    dst = fine_offset(ix, iy, iz, ss, nm, dim)
                    if dim == 1:
                        post_ops.append((OP_BC_SCALAR, 0, 0, 0, src, dst, 0, 0))
                    else:
                        post_ops.append((OP_BC_VECTOR, 0, 0, negate_mask, src, dst, 0, 0))

    # ==================================================================
    # s=10..12: outflow — zero-gradient (copy nearest interior cell)
    # ==================================================================
    if 10 <= s <= 12:
        blk_src = blk_src or [make_src(is_self=1, self_idx=0)]
        out_axis = s - 10
        side = 1 if dirs[out_axis] > 0 else 0
        for iz in range(gs[2][0], gs[2][1]):
            for iy in range(gs[1][0], gs[1][1]):
                for ix in range(gs[0][0], gs[0][1]):
                    g = [ix, iy, iz]
                    interior = list(g)
                    interior[out_axis] = 0 if side == 0 else BS - 1
                    src = fine_offset(interior[0], interior[1], interior[2], ss, nm, dim)
                    dst = fine_offset(ix, iy, iz, ss, nm, dim)
                    pre_ops.append((OP_BC_SCALAR, 0, 0, 0, src, dst, 0, 0))

    # ==================================================================
    # s=13..15: inflow — fill from constant buffer (dst[2] = bc_const)
    # ==================================================================
    if 13 <= s <= 15:
        blk_src = blk_src or [make_src(is_self=1, self_idx=2)]
        for iz in range(gs[2][0], gs[2][1]):
            for iy in range(gs[1][0], gs[1][1]):
                for ix in range(gs[0][0], gs[0][1]):
                    dst = fine_offset(ix, iy, iz, ss, nm, dim)
                    pre_ops.append((OP_BC_FIXED, 0, 0, 0, 0, dst, 0, 0))

    return {'blk_src': blk_src, 'pre_ops': pre_ops, 'post_ops': post_ops}


def pack_entry(e):
    buf = bytearray()
    n_blk = len(e['blk_src'])
    buf += struct.pack('<b', n_blk)
    for b in range(4):
        buf += pack_src(e['blk_src'][b]) if b < n_blk else b'\x00' * 12
    buf += b'\x00' * 3  # pad
    pre = list(e['pre_ops'])
    post = list(e['post_ops'])
    if len(post) > MAX_POST:
        pre = pre + post[MAX_POST:]
        post = post[:MAX_POST]
    n_pre, n_post = len(pre), len(post)
    assert n_pre <= MAX_PRE, f"n_pre={n_pre} > {MAX_PRE}"
    assert n_post <= MAX_POST, f"n_post={n_post} > {MAX_POST}"
    buf += struct.pack('<ii', n_pre, n_post)
    for i in range(MAX_PRE):
        buf += pack_op(pre[i]) if i < n_pre else ZERO_OP
    for i in range(MAX_POST):
        buf += pack_op(post[i]) if i < n_post else ZERO_OP
    assert len(buf) == ENTRY_SIZE
    return bytes(buf)


ZERO_ENTRY = b'\x00' * ENTRY_SIZE


def build_and_write(fname, ss, dim):
    max_pre = max_post = 0
    with open(fname, 'wb') as f:
        for cx in range(-1, 2):
            for cy in range(-1, 2):
                for cz in range(-1, 2):
                    for xp in range(2):
                        for yp in range(2):
                            for zp in range(2):
                                for s in range(16):
                                    if cx == 0 and cy == 0 and cz == 0:
                                        f.write(ZERO_ENTRY)
                                        continue
                                    e = build_entry(cx, cy, cz, xp, yp, zp, s, ss, dim)
                                    np_ = len(e['pre_ops']) + max(0, len(e['post_ops']) - MAX_POST)
                                    nq_ = min(len(e['post_ops']), MAX_POST)
                                    max_pre = max(max_pre, np_)
                                    max_post = max(max_post, nq_)
                                    f.write(pack_entry(e))
    sz = 3*3*3*2*2*2*16 * ENTRY_SIZE
    print(f'{fname}: {sz} bytes ({sz//1024} KB), max_pre={max_pre}, max_post={max_post}')


def main():
    for ss, dim in [(1, 1), (1, 3), (4, 1)]:
        build_and_write(f'tab3d_ss{ss}_dim{dim}.bin', ss, dim)


if __name__ == '__main__':
    main()
