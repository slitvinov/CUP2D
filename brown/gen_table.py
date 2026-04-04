#!/usr/bin/env python3
"""
Generate 2D binary dispatch tables for AMR ghost fill/interpolation/BC.

Status codes (N_STATUS=10):
  0: same-level neighbor
  1: finer neighbor (children)
  2: coarser neighbor
  3: x-wall/symmetry   4: y-wall/symmetry   5: xy-corner
  6: x-outflow          7: y-outflow
  8: x-inflow           9: y-inflow

Table layout: [cx+1][cy+1][xparity][yparity][status] -> entry
  3 * 3 * 2 * 2 * 10 = 360 entries per (ss, dim) config.
"""

import struct

BS = 8
MAX_PRE = 32
MAX_POST = 48
MAX_OPS = MAX_PRE + MAX_POST

OP_COPY = 0
OP_AVG = 1
OP_INTERP9 = 2
OP_INTERP3 = 3
OP_LELI = 4
OP_BC_SCALAR = 5
OP_BC_VECTOR = 6
OP_BC_CORNER = 7
OP_BC_FIXED = 8

N_STATUS = 10

# Child neighbor patterns: (cx, cy, row_stride, col_stride, offsets, n_blk)
CHILD_NB = [
    (-1, -1, 3, 1, [(-1, -1)], 1),
    ( 0, -1, 1, 1, [( 0, -1), (1, -1)], 2),
    ( 1, -1, 3, 1, [( 2, -1)], 1),
    (-1,  0, 1, 2, [(-1,  0), (-1, 1)], 2),
    ( 1,  0, 1, 2, [( 2,  0), (2, 1)], 2),
    (-1,  1, 3, 1, [(-1,  2)], 1),
    ( 0,  1, 1, 1, [( 0,  2), (1, 2)], 2),
    ( 1,  1, 3, 1, [( 2,  2)], 1),
]

LI_LE_TAB = [
    (0, +1, [(0, 0, -1, 0, -2), (1, 0, -2, 0, -3)]),
    (0, -1, [(1, 0, +2, 0, +3), (0, 0, +1, 0, +2)]),
    (+1, 0, [(0, -1, 0, -2, 0), (1, -2, 0, -3, 0)]),
    (-1, 0, [(1, +2, 0, +3, 0), (0, +1, 0, +2, 0)]),
]

# 1D Lagrange interpolation weights (×32) from 3 coarse cells at positions pts,
# evaluated at fine subcell offset t = ±1/4.
# branch: 0 = left edge (pts 0,1,2), 1 = right edge (-2,-1,0), 2 = interior (-1,0,1)
_PTS = {0: (0, 1, 2), 1: (-2, -1, 0), 2: (-1, 0, 1)}
def _lagrange3(pts, t):
    """3-point Lagrange weights at t, scaled by 32."""
    p = list(pts)
    return tuple(int(round(32 * (t-p[j])*(t-p[k]) / ((p[i]-p[j])*(p[i]-p[k]))))
                 for i, j, k in [(0,1,2),(1,0,2),(2,0,1)])
FACE_W = {(b, sub): _lagrange3(_PTS[b], 0.25 * (2*sub - 1))
           for b in range(3) for sub in range(2)}  # sub: 0→t=-1/4, 1→t=+1/4
# FACE_DSIGN[dirn][k]: sign pattern for 4 subcells in 2×2 block.
# k = (tangent_sub, normal_sub) ordering depends on dirn.
# dirn=0 (x-face): subcells (ix_sub, iy_sub) → k = iy_sub*2 + ix_sub, sign = (-1)^iy_sub
# dirn=1 (y-face): subcells (ix_sub, iy_sub) → k = iy_sub*2 + ix_sub, sign = (-1)^ix_sub
# k indexes 4 subcells: k=0:(0,0), k=1:(1,0), k=2:(0,1), k=3:(1,1)
# dirn=0 (x-face): tangent=y, sign alternates with k&1 (ix_sub)
# dirn=1 (y-face): tangent=x, sign alternates with k>>1 (iy_sub)
FACE_DSIGN = [[1, -1, 1, -1], [1, 1, -1, -1]]


def get_child_pattern(cx, cy):
    for e in CHILD_NB:
        if e[0] == cx and e[1] == cy:
            return e
    return None


def ghost_bounds(c, ss):
    s = -ss if c < 0 else (0 if c == 0 else BS)
    e = 0 if c < 0 else (BS if c == 0 else BS + ss)
    return s, e


def coarse_bounds(c, ss, coff):
    nc = BS // 2 + ss + 3
    s = coff if c < 0 else (0 if c == 0 else BS // 2)
    e = 0 if c < 0 else (BS // 2 if c == 0 else min(BS // 2 + (ss + 1) // 2 + 1, coff + nc))
    return s, e


def make_blksrc(level_delta=0, xi_mul=0, yi_mul=0, xi_add=0, yi_add=0,
                xi_shift=0, yi_shift=0, is_self=0, self_idx=0):
    return dict(level_delta=level_delta, xi_mul=xi_mul, yi_mul=yi_mul,
                xi_add=xi_add, yi_add=yi_add, xi_shift=xi_shift,
                yi_shift=yi_shift, is_self=is_self, self_idx=self_idx)


def build_entry(cx, cy, xp, yp, s, ss, dim):
    coff = (-ss - 1) // 2 - 1
    nm = 2 * ss + BS
    nc = BS // 2 + ss + 3
    eC = (ss + 1) // 2 + 2

    fs0, fe0 = ghost_bounds(cx, ss)
    fs1, fe1 = ghost_bounds(cy, ss)
    cs0, ce0 = coarse_bounds(cx, ss, coff)
    cs1, ce1 = coarse_bounds(cy, ss, coff)

    cstart = [0, 0]
    for d in range(2):
        cc = cx if d == 0 else cy
        coord = xp if d == 0 else yp
        base_d = (coord + cc + 2) % 2
        ce_d = 1 if (cc != 0) and (coord == (1 if cc < 0 else 0)) else 0
        cstart[d] = (max(cc, 0) * BS // 2 +
                     (1 - abs(cc)) * base_d * BS // 2 - cc * BS +
                     ce_d * cc * BS // 2)

    sC0 = (-ss - 1) // 2 if cx < 0 else (0 if cx == 0 else BS // 2)
    sC1 = (-ss - 1) // 2 if cy < 0 else (0 if cy == 0 else BS // 2)
    is_face = abs(cx) + abs(cy) == 1

    e = dict(cx=cx, cy=cy, n_blk=0, blk_src=[],
             fill=[], cbc=[], interp_ops=[], fbc=[])

    # ---- Outflow (s=6,7): zero-gradient ----
    if s in (6, 7):
        axis = s - 6
        side = 1 if (cx if axis == 0 else cy) > 0 else 0
        e['n_blk'] = 1
        e['blk_src'] = [make_blksrc(is_self=1, self_idx=0)]
        for iy in range(fs1, fe1):
            for ix in range(fs0, fe0):
                interior = [ix, iy]
                interior[axis] = 0 if side == 0 else BS - 1
                src = dim * ((interior[0] + ss) + nm * (interior[1] + ss))
                dst = dim * ((ix + ss) + nm * (iy + ss))
                e['fill'].append((OP_BC_SCALAR, 0, 0, 0, src, dst, 0, 0))
        return e

    # ---- Inflow (s=8,9): constant value ----
    if s in (8, 9):
        e['n_blk'] = 1
        e['blk_src'] = [make_blksrc(is_self=1, self_idx=2)]
        for iy in range(fs1, fe1):
            for ix in range(fs0, fe0):
                dst = dim * ((ix + ss) + nm * (iy + ss))
                e['fill'].append((OP_BC_FIXED, 0, 0, 0, 0, dst, 0, 0))
        return e

    # ---- Block sources ----
    if s == 0:
        e['n_blk'] = 1
        e['blk_src'] = [make_blksrc(xi_mul=1, yi_mul=1, xi_add=cx, yi_add=cy)]
    elif s == 1:
        pat = get_child_pattern(cx, cy)
        e['n_blk'] = pat[5]
        e['blk_src'] = [make_blksrc(level_delta=1, xi_mul=2, yi_mul=2,
                                     xi_add=pat[4][b][0], yi_add=pat[4][b][1])
                        for b in range(pat[5])]
    elif s == 2:
        e['n_blk'] = 2
        e['blk_src'] = [
            make_blksrc(level_delta=-1, xi_mul=1, yi_mul=1,
                        xi_add=cx, yi_add=cy, xi_shift=1, yi_shift=1),
            make_blksrc(is_self=1, self_idx=0),
        ]

    # ---- Fill ops ----
    if s == 0:
        for iy in range(fs1, fe1):
            src_off = dim * (BS * (iy - cy * BS) + (fs0 - cx * BS))
            dst_off = dim * ((fs0 + ss) + (iy + ss) * nm)
            cols = fe0 - fs0
            e['fill'].append((OP_COPY, 0, 0, 0, src_off, dst_off, cols, 0))
        if dim > 0:
            lcs0, lcs1 = cs0, cs1
            lce0 = ce0 if cx < 1 else min(BS // 2 + eC - 1, coff + nc)
            lce1 = ce1 if cy < 1 else min(BS // 2 + eC - 1, coff + nc)
            cols = lce0 - lcs0
            if cols > 0:
                s0 = lcs0 + max(cx, 0) * (BS // 2) - cx * BS + min(0, cx) * cols
                s1_v = lcs1 + max(cy, 0) * (BS // 2) - cy * BS + min(0, cy) * (lce1 - lcs1)
                di = lcs0 - coff
                for iy in range(lcs1, lce1):
                    y0 = 2 * (iy - lcs1) + s1_v
                    src_off = dim * (BS * y0 + s0)
                    dst_off = dim * (di + (iy - coff) * nc)
                    e['fill'].append((OP_AVG, 0, 1, 0, src_off, dst_off, cols, BS))

    elif s == 1:
        pat = get_child_pattern(cx, cy)
        width = abs(cx) * (fe0 - fs0) + (1 - abs(cx)) * ((fe0 - fs0) // 2)
        B = 0
        for cnt in range(pat[5]):
            aux = (B % 2) if abs(cx) == 1 else (B // 2)
            di = abs(cx) * (fs0 + ss) + (1 - abs(cx)) * (fs0 + ss + (B % 2) * (fe0 - fs0) // 2)
            sx = fs0 - cx * BS + min(0, cx) * (fe0 - fs0)
            iy2 = fs1
            while iy2 < fe1:
                sy = 2 * (iy2 - cy * BS) + min(0, cy) * BS if abs(cy) == 1 else iy2
                dk = di + (abs(cy) * (iy2 + ss) +
                           (1 - abs(cy)) * (iy2 // 2 + ss + aux * (fe1 - fs1) // 2)) * nm
                src_off = dim * (BS * sy + sx)
                dst_off = dim * dk
                e['fill'].append((OP_AVG, cnt, 0, 0, src_off, dst_off, width, BS))
                iy2 += pat[3]
            B += pat[2]

    elif s == 2:
        for iy in range(cs1, ce1):
            src_off = dim * (BS * (iy + cstart[1]) + cs0 + cstart[0])
            dst_off = dim * (cs0 - coff + (iy - coff) * nc)
            cols = ce0 - cs0
            e['fill'].append((OP_COPY, 0, 1, 0, src_off, dst_off, cols, 0))
        for j in range(BS // 2):
            for i in range(BS // 2):
                if 1 < i < BS // 2 - 2 and 2 < j < BS // 2 - 2:
                    continue
                src_off = dim * (2 * i + ss + nm * (2 * j + ss))
                dst_off = dim * (i - coff + (j - coff) * nc)
                e['fill'].append((OP_AVG, 1, 1, 0, src_off, dst_off, 1, nm))

    # ---- Interpolation ops (coarser neighbor) ----
    if s == 2 and is_face:
        s0_i, s1_i = fs0, fs1
        e0, e1 = fe0, fe1
        for iy in range(s1_i, e1, 2):
            YY = (iy - s1_i - min(0, cy) * ((e1 - s1_i) % 2)) // 2 + sC1 - coff
            y = abs(iy - s1_i - min(0, cy) * ((e1 - s1_i) % 2)) % 2
            iyp = -1 if abs(iy) % 2 == 1 else 1
            dy_val = 0.25 * (2 * y - 1)
            for ix in range(s0_i, e0, 2):
                XX = (ix - s0_i - min(0, cx) * ((e0 - s0_i) % 2)) // 2 + sC0 - coff
                x = abs(ix - s0_i - min(0, cx) * ((e0 - s0_i) % 2)) % 2
                ixp = -1 if abs(ix) % 2 == 1 else 1
                dx_val = 0.25 * (2 * x - 1)
                if ix < -2 or iy < -2 or ix > BS + 1 or iy > BS + 1:
                    continue
                i1 = dim * (XX + nc * YY)
                j0 = dim * ((ix + ss) + nm * (iy + ss))
                dirn = 0 if cx != 0 else 1
                stride = nc if dirn == 0 else 1
                CC = YY if dirn == 0 else XX
                t = dy_val if dirn == 0 else dx_val
                branch = 0 if CC + coff == 0 else (1 if CC + coff == BS // 2 - 1 else 2)
                ok1 = s1_i <= iy + iyp < e1
                ok2 = s0_i <= ix + ixp < e0
                cs_off = stride * dim
                pts = _PTS[branch]
                srcs = tuple(i1 + p * cs_off for p in pts)
                # Lagrange weights: sub=1 → t=+1/4, sub=0 → t=-1/4
                wvp = FACE_W[branch, 1]  # t = +1/4
                wvm = FACE_W[branch, 0]  # t = -1/4
                if t < 0:
                    wvp, wvm = wvm, wvp
                sign_mask = sum((1 << k) for k in range(4) if FACE_DSIGN[dirn][k] < 0)
                dests = [j0, j0 + nm * dim * iyp, j0 + dim * ixp,
                         j0 + dim * ixp + nm * dim * iyp]
                ok = [True, ok1, ok2, ok1 and ok2]
                for k in range(4):
                    if not ok[k]:
                        continue
                    w = wvm if (sign_mask & (1 << k)) else wvp
                    e['interp_ops'].append((OP_INTERP3, w[0], w[1], w[2],
                                            srcs[0], dests[k], srcs[1], srcs[2]))

        li = next((j for j in range(4)
                    if LI_LE_TAB[j][0] == cx and LI_LE_TAB[j][1] == cy), -1)
        if li >= 0:
            for iy in range(s1_i, e1):
                for ix in range(s0_i, e0):
                    if ix < -2 or iy < -2 or ix > BS + 1 or iy > BS + 1:
                        continue
                    ka = dim * ((ix + ss) + nm * (iy + ss))
                    x = abs(ix - s0_i - min(0, cx) * ((e0 - s0_i) % 2)) % 2
                    y = abs(iy - s1_i - min(0, cy) * ((e1 - s1_i) % 2)) % 2
                    p = x if cx != 0 else y
                    is_LE, b_dx, b_dy, c_dx, c_dy = LI_LE_TAB[li][2][p]
                    kb = dim * ((ix + ss + b_dx) + nm * (iy + ss + b_dy))
                    kc = dim * ((ix + ss + c_dx) + nm * (iy + ss + c_dy))
                    e['interp_ops'].append((OP_LELI, 0, 0, is_LE, ka, kb, kc, 0))

    elif s == 2 and not is_face:
        for iy in range(fs1, fe1):
            for ix in range(fs0, fe0):
                YY = (iy - fs1 - min(0, cy) * ((fe1 - fs1) % 2)) // 2 + sC1
                XX = (ix - fs0 - min(0, cx) * ((fe0 - fs0) % 2)) // 2 + sC0
                src_c = dim * (XX - coff + nc * (YY - coff))
                dst_m = dim * (ix + ss + nm * (iy + ss))
                x = abs(ix - fs0 - min(0, cx) * ((fe0 - fs0) % 2)) % 2
                y = abs(iy - fs1 - min(0, cy) * ((fe1 - fs1) % 2)) % 2
                e['interp_ops'].append((OP_INTERP9, 0, 0, x | (y << 1),
                                        src_c, dst_m, 0, 0))

    # ---- Wall/symmetry BC (s=3,4,5) ----
    # Mirror formula: m(i, side, L) = 2*side*L - 1 - i
    # s=3: reflect x, s=4: reflect y, s=5: reflect both
    if s in (3, 4, 5):
        refl = [s != 4, s != 3]  # which axes get reflected
        sd = [(1 + cx) // 2, (1 + cy) // 2]  # side per axis: 0=left/bottom, 1=right/top
        nrefl = refl[0] + refl[1]
        op = OP_BC_SCALAR if dim == 1 else (OP_BC_CORNER if nrefl == 2 else OP_BC_VECTOR)
        flags = 0 if nrefl == 2 else (0 if refl[0] else 1)
        cc = [cx, cy]

        # Ghost ranges: for wall faces, restrict to wall-normal ghost cells
        gs = [fs0, fs1]; ge = [fe0, fe1]
        for k in range(2):
            if refl[k]:
                gs[k] = -ss if sd[k] == 0 else BS
                ge[k] = 0 if sd[k] == 0 else BS + ss

        # Fine buffer: mirror = 2*sd*BS - 1 - i + ss, identity = i + ss
        # Corner (s=5) uses full ghost region; faces use wall-restricted range
        if s == 5:
            fgs = [fs0, fs1]; fge = [fe0, fe1]
        else:
            fgs = [gs[0] if refl[0] else fs0, gs[1] if refl[1] else fs1]
            fge = [ge[0] if refl[0] else fe0, ge[1] if refl[1] else fe1]
        for iy in range(fgs[1], fge[1]):
            for ix in range(fgs[0], fge[0]):
                p = [ix, iy]
                mp = [2*sd[k]*BS - 1 - p[k] + ss if refl[k] else p[k] + ss for k in range(2)]
                i0 = (ix + ss) + nm * (iy + ss)
                i1 = mp[0] + nm * mp[1]
                e['fbc'].append((op, 0, 0, flags, dim * i1, dim * i0, 0, 0))

        # Coarse buffer: mirror = 2*sd*(BS//2) - 1 - i - sI, identity = i - sI
        sI = (-ss - 1) // 2 - 1
        eI0 = (ss + 1) // 2 + 2
        H = BS // 2
        if s == 5:
            cgs = [cs0, cs1]; cge = [ce0, ce1]
        else:
            cgs = [0, 0]; cge = [0, 0]
            for k in range(2):
                cs_k, ce_k = (cs0, ce0) if k == 0 else (cs1, ce1)
                if refl[k]:
                    cgs[k] = sI if sd[k] == 0 else H
                    cge[k] = 0 if sd[k] == 0 else min(H + eI0 - 1, sI + nc)
                else:
                    cgs[k] = cs_k
                    cge[k] = ce_k
        for iy in range(cgs[1], cge[1]):
            for ix in range(cgs[0], cge[0]):
                p = [ix, iy]
                mp = [2*sd[k]*H - 1 - p[k] - sI if refl[k] else p[k] - sI for k in range(2)]
                i0 = (ix - sI) + nc * (iy - sI)
                i1 = mp[0] + nc * mp[1]
                e['cbc'].append((op, 0, 1, flags, dim * i1, dim * i0, 0, 0))

    return e


# ---- Binary serialization ----

ENTRY_SIZE = 20 + 8 + MAX_OPS * 20


def pack_op(op):
    return struct.pack('<bbbb iiii', *op)


def pack_blksrc(b):
    return struct.pack('<bbbbbbb?b',
                       b['level_delta'], b['xi_mul'], b['yi_mul'],
                       b['xi_add'], b['yi_add'], b['xi_shift'], b['yi_shift'],
                       bool(b['is_self']), b['self_idx'])


ZERO_OP = pack_op((0, 0, 0, 0, 0, 0, 0, 0))
ZERO_BLKSRC = make_blksrc()


def pack_entry(e):
    buf = bytearray()
    buf += struct.pack('<b', e['n_blk'])
    blks = list(e['blk_src'])
    while len(blks) < 2:
        blks.append(ZERO_BLKSRC)
    for b in blks:
        buf += pack_blksrc(b)
    buf += b'\x00'
    assert len(buf) == 20

    pre_ops = e['fill'] + e['cbc']
    post_ops = e['interp_ops'] + e['fbc']
    assert len(pre_ops) <= MAX_PRE, f"pre overflow: {len(pre_ops)} > {MAX_PRE}"
    assert len(post_ops) <= MAX_POST, f"post overflow: {len(post_ops)} > {MAX_POST}"

    buf += struct.pack('<ii', len(pre_ops), len(post_ops))
    for op in pre_ops:
        buf += pack_op(op)
    buf += ZERO_OP * (MAX_PRE - len(pre_ops))
    for op in post_ops:
        buf += pack_op(op)
    buf += ZERO_OP * (MAX_POST - len(post_ops))
    assert len(buf) == ENTRY_SIZE
    return bytes(buf)


def build_and_write(fname, ss, dim):
    max_pre = max_post = 0
    with open(fname, 'wb') as f:
        for cx in range(-1, 2):
            for cy in range(-1, 2):
                for xp in range(2):
                    for yp in range(2):
                        for s in range(N_STATUS):
                            if cx == 0 and cy == 0:
                                f.write(b'\x00' * ENTRY_SIZE)
                            else:
                                e = build_entry(cx, cy, xp, yp, s, ss, dim)
                                pre = len(e['fill']) + len(e['cbc'])
                                post = len(e['interp_ops']) + len(e['fbc'])
                                max_pre = max(max_pre, pre)
                                max_post = max(max_post, post)
                                f.write(pack_entry(e))
    n_entries = 3 * 3 * 2 * 2 * N_STATUS
    sz = n_entries * ENTRY_SIZE
    print(f'{fname}: {sz} bytes ({sz // 1024} KB), max_pre={max_pre}, max_post={max_post}')



if __name__ == '__main__':
    for ss, dim in [(1, 1), (1, 2), (2, 1), (4, 1)]:
        build_and_write(f'tab_ss{ss}_dim{dim}.bin', ss, dim)
