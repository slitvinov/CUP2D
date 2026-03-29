#!/usr/bin/env python3
"""Generate binary dispatch table for AMR ghost fill/interpolation/BC."""

import struct
import sys

BS = 8
MAX_PRE = 32
MAX_POST = 48
MAX_OPS = MAX_PRE + MAX_POST

childNeighborTable = [
    (-1, -1, 3, 1, [(-1, -1)], 1),
    ( 0, -1, 1, 1, [( 0, -1), (1, -1)], 2),
    ( 1, -1, 3, 1, [( 2, -1)], 1),
    (-1,  0, 1, 2, [(-1,  0), (-1, 1)], 2),
    ( 1,  0, 1, 2, [( 2,  0), (2, 1)], 2),
    (-1,  1, 3, 1, [(-1,  2)], 1),
    ( 0,  1, 1, 1, [( 0,  2), (1, 2)], 2),
    ( 1,  1, 3, 1, [( 2,  2)], 1),
]

liLeTab = [
    (0, +1, [(0, 0, -1, 0, -2), (1, 0, -2, 0, -3)]),
    (0, -1, [(1, 0, +2, 0, +3), (0, 0, +1, 0, +2)]),
    (+1, 0, [(0, -1, 0, -2, 0), (1, -2, 0, -3, 0)]),
    (-1, 0, [(1, +2, 0, +3, 0), (0, +1, 0, +2, 0)]),
]

face_dsign = [
    [+1, -1, +1, -1],
    [+1, +1, -1, -1],
]

OP_COPY = 0; OP_AVG = 1; OP_INTERP9 = 2; OP_INTERP3 = 3
OP_LELI = 4; OP_BC_SCALAR = 5; OP_BC_VECTOR = 6

# Face interpolation weights (numerators / 32)
FACE_VP = {0: (21, 14, -3), 1: (5, -18, 45), 2: (-3, 30, 5)}
FACE_VM = {0: (45, -18, 5), 1: (-3, 14, 21), 2: (5, 30, -3)}

def get_child_pattern(cx, cy):
    for e in childNeighborTable:
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
        CoarseEdge_d = 1 if (cc != 0) and (coord == (1 if cc < 0 else 0)) else 0
        cstart[d] = (max(cc, 0) * BS // 2 +
                     (1 - abs(cc)) * base_d * BS // 2 - cc * BS +
                     CoarseEdge_d * cc * BS // 2)

    sC0 = (-ss - 1) // 2 if cx < 0 else (0 if cx == 0 else BS // 2)
    sC1 = (-ss - 1) // 2 if cy < 0 else (0 if cy == 0 else BS // 2)
    is_face = abs(cx) + abs(cy) == 1

    e = {
        'cx': cx, 'cy': cy,
        'fs': [fs0, fs1], 'fe': [fe0, fe1],
        'cs': [cs0, cs1], 'ce': [ce0, ce1],
        'cstart': cstart, 'sC': [sC0, sC1],
        'n_blk': 0, 'blk_src': [],
        'fill': [], 'cbc': [], 'interp_ops': [], 'fbc': [],
    }

    # BlkSrc
    if s == 0:
        e['n_blk'] = 1
        e['blk_src'] = [{'level_delta': 0, 'xi_mul': 1, 'yi_mul': 1,
                          'xi_add': cx, 'yi_add': cy,
                          'xi_shift': 0, 'yi_shift': 0,
                          'is_self': 0, 'self_idx': 0}]
    elif s == 1:
        pat = get_child_pattern(cx, cy)
        e['n_blk'] = pat[5]
        e['blk_src'] = []
        for b in range(pat[5]):
            e['blk_src'].append({
                'level_delta': 1, 'xi_mul': 2, 'yi_mul': 2,
                'xi_add': pat[4][b][0], 'yi_add': pat[4][b][1],
                'xi_shift': 0, 'yi_shift': 0,
                'is_self': 0, 'self_idx': 0})
    elif s == 2:
        e['n_blk'] = 2
        e['blk_src'] = [
            {'level_delta': -1, 'xi_mul': 1, 'yi_mul': 1,
             'xi_add': cx, 'yi_add': cy,
             'xi_shift': 1, 'yi_shift': 1,
             'is_self': 0, 'self_idx': 0},
            {'level_delta': 0, 'xi_mul': 0, 'yi_mul': 0,
             'xi_add': 0, 'yi_add': 0,
             'xi_shift': 0, 'yi_shift': 0,
             'is_self': 1, 'self_idx': 0},
        ]

    # Fill ops (always ua=1 behavior)
    if s == 0:
        # Always fill ghost cells (ua=1: no skip for corners)
        for iy in range(fs1, fe1):
            src_off = dim * (BS * (iy - cy * BS) + (fs0 - cx * BS))
            dst_off = dim * ((fs0 + ss) + (iy + ss) * nm)
            cols = fe0 - fs0
            e['fill'].append((OP_COPY, 0, 0, 0, src_off, dst_off, cols, 0))
        # Always fill coarse buffer (ua=1)
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
        width = (abs(cx) * (fe0 - fs0) +
                 (1 - abs(cx)) * ((fe0 - fs0) // 2))
        B = 0
        for cnt in range(pat[5]):
            aux = (B % 2) if abs(cx) == 1 else (B // 2)
            di = (abs(cx) * (fs0 + ss) +
                  (1 - abs(cx)) * (fs0 + ss + (B % 2) * (fe0 - fs0) // 2))
            sx = fs0 - cx * BS + min(0, cx) * (fe0 - fs0)
            iy2 = fs1
            while iy2 < fe1:
                sy = (2 * (iy2 - cy * BS) + min(0, cy) * BS if abs(cy) == 1
                      else iy2)
                dk = (di + (abs(cy) * (iy2 + ss) +
                            (1 - abs(cy)) * (iy2 // 2 + ss +
                                             aux * (fe1 - fs1) // 2)) * nm)
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
                if i > 1 and i < BS // 2 - 2 and j > 2 and j < BS // 2 - 2:
                    continue
                ix0 = 2 * i + ss
                iy0 = 2 * j + ss
                src_off = dim * (ix0 + nm * iy0)
                dst_off = dim * (i - coff + (j - coff) * nc)
                e['fill'].append((OP_AVG, 1, 1, 0, src_off, dst_off, 1, nm))

    # Interp ops
    if s == 2 and is_face:
        s0_i, s1_i = fs0, fs1
        e0, e1 = fe0, fe1
        for iy in range(s1_i, e1, 2):
            YY = ((iy - s1_i - min(0, cy) * ((e1 - s1_i) % 2)) // 2 +
                  sC1 - coff)
            y = abs(iy - s1_i - min(0, cy) * ((e1 - s1_i) % 2)) % 2
            iyp = -1 if (abs(iy) % 2 == 1) else 1
            dy_val = 0.25 * (2 * y - 1)
            for ix in range(s0_i, e0, 2):
                XX = ((ix - s0_i - min(0, cx) * ((e0 - s0_i) % 2)) // 2 +
                      sC0 - coff)
                x = abs(ix - s0_i - min(0, cx) * ((e0 - s0_i) % 2)) % 2
                ixp = -1 if (abs(ix) % 2 == 1) else 1
                dx_val = 0.25 * (2 * x - 1)
                if ix < -2 or iy < -2 or ix > BS + 1 or iy > BS + 1:
                    continue
                i1 = dim * (XX + nc * YY)
                j0 = dim * ((ix + ss) + nm * (iy + ss))
                dirn = 0 if cx != 0 else 1
                stride = nc if dirn == 0 else 1
                CC = YY if dirn == 0 else XX
                t = dy_val if dirn == 0 else dx_val
                if CC + coff == 0:
                    branch = 0
                elif CC + coff == (BS // 2) - 1:
                    branch = 1
                else:
                    branch = 2
                ok1 = iy + iyp >= s1_i and iy + iyp < e1
                ok2 = ix + ixp >= s0_i and ix + ixp < e0
                sign_mask = 0
                for k in range(4):
                    if face_dsign[dirn][k] < 0:
                        sign_mask |= (1 << k)

                cs_off = stride * dim
                if branch == 0:
                    srcs = (i1, i1 + cs_off, i1 + 2 * cs_off)
                elif branch == 1:
                    srcs = (i1 - 2 * cs_off, i1 - cs_off, i1)
                else:
                    srcs = (i1 - cs_off, i1, i1 + cs_off)

                wvp = FACE_VP[branch]
                wvm = FACE_VM[branch]
                if t < 0:
                    wvp, wvm = wvm, wvp

                dests = [j0,
                         j0 + nm * dim * iyp,
                         j0 + dim * ixp,
                         j0 + dim * ixp + nm * dim * iyp]
                ok = [True, ok1, ok2, ok1 and ok2]
                for k in range(4):
                    if not ok[k]:
                        continue
                    w = wvm if (sign_mask & (1 << k)) else wvp
                    e['interp_ops'].append((OP_INTERP3, w[0], w[1], w[2],
                                            srcs[0], dests[k], srcs[1], srcs[2]))

        li = -1
        for j in range(4):
            if liLeTab[j][0] == cx and liLeTab[j][1] == cy:
                li = j
                break
        if li >= 0:
            for iy in range(s1_i, e1):
                for ix in range(s0_i, e0):
                    if ix < -2 or iy < -2 or ix > BS + 1 or iy > BS + 1:
                        continue
                    ka = dim * ((ix + ss) + nm * (iy + ss))
                    x = abs(ix - s0_i - min(0, cx) * ((e0 - s0_i) % 2)) % 2
                    y = abs(iy - s1_i - min(0, cy) * ((e1 - s1_i) % 2)) % 2
                    p = x if cx != 0 else y
                    sub = liLeTab[li][2][p]
                    is_LE, b_dx, b_dy, c_dx, c_dy = sub
                    kb = dim * ((ix + ss + b_dx) + nm * (iy + ss + b_dy))
                    kc = dim * ((ix + ss + c_dx) + nm * (iy + ss + c_dy))
                    e['interp_ops'].append((OP_LELI, 0, 0, is_LE, ka, kb, kc, 0))

    elif s == 2 and not is_face:
        for iy in range(fs1, fe1):
            for ix in range(fs0, fe0):
                YY = ((iy - fs1 - min(0, cy) * ((fe1 - fs1) % 2)) // 2 + sC1)
                XX = ((ix - fs0 - min(0, cx) * ((fe0 - fs0) % 2)) // 2 + sC0)
                src_c = dim * (XX - coff + nc * (YY - coff))
                dst_m = dim * (ix + ss + nm * (iy + ss))
                x = abs(ix - fs0 - min(0, cx) * ((fe0 - fs0) % 2)) % 2
                y = abs(iy - fs1 - min(0, cy) * ((fe1 - fs1) % 2)) % 2
                e['interp_ops'].append((OP_INTERP9, 0, 0,
                                        x | (y << 1), src_c, dst_m, 0, 0))

    # BC ops
    if s >= 3:
        bc_dirs = []
        if s == 3 or s == 5:
            bc_dirs.append(0)
        if s == 4 or s == 5:
            bc_dirs.append(1)
        for dirn in bc_dirs:
            side = 1 if (cx if dirn == 0 else cy) > 0 else 0
            gs = [0, 0]; ge = [0, 0]
            gs[0] = (-ss if side == 0 else BS) if dirn == 0 else fs0
            gs[1] = (-ss if side == 0 else BS) if dirn == 1 else fs1
            ge[0] = (0 if side == 0 else BS + ss) if dirn == 0 else fe0
            ge[1] = (0 if side == 0 else BS + ss) if dirn == 1 else fe1
            mirror = 0 if side == 0 else BS - 1
            for iy in range(gs[1], ge[1]):
                for ix in range(gs[0], ge[0]):
                    mx = (mirror if dirn == 0 else ix) + ss
                    my = (mirror if dirn == 1 else iy) + ss
                    i0 = (ix + ss) + nm * (iy + ss)
                    i1 = mx + nm * my
                    if dim == 1:
                        e['fbc'].append((OP_BC_SCALAR, 0, 0, 0, dim*i1, dim*i0, 0, 0))
                    else:
                        e['fbc'].append((OP_BC_VECTOR, 0, 0, dirn, dim*i1, dim*i0, 0, 0))
            sI = (-ss - 1) // 2 - 1
            eI0 = (ss + 1) // 2 + 2
            cgs = [0, 0]; cge = [0, 0]
            cgs[0] = (sI if side == 0 else BS // 2) if dirn == 0 else cs0
            cgs[1] = (sI if side == 0 else BS // 2) if dirn == 1 else cs1
            cge[0] = (0 if side == 0 else min(BS // 2 + eI0 - 1, sI + nc)) if dirn == 0 else ce0
            cge[1] = (0 if side == 0 else min(BS // 2 + eI0 - 1, sI + nc)) if dirn == 1 else ce1
            cmirror = 0 if side == 0 else BS // 2 - 1
            for iy in range(cgs[1], cge[1]):
                for ix in range(cgs[0], cge[0]):
                    mx = (cmirror if dirn == 0 else ix) - sI
                    my = (cmirror if dirn == 1 else iy) - sI
                    i0 = (ix - sI) + nc * (iy - sI)
                    i1 = mx + nc * my
                    if dim == 1:
                        e['cbc'].append((OP_BC_SCALAR, 0, 1, 0, dim*i1, dim*i0, 0, 0))
                    else:
                        e['cbc'].append((OP_BC_VECTOR, 0, 1, dirn, dim*i1, dim*i0, 0, 0))
    return e

# Binary serialization
# Op: 4 bytes (type,blk_idx,dst_idx,flags) + 4 ints = 20 bytes
# BlkSrc: 9 bytes
# TabEntry layout:
#   0: n_blk(1) blk_src[2](18) pad(1) = 20 bytes
#   20: n_pre(4) n_post(4) = 8 bytes
#   28: ops[MAX_OPS](MAX_OPS*20)
#   Total: 20 + 8 + MAX_OPS*20

ENTRY_SIZE = 20 + 8 + MAX_OPS * 20

def pack_op(op):
    return struct.pack('<bbbb iiii', op[0], op[1], op[2], op[3], op[4], op[5], op[6], op[7])

def pack_blksrc(b):
    return struct.pack('<bbbbbbb?b',
        b['level_delta'], b['xi_mul'], b['yi_mul'],
        b['xi_add'], b['yi_add'], b['xi_shift'], b['yi_shift'],
        bool(b['is_self']), b['self_idx'])

ZERO_OP = pack_op((0,0,0,0,0,0,0,0))

def pack_entry(e):
    buf = bytearray()
    buf += struct.pack('<b', e['n_blk'])
    blks = list(e['blk_src'])
    while len(blks) < 2:
        blks.append({'level_delta':0,'xi_mul':0,'yi_mul':0,'xi_add':0,'yi_add':0,'xi_shift':0,'yi_shift':0,'is_self':0,'self_idx':0})
    for b in blks:
        buf += pack_blksrc(b)
    assert len(buf) == 19
    buf += b'\x00'
    assert len(buf) == 20

    pre_ops = e['fill'] + e['cbc']
    post_ops = e['interp_ops'] + e['fbc']
    assert len(pre_ops) <= MAX_PRE, f"pre overflow: {len(pre_ops)} > {MAX_PRE} at ({e['cx']},{e['cy']})"
    assert len(post_ops) <= MAX_POST, f"post overflow: {len(post_ops)} > {MAX_POST} at ({e['cx']},{e['cy']})"

    buf += struct.pack('<ii', len(pre_ops), len(post_ops))

    for op in pre_ops:
        buf += pack_op(op)
    for _ in range(MAX_PRE - len(pre_ops)):
        buf += ZERO_OP
    for op in post_ops:
        buf += pack_op(op)
    for _ in range(MAX_POST - len(post_ops)):
        buf += ZERO_OP

    assert len(buf) == ENTRY_SIZE
    return bytes(buf)

def build_and_write(fname, ss, dim):
    max_pre = 0
    max_post = 0
    with open(fname, 'wb') as f:
        for cxi in range(3):
            cx = cxi - 1
            for cyi in range(3):
                cy = cyi - 1
                for xp in range(2):
                    for yp in range(2):
                        for s in range(6):
                            if cx == 0 and cy == 0:
                                f.write(b'\x00' * ENTRY_SIZE)
                            else:
                                e = build_entry(cx, cy, xp, yp, s, ss, dim)
                                pre = len(e['fill']) + len(e['cbc'])
                                post = len(e['interp_ops']) + len(e['fbc'])
                                max_pre = max(max_pre, pre)
                                max_post = max(max_post, post)
                                f.write(pack_entry(e))
    sz = 216 * ENTRY_SIZE
    print(f'{fname}: {sz} bytes ({sz/1024:.0f} KB), max_pre={max_pre}, max_post={max_post}')

# Poisson table (unchanged)
MAX_POISSON_OPS = 16
POISSON_ENTRY_SIZE = 4 + MAX_POISSON_OPS * 8

P_INTERP_OFF = [[-2,-1,0], [2,1,0], [-1,1,0]]
P_INTERP_D1 = [[1/8, -1/2, 3/8], [-1/8, 1/2, -3/8], [-1/8, 1/8, 0]]
P_INTERP_D2 = [[1/32, -1/16, 1/32], [1/32, -1/16, 1/32], [1/32, 1/32, -1/16]]

def poisson_interp_ops(add, c_blk, cix, ciy, f_blk, fc_ix, fc_iy, ff_ix, ff_iy,
                       signInt, signTaylor, dir):
    add(f_blk, fc_ix, fc_iy, signInt * 2/3)
    add(f_blk, ff_ix, ff_iy, -signInt * 1/5)
    tf = signInt * 8/15
    add(c_blk, cix, ciy, tf)
    tang = ciy if dir == 0 else cix
    c = (0 if (tang == BS-1 or tang == BS//2-1)
         else 1 if (tang == 0 or tang == BS//2) else 2)
    for i in range(3):
        off = P_INTERP_OFF[c][i]
        if off == 0:
            ox, oy = cix, ciy
        elif dir == 0:
            ox, oy = cix, ciy + off
        else:
            ox, oy = cix + off, ciy
        add(c_blk, ox, oy, signTaylor * tf * P_INTERP_D1[c][i])
        add(c_blk, ox, oy, tf * P_INTERP_D2[c][i])

def build_poisson_entry(edge, tc, parity, state):
    dir = edge >> 1
    side = edge & 1
    sign = 2 * side - 1
    dx, dy = (1 - dir) * sign, dir * sign
    ix = (0 if side == 0 else BS - 1) if dir == 0 else tc
    iy = tc if dir == 0 else (0 if side == 0 else BS - 1)

    ops = {}
    def add(blk, cx, cy, coeff):
        key = (blk, cx, cy)
        ops[key] = ops.get(key, 0.0) + coeff

    if state == 0:
        add(0, ix + dx, iy + dy, 1.0)
        add(0, ix, iy, -1.0)
    elif state == 1:
        ne = (1 - side) * (BS - 1)
        if dir == 0:
            add(1, ne, tc, 1.0)
        else:
            add(1, tc, ne, 1.0)
        add(0, ix, iy, -1.0)
    elif state == 2:
        cix = (1 - side) * (BS - 1) if dir == 0 else tc // 2 + parity * (BS // 2)
        ciy = tc // 2 + parity * (BS // 2) if dir == 0 else (1 - side) * (BS - 1)
        signTaylor = -1.0 if tc % 2 == 0 else 1.0
        poisson_interp_ops(add, 2, cix, ciy, 0, ix, iy, ix - dx, iy - dy,
                           1.0, signTaylor, dir)
        add(0, ix, iy, -1.0)
    elif state == 3:
        fe0 = BS - 1 if side == 0 else 0
        fe1 = BS - 2 if side == 0 else 1
        ft = (tc % (BS // 2)) * 2
        for dt in range(2):
            fc_ix, fc_iy = (fe0, ft + dt) if dir == 0 else (ft + dt, fe0)
            ff_ix, ff_iy = (fe1, ft + dt) if dir == 0 else (ft + dt, fe1)
            add(3, fc_ix, fc_iy, 1.0)
            signTaylor_f = -1.0 if dt == 0 else 1.0
            poisson_interp_ops(add, 0, ix, iy, 3, fc_ix, fc_iy, ff_ix, ff_iy,
                               -1.0, signTaylor_f, dir)

    return [(k[0], k[1], k[2], v) for k, v in ops.items() if abs(v) > 1e-15]

def pack_poisson_op(op):
    return struct.pack('<bbbx f', op[0], op[1], op[2], op[3])

ZERO_POP = b'\x00' * 8

def pack_poisson_entry(ops):
    assert len(ops) <= MAX_POISSON_OPS, f"poisson ops overflow: {len(ops)}"
    buf = struct.pack('<i', len(ops))
    for op in ops:
        buf += pack_poisson_op(op)
    buf += ZERO_POP * (MAX_POISSON_OPS - len(ops))
    assert len(buf) == POISSON_ENTRY_SIZE
    return buf

def build_poisson_table():
    fname = 'tab_poisson.bin'
    max_ops = 0
    with open(fname, 'wb') as f:
        for edge in range(4):
            for tc in range(BS):
                for parity in range(2):
                    for state in range(4):
                        ops = build_poisson_entry(edge, tc, parity, state)
                        max_ops = max(max_ops, len(ops))
                        f.write(pack_poisson_entry(ops))
    sz = 4 * BS * 2 * 4 * POISSON_ENTRY_SIZE
    print(f'{fname}: {sz} bytes ({sz/1024:.0f} KB), max ops={max_ops}')

def main():
    combos = [(1, 1), (1, 2), (3, 2), (4, 1)]
    for ss, dim in combos:
        fname = f'tab_ss{ss}_dim{dim}.bin'
        build_and_write(fname, ss, dim)
    build_poisson_table()

if __name__ == '__main__':
    main()
