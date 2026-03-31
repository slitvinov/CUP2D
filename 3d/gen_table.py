#!/usr/bin/env python3
"""Generate 3D binary dispatch tables for AMR ghost fill/interpolation/BC."""

import struct
import sys

BS = 8
MAX_PRE = 300
MAX_POST = 64
MAX_OPS = MAX_PRE + MAX_POST

OP_COPY = 0; OP_AVG = 1; OP_INTERP = 2; OP_BC_SCALAR = 3; OP_BC_VECTOR = 4

def ghost_bounds(c, ss):
    if c < 0: return (-ss, 0)
    if c > 0: return (BS, BS + ss)
    return (0, BS)

def bc_status(cx, cy, cz):
    """Return BC status for a wall direction."""
    xbc = cx != 0
    ybc = cy != 0
    zbc = cz != 0
    n = xbc + ybc + zbc
    if n == 3: return 9   # corner
    if n == 2:
        if not xbc: return 8  # yz-edge
        if not ybc: return 7  # xz-edge
        return 6              # xy-edge
    if xbc: return 3
    if ybc: return 4
    return 5  # z-wall

def gen_entry(cx, cy, cz, xp, yp, zp, s, ss, dim):
    """Generate a table entry for one neighbor configuration."""
    # Center direction is never used - return empty entry
    if cx == 0 and cy == 0 and cz == 0:
        return {'cx':0,'cy':0,'cz':0,'n_blk':0,'blk_src':[],'fill':[],'fbc':[]}
    nm = 2 * ss + BS
    nc = BS // 2 + ss + 3

    fs = [ghost_bounds(c, ss) for c in (cx, cy, cz)]
    fs0, fe0 = fs[0]
    fs1, fe1 = fs[1]
    fs2, fe2 = fs[2]

    e = {
        'cx': cx, 'cy': cy, 'cz': cz,
        'n_blk': 0, 'blk_src': [],
        'fill': [], 'fbc': [],
    }

    # BlkSrc
    if s == 0:
        e['n_blk'] = 1
        e['blk_src'] = [{'level_delta': 0,
            'xi_mul': 1, 'yi_mul': 1, 'zi_mul': 1,
            'xi_add': cx, 'yi_add': cy, 'zi_add': cz,
            'xi_shift': 0, 'yi_shift': 0, 'zi_shift': 0,
            'is_self': 0, 'self_idx': 0}]
    elif s == 2:
        e['n_blk'] = 1
        e['blk_src'] = [{'level_delta': -1,
            'xi_mul': 1, 'yi_mul': 1, 'zi_mul': 1,
            'xi_add': cx, 'yi_add': cy, 'zi_add': cz,
            'xi_shift': 1, 'yi_shift': 1, 'zi_shift': 1,
            'is_self': 0, 'self_idx': 0}]
    elif s == 1:
        nch = sum(1 for c in (cx,cy,cz) if c == 0)
        nch = 1 << nch  # 4 for face, 2 for edge, 1 for corner
        # For now limit to 4 children max
        nch = min(nch, 4)
        e['n_blk'] = nch
        e['blk_src'] = []
        for b in range(nch):
            e['blk_src'].append({'level_delta': 1,
                'xi_mul': 2, 'yi_mul': 2, 'zi_mul': 2,
                'xi_add': 0, 'yi_add': 0, 'zi_add': 0,
                'xi_shift': 0, 'yi_shift': 0, 'zi_shift': 0,
                'is_self': 0, 'self_idx': 0})
    else:
        # BC: self-reference
        e['n_blk'] = 1
        e['blk_src'] = [{'level_delta': 0,
            'xi_mul': 0, 'yi_mul': 0, 'zi_mul': 0,
            'xi_add': 0, 'yi_add': 0, 'zi_add': 0,
            'xi_shift': 0, 'yi_shift': 0, 'zi_shift': 0,
            'is_self': 1, 'self_idx': 0}]

    # Fill ops for s=0 (same level): copy slabs from neighbor
    if s == 0:
        for iz in range(fs2, fe2):
            for iy in range(fs1, fe1):
                src_off = dim * ((iz - cz*BS)*BS*BS + (iy - cy*BS)*BS + (fs0 - cx*BS))
                dst_off = dim * ((iz + ss)*nm*nm + (iy + ss)*nm + (fs0 + ss))
                cols = fe0 - fs0
                e['fill'].append((OP_COPY, 0, 0, 0, src_off, dst_off, cols, 0))

    # Fill ops for s=2 (coarser): piecewise constant injection
    elif s == 2:
        for iz in range(fs2, fe2):
            for iy in range(fs1, fe1):
                for ix in range(fs0, fe0):
                    # Map to coarser block coordinates
                    parity = [xp, yp, zp]
                    src_ix = (ix + parity[0] * (BS//2)) // 2
                    src_iy = (iy + parity[1] * (BS//2)) // 2
                    src_iz = (iz + parity[2] * (BS//2)) // 2
                    # Clamp to [0, BS-1]
                    src_ix = max(0, min(BS-1, src_ix))
                    src_iy = max(0, min(BS-1, src_iy))
                    src_iz = max(0, min(BS-1, src_iz))
                    src_off = dim * ((src_iz*BS + src_iy)*BS + src_ix)
                    dst_off = dim * ((iz + ss)*nm*nm + (iy + ss)*nm + (ix + ss))
                    e['fill'].append((OP_COPY, 0, 0, 0, src_off, dst_off, 1, 0))

    # Fill ops for s=1 (finer): average 2x2x2 fine cells
    elif s == 1:
        # For each ghost cell, average from the appropriate fine child
        for iz in range(fs2, fe2):
            for iy in range(fs1, fe1):
                for ix in range(fs0, fe0):
                    dst_off = dim * ((iz + ss)*nm*nm + (iy + ss)*nm + (ix + ss))
                    # Determine which child block and fine cell
                    # The fine block's cell at (2*ix, 2*iy, 2*iz) relative
                    # For simplicity, use first child with piecewise constant
                    fi = ix * 2 - (cx if cx != 0 else 0) * BS
                    fj = iy * 2 - (cy if cy != 0 else 0) * BS
                    fk = iz * 2 - (cz if cz != 0 else 0) * BS
                    fi = max(0, min(BS-1, fi))
                    fj = max(0, min(BS-1, fj))
                    fk = max(0, min(BS-1, fk))
                    src_off = dim * ((fk*BS + fj)*BS + fi)
                    e['fill'].append((OP_COPY, 0, 0, 0, src_off, dst_off, 1, 0))

    # BC ops
    if s >= 3:
        bc_axes = []
        if s in (3, 6, 7, 9): bc_axes.append(0)  # x
        if s in (4, 6, 8, 9): bc_axes.append(1)  # y
        if s in (5, 7, 8, 9): bc_axes.append(2)  # z
        negate_mask = 0
        for ax in bc_axes:
            negate_mask |= (1 << ax)
        sides = [1 if c > 0 else 0 for c in (cx, cy, cz)]

        for iz in range(fs2, fe2):
            for iy in range(fs1, fe1):
                for ix in range(fs0, fe0):
                    coords = [ix, iy, iz]
                    mirror = list(coords)
                    for ax in bc_axes:
                        side = sides[ax]
                        g = coords[ax]
                        mirror[ax] = -g - 1 if side == 0 else 2*BS - 1 - g
                    mx, my, mz = mirror
                    src_off = dim * ((mz + ss)*nm*nm + (my + ss)*nm + (mx + ss))
                    dst_off = dim * ((iz + ss)*nm*nm + (iy + ss)*nm + (ix + ss))
                    if dim == 1:
                        e['fbc'].append((OP_BC_SCALAR, 0, 0, 0, src_off, dst_off, 0, 0))
                    else:
                        e['fbc'].append((OP_BC_VECTOR, 0, 0, negate_mask, src_off, dst_off, 0, 0))
    return e


def serialize_op(op):
    """Pack one LbOp: type(1) blk_idx(1) dst_idx(1) flags(1) src_off(4) dst_off(4) p1(4) p2(4) = 20 bytes"""
    return struct.pack('<bbbb iiii', op[0], op[1], op[2], op[3], op[4], op[5], op[6], op[7])

def serialize_blk_src(bs):
    """Pack one LbSrc: 12 bytes"""
    return struct.pack('<bbbbbbbbbbbb',
        bs['level_delta'], bs['xi_mul'], bs['yi_mul'], bs['zi_mul'],
        bs['xi_add'], bs['yi_add'], bs['zi_add'],
        bs['xi_shift'], bs['yi_shift'], bs['zi_shift'],
        bs['is_self'], bs['self_idx'])

def serialize_entry(e):
    """Serialize one LbTab entry."""
    # n_blk (1 byte)
    data = struct.pack('<b', e['n_blk'])
    # blk_src[4] (4 * 12 = 48 bytes)
    for b in range(4):
        if b < len(e['blk_src']):
            data += serialize_blk_src(e['blk_src'][b])
        else:
            data += b'\x00' * 12
    # pad (3 bytes to align n_pre to 4-byte boundary)
    data += b'\x00' * 3
    # n_pre, n_post (4+4 bytes)
    pre_ops = e['fill'] + e.get('fbc', [])
    post_ops = []
    n_pre = len(pre_ops)
    n_post = len(post_ops)
    assert n_pre <= MAX_PRE, f"n_pre={n_pre} > {MAX_PRE}"
    assert n_post <= MAX_POST, f"n_post={n_post} > {MAX_POST}"
    data += struct.pack('<ii', n_pre, n_post)
    # ops[MAX_OPS] (MAX_OPS * 20 bytes)
    for i in range(MAX_PRE):
        if i < n_pre:
            data += serialize_op(pre_ops[i])
        else:
            data += b'\x00' * 20
    for i in range(MAX_POST):
        if i < n_post:
            data += serialize_op(post_ops[i])
        else:
            data += b'\x00' * 20
    return data

def gen_tables(ss, dim):
    """Generate all table entries for given (ss, dim) config."""
    all_data = b''
    max_pre = 0
    # Layout: [cx+1][cy+1][cz+1][xp%2][yp%2][zp%2][status]
    # = 3 * 3 * 3 * 2 * 2 * 2 * 10 entries
    for cx in range(-1, 2):
        for cy in range(-1, 2):
            for cz in range(-1, 2):
                for xp in range(2):
                    for yp in range(2):
                        for zp in range(2):
                            for s in range(10):
                                e = gen_entry(cx, cy, cz, xp, yp, zp, s, ss, dim)
                                n_pre = len(e['fill']) + len(e.get('fbc', []))
                                max_pre = max(max_pre, n_pre)
                                all_data += serialize_entry(e)
    return all_data, max_pre

configs = [(1, 1), (1, 3), (4, 1)]
for ss, dim in configs:
    data, max_pre = gen_tables(ss, dim)
    fname = f'tab3d_ss{ss}_dim{dim}.bin'
    with open(fname, 'wb') as f:
        f.write(data)
    print(f'{fname}: {len(data)} bytes ({len(data)//1024} KB), max_pre={max_pre}')
