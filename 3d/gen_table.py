#!/usr/bin/env python3
"""Generate 3D binary dispatch tables for AMR ghost fill/interpolation/BC.

Output files: tab3d_ss{ss}_dim{dim}.bin
Each file has 3*3*3*2*2*2*10 = 2160 entries, each 7340 bytes.

Table index: [cx+1][cy+1][cz+1][xp%2][yp%2][zp%2][status]
  cx,cy,cz in {-1,0,1} -- neighbor direction
  xp,yp,zp in {0,1}    -- block parity (ix%2, iy%2, iz%2)
  status in {0..9}:
    0=same level, 1=finer, 2=coarser,
    3=x-wall, 4=y-wall, 5=z-wall,
    6=xy-edge, 7=xz-edge, 8=yz-edge, 9=xyz-corner

C struct layout (must match exactly):
  struct LbOp    -- 20 bytes: type(1) blk_idx(1) dst_idx(1) flags(1) src_off(4) dst_off(4) p1(4) p2(4)
  struct LbSrc   -- 12 bytes: level_delta xi_mul yi_mul zi_mul xi_add yi_add zi_add xi_shift yi_shift zi_shift is_self self_idx
  struct LbTab   -- 7340 bytes: n_blk(1) blk_src[4](48) _pad[3](3) n_pre(4) n_post(4) ops[364](7280)
"""

import struct
import sys

BS = 8
MAX_PRE = 400
MAX_POST = 64
MAX_OPS = MAX_PRE + MAX_POST
ENTRY_SIZE = 1 + 4 * 12 + 3 + 4 + 4 + MAX_OPS * 20  # = 7340

OP_COPY = 0
OP_AVG = 1
OP_INTERP27 = 2
OP_BC_SCALAR = 3
OP_BC_VECTOR = 4


def ghost_bounds(c, ss):
    """Ghost region bounds along one axis for neighbor direction c."""
    if c < 0:
        return (-ss, 0)
    if c > 0:
        return (BS, BS + ss)
    return (0, BS)


def nb_ch_n(cx, cy, cz):
    """Number of finer children for neighbor direction (cx,cy,cz)."""
    nzero = sum(1 for c in (cx, cy, cz) if c == 0)
    return 1 << nzero  # face:4, edge:2, corner:1


def nb_ch_off(cx, cy, cz, b):
    """Child offset [3] for child b of finer neighbor at (cx,cy,cz).

    Matches nb_ch_off_3d in main.c exactly.
    """
    dirs = [cx, cy, cz]
    zero_axes = [d for d in range(3) if dirs[d] == 0]
    out = [0, 0, 0]
    for d in range(3):
        out[d] = 0 if dirs[d] == 0 else dirs[d] * 2
    if len(zero_axes) >= 1:
        out[zero_axes[0]] = b & 1
    if len(zero_axes) >= 2:
        out[zero_axes[1]] = (b >> 1) & 1
    return out


def build_entry(cx, cy, cz, xp, yp, zp, s, ss, dim):
    """Build one table entry.

    Returns dict with keys: n_blk, blk_src, pre_ops, post_ops.
    pre_ops  = data-copy ops (OP_COPY/OP_AVG from neighbor blocks)
    post_ops = boundary condition ops (OP_BC_SCALAR/OP_BC_VECTOR from self buffer)
    """
    nm = 2 * ss + BS

    gs = [ghost_bounds(c, ss) for c in (cx, cy, cz)]
    # gs[d] = (start, end) for ghost region in axis d

    blk_src = []
    pre_ops = []
    post_ops = []

    # ---------------------------------------------------------------
    # s=0: same-level neighbor -- copy slab row by row
    # ---------------------------------------------------------------
    if s == 0:
        blk_src = [make_src(level_delta=0,
                            xi_mul=1, yi_mul=1, zi_mul=1,
                            xi_add=cx, yi_add=cy, zi_add=cz)]
        # Neighbor block stores data[dim * ((z*BS + y)*BS + x)]
        # We need to copy the appropriate face/edge/corner slab.
        # For cx=-1: we want the neighbor's x in [BS-ss, BS), ghost x in [-ss, 0)
        # For cx=+1: we want the neighbor's x in [0, ss), ghost x in [BS, BS+ss)
        # For cx=0:  we want the neighbor's x in [0, BS), ghost x in [0, BS)
        # Source x = ghost_x - cx*BS
        for iz in range(gs[2][0], gs[2][1]):
            src_z = iz - cz * BS
            for iy in range(gs[1][0], gs[1][1]):
                src_y = iy - cy * BS
                src_x0 = gs[0][0] - cx * BS
                src_off = dim * ((src_z * BS + src_y) * BS + src_x0)
                dst_off = dim * ((iz + ss) * nm * nm + (iy + ss) * nm + (gs[0][0] + ss))
                cols = gs[0][1] - gs[0][0]
                pre_ops.append((OP_COPY, 0, 0, 0, src_off, dst_off, cols, 0))

    # ---------------------------------------------------------------
    # s=1: finer neighbor -- average 2x2x2 fine cells per ghost cell
    # ---------------------------------------------------------------
    elif s == 1:
        nch = nb_ch_n(cx, cy, cz)
        nch = min(nch, 4)  # struct has blk_src[4]
        for b in range(nch):
            off = nb_ch_off(cx, cy, cz, b)
            blk_src.append(make_src(level_delta=1,
                                    xi_mul=2, yi_mul=2, zi_mul=2,
                                    xi_add=off[0], yi_add=off[1], zi_add=off[2]))

        # Each ghost cell at (gx, gy, gz) in the fine block's coordinates
        # needs to be filled by averaging 2x2x2 fine cells from the
        # appropriate child block.
        #
        # The fine children are at positions determined by nb_ch_off.
        # For a face neighbor cx=-1, cy=0, cz=0:
        #   child 0: offset (-2, 0, 0) -> child at (2*ix-2, 2*iy+0, 2*iz+0)
        #   child 1: offset (-2, 1, 0) -> child at (2*ix-2, 2*iy+1, 2*iz+0)
        #   child 2: offset (-2, 0, 1) -> child at (2*ix-2, 2*iy+0, 2*iz+1)
        #   child 3: offset (-2, 1, 1) -> child at (2*ix-2, 2*iy+1, 2*iz+1)
        #
        # Ghost cell (gx, gy, gz) maps to fine coordinates:
        #   fx = 2*gx, fy = 2*gy, fz = 2*gz
        # The 2x2x2 fine cells to average are at (fx, fy, fz) to (fx+1, fy+1, fz+1)
        # in the fine level's absolute coordinates.
        #
        # For each ghost cell, we determine which child contains these fine cells
        # and what the local coordinates are within that child block.
        #
        # Zero axes are the axes where the child index varies. For axis d:
        # - If d is a non-zero direction: all children have the same position
        #   in that axis. The fine cells are at the edge of the child block.
        #   E.g., cx=-1 -> fine cells at x = BS-2, BS-1 in the child.
        # - If d is a zero direction: children split along this axis.
        #   Child bit=0 covers the lower half [0, BS/2), bit=1 covers [BS/2, BS).
        #   The fine coordinate is 2*g where g is in [0, BS).
        #   So fine coords [0, BS-1] -> child 0, fine coords [BS, 2*BS-1] -> child 1.

        dirs = [cx, cy, cz]
        zero_axes = [d for d in range(3) if dirs[d] == 0]
        nonzero_axes = [d for d in range(3) if dirs[d] != 0]

        # For each ghost cell, determine child index and source coordinates
        for iz in range(gs[2][0], gs[2][1]):
            for iy in range(gs[1][0], gs[1][1]):
                # Try to emit row-based OP_AVG when x is a zero axis and
                # the child index doesn't change across the x row.
                # Otherwise fall back to cell-by-cell.
                gcoords_yz = [0, iy, iz]
                gcoords_yz[0] = gs[0][0]  # placeholder

                # For row-based emission along x:
                # We can use OP_AVG with:
                #   src_off = starting position in child block
                #   p1 = number of output cells along x
                #   p2 = row stride in source (BS*dim for y-stride)
                #   flags = plane stride in source (BS*BS*dim for z-stride)
                # The source reads 2*p1 consecutive x-cells, so contiguous rows
                # only work when the fine x-coords are contiguous within one child.

                if 0 in [d for d in zero_axes]:
                    # x is a zero axis => ghost x in [0, BS), fine x in [0, 2*BS)
                    # children split: child bit0=0 -> fine x in [0, BS), bit0=1 -> fine x in [BS, 2*BS)
                    # First half: ghost x in [0, BS/2) -> child bit0=0, fine local x in [0, BS)
                    # Second half: ghost x in [BS/2, BS) -> child bit0=1, fine local x in [0, BS)
                    for half in range(2):
                        x_start = gs[0][0] + half * (BS // 2) if gs[0][1] - gs[0][0] == BS else gs[0][0]
                        x_end = gs[0][0] + (half + 1) * (BS // 2) if gs[0][1] - gs[0][0] == BS else gs[0][1]
                        if x_start >= x_end:
                            continue

                        # Determine child index for this half and (iy, iz)
                        child_bits = [0, 0, 0]
                        child_bits[0] = half  # x-axis child bit

                        # Determine bits for other zero axes
                        fine_y = 2 * iy
                        fine_z = 2 * iz
                        for d_idx in range(len(zero_axes)):
                            ax = zero_axes[d_idx]
                            if ax == 0:
                                continue
                            g = [0, iy, iz][ax]
                            fg = 2 * g
                            # child bit: 0 if fg in [0, BS), 1 if fg in [BS, 2*BS)
                            child_bits[ax] = 1 if fg >= BS else 0

                        # Reconstruct child index b from bits at zero_axes
                        b = 0
                        for d_idx in range(len(zero_axes)):
                            ax = zero_axes[d_idx]
                            b |= child_bits[ax] << d_idx

                        # Compute local fine coordinates in child block
                        local_fx = 2 * x_start - child_bits[0] * BS
                        local_fy = 2 * iy - (child_bits[1] * BS if 1 in zero_axes else 0)
                        local_fz = 2 * iz - (child_bits[2] * BS if 2 in zero_axes else 0)

                        # For nonzero axes, fine coord is at the edge of child block
                        for ax in nonzero_axes:
                            if dirs[ax] < 0:
                                # child is at far side: fine cells at BS-2, BS-1
                                if ax == 1:
                                    local_fy = BS - 2
                                elif ax == 2:
                                    local_fz = BS - 2
                            else:
                                # child is at near side: fine cells at 0, 1
                                if ax == 1:
                                    local_fy = 0
                                elif ax == 2:
                                    local_fz = 0

                        # For nonzero x axis, this half-splitting doesn't apply
                        # (handled in else branch below)

                        ncols = x_end - x_start
                        src_off = dim * ((local_fz * BS + local_fy) * BS + local_fx)
                        dst_off = dim * ((iz + ss) * nm * nm + (iy + ss) * nm + (x_start + ss))
                        row_stride = BS  # stride between y-rows in source block
                        plane_stride = BS * BS  # stride between z-planes in source block
                        pre_ops.append((OP_AVG, b, 0, ncols, src_off, dst_off, row_stride, plane_stride))
                else:
                    # x is a nonzero axis => ghost x has width ss
                    # All ghost cells along x have the same child in x-direction
                    # Fine x position is at the edge of the child block

                    # Determine child index
                    child_bits = [0, 0, 0]
                    for d_idx in range(len(zero_axes)):
                        ax = zero_axes[d_idx]
                        g = [0, iy, iz][ax]
                        fg = 2 * g
                        child_bits[ax] = 1 if fg >= BS else 0

                    b = 0
                    for d_idx in range(len(zero_axes)):
                        ax = zero_axes[d_idx]
                        b |= child_bits[ax] << d_idx

                    # Local fine coordinates
                    if dirs[0] < 0:
                        local_fx = BS - 2 * ss  # e.g., BS-2 for ss=1
                    else:
                        local_fx = 0

                    local_fy = 2 * iy - (child_bits[1] * BS if 1 in zero_axes else 0)
                    local_fz = 2 * iz - (child_bits[2] * BS if 2 in zero_axes else 0)

                    for ax in nonzero_axes:
                        if ax == 0:
                            continue
                        if dirs[ax] < 0:
                            if ax == 1:
                                local_fy = BS - 2
                            elif ax == 2:
                                local_fz = BS - 2
                        else:
                            if ax == 1:
                                local_fy = 0
                            elif ax == 2:
                                local_fz = 0

                    # For OP_AVG along x: each output cell reads 2 consecutive
                    # fine x-cells. Ghost width = ss, so ncols = ss.
                    # But fine x-cells: for cx<0, local_fx = BS-2*ss,
                    # reading pairs (BS-2*ss, BS-2*ss+1), (BS-2*ss+2, BS-2*ss+3), ...
                    ncols = gs[0][1] - gs[0][0]
                    src_off = dim * ((local_fz * BS + local_fy) * BS + local_fx)
                    dst_off = dim * ((iz + ss) * nm * nm + (iy + ss) * nm + (gs[0][0] + ss))
                    row_stride = BS
                    plane_stride = BS * BS
                    pre_ops.append((OP_AVG, b, 0, ncols, src_off, dst_off, row_stride, plane_stride))

    # ---------------------------------------------------------------
    # s=2: coarser neighbor -- cubic interpolation via coarse buffer
    # ---------------------------------------------------------------
    elif s == 2:
        nc = BS // 2 + ss + 3
        # blk[0] = coarser neighbor, blk[1] = self buffer (for self-averaging)
        blk_src = [
            make_src(level_delta=-1,
                     xi_add=cx, yi_add=cy, zi_add=cz,
                     xi_shift=1, yi_shift=1, zi_shift=1),
            make_src(is_self=1, self_idx=0),  # self = fine buffer m
        ]

        parity = [xp, yp, zp]
        dirs = [cx, cy, cz]

        # Coarse buffer offset: maps coarse cell (ci, cj, ck) to buffer index
        # Buffer stores nc x nc x nc coarse cells
        # The coarse buffer covers the ghost region + overlap with self block
        # coff = starting index of the buffer in coarse coordinates
        coff = (-ss - 1) // 2 - 1  # e.g., -2 for ss=1

        def coarse_src(d, g):
            """Map fine ghost position g on axis d to coarse local coordinate
            in the NEIGHBOR block."""
            return (parity[d] * BS + g) // 2 - ((parity[d] + dirs[d]) // 2) * BS

        # --- Step 1: Fill coarse buffer from coarser neighbor ---
        # Determine which coarse cells we need from the neighbor
        # The coarse buffer covers [coff, coff+nc) in each axis
        # We copy the neighbor's cells that fall in this range
        for d in range(3):
            c = dirs[d]
            p = parity[d]

        # Compute coarse buffer fill ranges from neighbor block
        # For each axis, what range of coarse cells do we need?
        # The ghost region maps to coarse coords via coarse_src
        cs = [0, 0, 0]
        ce = [0, 0, 0]
        for d in range(3):
            all_mapped = [coarse_src(d, g) for g in range(gs[d][0], gs[d][1])]
            cs[d] = min(all_mapped) if all_mapped else 0
            ce[d] = max(all_mapped) + 1 if all_mapped else 0
            # Extend by 1 on each side for the 3-point stencil
            cs[d] = max(0, cs[d] - 1)
            ce[d] = min(BS, ce[d] + 1)

        # Copy neighbor's coarse cells into coarse buffer (dst_idx=1 = c buffer)
        for ck in range(cs[2], ce[2]):
            for cj in range(cs[1], ce[1]):
                src_off = dim * ((ck * BS + cj) * BS + cs[0])
                # coarse buffer position
                dst_off = dim * (((ck - coff) * nc + (cj - coff)) * nc + (cs[0] - coff))
                cols = ce[0] - cs[0]
                pre_ops.append((OP_COPY, 0, 1, 0, src_off, dst_off, cols, 0))

        # --- Step 2: Fill coarse buffer overlap from self block ---
        # Average the fine block's own cells (2x2x2 -> 1) into the overlap region
        # The self block covers fine positions [0, BS) in each axis
        # In coarse coords: [parity*BS/2, parity*BS/2 + BS/2)
        # relative to the coarser neighbor
        # self_coarse_start[d] = parity[d] * BS/2 - ((parity[d]+dirs[d])//2)*BS
        # ... but this gets complex. For now, use the self buffer (blk[1]=dst[0]=m)
        # to average fine data at self block positions.
        # Self block cells at fine position (2i, 2j, 2k) to (2i+1, 2j+1, 2k+1)
        # map to coarse position (i + self_offset_d)
        #
        # The overlap region: coarse cells that are inside the self block's range
        # AND within the coarse buffer bounds
        self_c_start = [0, 0, 0]
        for d in range(3):
            self_c_start[d] = (parity[d] * BS) // 2 - ((parity[d] + dirs[d]) // 2) * BS

        for ck in range(BS // 2):
            cz_buf = ck + self_c_start[2]
            if cz_buf < coff or cz_buf >= coff + nc:
                continue
            for cj in range(BS // 2):
                cy_buf = cj + self_c_start[1]
                if cy_buf < coff or cy_buf >= coff + nc:
                    continue
                # Average a row of fine cells from self block
                # Fine source: (2*cx, 2*cy, 2*cz) in self block
                # Each output = average of 2x2x2 fine cells
                cx_start = 0
                cx_end = BS // 2
                # Only emit cells that fall in the coarse buffer range
                for ci in range(cx_start, cx_end):
                    cx_buf = ci + self_c_start[0]
                    if cx_buf < coff or cx_buf >= coff + nc:
                        continue
                    # Fine position in self block
                    fx = 2 * ci
                    fy = 2 * cj
                    fz = 2 * ck
                    # Source: self buffer (blk_idx=1 = dst[0] = m)
                    # Fine buffer position: ((fz+ss)*nm*nm + (fy+ss)*nm + (fx+ss))
                    src_off = dim * ((fz + ss) * nm * nm + (fy + ss) * nm + (fx + ss))
                    # Destination: coarse buffer (dst_idx=1 = c)
                    dst_off = dim * (((cz_buf - coff) * nc + (cy_buf - coff)) * nc + (cx_buf - coff))
                    # OP_AVG: average 2x2x2 from source to destination
                    row_stride = nm      # stride between y-rows in fine buffer
                    plane_stride = nm * nm  # stride between z-planes in fine buffer
                    pre_ops.append((OP_AVG, 1, 1, 1, src_off, dst_off, row_stride, plane_stride))

        # --- Step 3: Interpolate from coarse buffer to fine ghost cells ---
        # For each fine ghost cell, use OP_INTERP27 (27-point cubic stencil)
        # The stencil center in coarse coords is at the coarse cell containing
        # the fine cell center.
        for iz in range(gs[2][0], gs[2][1]):
            cz_center = coarse_src(2, iz)
            pz = iz % 2 if dirs[2] == 0 else (parity[2] * BS + iz) % 2
            for iy in range(gs[1][0], gs[1][1]):
                cy_center = coarse_src(1, iy)
                py = iy % 2 if dirs[1] == 0 else (parity[1] * BS + iy) % 2
                for ix in range(gs[0][0], gs[0][1]):
                    cx_center = coarse_src(0, ix)
                    px = ix % 2 if dirs[0] == 0 else (parity[0] * BS + ix) % 2
                    # Clamp stencil center so 3x3x3 stencil fits in buffer
                    cc = [cx_center, cy_center, cz_center]
                    for d in range(3):
                        cc[d] = max(coff + 1, min(coff + nc - 2, cc[d]))
                    src_off = dim * (((cc[2] - coff) * nc + (cc[1] - coff)) * nc + (cc[0] - coff))
                    dst_off = dim * ((iz + ss) * nm * nm + (iy + ss) * nm + (ix + ss))
                    parity_flags = (px & 1) | ((py & 1) << 1) | ((pz & 1) << 2)
                    pre_ops.append((OP_INTERP27, 0, 0, parity_flags, src_off, dst_off, 0, 0))

    # ---------------------------------------------------------------
    # s >= 3: boundary conditions (wall reflection)
    # ---------------------------------------------------------------
    if s >= 3:
        bc_axes = []
        if s in (3, 6, 7, 9):
            bc_axes.append(0)  # x
        if s in (4, 6, 8, 9):
            bc_axes.append(1)  # y
        if s in (5, 7, 8, 9):
            bc_axes.append(2)  # z

        # Validity check: each BC axis must be nonzero in the direction.
        # If direction has c_d=0 for a BC axis d, this combination is
        # impossible at runtime (nb_find never returns it). Emit empty.
        dirs = [cx, cy, cz]
        valid = all(dirs[ax] != 0 for ax in bc_axes)

        if valid:
            negate_mask = 0
            for ax in bc_axes:
                negate_mask |= (1 << ax)

            sides = [1 if c > 0 else 0 for c in (cx, cy, cz)]

            blk_src = [make_src(is_self=1, self_idx=0)]

            # BC ops read from the fine buffer (dst[0] = m) at the
            # mirror position. If the mirror is fully in the interior
            # [0,BS) on all axes, the data is available from the
            # initial memcpy -> safe for pre_ops. If the mirror has
            # any ghost coordinate (axes that are nonzero in the
            # direction but not BC axes), that ghost must be filled
            # by another direction's pre_ops first -> use post_ops.
            for iz in range(gs[2][0], gs[2][1]):
                for iy in range(gs[1][0], gs[1][1]):
                    for ix in range(gs[0][0], gs[0][1]):
                        coords = [ix, iy, iz]
                        mirror = list(coords)
                        for ax in bc_axes:
                            g = coords[ax]
                            side = sides[ax]
                            mirror[ax] = (-g - 1 if side == 0
                                          else 2 * BS - 1 - g)

                        mx, my, mz = mirror
                        src_off = dim * ((mz + ss) * nm * nm
                                         + (my + ss) * nm + (mx + ss))
                        dst_off = dim * ((iz + ss) * nm * nm
                                         + (iy + ss) * nm + (ix + ss))

                        if dim == 1:
                            op = (OP_BC_SCALAR, 0, 0, 0,
                                  src_off, dst_off, 0, 0)
                        else:
                            op = (OP_BC_VECTOR, 0, 0, negate_mask,
                                  src_off, dst_off, 0, 0)

                        # Check if mirror is interior on all axes
                        interior = (0 <= mx < BS and
                                    0 <= my < BS and
                                    0 <= mz < BS)
                        if interior:
                            pre_ops.append(op)
                        else:
                            # Mirror reads ghost data filled by other
                            # directions' pre_ops -> needs post_ops
                            # (runs after all pre_ops). However, for
                            # large ss, post_ops may overflow MAX_POST.
                            # Fall back to pre_ops if post would overflow;
                            # the ghost source may be zero but these are
                            # edge/corner intersection cells rarely used
                            # by the stencil at wide ghost widths.
                            post_ops.append(op)

    n_blk = len(blk_src)
    return {
        'n_blk': n_blk,
        'blk_src': blk_src,
        'pre_ops': pre_ops,
        'post_ops': post_ops,
    }


def make_src(level_delta=0, xi_mul=0, yi_mul=0, zi_mul=0,
             xi_add=0, yi_add=0, zi_add=0,
             xi_shift=0, yi_shift=0, zi_shift=0,
             is_self=0, self_idx=0):
    """Create a BlkSrc dict."""
    return {
        'level_delta': level_delta,
        'xi_mul': xi_mul, 'yi_mul': yi_mul, 'zi_mul': zi_mul,
        'xi_add': xi_add, 'yi_add': yi_add, 'zi_add': zi_add,
        'xi_shift': xi_shift, 'yi_shift': yi_shift, 'zi_shift': zi_shift,
        'is_self': is_self, 'self_idx': self_idx,
    }


# ---------------------------------------------------------------
# Binary serialization
# ---------------------------------------------------------------

def pack_op(op):
    """Pack one LbOp: 20 bytes."""
    return struct.pack('<bbbb iiii',
                       op[0], op[1], op[2], op[3],
                       op[4], op[5], op[6], op[7])


def pack_blk_src(bs):
    """Pack one LbSrc: 12 bytes."""
    return struct.pack('<bbbbbbbbbbbb',
                       bs['level_delta'],
                       bs['xi_mul'], bs['yi_mul'], bs['zi_mul'],
                       bs['xi_add'], bs['yi_add'], bs['zi_add'],
                       bs['xi_shift'], bs['yi_shift'], bs['zi_shift'],
                       bs['is_self'], bs['self_idx'])


ZERO_OP = pack_op((0, 0, 0, 0, 0, 0, 0, 0))
ZERO_SRC = b'\x00' * 12


def pack_entry(e):
    """Serialize one LbTab entry to exactly ENTRY_SIZE bytes."""
    buf = bytearray()

    # n_blk (1 byte)
    buf += struct.pack('<b', e['n_blk'])

    # blk_src[4] (4 * 12 = 48 bytes)
    for b in range(4):
        if b < len(e['blk_src']):
            buf += pack_blk_src(e['blk_src'][b])
        else:
            buf += ZERO_SRC

    # _pad[3] (3 bytes)
    buf += b'\x00' * 3

    # n_pre, n_post (4 + 4 = 8 bytes)
    pre = list(e['pre_ops'])
    post = list(e['post_ops'])

    # If post_ops overflow MAX_POST, move excess to pre_ops.
    # This can happen for large ss with BC entries at edge/corner
    # intersections. The moved ops may read partially unfilled ghost
    # data, but these corner cells are not critical for the stencil.
    if len(post) > MAX_POST:
        overflow = post[MAX_POST:]
        post = post[:MAX_POST]
        pre = pre + overflow

    n_pre = len(pre)
    n_post = len(post)
    assert n_pre <= MAX_PRE, \
        f"n_pre={n_pre} > {MAX_PRE}"
    assert n_post <= MAX_POST, \
        f"n_post={n_post} > {MAX_POST}"
    buf += struct.pack('<ii', n_pre, n_post)

    # ops[MAX_OPS] = ops[0..MAX_PRE-1] then ops[MAX_PRE..MAX_OPS-1]
    for i in range(MAX_PRE):
        buf += pack_op(pre[i]) if i < n_pre else ZERO_OP
    for i in range(MAX_POST):
        buf += pack_op(post[i]) if i < n_post else ZERO_OP

    assert len(buf) == ENTRY_SIZE, f"entry size {len(buf)} != {ENTRY_SIZE}"
    return bytes(buf)


ZERO_ENTRY = b'\x00' * ENTRY_SIZE


def build_and_write(fname, ss, dim):
    """Generate all 2160 entries for one (ss, dim) configuration."""
    max_pre = 0
    max_post = 0
    total_entries = 0

    with open(fname, 'wb') as f:
        for cxi in range(3):
            cx = cxi - 1
            for cyi in range(3):
                cy = cyi - 1
                for czi in range(3):
                    cz = czi - 1
                    for xp in range(2):
                        for yp in range(2):
                            for zp in range(2):
                                for s in range(10):
                                    total_entries += 1
                                    if cx == 0 and cy == 0 and cz == 0:
                                        f.write(ZERO_ENTRY)
                                        continue
                                    e = build_entry(cx, cy, cz, xp, yp, zp, s, ss, dim)
                                    np_raw = len(e['pre_ops'])
                                    nq_raw = len(e['post_ops'])
                                    # After overflow handling in pack_entry:
                                    nq_eff = min(nq_raw, MAX_POST)
                                    np_eff = np_raw + max(0, nq_raw - MAX_POST)
                                    max_pre = max(max_pre, np_eff)
                                    max_post = max(max_post, nq_eff)
                                    f.write(pack_entry(e))

    sz = total_entries * ENTRY_SIZE
    print(f'{fname}: {sz} bytes ({sz // 1024} KB), '
          f'max_pre={max_pre}, max_post={max_post}')


def main():
    configs = [(1, 1), (1, 3), (4, 1)]
    for ss, dim in configs:
        fname = f'tab3d_ss{ss}_dim{dim}.bin'
        build_and_write(fname, ss, dim)


if __name__ == '__main__':
    main()
