#!/usr/bin/env python3
"""
Extract 2D slice from 3D dump and plot density contours + mesh.
Reproduces Khokhlov Fig. 11 (XY plane at z=Lz).
Usage: slice.py [-z Z] [-l level] dump1 dump2 ...
"""
import numpy as np
import sys
import os
import re
import xml.etree.ElementTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.expanduser('~/cudaAmrIsoSurfaceExtraction'))
import amriso

BS = 8

def load_and_slice(path, z_cut):
    path = re.sub(r'\.(xdmf2|xyz\.raw)$', '', path)
    root = xml.etree.ElementTree.parse(path + '.xdmf2')
    time = float(root.find('.//Time').get('Value'))
    info = root.find(".//Information[@Name='Step']")
    step = int(info.get('Value')) if info is not None else 0

    xyz = np.fromfile(path + '.xyz.raw', dtype=np.float32).reshape(-1, 8, 3)
    rho = np.fromfile(path + '.rho.raw', dtype=np.float64)
    ncell = len(rho)

    # Find cells whose z-range includes z_cut
    z0 = xyz[:, 0, 2]  # vertex 0 z
    z1 = xyz[:, 4, 2]  # vertex 4 z (top face)
    mask = (z0 <= z_cut + 1e-10) & (z_cut < z1 + 1e-10)

    sel_xyz = xyz[mask]
    sel_rho = rho[mask]

    # For the slice, use x,y from vertices 0 and 2
    x0 = sel_xyz[:, 0, 0]
    y0 = sel_xyz[:, 0, 1]
    x1 = sel_xyz[:, 6, 0]  # vertex 6 = (x1,y1,z1)
    y1 = sel_xyz[:, 6, 1]

    # Mirror across y=Ly symmetry plane to show full tube cross-section
    Ly = 0.25
    x0m = sel_xyz[:, 0, 0]
    y0m = 2*Ly - sel_xyz[:, 6, 1]  # mirror y
    x1m = sel_xyz[:, 6, 0]
    y1m = 2*Ly - sel_xyz[:, 0, 1]
    x0_all = np.concatenate([x0, x0m])
    y0_all = np.concatenate([y0, y0m])
    x1_all = np.concatenate([x1, x1m])
    y1_all = np.concatenate([y1, y1m])
    rho_all = np.concatenate([sel_rho, sel_rho])

    return path, time, step, x0_all, y0_all, x1_all, y1_all, rho_all


args = sys.argv[1:]
z_cut = None
while args and args[0].startswith('-'):
    if args[0] == '-z':
        z_cut = float(args[1]); args = args[2:]
    else:
        args = args[1:]

paths = sorted(args)
if not paths:
    sys.exit('Usage: slice.py [-z Z] dump1 dump2 ...')

def plot_isolines(ax, geo2d, rho2d, xmax, ymax, levels, color='k'):
    """Draw AMR-aware isolines using amriso.extract2d."""
    for lv in levels:
        try:
            xy, seg, attr = amriso.extract2d(geo2d, rho2d, rho2d, lv)
            if len(seg) == 0:
                continue
            c = color if isinstance(color, str) else color(lv)
            segs = [[(xy[s[0], 0], xy[s[0], 1]),
                     (xy[s[1], 0], xy[s[1], 1])]
                    for s in seg if s[0] < len(xy) and s[1] < len(xy)]
            if segs:
                ax.add_collection(LineCollection(segs, linewidths=0.5, colors=c))
        except (IndexError, ValueError):
            pass


for p in paths:
    base, t, step, x0, y0, x1, y1, rho = load_and_slice(p, z_cut if z_cut is not None else 0.24)
    h_all = x1 - x0
    xmax = float(x1.max())
    ymax = float(y1.max())
    label = f'step {step}' if step else f't={t:.3e}'
    levels = np.arange(0.05, 3.0, 0.05)

    # Build 2D quad geometry for pyiso2d: float32[ncell][4][2]
    # Vertex order for pyiso2d (marching squares): BL, BR, TR, TL
    ncell = len(rho)
    geo2d = np.zeros((ncell, 4, 2), dtype=np.float32)
    geo2d[:, 0, 0] = x0;  geo2d[:, 0, 1] = y0   # BL
    geo2d[:, 1, 0] = x1;  geo2d[:, 1, 1] = y0   # BR
    geo2d[:, 2, 0] = x1;  geo2d[:, 2, 1] = y1   # TR
    geo2d[:, 3, 0] = x0;  geo2d[:, 3, 1] = y1   # TL
    rho2d = rho.astype(np.float32)
    geo_flat = geo2d.reshape(-1).copy()

    # Split levels: bubble (rho < 1) in blue, shock (rho > 1) in red
    bubble_levels = levels[levels < 1.0]
    shock_levels = levels[levels >= 1.0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    # Top: density isolines colored by type
    plot_isolines(ax1, geo_flat, rho2d, xmax, ymax, bubble_levels, color='steelblue')
    plot_isolines(ax1, geo_flat, rho2d, xmax, ymax, shock_levels, color='orangered')
    ax1.set_xlim(0, xmax); ax1.set_ylim(0, ymax)
    ax1.set_aspect('equal')
    ax1.set_title(f'{label} — density')

    # Bottom: mesh + isolines
    blocks = set()
    for i in range(ncell):
        h = float(h_all[i])
        bh = BS * h
        bx = round(x0[i] / bh) * bh
        by = round(y0[i] / bh) * bh
        blocks.add((bx, by, bh * BS))
    segs = []
    for bx, by, bs in blocks:
        segs.append(((bx, by), (bx+bs, by)))
        segs.append(((bx+bs, by), (bx+bs, by+bs)))
        segs.append(((bx, by+bs), (bx+bs, by+bs)))
        segs.append(((bx, by), (bx, by+bs)))
    ax2.add_collection(LineCollection(segs, linewidths=0.2, colors='lightgray'))
    plot_isolines(ax2, geo_flat, rho2d, xmax, ymax, bubble_levels, color='steelblue')
    plot_isolines(ax2, geo_flat, rho2d, xmax, ymax, shock_levels, color='orangered')
    ax2.set_xlim(0, xmax); ax2.set_ylim(0, ymax)
    ax2.set_aspect('equal')
    ax2.set_title(f'{label} — mesh + contours')

    plt.tight_layout()
    out = base + '.slice.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    sys.stderr.write(f'{out}\n')
