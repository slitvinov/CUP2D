#!/usr/bin/env python3
"""Build AMR panels using amriso for isoline extraction:
  1) comparison_amr.png: paper (256) vs AMR isolines
  2) panel_amr.png: mesh + isolines for t=0.8 and t=1.2
"""
import amriso
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image

BS = 8

def load_dump(prefix):
    coords = np.fromfile(f"{prefix}.xyz.raw", dtype=np.float32)
    vort = np.fromfile(f"{prefix}.vort.raw", dtype=np.float64)
    ncells = vort.size
    coords = coords.reshape(ncells, 4, 2)
    return coords, vort.astype(np.float32)


def get_block_info(coords):
    ncells = len(coords)
    nblk = ncells // (BS * BS)
    blocks = []
    hs = set()
    for b in range(nblk):
        c0 = b * BS * BS
        ox = coords[c0, 0, 0]
        oy = coords[c0, 0, 1]
        h = coords[c0, 3, 0] - coords[c0, 0, 0]
        bw = BS * h
        blocks.append((float(ox), float(oy), float(h), float(bw)))
        hs.add(round(float(h), 10))
    return blocks, sorted(hs, reverse=True)


def gather_to_finest(coords, scalar):
    """Gather AMR data to finest uniform grid with bilinear interpolation."""
    blocks, hs = get_block_info(coords)
    hf = min(hs)
    Ng = round(1.0 / hf)
    # Place cell-center values onto a grid keyed by (x_center, y_center)
    nblk = len(blocks)
    vort = scalar.reshape(nblk, BS, BS)
    # Collect all cell centers and values
    cx_all, cy_all, val_all, h_all = [], [], [], []
    for b, (ox, oy, h, bw) in enumerate(blocks):
        for j in range(BS):
            for i in range(BS):
                cx_all.append(ox + (i + 0.5) * h)
                cy_all.append(oy + (j + 0.5) * h)
                val_all.append(vort[b, j, i])
                h_all.append(h)
    cx_all = np.array(cx_all)
    cy_all = np.array(cy_all)
    val_all = np.array(val_all, dtype=np.float32)
    # For fine cells: use scipy griddata for smooth interpolation
    from scipy.interpolate import griddata
    xf = np.linspace(hf * 0.5, 1.0 - hf * 0.5, Ng)
    yf = np.linspace(hf * 0.5, 1.0 - hf * 0.5, Ng)
    Xf, Yf = np.meshgrid(xf, yf)
    grid = griddata((cx_all, cy_all), val_all, (Xf, Yf), method='linear', fill_value=0.0)
    grid = grid.astype(np.float32)
    # Build uniform coords for amriso
    nc = Ng * Ng
    uc = np.zeros((nc, 4, 2), dtype=np.float32)
    for j in range(Ng):
        for i in range(Ng):
            c = j * Ng + i
            x0, y0 = i * hf, j * hf
            x1, y1 = x0 + hf, y0 + hf
            uc[c, 0] = [x0, y0]
            uc[c, 1] = [x0, y1]
            uc[c, 2] = [x1, y1]
            uc[c, 3] = [x1, y0]
    return uc, grid.ravel(), Ng


def plot_isolines(ax, coords, scalar, levels):
    """Plot isolines on gathered uniform grid (no block artifacts)."""
    uc, us, Ng = gather_to_finest(coords, scalar)
    segs_all = []
    for iso in levels:
        xy, seg, attr = amriso.extract2d(uc, us, us, iso)
        if len(seg) > 0:
            xy = np.array(xy)
            seg = np.array(seg)
            lines = xy[seg]
            segs_all.extend(lines.tolist())
    if segs_all:
        lc = LineCollection(segs_all, colors='k', linewidths=1.2)
        ax.add_collection(lc)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])


def plot_mesh(ax, blocks, hs):
    segs = []
    colors_list = []
    widths = []
    h_max = max(hs)
    h_min = min(hs)
    for ox, oy, h, bw in blocks:
        x0, y0, x1, y1 = ox, oy, ox + bw, oy + bw
        if h == h_max:
            lw, c = 0.2, '#cccccc'
        elif h == h_min:
            lw, c = 0.8, '#000000'
        else:
            lw, c = 0.5, '#666666'
        for s in [[(x0,y0),(x1,y0)], [(x1,y0),(x1,y1)], [(x1,y1),(x0,y1)], [(x0,y1),(x0,y0)]]:
            segs.append(s)
            colors_list.append(c)
            widths.append(lw)
    lc = LineCollection(segs, colors=colors_list, linewidths=widths)
    ax.add_collection(lc)


levels = np.arange(-36, 37, 6)
levels = levels[levels != 0]

run = "run_amr"
paper_label = 'C'

# Auto-detect dump indices closest to t=0.8 and t=1.2
import glob, xml.etree.ElementTree as ET
dumps = {}
dump_times = {}
for xf in sorted(glob.glob(f"{run}/vel.*.xdmf2")):
    idx = int(xf.split('.')[-2])
    tree = ET.parse(xf)
    t = float(tree.find('.//{http://www.w3.org/2001/XMLSchema}' if False else './/Time').get('Value'))
    dump_times[idx] = t

# Find closest to 0.8 and 1.2
targets = [0.8, 1.2]
best = {}
for tgt in targets:
    bi, bd = None, 1e9
    for idx, t in dump_times.items():
        if abs(t - tgt) < bd:
            bd = abs(t - tgt)
            bi = idx
    if bi is not None and bd < 0.15:
        best[tgt] = bi

for tgt, idx in best.items():
    prefix = f"{run}/vel.{idx:08d}"
    try:
        dumps[idx] = load_dump(prefix)
    except Exception as e:
        print(f"Warning: {e}")

time_data = [(best.get(0.8, -1), 't=0.8'), (best.get(1.2, -1), 't=1.2')]
print(f"Using dumps: {best}")

if not dumps:
    print("No data")
    exit(1)

blocks0, hs0 = get_block_info(list(dumps.values())[0][0])
N_min = round(1.0 / max(hs0))
N_max = round(1.0 / min(hs0))

for idx, (coords, scalar) in dumps.items():
    blocks, hs = get_block_info(coords)
    Nmin = round(1.0 / max(hs))
    Nmax = round(1.0 / min(hs))
    nblk = len(blocks)
    print(f"dump {idx}: {nblk} blocks, levels {Nmin}-{Nmax}, vort [{scalar.min():.1f}, {scalar.max():.1f}]")

# --- comparison_amr.png ---
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for col_pair, (dump_idx, tlabel) in enumerate(time_data):
    pcol = col_pair * 2
    ccol = col_pair * 2 + 1
    img_path = f"/tmp/brown_imgs/fig{dump_idx}_{paper_label}.png"
    try:
        img = Image.open(img_path)
        axes[pcol].imshow(img, cmap='gray', aspect='equal')
    except:
        pass
    axes[pcol].set_xticks([])
    axes[pcol].set_yticks([])
    axes[pcol].set_title(f"Paper 256 {tlabel}", fontsize=11)

    if dump_idx in dumps:
        coords, scalar = dumps[dump_idx]
        plot_isolines(axes[ccol], coords, scalar, levels)
    axes[ccol].set_title(f"AMR {tlabel}", fontsize=11)

fig.suptitle(f"AMR levels {N_min}-{N_max}", fontsize=11)
fig.tight_layout()
fig.savefig("comparison_amr.png", dpi=200, bbox_inches='tight')
print("Saved comparison_amr.png")

# --- panel_amr.png ---
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 12))
for col, (dump_idx, tlabel) in enumerate(time_data):
    if dump_idx not in dumps:
        continue
    coords, scalar = dumps[dump_idx]
    blocks, hs = get_block_info(coords)
    nblk = len(blocks)

    plot_isolines(axes2[0, col], coords, scalar, levels)
    axes2[0, col].set_title(f"Vorticity {tlabel} ({nblk} blocks)", fontsize=11)

    plot_mesh(axes2[1, col], blocks, hs)
    plot_isolines(axes2[1, col], coords, scalar, levels)
    axes2[1, col].set_title(f"Mesh + vorticity {tlabel}", fontsize=11)

fig2.suptitle(f"AMR: levels {N_min}-{N_max}", fontsize=12)
fig2.tight_layout()
fig2.savefig("panel_amr.png", dpi=200, bbox_inches='tight')
print("Saved panel_amr.png")
