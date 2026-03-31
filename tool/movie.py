#!/usr/bin/env python3
"""
Create side-by-side mp4: density isolines (left) + AMR grid (right).
Uses pyiso2d for proper AMR-aware isoline extraction.
Usage: movie.py vel.00000000 vel.00000001 ...
"""
import sys
import os
import re
import subprocess
import shutil
import numpy as np
import xml.etree.ElementTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.expanduser('~/cudaAmrIsoSurfaceExtraction'))
import pyiso2d

BS = 8


def load_dump(path):
    path = re.sub(r'\.(xdmf2|xyz\.raw)$', '', path)
    root = xml.etree.ElementTree.parse(path + '.xdmf2')
    time = float(root.find('Domain/Grid/Time').get('Value'))
    info = root.find(".//Information[@Name='Step']")
    step = int(info.get('Value')) if info is not None else None
    geo = np.fromfile(path + '.xyz.raw', dtype=np.float32)
    rho = np.fromfile(path + '.rho.raw', dtype=np.float64).astype(np.float32)
    xyz = geo.reshape(-1, 4, 2)
    ncell = len(rho)
    hmin = min(xyz[i, 2, 0] - xyz[i, 0, 0] for i in range(ncell))
    N = int(round(1.0 / hmin))
    # reorder vertices for pyiso2d: ParaView order (BL,TL,TR,BR) -> marching squares (BL,BR,TR,TL)
    geo_iso = geo.reshape(-1, 4, 2).copy()
    geo_iso[:, [1, 3]] = geo_iso[:, [3, 1]]
    return path, time, step, N, geo_iso.reshape(-1), rho, xyz


def plot_isolines(ax, geo, rho, N, levels):
    for lv in levels:
        try:
            xy, seg, attr = pyiso2d.extract(geo, rho, rho, lv)
            if len(seg) == 0:
                continue
            segs = [[(xy[s[0], 0] * N, xy[s[0], 1] * N),
                     (xy[s[1], 0] * N, xy[s[1], 1] * N)] for s in seg
                    if s[0] < len(xy) and s[1] < len(xy)]
            if segs:
                lc = LineCollection(segs, linewidths=0.4, colors='k')
                ax.add_collection(lc)
        except (IndexError, ValueError):
            pass


def plot_blocks(ax, xyz, N):
    blocks = set()
    for i in range(len(xyz)):
        h = float(xyz[i, 2, 0] - xyz[i, 0, 0])
        bh = BS * h
        bx = round(xyz[i, 0, 0] / bh) * bh
        by = round(xyz[i, 0, 1] / bh) * bh
        blocks.add((round(bx * N), round(by * N), round(bh * N)))
    segs = []
    for bx, by, bs in blocks:
        segs.append(((bx, by), (bx + bs, by)))
        segs.append(((bx + bs, by), (bx + bs, by + bs)))
        segs.append(((bx, by + bs), (bx + bs, by + bs)))
        segs.append(((bx, by), (bx, by + bs)))
    lc = LineCollection(segs, linewidths=0.3, colors='k')
    ax.add_collection(lc)


paths = sorted(sys.argv[1:])
if not paths:
    sys.exit('Usage: movie.py vel.00000000 vel.00000001 ...')

tmpdir = '_movie_frames'
os.makedirs(tmpdir, exist_ok=True)

rho_levels = np.arange(0.05, 7.5, 0.1)

for idx, p in enumerate(paths):
    base, t, step, N, geo, rho, xyz = load_dump(p)
    label = f'step {step}' if step is not None else f't = {t:.3e}'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Left: density isolines via pyiso2d
    plot_isolines(ax1, geo, rho, N, rho_levels)
    ax1.set_xlim(0, N)
    ax1.set_ylim(0, N)
    ax1.set_aspect('equal')
    ax1.set_title(f'{label} — density')

    # Right: AMR blocks
    plot_blocks(ax2, xyz, N)
    nblk = len(set(
        (round(xyz[i, 0, 0] / (BS * (xyz[i, 2, 0] - xyz[i, 0, 0])) *
               BS * (xyz[i, 2, 0] - xyz[i, 0, 0]) * N),
         round(xyz[i, 0, 1] / (BS * (xyz[i, 2, 1] - xyz[i, 0, 1])) *
               BS * (xyz[i, 2, 1] - xyz[i, 0, 1]) * N))
        for i in range(len(xyz))
    ))
    ax2.set_xlim(0, N)
    ax2.set_ylim(0, N)
    ax2.set_aspect('equal')
    ax2.set_title(f'{label} — mesh')

    plt.tight_layout()
    out = os.path.join(tmpdir, f'frame_{idx:04d}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    sys.stderr.write(f'{out}\n')

mp4 = 'movie.mp4'
subprocess.run([
    'ffmpeg', '-y', '-framerate', '6', '-i', f'{tmpdir}/frame_%04d.png',
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
    '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
    mp4
], check=True, capture_output=True)
sys.stderr.write(f'saved {mp4}\n')
shutil.rmtree(tmpdir)
