#!/usr/bin/env python3
"""
Create mp4 showing 3D AMR block wireframes, mirrored across symmetry planes.
Usage: movie3d_grid.py vel.00000000 vel.00000001 ...
"""
import numpy as np
import sys
import os
import re
import subprocess
import xml.etree.ElementTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

BS = 8
Ly, Lz = 0.25, 0.25


def load_blocks(path):
    path = re.sub(r'\.(xdmf2|xyz\.raw)$', '', path)
    root = xml.etree.ElementTree.parse(path + '.xdmf2')
    time = float(root.find('.//Time').get('Value'))
    info = root.find(".//Information[@Name='Step']")
    step = int(info.get('Value')) if info is not None else 0
    xyz = np.fromfile(path + '.xyz.raw', dtype=np.float32).reshape(-1, 8, 3)
    ncell = len(xyz)
    # Extract unique blocks: group cells by block origin and size
    blocks = set()
    for i in range(ncell):
        h = float(xyz[i, 6, 0] - xyz[i, 0, 0])
        bh = BS * h
        bx = np.floor(xyz[i, 0, 0] / bh + 1e-9) * bh
        by = np.floor(xyz[i, 0, 1] / bh + 1e-9) * bh
        bz = np.floor(xyz[i, 0, 2] / bh + 1e-9) * bh
        blocks.add((float(bx), float(by), float(bz), bh))
    return step, time, blocks


def mirror_blocks(blocks):
    """Mirror blocks across y=Ly and z=Lz."""
    out = set()
    for bx, by, bz, bh in blocks:
        out.add((bx, by, bz, bh))
        out.add((bx, 2*Ly - by - bh, bz, bh))
        out.add((bx, by, 2*Lz - bz - bh, bh))
        out.add((bx, 2*Ly - by - bh, 2*Lz - bz - bh, bh))
    return out


def cube_edges(x0, y0, z0, s):
    """12 edges of a cube at (x0,y0,z0) with side s."""
    x1, y1, z1 = x0+s, y0+s, z0+s
    return [
        ((x0,y0,z0),(x1,y0,z0)), ((x0,y1,z0),(x1,y1,z0)),
        ((x0,y0,z1),(x1,y0,z1)), ((x0,y1,z1),(x1,y1,z1)),
        ((x0,y0,z0),(x0,y1,z0)), ((x1,y0,z0),(x1,y1,z0)),
        ((x0,y0,z1),(x0,y1,z1)), ((x1,y0,z1),(x1,y1,z1)),
        ((x0,y0,z0),(x0,y0,z1)), ((x1,y0,z0),(x1,y0,z1)),
        ((x0,y1,z0),(x0,y1,z1)), ((x1,y1,z0),(x1,y1,z1)),
    ]


paths = sorted(sys.argv[1:])
if not paths:
    sys.exit('Usage: movie3d_grid.py vel.00000000 ...')

tmpdir = '_movie3d_grid'
os.makedirs(tmpdir, exist_ok=True)

for idx, p in enumerate(paths):
    step, time, blocks = load_blocks(p)
    blocks = mirror_blocks(blocks)
    label = f'step {step}' if step else f't={time:.3e}'

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Color by level (block size)
    sizes = sorted(set(bh for _, _, _, bh in blocks), reverse=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sizes)))
    size_color = {s: colors[i] for i, s in enumerate(sizes)}

    for bx, by, bz, bh in blocks:
        edges = cube_edges(bx, by, bz, bh)
        lw = 0.3 if bh > sizes[-1] else 0.6
        lc = Line3DCollection(edges, linewidths=lw, colors=[size_color[bh]]*12, alpha=0.6)
        ax.add_collection3d(lc)

    Lx = 1.0
    ax.set_xlim(0, Lx); ax.set_ylim(0, 2*Ly); ax.set_zlim(0, 2*Lz)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_box_aspect([Lx, 2*Ly, 2*Lz])
    ax.set_title(f'{label} ({len(blocks)} blocks)')
    ax.view_init(elev=20, azim=-60)
    out = os.path.join(tmpdir, f'frame_{idx:04d}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    sys.stderr.write(f'{out}\n')

mp4 = 'movie3d_grid.mp4'
subprocess.run([
    'ffmpeg', '-y', '-framerate', '6', '-i', f'{tmpdir}/frame_%04d.png',
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
    '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
    mp4
], check=True, capture_output=True)
sys.stderr.write(f'saved {mp4}\n')
