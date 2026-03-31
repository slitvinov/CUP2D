#!/usr/bin/env python3
"""
Plot AMR block boundaries (not individual cells).
Usage: grid.py dump1 dump2 ...
"""
import numpy as np
import sys
import re
import xml.etree.ElementTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection, LineCollection

BS = 8

for path in sys.argv[1:]:
    path = re.sub(r'\.(xdmf2|xyz\.raw)$', '', path)
    root = xml.etree.ElementTree.parse(path + '.xdmf2')
    time = float(root.find('Domain/Grid/Time').get('Value'))
    info = root.find(".//Information[@Name='Step']")
    step = int(info.get('Value')) if info is not None else None
    xyz = np.memmap(path + '.xyz.raw', 'float32', 'r').reshape(-1, 4, 2)
    ncell = len(xyz)
    hmin = min(xyz[i, 2, 0] - xyz[i, 0, 0] for i in range(ncell))
    N = int(round(1.0 / hmin))

    # collect unique blocks
    blocks = set()
    for i in range(ncell):
        h = float(xyz[i, 2, 0] - xyz[i, 0, 0])
        bh = BS * h
        bx = round(xyz[i, 0, 0] / bh) * bh
        by = round(xyz[i, 0, 1] / bh) * bh
        blocks.add((round(bx * N), round(by * N), round(bh * N)))

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    segs = []
    for bx, by, bs in blocks:
        segs.append(((bx, by), (bx + bs, by)))
        segs.append(((bx + bs, by), (bx + bs, by + bs)))
        segs.append(((bx, by + bs), (bx + bs, by + bs)))
        segs.append(((bx, by), (bx, by + bs)))
    lc = LineCollection(segs, linewidths=0.3, colors='k')
    ax.add_collection(lc)
    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_aspect('equal')
    label = f'step {step}' if step is not None else f't = {time:.3e}'
    ax.set_title(label)
    out = path + '.blk.png'
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    sys.stderr.write(f'{out}\n')
