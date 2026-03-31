#!/usr/bin/env python3
"""
Generate individual panels: density contours and AMR grid.
Usage: panel.py [-grid] [-pres] dump_path
  -grid: also generate grid plot showing AMR mesh
  -pres: plot pressure contours instead of density
"""
import numpy as np
import sys
import re
import xml.etree.ElementTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def load_dump(path):
    path = re.sub(r'\.(xdmf2|xyz\.raw)$', '', path)
    root = xml.etree.ElementTree.parse(path + '.xdmf2')
    time = float(root.find('Domain/Grid/Time').get('Value'))
    info = root.find(".//Information[@Name='Step']")
    step = int(info.get('Value')) if info is not None else None
    xyz = np.memmap(path + '.xyz.raw', 'float32', 'r').reshape(-1, 4, 2)
    rho = np.memmap(path + '.rho.raw', 'float64', 'r')
    pres = np.memmap(path + '.pres.raw', 'float64', 'r')
    ncell = len(xyz)
    hmin = min(xyz[i, 2, 0] - xyz[i, 0, 0] for i in range(ncell))
    N = int(round(1.0 / hmin))
    rho_grid = np.full((N, N), np.nan)
    pres_grid = np.full((N, N), np.nan)
    for i in range(ncell):
        ix0 = int(round(xyz[i, 0, 0] * N))
        iy0 = int(round(xyz[i, 0, 1] * N))
        ix1 = int(round(xyz[i, 2, 0] * N))
        iy1 = int(round(xyz[i, 2, 1] * N))
        rho_grid[iy0:iy1, ix0:ix1] = rho[i]
        pres_grid[iy0:iy1, ix0:ix1] = pres[i]
    return path, time, step, N, xyz, rho_grid, pres_grid


def plot_contour(ax, N, grid, levels, title):
    x = np.linspace(0, N, N)
    ax.contour(x, x, grid, levels=levels, colors='k', linewidths=0.4)
    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)


def plot_grid(ax, N, xyz):
    """Draw AMR cell edges."""
    segs = []
    for i in range(len(xyz)):
        x0, y0 = xyz[i, 0] * N
        x1, y1 = xyz[i, 2] * N
        segs.append(((x0, y0), (x1, y0)))
        segs.append(((x1, y0), (x1, y1)))
        segs.append(((x0, y1), (x1, y1)))
        segs.append(((x0, y0), (x0, y1)))
    lc = LineCollection(segs, linewidths=0.15, colors='k')
    ax.add_collection(lc)
    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_aspect('equal')


args = sys.argv[1:]
do_grid = '-grid' in args
do_pres = '-pres' in args
paths = [a for a in args if not a.startswith('-')]

rho_levels = np.arange(0.05, 7.5, 0.1)

for p in paths:
    base, t, step, N, xyz, rho_g, pres_g = load_dump(p)
    label = f'step {step}' if step is not None else f't = {t:.3e}'
    sys.stderr.write(f'{base}: {label} N={N}\n')

    # density contour
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plot_contour(ax, N, rho_g, rho_levels, label)
    out = base + '.rho.png'
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    sys.stderr.write(f'  -> {out}\n')

    if do_pres:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        p_max = np.nanmax(pres_g)
        p_levels = np.arange(0, p_max + 1000, 1000)
        if len(p_levels) > 200:
            p_levels = np.arange(0, p_max + p_max / 100, p_max / 100)
        plot_contour(ax, N, pres_g, p_levels, f'{label} (pressure)')
        out = base + '.pres.png'
        plt.tight_layout()
        plt.savefig(out, dpi=200, bbox_inches='tight', pad_inches=0.05)
        plt.close()
        sys.stderr.write(f'  -> {out}\n')

    if do_grid:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        plot_contour(ax, N, rho_g, rho_levels, f'{label} (mesh)')
        plot_grid(ax, N, xyz)
        out = base + '.grid.png'
        plt.tight_layout()
        plt.savefig(out, dpi=200, bbox_inches='tight', pad_inches=0.05)
        plt.close()
        sys.stderr.write(f'  -> {out}\n')
