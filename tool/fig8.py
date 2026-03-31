#!/usr/bin/env python3
"""
Generate Figure 8 from Khokhlov (1998) - Cylindrical strong point explosion.
Usage: fig8.py dump1 dump2 dump3 dump4 dump5
  Panels a-e: density contours at 5 timesteps
  Panel f: pressure contours at the last timestep (dump5)
"""
import numpy as np
import sys
import re
import xml.etree.ElementTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
    return time, step, N, rho_grid, pres_grid


paths = sys.argv[1:]
if len(paths) < 5:
    sys.exit('Usage: fig8.py dump1 dump2 dump3 dump4 dump5')

fig, axes = plt.subplots(3, 2, figsize=(10, 15))
labels = 'abcdef'
rho_levels = np.arange(0.05, 7.5, 0.1)

for idx in range(5):
    row, col = idx // 2, idx % 2
    ax = axes[row][col]
    t, step, N, rho_g, pres_g = load_dump(paths[idx])
    sys.stderr.write(f'panel {labels[idx]}: t={t:.3e} step={step} N={N}\n')
    x = np.linspace(0, N, N)
    ax.contour(x, x, rho_g, levels=rho_levels, colors='k', linewidths=0.4)
    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_aspect('equal')
    title = f'step {step}' if step is not None else f't = {t:.3e}'
    ax.set_title(title, fontsize=10)
    ax.text(0.97, 0.97, labels[idx], transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='right')

# Panel f: pressure from last dump
ax = axes[2][1]
t, step, N, rho_g, pres_g = load_dump(paths[4])
p_max = np.nanmax(pres_g)
pres_levels = np.arange(0, p_max + 1000, 1000)
if len(pres_levels) > 200:
    pres_levels = np.arange(0, p_max + p_max / 100, p_max / 100)
x = np.linspace(0, N, N)
ax.contour(x, x, pres_g, levels=pres_levels, colors='k', linewidths=0.4)
ax.set_xlim(0, N)
ax.set_ylim(0, N)
ax.set_aspect('equal')
title = f'step {step}' if step is not None else f't = {t:.3e}'
ax.set_title(f'{title} (pressure)', fontsize=10)
ax.text(0.97, 0.97, 'f', transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='right')

plt.tight_layout()
plt.savefig('fig8.png', dpi=200, bbox_inches='tight')
sys.stderr.write('fig8.py: saved fig8.png\n')
