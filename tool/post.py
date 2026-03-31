#!/usr/bin/env python3
import numpy as np
import sys
import os
import re
import xml.etree.ElementTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

for path in sys.argv[1:]:
    path = re.sub("[.]xdmf2$", "", path)
    path = re.sub("[.]xyz[.]raw$", "", path)
    png_path = path + ".png"
    xdmf_path = path + ".xdmf2"
    xyz_path = path + ".xyz.raw"
    rho_path = path + ".rho.raw"
    if not os.path.isfile(png_path):
        sys.stderr.write(f"post.py: {path}\n")
        root = xml.etree.ElementTree.parse(xdmf_path)
        time = root.find("Domain/Grid/Time").get("Value")
        xyz = np.memmap(xyz_path, "float32", "r").reshape(-1, 4, 2)
        rho = np.memmap(rho_path, "float64", "r")
        ncell = len(xyz)
        hmin = min(xyz[i, 2, 0] - xyz[i, 0, 0] for i in range(ncell))
        N = int(round(1.0 / hmin))
        grid = np.full((N, N), np.nan)
        for i in range(ncell):
            ix0 = int(round(xyz[i, 0, 0] * N))
            iy0 = int(round(xyz[i, 0, 1] * N))
            ix1 = int(round(xyz[i, 2, 0] * N))
            iy1 = int(round(xyz[i, 2, 1] * N))
            grid[iy0:iy1, ix0:ix1] = rho[i]
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        levels = np.arange(0.05, 7.5, 0.1)
        ax.contour(np.linspace(0, N, N), np.linspace(0, N, N),
                   grid, levels=levels, colors='k', linewidths=0.5)
        ax.set_xlim(0, N)
        ax.set_ylim(0, N)
        ax.set_aspect('equal')
        ax.set_title(f't = {float(time):.3e}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.tight_layout()
        plt.savefig(png_path, dpi=200, bbox_inches='tight', pad_inches=0.05)
        plt.close()
        sys.stderr.write(f"post.py: {png_path}\n")
