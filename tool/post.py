#!/usr/bin/env python3
import numpy as np
import sys
import os
import re
import xml.etree.ElementTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches

plt.rcParams['image.cmap'] = 'viridis'
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
        xyz = np.memmap(xyz_path, "float32", "r")
        xyz = xyz.reshape(-1, 4, 2)
        ncell = len(xyz)
        rho = np.memmap(rho_path, "float64", "r")
        patches = []
        color = []
        for i in range(ncell):
            x = xyz[i, 0, 0]
            y = xyz[i, 0, 1]
            lx = xyz[i, 2, 0] - x
            ly = xyz[i, 2, 1] - y
            color.append(rho[i])
            patches.append(matplotlib.patches.Rectangle((x, y), lx, ly))
        p = matplotlib.collections.PatchCollection(patches,
                                                   edgecolor='black',
                                                   linewidth=0.1)
        p.set_array(color)
        p.set_clim(0, max(color))
        plt.gca().add_collection(p)
        plt.axis("scaled")
        plt.gca().set_xlim(0, 1)
        plt.gca().set_ylim(0, 1)
        plt.colorbar(p, label='density')
        plt.title(f't = {float(time):.4f}')
        plt.tight_layout()
        plt.savefig(png_path, dpi=400, bbox_inches='tight', pad_inches=0)
        plt.close()
        sys.stderr.write(f"post.py: {png_path}\n")
