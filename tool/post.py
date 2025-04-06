#!/usr/bin/env python3
import numpy as np
import sys
import os
import re
import xml.etree.ElementTree
import matplotlib.pyplot as plt
import matplotlib.patches
import statistics

plt.rcParams['image.cmap'] = 'RdBu'
for path in sys.argv[1:]:
    path = re.sub("[.]xdmf2$", "", path)
    path = re.sub("[.]chi\.raw$", "", path)
    path = re.sub("[.]xyz\.raw$", "", path)
    png_path = path + ".png"
    xdmf_path = path + ".xdmf2"
    xyz_path = path + ".xyz.raw"
    chi_path = path + ".chi.raw"
    vel_path = path + ".vel.raw"
    tmp_path = path + ".tmp.raw"
    rank_path = path + ".rank.raw"
    if not os.path.isfile(png_path):
        sys.stderr.write(f"post.py: {path}\n")
        root = xml.etree.ElementTree.parse(xdmf_path)
        time = root.find("Domain/Grid/Time").get("Value")
        xyz = np.memmap(xyz_path, "float32", "r")
        xyz = xyz.reshape(-1, 4, 2)
        ncell = len(xyz)
        rank = np.memmap(rank_path, "uint32", "r")
        chi = np.memmap(chi_path, "float64", "r")
        tmp = np.memmap(tmp_path, "float64", "r")
        vel = np.memmap(vel_path, "float64", "r")
        vel = vel.reshape(-1, 2)
        patches = []
        color = []
        for i in range(ncell):
            if chi[i] < 0.5 and rank[i] == 0:
                x = xyz[i, 0, 0]
                y = xyz[i, 0, 1]
                lx = xyz[i, 2, 0] - x
                ly = xyz[i, 2, 1] - y
                color.append(tmp[i])
                patches.append(matplotlib.patches.Rectangle((x, y), lx, ly))
        p = matplotlib.collections.PatchCollection(patches,
                                                   edgecolor='black',
                                                   linewidth=0.1)
        vmax = np.nanquantile(np.abs(color), 0.95)
        p.set_array(color)
        p.set_norm(matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax))
        plt.gca().add_collection(p)
        plt.axis("scaled")
        plt.gca().set_xlim(0, 1)
        plt.gca().set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(png_path, dpi=400, bbox_inches='tight', pad_inches=0)
        plt.close()
        sys.stderr.write(f"post.py: {png_path}\n")
