#!/usr/bin/env python3
import numpy as np
import sys
import os
import re
import xml.etree.ElementTree
import matplotlib.pyplot as plt
import matplotlib.patches
import statistics


def plot(path):
    path = re.sub("[.]xdmf2$", "", path)
    path = re.sub("[.]chi\.raw$", "", path)
    path = re.sub("[.]xyz\.raw$", "", path)
    xdmf_path = path + ".xdmf2"
    xyz_path = path + ".xyz.raw"
    chi_path = path + ".chi.raw"
    vel_path = path + ".vel.raw"
    png_path = path + ".png"
    root = xml.etree.ElementTree.parse(xdmf_path)
    time = root.find("Domain/Grid/Time").get("Value")
    xyz = np.memmap(xyz_path, "float32", "r")
    xyz = xyz.reshape(-1, 4, 2)
    ncell = len(xyz)
    chi = np.memmap(chi_path, "float64", "r")
    vel = np.memmap(vel_path, "float64", "r")
    vel = vel.reshape(-1, 2)
    patches = []
    for i in range(ncell):
        x = xyz[i, 0, 0]
        y = xyz[i, 0, 1]
        lx = xyz[i, 2, 0] - x
        ly = xyz[i, 2, 1] - y
        patches.append(matplotlib.patches.Rectangle((x, y), lx, ly))
    print(min(chi), max(chi), statistics.variance(chi))
    plt.axis((0, 1, 0, 1))
    # plt.axis("off")
    p = matplotlib.collections.PatchCollection(patches)

    color = np.sum(vel**2, 1)
    color[chi > 0.5] = None

    p.set_array(color)
    plt.gca().add_collection(p)
    plt.tight_layout()
    plt.savefig(png_path, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close()
    sys.stderr.write(f"post.py: {png_path}\n")


plt.rcParams['image.cmap'] = 'jet'
for path in sys.argv[1:]:
    sys.stderr.write(f"post.py: {path}\n")
    plot(path)
