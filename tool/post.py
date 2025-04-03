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
    path = re.sub("[.]attr\.raw$", "", path)
    path = re.sub("[.]xyz\.raw$", "", path)
    xdmf_path = path + ".xdmf2"
    xyz_path = path + ".xyz.raw"
    attr_path = path + ".chi.raw"
    png_path = path + ".png"
    root = xml.etree.ElementTree.parse(xdmf_path)
    time = root.find("Domain/Grid/Time").get("Value")
    xyz = np.memmap(xyz_path, "float32", "r")
    ncell = xyz.size // (2 * 4)
    assert ncell * 2 * 4 == xyz.size
    attr = np.memmap(attr_path, "float64", "r")
    attr = attr.reshape((ncell, -1))
    xyz = xyz.reshape(ncell, -1, 2)
    patches = []
    for i in range(ncell):
        x = xyz[i, 0, 0]
        y = xyz[i, 0, 1]
        lx = xyz[i, 2, 0] - x
        ly = xyz[i, 2, 1] - y
        patches.append(matplotlib.patches.Rectangle((x, y), lx, ly))
    print(min(attr[:, 0]), max(attr[:, 0]),
          statistics.variance(attr[:, 0]))
    plt.axis((0, 1, 0, 1))
    plt.axis("scaled")
    plt.axis("off")
    p = matplotlib.collections.PatchCollection(patches)
    color = np.sum(attr**2, 1)
    p.set_array(color)
    plt.gca().add_collection(p)
    plt.tight_layout()
    plt.savefig(png_path, dpi=400, bbox_inches='tight', pad_inches=0)

plt.rcParams['image.cmap'] = 'jet'
for path in sys.argv[1:]:
    sys.stderr.write("post.py: %s\n" % path)
    plot(path)
