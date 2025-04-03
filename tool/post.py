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
    patches = []
    xx = [ ]
    yy = [ ]
    for i in range(ncell):
        j = 2 * 4 * i
        x = xyz[j]
        y = xyz[j + 1]
        lx = xyz[j + 4] - x
        ly = xyz[j + 5] - y
        patches.append(matplotlib.patches.Rectangle((x, y), lx, ly))
        xx.append(x)
        yy.append(y)
    print(min(attr[:, 0]), max(attr[:, 0]),
          statistics.variance(attr[:, 0]))
    # plt.axis((0, 1, 0, 1))
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
