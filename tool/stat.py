#!/usr/bin/env python3
import numpy as np
import sys
import os
import re
import xml.etree.ElementTree

for path in sys.argv[1:]:
    path = re.sub("[.]xdmf2$", "", path)
    path = re.sub("[.]attr\.raw$", "", path)
    path = re.sub("[.]xyz\.raw$", "", path)
    xdmf_path = path + ".xdmf2"
    xyz_path = path + ".xyz.raw"
    attr_path = path + ".chi.raw"
    root = xml.etree.ElementTree.parse(xdmf_path)
    time = root.find("Domain/Grid/Time").get("Value")
    xyz = np.memmap(xyz_path, np.dtype("float32"), "r")
    ncell = xyz.size // (2 * 4)
    assert ncell * 2 * 4 == xyz.size
    attr = np.memmap(attr_path, np.dtype("float64"), "r")
    attr = attr.reshape((ncell, -1))
    print(
        f"{np.mean(attr):+.2e} {np.std(attr):+.2e} {np.mean(xyz):+.2e} {np.std(attr):+.2e}"
    )
