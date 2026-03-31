#!/usr/bin/env python3
"""
Visualize 3D iso-surfaces extracted by iso3d.
Usage: view.py iso_prefix
  Reads iso_prefix.vert.raw (float32[nvert][3]) and iso_prefix.tri.raw (int32[ntri][3])
"""
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

prefix = sys.argv[1] if len(sys.argv) > 1 else 'iso_test'
verts = np.fromfile(f'{prefix}.xyz.raw', dtype=np.float32).reshape(-1, 3)
tris = np.fromfile(f'{prefix}.tri.raw', dtype=np.int32).reshape(-1, 3)
print(f'{len(verts)} vertices, {len(tris)} triangles')

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
mesh = Poly3DCollection(verts[tris], alpha=0.3, edgecolor='k', linewidth=0.1)
mesh.set_facecolor('steelblue')
ax.add_collection3d(mesh)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title(f'{prefix}: {len(tris)} triangles')
plt.savefig(f'{prefix}.png', dpi=150, bbox_inches='tight')
print(f'saved {prefix}.png')
