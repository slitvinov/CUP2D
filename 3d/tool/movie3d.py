#!/usr/bin/env python3
"""
Create mp4 from 3D dumps. Mirrors domain across symmetry planes (y,z)
to show full tube cross-section. Multiple iso-levels for shock + bubble.
Usage: movie3d.py [bubble_iso shock_iso] vel.00000000 vel.00000001 ...
"""
import numpy as np
import sys
import os
import re
import subprocess
import shutil
import xml.etree.ElementTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.expanduser('~/cudaAmrIsoSurfaceExtraction'))
import amriso


def mirror_data(xyz, rho, Ly, Lz):
    """Mirror data across y=Ly and z=Lz symmetry planes to get full domain."""
    ncell = len(rho)
    # Original: [0,Lx] x [0,Ly] x [0,Lz]
    # Mirror y: reflect y -> 2*Ly - y
    # Mirror z: reflect z -> 2*Lz - z
    # 4 quadrants: (orig), (y-mirror), (z-mirror), (yz-mirror)
    all_xyz = [xyz]
    all_rho = [rho]

    # y-mirror
    ym = xyz.copy()
    ym[:, :, 1] = 2 * Ly - ym[:, :, 1]
    # Fix vertex order (flip y-normals) by swapping pairs
    ym[:, [0,3,4,7], :], ym[:, [1,2,5,6], :] = ym[:, [3,0,7,4], :].copy(), ym[:, [2,1,6,5], :].copy()
    all_xyz.append(ym)
    all_rho.append(rho.copy())

    # z-mirror
    zm = xyz.copy()
    zm[:, :, 2] = 2 * Lz - zm[:, :, 2]
    zm[:, [0,1,2,3], :], zm[:, [4,5,6,7], :] = zm[:, [4,5,6,7], :].copy(), zm[:, [0,1,2,3], :].copy()
    all_xyz.append(zm)
    all_rho.append(rho.copy())

    # yz-mirror
    yzm = xyz.copy()
    yzm[:, :, 1] = 2 * Ly - yzm[:, :, 1]
    yzm[:, :, 2] = 2 * Lz - yzm[:, :, 2]
    # swap both
    yzm[:, [0,3,4,7], :], yzm[:, [1,2,5,6], :] = yzm[:, [3,0,7,4], :].copy(), yzm[:, [2,1,6,5], :].copy()
    yzm[:, [0,1,2,3], :], yzm[:, [4,5,6,7], :] = yzm[:, [4,5,6,7], :].copy(), yzm[:, [0,1,2,3], :].copy()
    all_xyz.append(yzm)
    all_rho.append(rho.copy())

    return np.concatenate(all_xyz), np.concatenate(all_rho)


def extract_iso(geo_flat, rho_flat, level):
    """Extract isosurface using amriso.extract3d. Returns (verts, tris) or (None, None)."""
    try:
        xyz, tri, attr = amriso.extract3d(geo_flat, rho_flat, rho_flat, level)
        if len(tri) > 0:
            return xyz, tri
    except Exception:
        pass
    return None, None


def render_frame(dump, bubble_iso, shock_iso, out_png):
    path = re.sub(r'\.(xdmf2|xyz\.raw)$', '', dump)
    root = xml.etree.ElementTree.parse(path + '.xdmf2')
    time = float(root.find('.//Time').get('Value'))
    info = root.find(".//Information[@Name='Step']")
    step = int(info.get('Value')) if info is not None else 0

    xyz = np.fromfile(path + '.xyz.raw', dtype=np.float32).reshape(-1, 8, 3)
    rho = np.fromfile(path + '.rho.raw', dtype=np.float64)
    Ly = 0.25; Lz = 0.25
    xyz_full, rho_full = mirror_data(xyz, rho, Ly, Lz)
    geo_flat = xyz_full.astype(np.float32).reshape(-1)
    rho_flat = rho_full.astype(np.float32)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Bubble isosurface
    v, t = extract_iso(geo_flat, rho_flat, bubble_iso)
    if v is not None and len(t) > 0:
        mesh = Poly3DCollection(v[t], alpha=0.5, linewidth=0)
        mesh.set_facecolor('steelblue')
        ax.add_collection3d(mesh)

    # Shock isosurface
    v, t = extract_iso(geo_flat, rho_flat, shock_iso)
    if v is not None and len(t) > 0:
        mesh = Poly3DCollection(v[t], alpha=0.15, linewidth=0)
        mesh.set_facecolor('orangered')
        ax.add_collection3d(mesh)

    Lx = 1.0
    ax.set_xlim(0, Lx); ax.set_ylim(0, 2*Ly); ax.set_zlim(0, 2*Lz)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_box_aspect([Lx, 2*Ly, 2*Lz])
    label = f'step {step}' if step else f't={time:.3e}'
    ax.set_title(label)
    ax.view_init(elev=20, azim=-60)
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()


args = sys.argv[1:]
bubble_iso = 0.5
shock_iso = 1.3
if args and not args[0].startswith('vel'):
    bubble_iso = float(args.pop(0))
if args and not args[0].startswith('vel'):
    shock_iso = float(args.pop(0))

paths = sorted(args)
if not paths:
    sys.exit('Usage: movie3d.py [bubble_iso shock_iso] vel.00000000 ...')

tmpdir = '_movie3d_frames'
os.makedirs(tmpdir, exist_ok=True)

for idx, p in enumerate(paths):
    out = os.path.join(tmpdir, f'frame_{idx:04d}.png')
    render_frame(p, bubble_iso, shock_iso, out)
    sys.stderr.write(f'{out}\n')

mp4 = 'movie3d.mp4'
subprocess.run([
    'ffmpeg', '-y', '-framerate', '6', '-i', f'{tmpdir}/frame_%04d.png',
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
    '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
    mp4
], check=True, capture_output=True)
sys.stderr.write(f'saved {mp4}\n')
