#!/usr/bin/env python3
"""Build side-by-side comparison: paper Fig 2,3 vs computed results."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

BS = 8

def read_vort(path, N):
    nblk = (N // BS) ** 2
    data = np.fromfile(path, dtype=np.float64)
    assert data.size == nblk * BS * BS, f"Expected {nblk*BS*BS}, got {data.size}"
    nb = N // BS
    grid = np.zeros((N, N))
    for b in range(nblk):
        bx = b % nb
        by = b // nb
        blk = data[b * BS * BS : (b + 1) * BS * BS].reshape(BS, BS)
        grid[by * BS:(by + 1) * BS, bx * BS:(bx + 1) * BS] = blk
    return grid


def plot_vort(ax, w, N):
    h = 1.0 / N
    x = np.linspace(h / 2, 1 - h / 2, N)
    y = np.linspace(h / 2, 1 - h / 2, N)
    X, Y = np.meshgrid(x, y)
    levels = np.arange(-36, 37, 6)
    levels = levels[levels != 0]
    ax.contour(X, Y, w, levels=levels, colors='k', linewidths=1.2, linestyles='solid')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])


fig, axes = plt.subplots(3, 4, figsize=(16, 12))

resolutions = [64, 128, 256]
paper_labels = {64: 'A', 128: 'B', 256: 'C'}

for row, N in enumerate(resolutions):
    run_dir = f"run{N}"
    label = paper_labels[N]

    for col_pair, (fig_name, dump_idx, time_label) in enumerate([
        ('fig2', 2, 't=0.8'), ('fig3', 3, 't=1.2')
    ]):
        paper_col = col_pair * 2
        comp_col = col_pair * 2 + 1

        # Paper image
        img_path = f"ref/{fig_name}_{label}.png"
        try:
            img = Image.open(img_path)
            axes[row, paper_col].imshow(img, cmap='gray', aspect='equal')
        except:
            pass
        axes[row, paper_col].set_xticks([])
        axes[row, paper_col].set_yticks([])

        # Computed
        vort_path = f"{run_dir}/{dump_idx:08d}.vort.raw"
        try:
            w = read_vort(vort_path, N)
            plot_vort(axes[row, comp_col], w, N)
        except:
            pass

    # Row label
    axes[row, 0].set_ylabel(f"{N}", fontsize=12, fontweight='bold')

# Column headers only
axes[0, 0].set_title("Paper t=0.8", fontsize=11)
axes[0, 1].set_title("Computed t=0.8", fontsize=11)
axes[0, 2].set_title("Paper t=1.2", fontsize=11)
axes[0, 3].set_title("Computed t=1.2", fontsize=11)

fig.tight_layout()
fig.savefig("comparison.png", dpi=200, bbox_inches='tight')
print("Saved comparison.png")
