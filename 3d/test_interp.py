#!/usr/bin/env python3
"""Test the OP_INTERP27 weights for correctness.

The 1D cubic interpolation weights {5, 30, -3}/32 should exactly
reproduce quadratic polynomials. In 3D tensor product, they should
exactly reproduce tri-quadratic polynomials.

Test: f(x,y,z) = x^2 + y^2 + z^2 on a coarse 3x3x3 stencil.
Interpolate to sub-cell positions and compare with exact values.
"""
import numpy as np

W1 = np.array([[5, 30, -3], [-3, 30, 5]], dtype=float) / 32.0

# Test 1: 1D quadratic reproduction
print("=== 1D test: f(x) = x^2 ===")
# Coarse cell centers at x = -1, 0, +1
x_coarse = np.array([-1.0, 0.0, 1.0])
f_coarse = x_coarse ** 2  # = [1, 0, 1]

# Sub-cell positions: -1/4 (parity 0) and +1/4 (parity 1)
for p in range(2):
    x_fine = -0.25 if p == 0 else 0.25
    interp = np.dot(W1[p], f_coarse)
    exact = x_fine ** 2
    print(f"  parity {p}: x={x_fine:+.2f}, interp={interp:.6f}, exact={exact:.6f}, err={abs(interp-exact):.1e}")

# Test 2: 3D tensor product for f(x,y,z) = x^2 + y^2 + z^2
print("\n=== 3D test: f(x,y,z) = x^2 + y^2 + z^2 ===")
max_err = 0
for pz in range(2):
    for py in range(2):
        for px in range(2):
            # Fine position
            xf = -0.25 if px == 0 else 0.25
            yf = -0.25 if py == 0 else 0.25
            zf = -0.25 if pz == 0 else 0.25
            exact = xf**2 + yf**2 + zf**2
            # 27-point interpolation
            interp = 0
            for kk in range(3):
                for jj in range(3):
                    for ii in range(3):
                        xc, yc, zc = ii - 1.0, jj - 1.0, kk - 1.0
                        fc = xc**2 + yc**2 + zc**2
                        w = W1[px][ii] * W1[py][jj] * W1[pz][kk]
                        interp += w * fc
            err = abs(interp - exact)
            max_err = max(max_err, err)
            print(f"  parity ({px},{py},{pz}): interp={interp:.6f}, exact={exact:.6f}, err={err:.1e}")
print(f"  Max error: {max_err:.1e}")

# Test 3: Compare with piecewise constant
print("\n=== Convergence comparison ===")
for name, order in [("piecewise constant", 0), ("cubic interp", 2)]:
    # f(x) = sin(pi*x) on cells of size h
    for h_inv in [4, 8, 16, 32]:
        h = 1.0 / h_inv
        # Coarse cells at centers -h, 0, +h
        xc = np.array([-h, 0, h])
        fc = np.sin(np.pi * xc)
        # Fine cell at h/4 from center
        x_fine = h / 4.0
        exact = np.sin(np.pi * x_fine)
        if order == 0:
            interp = fc[1]  # piecewise constant = center value
        else:
            interp = np.dot(W1[1], fc)  # parity 1 = +1/4 position
        err = abs(interp - exact)
        if h_inv == 4:
            prev_err = err
            print(f"  {name}: h=1/{h_inv:2d}, err={err:.2e}")
        else:
            rate = np.log2(prev_err / err)
            print(f"  {name}: h=1/{h_inv:2d}, err={err:.2e}, rate={rate:.1f}")
            prev_err = err
