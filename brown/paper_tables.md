# Brown & Minion 1995 — Extracted Tables

Source: "Performance of Under-resolved Two-Dimensional Incompressible Flow Simulations", JCP 122, 165-183 (1995)

All L2 errors estimated via Richardson extrapolation between successive mesh pairs. Rates from adjacent error pairs.

## Table I: Projection Method — Thick Shear Layer (rho=30, nu=1/10000)

| Time | 32-64 | Rate | 64-128 | Rate | 128-256 | Rate | 256-512 |
|------|-------|------|--------|------|---------|------|---------|
| 0.4 | 1.34E-2 | 2.43 | 2.49E-3 | 1.89 | 6.73E-4 | 1.88 | 1.82E-4 |
| 0.8 | 7.05E-2 | 2.62 | 1.14E-2 | 2.53 | 1.98E-3 | 2.08 | 4.67E-4 |
| 1.2 | 7.51E-2 | 2.12 | 1.72E-2 | 2.50 | 3.05E-3 | 2.20 | 6.61E-4 |

## Table II: Centered Finite-Difference — Thick Shear Layer (rho=30, nu=1/10000)

| Time | 32-64 | Rate | 64-128 | Rate | 128-256 | Rate | 256-512 |
|------|-------|------|--------|------|---------|------|---------|
| 0.4 | 1.21E-2 | 2.80 | 1.74E-3 | 3.24 | 1.84E-4 | 3.53 | 1.59E-5 |
| 0.8 | 6.68E-2 | 2.49 | 1.19E-2 | 2.00 | 2.97E-3 | 1.82 | 8.40E-4 |
| 1.2 | 7.39E-2 | 1.90 | 1.98E-2 | 1.52 | 6.89E-3 | 1.28 | 2.83E-3 |

## Table III: Projection Method — Thin Shear Layer (rho=100, nu=1/10000)

| Time | 32-64 | Rate | 64-128 | Rate | 128-256 | Rate | 256-512 |
|------|-------|------|--------|------|---------|------|---------|
| 0.4 | 1.07E-1 | 0.68 | 6.67E-2 | 1.70 | 2.04E-2 | 2.77 | 2.98E-3 |
| 0.8 | 2.39E-1 | 0.13 | 2.18E-1 | 0.51 | 1.53E-1 | 2.46 | 2.78E-2 |
| 1.2 | 1.91E-1 | -0.16 | 2.14E-1 | -0.22 | 2.50E-1 | 2.01 | 6.18E-2 |

## Table IV: Projection Method — Thin Shear Layer (rho=100, nu=1/20000)

| Time | 32-64 | Rate | 64-128 | Rate | 128-256 | Rate | 256-512 |
|------|-------|------|--------|------|---------|------|---------|
| 0.4 | 1.15E-1 | 0.58 | 7.68E-2 | 1.15 | 3.45E-2 | 2.59 | 5.72E-3 |
| 0.8 | 2.45E-1 | -0.15 | 2.71E-1 | 0.52 | 1.89E-1 | 0.98 | 9.68E-2 |
| 1.2 | 2.01E-1 | -0.18 | 2.28E-1 | -0.29 | 2.78E-1 | 0.67 | 1.74E-1 |

## Table V: Local L2 Errors — Thin Shear Layer, Spurious Vortex Region (rho=100, nu=1/10000)

Errors compared with 1024x1024 reference solution.

| Time | 32 | Rate | 64 | Rate | 128 | Rate | 256 | Rate | 512 |
|------|-----|------|-----|------|------|------|------|------|------|
| 0.4 | 2.92 | 0.29 | 2.39 | 1.66 | 7.57E-1 | 3.74 | 5.67E-2 | 1.66 | 1.80E-2 |
| 0.8 | 3.57 | 0.10 | 3.34 | 0.18 | 2.94 | 1.35 | 1.15 | 7.92 | 4.76E-2 |
| 1.2 | 3.18 | -0.07 | 3.34 | 0.26 | 2.78 | 1.01 | 1.38 | 9.71 | 1.65E-2 |

## Table VI: Local L2 Errors — Thin Shear Layer, Spurious Vortex Region (rho=100, nu=1/20000)

Errors compared with 1024x1024 reference solution.

| Time | 32 | Rate | 64 | Rate | 128 | Rate | 256 | Rate | 512 |
|------|-----|------|-----|------|------|------|------|------|------|
| 0.4 | 3.48 | 0.13 | 3.17 | 0.95 | 1.64 | 3.23 | 1.75E-2 | 2.54 | 3.01E-2 |
| 0.8 | 3.83 | 0.07 | 3.66 | 0.12 | 3.36 | 0.12 | 3.09 | 3.24 | 3.26E-2 |
| 1.2 | 3.27 | -0.11 | 3.52 | 0.17 | 3.12 | 0.30 | 2.53 | 3.33 | 5.25E-2 |
