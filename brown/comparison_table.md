# Table I Comparison: My Code vs Paper

Thick shear layer, rho=30, nu=1/10000
Paper: CFL=0.9, full Godunov+MAC+transverse
Mine: CFL=0.7, Godunov+MAC, no transverse, minmod neighbor slopes

## My Results

| Time | 32-64 | Rate | 64-128 | Rate | 128-256 |
|------|-------|------|--------|------|---------|
| 0.4 | 1.67E-2 | — | 4.91E-3 | 1.77 | 1.95E-3 |
| 0.8 | 8.36E-2 | — | 3.01E-2 | 1.47 | 1.59E-2 |
| 1.2 | 9.85E-2 | — | 4.31E-2 | 1.19 | 2.45E-2 |

## Paper Results

| Time | 32-64 | Rate | 64-128 | Rate | 128-256 | Rate | 256-512 |
|------|-------|------|--------|------|---------|------|---------|
| 0.4 | 1.34E-2 | 2.43 | 2.49E-3 | 1.89 | 6.73E-4 | 1.88 | 1.82E-4 |
| 0.8 | 7.05E-2 | 2.62 | 1.14E-2 | 2.53 | 1.98E-3 | 2.08 | 4.67E-4 |
| 1.2 | 7.51E-2 | 2.12 | 1.72E-2 | 2.50 | 3.05E-3 | 2.20 | 6.61E-4 |

## Error Ratio (mine / paper)

| Time | 32-64 | 64-128 | 128-256 |
|------|-------|--------|---------|
| 0.4 | 1.2x | 2.0x | 2.9x |
| 0.8 | 1.2x | 2.6x | 8.0x |
| 1.2 | 1.3x | 2.5x | 8.0x |

## Analysis

- At coarse resolution (32-64): errors are comparable (1.2-1.3x paper)
- At fine resolution (128-256): errors are 3-8x larger than paper
- Convergence rate: ~1.0-1.8 vs paper's ~2.0-2.6
- Root causes: missing transverse correction (drops to ~1st order in splitting),
  minmod neighbor slopes at block boundaries (reduces to 2nd order locally)
