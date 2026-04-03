# Brown & Minion 1995 — Complete Data Extraction

**Source:** D. L. Brown and M. L. Minion, "Performance of Under-resolved Two-Dimensional Incompressible Flow Simulations," *J. Comput. Phys.* **122**, 165–183 (1995).

Machine-readable data: [`paper_data.txt`](paper_data.txt)

---

## 1. Problem Setup (Sec. 2.1, 3)

### Initial Conditions (Eq. 27–28)

```
u(x,y) = tanh(rho*(y - 0.25))    for y <= 0.5
         tanh(rho*(0.75 - y))     for y >  0.5
v(x,y) = delta * sin(2*pi*x)
```

| Parameter | Thick layer | Thin layer |
|-----------|------------|------------|
| rho | 30 | 100 |
| delta | 0.05 | 0.05 |
| nu | 1/10,000 | 1/10,000 or 1/20,000 or 1/40,000 |
| Domain | [0,1]^2 periodic | [0,1]^2 periodic |
| CFL (Godunov) | 0.9 | 0.9 |
| CFL (centered) | 0.7 | 0.7 |

### Contour Levels

| Case | Range | Step |
|------|-------|------|
| Thick (rho=30) | -36 to 36 | 6 |
| Thin (rho=100) | -70 to 70 | 10 |

---

## 2. Convergence Tables

### Table I — Projection Method, Thick Layer (rho=30, nu=1/10,000) [p. 173]

L2 velocity error via Richardson extrapolation between successive mesh pairs.

| Time | 32–64 | Rate | 64–128 | Rate | 128–256 | Rate | 256–512 |
|------|-------|------|--------|------|---------|------|---------|
| 0.4 | 1.34E-2 | 2.43 | 2.49E-3 | 1.89 | 6.73E-4 | 1.88 | 1.82E-4 |
| 0.8 | 7.05E-2 | 2.62 | 1.14E-2 | 2.53 | 1.98E-3 | 2.08 | 4.67E-4 |
| 1.2 | 7.51E-2 | 2.12 | 1.72E-2 | 2.50 | 3.05E-3 | 2.20 | 6.61E-4 |

**Key finding:** Second-order convergence (~2.0–2.6 rate) at all times.

### Table II — Centered FD (4th-order), Thick Layer (rho=30, nu=1/10,000) [p. 175]

| Time | 64–128 | Rate | 128–256 | Rate | 256–512 | Rate | 512–1024 |
|------|--------|------|---------|------|---------|------|----------|
| 0.4 | 0.268 | 3.49 | 2.38E-2 | 3.86 | 1.64E-3 | 3.29 | 1.67E-4 |
| 0.8 | 1.83 | 3.72 | 0.277 | 3.53 | 2.41E-2 | 3.80 | 1.74E-3 |

**Key finding:** Higher-order convergence (~3.3–3.9 rate) when resolved.

### Table III — Projection Method, Thin Layer (rho=100, nu=1/10,000) [p. 178]

| Time | 32–64 | Rate | 64–128 | Rate | 128–256 | Rate | 256–512 |
|------|-------|------|--------|------|---------|------|---------|
| 0.4 | 1.07E-1 | 0.68 | 6.67E-2 | 1.70 | 2.04E-2 | 2.77 | 2.98E-3 |
| 0.8 | 2.39E-1 | 0.13 | 2.18E-1 | 0.51 | 1.53E-1 | 2.46 | 2.78E-2 |
| 1.2 | 1.91E-1 | -0.16 | 2.14E-1 | -0.22 | 2.50E-1 | 2.01 | 6.18E-2 |

**Key finding:** No convergence at coarse grids (negative rates at 32–64 for t=1.2). Convergence only appears at 128–256 and finer. Spurious vortices dominate the error at coarse resolution.

### Table IV — Projection Method, Thin Layer (rho=100, nu=1/20,000) [p. 178]

| Time | 32–64 | Rate | 64–128 | Rate | 128–256 | Rate | 256–512 |
|------|-------|------|--------|------|---------|------|---------|
| 0.4 | 1.15E-1 | 0.58 | 7.68E-2 | 1.15 | 3.45E-2 | 2.59 | 5.72E-3 |
| 0.8 | 2.45E-1 | -0.15 | 2.71E-1 | 0.52 | 1.89E-1 | 0.98 | 9.68E-2 |
| 1.2 | 2.01E-1 | -0.18 | 2.28E-1 | -0.29 | 2.78E-1 | 0.67 | 1.74E-1 |

**Key finding:** Even worse convergence than Table III. Lower viscosity = more underresolved = worse artifacts.

### Table V — Local L2 Error in Spurious Vortex Region (rho=100, nu=1/10,000) [p. 178]

Errors computed vs 1024x1024 reference in a localized region containing the main spurious vortex.

| Time | 32 | Rate | 64 | Rate | 128 | Rate | 256 | Rate | 512 |
|------|-----|------|-----|------|------|------|------|------|------|
| 0.4 | 2.92 | 0.29 | 2.39 | 1.66 | 7.57E-1 | 3.74 | 5.67E-2 | 1.66 | 1.80E-2 |
| 0.8 | 3.57 | 0.10 | 3.34 | 0.18 | 2.94 | 1.35 | 1.15 | 7.92 | 4.76E-2 |
| 1.2 | 3.18 | -0.07 | 3.34 | 0.26 | 2.78 | 1.01 | 1.38 | 9.71 | 1.65E-2 |

**Key finding:** No convergence until 256–512 pair. The spurious vortex error is O(1) until the vortex is resolved, then drops dramatically.

### Table VI — Local L2 Error in Spurious Vortex Region (rho=100, nu=1/20,000) [p. 179]

| Time | 32 | Rate | 64 | Rate | 128 | Rate | 256 | Rate | 512 |
|------|-----|------|-----|------|------|------|------|------|------|
| 0.4 | 3.48 | 0.13 | 3.17 | 0.95 | 1.64 | 3.23 | 1.75E-2 | 2.54 | 3.01E-2 |
| 0.8 | 3.83 | 0.07 | 3.66 | 0.12 | 3.36 | 0.12 | 3.09 | 3.24 | 3.26E-2 |
| 1.2 | 3.27 | -0.11 | 3.52 | 0.17 | 3.12 | 0.30 | 2.53 | 3.33 | 5.25E-2 |

---

## 3. Time Histories (Digitized from Figures)

### Fig. 6A — Energy Decay, Godunov 512x512, rho=30, nu=1/10,000 [p. 174]

Energy = (1/2) * integral(u^2 + v^2) over domain.

| t | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 | 1.2 | 1.4 | 1.6 |
|---|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| E | 0.932 | 0.931 | 0.928 | 0.925 | 0.922 | 0.921 | 0.920 | 0.919 | 0.919 |

### Fig. 6B — Enstrophy (L2 vorticity), Godunov 512x512, rho=30, nu=1/10,000 [p. 174]

| t | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 | 1.2 | 1.4 | 1.6 |
|---|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| ||w||_2 | 7.8 | 7.8 | 7.9 | 8.0 | 8.2 | 8.4 | 8.6 | 8.7 | 8.8 |

### Fig. 7A — Energy Decay, Centered FD 512x512, rho=30, nu=1/10,000 [p. 175]

| t | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 | 1.2 | 1.4 | 1.6 |
|---|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| E | 0.932 | 0.930 | 0.928 | 0.925 | 0.922 | 0.921 | 0.920 | 0.919 | 0.919 |

**Key finding:** Energy decay nearly identical between Godunov and centered methods at 512x512. Both methods produce the same global energy behavior.

### Fig. 7C — Enstrophy, Centered FD 512x512 [p. 175]

| t | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 | 1.2 | 1.4 | 1.6 |
|---|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| ||w||_2 | 7.8 | 7.8 | 7.9 | 8.0 | 8.2 | 8.4 | 8.5 | 8.6 | 8.7 |

---

## 4. Algorithm Details (Sec. 2.2)

### Discrete Operators (Eq. 14–17)

- **Divergence** D(U): centered, `(u_{i+1,j} - u_{i-1,j} + v_{i,j+1} - v_{i,j-1}) / (2h)` (Eq. 14)
- **Gradient** G(phi): centered, `((phi_{i+1}-phi_{i-1})/(2h), (phi_{j+1}-phi_{j-1})/(2h))` (Eq. 15)
- **Laplacian** L = DG: wide stencil, `(-4*phi + phi_{i+2} + phi_{i-2} + phi_{j+2} + phi_{j-2}) / (4h^2)` (Eq. 17)
- **Standard Laplacian** Delta^5: `(-4*phi + phi_{i+1} + phi_{i-1} + phi_{j+1} + phi_{j-1}) / h^2`

### Projection Steps (Eq. 10–13)

1. Predict U* with Crank-Nicolson viscosity: `(U*-U^n)/dt = nu/2*Delta(U^n+U*) - advection - grad(p)` (Eq. 10)
2. Decompose: `U* = U^{n+1} + grad(phi)` (Eq. 11)
3. Update pressure: `grad(p^{n+1/2}) = grad(p^{n-1/2}) + grad(phi)/dt` (Eq. 12)
4. Solve: `L(phi) = D(U*)` using wide Laplacian (Eq. 13, 16, 17)

### Godunov Edge Prediction (Eq. 18–22)

1. Taylor expand to edge with 4th-order monotone slopes (Eq. 20)
2. Riemann solve via Burgers upwind (Eq. 21)
3. Add transverse + viscous + pressure corrections (Eq. 22)
4. MAC projection on edge velocities (Eq. 23)
5. Difference for advection flux (after Eq. 22)

### Key Numerical Choices

| Parameter | Value |
|-----------|-------|
| Slope limiter | 4th-order monotone (Eq. 20, right column) |
| Riemann solver | Burgers upwind (Eq. 21) |
| Viscous discretization | Crank-Nicolson (Eq. 10) |
| Poisson solver | Multigrid on 4 decoupled sub-stencils (Ref. [10]) |
| MAC projection | Standard 5-point Laplacian (Eq. 23) |
| Time integrator | Forward Euler for projection; RK4 for reference (Eq. 25) |
| Initial pressure | Iteration procedure from Ref. [1] |

---

## 5. Figures Summary

| Figure | Content | Resolution | Parameters | Page |
|--------|---------|-----------|------------|------|
| 1 | Cell variable locations | — | — | 168 |
| 2 | Thick layer vorticity, t=0.8, Godunov | 64,128,256,512 | rho=30, nu=1e-4 | 170 |
| 3 | Thick layer vorticity, t=1.2, Godunov | 64,128,256,512 | rho=30, nu=1e-4 | 171 |
| 4 | Thick layer vorticity, t=0.8, centered | 64,128,256,512 | rho=30, nu=1e-4 | 172 |
| 5 | Thick layer vorticity, t=1.2, centered | 64,128,256,512 | rho=30, nu=1e-4 | 173 |
| 6 | Energy/enstrophy/spectrum/limiters, Godunov | 512 | rho=30, nu=1e-4 | 174 |
| 7 | Energy/spectrum/enstrophy, centered | 512 | rho=30, nu=1e-4 | 175 |
| 8 | Thin layer t=0.8, Godunov+centered | 128,256,512 | rho=100, nu=5e-5 | 176 |
| 9 | Thin layer t=1.2, Godunov+centered | 128,256,512 | rho=100, nu=5e-5 | 177 |
| 10 | Centered FD instability | 256 | rho=100, nu=2.5e-5 | 178 |
| 11 | Thin layer t=0.8, proj+centered | 128,256,512 | rho=100, nu=1e-4 | 179 |
| 12 | Thin layer t=0.8,1.2 | 256,512 | rho=100, nu=2.5e-5 | 180 |
| 13 | Energy/enstrophy thin layer | 256,512 | rho=100, nu=5e-5 | 181 |
