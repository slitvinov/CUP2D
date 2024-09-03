#define OMPI_SKIP_MPICXX 1
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <gsl/gsl_linalg.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mpi.h>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "cuda.h"
enum { max_dim = 2 };
typedef double Real;
static const MPI_Datatype MPI_Real = MPI_DOUBLE;
static constexpr unsigned int sizes[] = {_BS_, _BS_, 1};
static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
static Real dist(Real a[2], Real b[2]) {
  return std::pow(a[0] - b[0], 2) + std::pow(a[1] - b[1], 2);
}
static void rotate2D(const Real Rmatrix2D[2][2], Real &x, Real &y) {
  Real p[2] = {x, y};
  x = Rmatrix2D[0][0] * p[0] + Rmatrix2D[0][1] * p[1];
  y = Rmatrix2D[1][0] * p[0] + Rmatrix2D[1][1] * p[1];
}
static Real dds(int i, int m, Real *a, Real *b) {
  if (i == 0)
    return (a[i + 1] - a[i]) / (b[i + 1] - b[i]);
  else if (i == m - 1)
    return (a[i] - a[i - 1]) / (b[i] - b[i - 1]);
  else
    return ((a[i + 1] - a[i]) / (b[i + 1] - b[i]) +
            (a[i] - a[i - 1]) / (b[i] - b[i - 1])) /
           2;
}
static double getA_local(int I1, int I2) {
  int j1 = I1 / _BS_;
  int i1 = I1 % _BS_;
  int j2 = I2 / _BS_;
  int i2 = I2 % _BS_;
  if (i1 == i2 && j1 == j2)
    return 4.0;
  else if (abs(i1 - i2) + abs(j1 - j2) == 1)
    return -1.0;
  else
    return 0.0;
}
static void pack(Real *srcbase, Real *dst, int dim, int xstart, int ystart,
                 int zstart, int xend, int yend, int zend, int BSX, int BSY) {
  if (dim == 1) {
    const int mod = (xend - xstart) % 4;
    for (int idst = 0, iz = zstart; iz < zend; ++iz)
      for (int iy = ystart; iy < yend; ++iy) {
        for (int ix = xstart; ix < xend - mod; ix += 4, idst += 4) {
          dst[idst + 0] = srcbase[ix + 0 + BSX * (iy + BSY * iz)];
          dst[idst + 1] = srcbase[ix + 1 + BSX * (iy + BSY * iz)];
          dst[idst + 2] = srcbase[ix + 2 + BSX * (iy + BSY * iz)];
          dst[idst + 3] = srcbase[ix + 3 + BSX * (iy + BSY * iz)];
        }
        for (int ix = xend - mod; ix < xend; ix++, idst++) {
          dst[idst] = srcbase[ix + BSX * (iy + BSY * iz)];
        }
      }
  } else {
    for (int idst = 0, iz = zstart; iz < zend; ++iz)
      for (int iy = ystart; iy < yend; ++iy)
        for (int ix = xstart; ix < xend; ++ix) {
          const Real *src = srcbase + dim * (ix + BSX * (iy + BSY * iz));
          for (int ic = 0; ic < dim; ic++, idst++)
            dst[idst] = src[ic];
        }
  }
}
static void unpack_subregion(Real *pack, Real *dstbase, int dim, int srcxstart,
                             int srcystart, int srczstart, int LX, int LY,
                             int dstxstart, int dstystart, int dstzstart,
                             int dstxend, int dstyend, int dstzend, int xsize,
                             int ysize) {
  if (dim == 1) {
    const int mod = (dstxend - dstxstart) % 4;
    for (int zd = dstzstart; zd < dstzend; ++zd)
      for (int yd = dstystart; yd < dstyend; ++yd) {
        const int offset = -dstxstart + srcxstart +
                           LX * (yd - dstystart + srcystart +
                                 LY * (zd - dstzstart + srczstart));
        const int offset_dst = xsize * (yd + ysize * zd);
        for (int xd = dstxstart; xd < dstxend - mod; xd += 4) {
          dstbase[xd + 0 + offset_dst] = pack[xd + 0 + offset];
          dstbase[xd + 1 + offset_dst] = pack[xd + 1 + offset];
          dstbase[xd + 2 + offset_dst] = pack[xd + 2 + offset];
          dstbase[xd + 3 + offset_dst] = pack[xd + 3 + offset];
        }
        for (int xd = dstxend - mod; xd < dstxend; ++xd) {
          dstbase[xd + offset_dst] = pack[xd + offset];
        }
      }
  } else {
    for (int zd = dstzstart; zd < dstzend; ++zd)
      for (int yd = dstystart; yd < dstyend; ++yd)
        for (int xd = dstxstart; xd < dstxend; ++xd) {
          Real *const dst = dstbase + dim * (xd + xsize * (yd + ysize * zd));
          const Real *src =
              pack + dim * (xd - dstxstart + srcxstart +
                            LX * (yd - dstystart + srcystart +
                                  LY * (zd - dstzstart + srczstart)));
          for (int c = 0; c < dim; ++c)
            dst[c] = src[c];
        }
  }
}
static void if2d_solve(unsigned Nm, Real *rS, Real *curv, Real *curv_dt,
                       Real *rX, Real *rY, Real *vX, Real *vY, Real *norX,
                       Real *norY, Real *vNorX, Real *vNorY) {
  rX[0] = 0.0;
  rY[0] = 0.0;
  norX[0] = 0.0;
  norY[0] = 1.0;
  Real ksiX = 1.0;
  Real ksiY = 0.0;
  vX[0] = 0.0;
  vY[0] = 0.0;
  vNorX[0] = 0.0;
  vNorY[0] = 0.0;
  Real vKsiX = 0.0;
  Real vKsiY = 0.0;
  for (unsigned i = 1; i < Nm; i++) {
    const Real dksiX = curv[i - 1] * norX[i - 1];
    const Real dksiY = curv[i - 1] * norY[i - 1];
    const Real dnuX = -curv[i - 1] * ksiX;
    const Real dnuY = -curv[i - 1] * ksiY;
    const Real dvKsiX =
        curv_dt[i - 1] * norX[i - 1] + curv[i - 1] * vNorX[i - 1];
    const Real dvKsiY =
        curv_dt[i - 1] * norY[i - 1] + curv[i - 1] * vNorY[i - 1];
    const Real dvNuX = -curv_dt[i - 1] * ksiX - curv[i - 1] * vKsiX;
    const Real dvNuY = -curv_dt[i - 1] * ksiY - curv[i - 1] * vKsiY;
    const Real ds = rS[i] - rS[i - 1];
    rX[i] = rX[i - 1] + ds * ksiX;
    rY[i] = rY[i - 1] + ds * ksiY;
    norX[i] = norX[i - 1] + ds * dnuX;
    norY[i] = norY[i - 1] + ds * dnuY;
    ksiX += ds * dksiX;
    ksiY += ds * dksiY;
    vX[i] = vX[i - 1] + ds * vKsiX;
    vY[i] = vY[i - 1] + ds * vKsiY;
    vNorX[i] = vNorX[i - 1] + ds * dvNuX;
    vNorY[i] = vNorY[i - 1] + ds * dvNuY;
    vKsiX += ds * dvKsiX;
    vKsiY += ds * dvKsiY;
    const Real d1 = ksiX * ksiX + ksiY * ksiY;
    const Real d2 = norX[i] * norX[i] + norY[i] * norY[i];
    if (d1 > std::numeric_limits<Real>::epsilon()) {
      const Real normfac = 1 / std::sqrt(d1);
      ksiX *= normfac;
      ksiY *= normfac;
    }
    if (d2 > std::numeric_limits<Real>::epsilon()) {
      const Real normfac = 1 / std::sqrt(d2);
      norX[i] *= normfac;
      norY[i] *= normfac;
    }
  }
}
static Real weno5_plus(Real um2, Real um1, Real u, Real up1, Real up2) {
  Real exponent = 2, e = 1e-6;
  Real b1 = 13.0 / 12.0 * pow((um2 + u) - 2 * um1, 2) +
            0.25 * pow((um2 + 3 * u) - 4 * um1, 2);
  Real b2 =
      13.0 / 12.0 * pow((um1 + up1) - 2 * u, 2) + 0.25 * pow(um1 - up1, 2);
  Real b3 = 13.0 / 12.0 * pow((u + up2) - 2 * up1, 2) +
            0.25 * pow((3 * u + up2) - 4 * up1, 2);
  Real g1 = 0.1, g2 = 0.6, g3 = 0.3;
  Real what1 = g1 / pow(b1 + e, exponent);
  Real what2 = g2 / pow(b2 + e, exponent);
  Real what3 = g3 / pow(b3 + e, exponent);
  Real aux = 1.0 / ((what1 + what3) + what2);
  Real w1 = what1 * aux, w2 = what2 * aux, w3 = what3 * aux;
  Real f1 = (11.0 / 6.0) * u + ((1.0 / 3.0) * um2 - (7.0 / 6.0) * um1);
  Real f2 = (5.0 / 6.0) * u + ((-1.0 / 6.0) * um1 + (1.0 / 3.0) * up1);
  Real f3 = (1.0 / 3.0) * u + ((+5.0 / 6.0) * up1 - (1.0 / 6.0) * up2);
  return (w1 * f1 + w3 * f3) + w2 * f2;
}
static Real weno5_minus(Real um2, Real um1, Real u, Real up1, Real up2) {
  Real exponent = 2, e = 1e-6;
  Real b1 = 13.0 / 12.0 * pow((um2 + u) - 2 * um1, 2) +
            0.25 * pow((um2 + 3 * u) - 4 * um1, 2);
  Real b2 =
      13.0 / 12.0 * pow((um1 + up1) - 2 * u, 2) + 0.25 * pow(um1 - up1, 2);
  Real b3 = 13.0 / 12.0 * pow((u + up2) - 2 * up1, 2) +
            0.25 * pow((3 * u + up2) - 4 * up1, 2);
  Real g1 = 0.3, g2 = 0.6, g3 = 0.1;
  Real what1 = g1 / pow(b1 + e, exponent);
  Real what2 = g2 / pow(b2 + e, exponent);
  Real what3 = g3 / pow(b3 + e, exponent);
  Real aux = 1.0 / ((what1 + what3) + what2);
  Real w1 = what1 * aux;
  Real w2 = what2 * aux;
  Real w3 = what3 * aux;
  Real f1 = (1.0 / 3.0) * u + ((-1.0 / 6.0) * um2 + (5.0 / 6.0) * um1);
  Real f2 = (5.0 / 6.0) * u + ((1.0 / 3.0) * um1 - (1.0 / 6.0) * up1);
  Real f3 = (11.0 / 6.0) * u + ((-7.0 / 6.0) * up1 + (1.0 / 3.0) * up2);
  return (w1 * f1 + w3 * f3) + w2 * f2;
}
static Real derivative(Real U, Real um3, Real um2, Real um1, Real u, Real up1,
                       Real up2, Real up3) {
  Real fp = 0.0;
  Real fm = 0.0;
  if (U > 0) {
    fp = weno5_plus(um2, um1, u, up1, up2);
    fm = weno5_plus(um3, um2, um1, u, up1);
  } else {
    fp = weno5_minus(um1, u, up1, up2, up3);
    fm = weno5_minus(um2, um1, u, up1, up2);
  }
  return (fp - fm);
}
static void compute_j(Real *Rc, Real *R, Real *N, Real *I, Real *J) {
  Real m00 = 1.0;
  Real m01 = 0.0;
  Real m02 = 0.0;
  Real m11 = 1.0;
  Real m12 = 0.0;
  Real m22 = I[5];
  Real a00 = m22 * m11 - m12 * m12;
  Real a01 = m02 * m12 - m22 * m01;
  Real a02 = m01 * m12 - m02 * m11;
  Real a11 = m22 * m00 - m02 * m02;
  Real a12 = m01 * m02 - m00 * m12;
  Real a22 = m00 * m11 - m01 * m01;
  Real determinant = 1.0 / ((m00 * a00) + (m01 * a01) + (m02 * a02));
  a00 *= determinant;
  a01 *= determinant;
  a02 *= determinant;
  a11 *= determinant;
  a12 *= determinant;
  a22 *= determinant;
  const Real aux_0 = (Rc[1] - R[1]) * N[2] - (Rc[2] - R[2]) * N[1];
  const Real aux_1 = (Rc[2] - R[2]) * N[0] - (Rc[0] - R[0]) * N[2];
  const Real aux_2 = (Rc[0] - R[0]) * N[1] - (Rc[1] - R[1]) * N[0];
  J[0] = a00 * aux_0 + a01 * aux_1 + a02 * aux_2;
  J[1] = a01 * aux_0 + a11 * aux_1 + a12 * aux_2;
  J[2] = a02 * aux_0 + a12 * aux_1 + a22 * aux_2;
}
static void collision(Real m1, Real m2, Real *I1, Real *I2, Real *v1, Real *v2,
                      Real *o1, Real *o2, Real *hv1, Real *hv2, Real *ho1,
                      Real *ho2, Real *C1, Real *C2, Real NX, Real NY, Real NZ,
                      Real CX, Real CY, Real CZ, Real *vc1, Real *vc2) {
  Real e = 1.0;
  Real N[3] = {NX, NY, NZ};
  Real C[3] = {CX, CY, CZ};
  Real k1[3] = {N[0] / m1, N[1] / m1, N[2] / m1};
  Real k2[3] = {-N[0] / m2, -N[1] / m2, -N[2] / m2};
  Real J1[3];
  Real J2[3];
  compute_j(C, C1, N, I1, J1);
  compute_j(C, C2, N, I2, J2);
  J2[0] = -J2[0];
  J2[1] = -J2[1];
  J2[2] = -J2[2];
  Real u1DEF[3];
  u1DEF[0] = vc1[0] - v1[0] - (o1[1] * (C[2] - C1[2]) - o1[2] * (C[1] - C1[1]));
  u1DEF[1] = vc1[1] - v1[1] - (o1[2] * (C[0] - C1[0]) - o1[0] * (C[2] - C1[2]));
  u1DEF[2] = vc1[2] - v1[2] - (o1[0] * (C[1] - C1[1]) - o1[1] * (C[0] - C1[0]));
  Real u2DEF[3];
  u2DEF[0] = vc2[0] - v2[0] - (o2[1] * (C[2] - C2[2]) - o2[2] * (C[1] - C2[1]));
  u2DEF[1] = vc2[1] - v2[1] - (o2[2] * (C[0] - C2[0]) - o2[0] * (C[2] - C2[2]));
  u2DEF[2] = vc2[2] - v2[2] - (o2[0] * (C[1] - C2[1]) - o2[1] * (C[0] - C2[0]));
  Real nom = e * ((vc1[0] - vc2[0]) * N[0] + (vc1[1] - vc2[1]) * N[1] +
                  (vc1[2] - vc2[2]) * N[2]) +
             ((v1[0] - v2[0] + u1DEF[0] - u2DEF[0]) * N[0] +
              (v1[1] - v2[1] + u1DEF[1] - u2DEF[1]) * N[1] +
              (v1[2] - v2[2] + u1DEF[2] - u2DEF[2]) * N[2]) +
             ((o1[1] * (C[2] - C1[2]) - o1[2] * (C[1] - C1[1])) * N[0] +
              (o1[2] * (C[0] - C1[0]) - o1[0] * (C[2] - C1[2])) * N[1] +
              (o1[0] * (C[1] - C1[1]) - o1[1] * (C[0] - C1[0])) * N[2]) -
             ((o2[1] * (C[2] - C2[2]) - o2[2] * (C[1] - C2[1])) * N[0] +
              (o2[2] * (C[0] - C2[0]) - o2[0] * (C[2] - C2[2])) * N[1] +
              (o2[0] * (C[1] - C2[1]) - o2[1] * (C[0] - C2[0])) * N[2]);
  Real denom = -(1.0 / m1 + 1.0 / m2) +
               +((J1[1] * (C[2] - C1[2]) - J1[2] * (C[1] - C1[1])) * (-N[0]) +
                 (J1[2] * (C[0] - C1[0]) - J1[0] * (C[2] - C1[2])) * (-N[1]) +
                 (J1[0] * (C[1] - C1[1]) - J1[1] * (C[0] - C1[0])) * (-N[2])) -
               ((J2[1] * (C[2] - C2[2]) - J2[2] * (C[1] - C2[1])) * (-N[0]) +
                (J2[2] * (C[0] - C2[0]) - J2[0] * (C[2] - C2[2])) * (-N[1]) +
                (J2[0] * (C[1] - C2[1]) - J2[1] * (C[0] - C2[0])) * (-N[2]));
  Real impulse = nom / (denom + 1e-21);
  hv1[0] = v1[0] + k1[0] * impulse;
  hv1[1] = v1[1] + k1[1] * impulse;
  hv1[2] = v1[2] + k1[2] * impulse;
  hv2[0] = v2[0] + k2[0] * impulse;
  hv2[1] = v2[1] + k2[1] * impulse;
  hv2[2] = v2[2] + k2[2] * impulse;
  ho1[0] = o1[0] + J1[0] * impulse;
  ho1[1] = o1[1] + J1[1] * impulse;
  ho1[2] = o1[2] + J1[2] * impulse;
  ho2[0] = o2[0] + J2[0] * impulse;
  ho2[1] = o2[1] + J2[1] * impulse;
  ho2[2] = o2[2] + J2[2] * impulse;
}
enum BCflag { freespace, periodic, wall };
static const struct {
  enum BCflag val;
  const char *key;
} bc_table[] = {
    {freespace, "freespace"}, {wall, "wall"}, {periodic, "periodic"}};
static BCflag bc_flag(const char *key) {
  size_t i;
  for (i = 0; i < sizeof bc_table / sizeof *bc_table; i++)
    if (strcmp(bc_table[i].key, key) == 0)
      return bc_table[i].val;
  fprintf(stderr, "main: error: unknown boundary '%s'\n", key);
  MPI_Abort(MPI_COMM_WORLD, 1);
  return periodic;
}
struct Shape;
struct SpaceCurve;
static struct {
  bool bAdaptChiGradient;
  int AdaptSteps{20};
  int bMeanConstraint;
  int bpdx;
  int bpdy;
  int dumpFreq;
  int levelMax;
  int levelStart;
  int maxPoissonIterations;
  int maxPoissonRestarts;
  int nsteps;
  int rank;
  int size;
  int step = 0;
  Real CFL;
  Real Ctol;
  Real dlm;
  Real dt;
  Real dumpTime;
  Real endTime;
  Real extents[2];
  Real h0;
  Real lambda;
  Real minH;
  Real nextDumpTime = 0;
  Real nu;
  Real PoissonTol;
  Real PoissonTolRel;
  Real Rtol;
  Real time = 0;
  Real uinfx = 0;
  Real uinfy = 0;
  struct SpaceCurve *space_curve;
  std::vector<Shape *> shapes;
  std::vector<int> bCollisionID;
  enum BCflag bcx, bcy;
} sim;
struct SpaceCurve {
  int BX;
  int BY;
  bool isRegular;
  int base_level;
  std::vector<std::vector<long long>> Zsave;
  std::vector<std::vector<int>> i_inverse;
  std::vector<std::vector<int>> j_inverse;
  long long AxestoTranspose(const int *X_in, int b) const {
    int x = X_in[0];
    int y = X_in[1];
    int n = 1 << b;
    int rx, ry, s, d = 0;
    for (s = n / 2; s > 0; s /= 2) {
      rx = (x & s) > 0;
      ry = (y & s) > 0;
      d += s * s * ((3 * rx) ^ ry);
      rot(n, &x, &y, rx, ry);
    }
    return d;
  }
  void TransposetoAxes(long long index, int *X, int b) const {
    int n = 1 << b;
    long long rx, ry, s, t = index;
    X[0] = 0;
    X[1] = 0;
    for (s = 1; s < n; s *= 2) {
      rx = 1 & (t / 2);
      ry = 1 & (t ^ rx);
      rot(s, &X[0], &X[1], rx, ry);
      X[0] += s * rx;
      X[1] += s * ry;
      t /= 4;
    }
  }
  void rot(long long n, int *x, int *y, long long rx, long long ry) const {
    if (ry == 0) {
      if (rx == 1) {
        *x = n - 1 - *x;
        *y = n - 1 - *y;
      }
      int t = *x;
      *x = *y;
      *y = t;
    }
  }
  SpaceCurve(int a_BX, int a_BY) : BX(a_BX), BY(a_BY) {
    const int n_max = std::max(BX, BY);
    base_level = (log(n_max) / log(2));
    if (base_level < (double)(log(n_max) / log(2)))
      base_level++;
    i_inverse.resize(sim.levelMax);
    j_inverse.resize(sim.levelMax);
    Zsave.resize(sim.levelMax);
    {
      const int l = 0;
      const int aux = pow(pow(2, l), 2);
      i_inverse[l].resize(BX * BY * aux, -1);
      j_inverse[l].resize(BX * BY * aux, -1);
      Zsave[l].resize(BX * BY * aux, -1);
    }
    isRegular = true;
#pragma omp parallel for collapse(2)
    for (int j = 0; j < BY; j++)
      for (int i = 0; i < BX; i++) {
        const int c[2] = {i, j};
        long long index = AxestoTranspose(c, base_level);
        long long substract = 0;
        for (long long h = 0; h < index; h++) {
          int X[2] = {0, 0};
          TransposetoAxes(h, X, base_level);
          if (X[0] >= BX || X[1] >= BY)
            substract++;
        }
        index -= substract;
        if (substract > 0)
          isRegular = false;
        i_inverse[0][index] = i;
        j_inverse[0][index] = j;
        Zsave[0][j * BX + i] = index;
      }
  }
  long long forward(const int l, const int i, const int j) const {
    const int aux = 1 << l;
    if (l >= sim.levelMax)
      return 0;
    long long retval;
    if (!isRegular) {
      const int I = i / aux;
      const int J = j / aux;
      const int c2_a[2] = {i - I * aux, j - J * aux};
      retval = AxestoTranspose(c2_a, l);
      retval += IJ_to_index(I, J) * aux * aux;
    } else {
      const int c2_a[2] = {i, j};
      retval = AxestoTranspose(c2_a, l + base_level);
    }
    return retval;
  }
  void inverse(long long Z, int l, int &i, int &j) const {
    if (isRegular) {
      int X[2] = {0, 0};
      TransposetoAxes(Z, X, l + base_level);
      i = X[0];
      j = X[1];
    } else {
      int aux = 1 << l;
      long long Zloc = Z % (aux * aux);
      int X[2] = {0, 0};
      TransposetoAxes(Zloc, X, l);
      long long index = Z / (aux * aux);
      int I, J;
      index_to_IJ(index, I, J);
      i = X[0] + I * aux;
      j = X[1] + J * aux;
    }
    return;
  }
  long long IJ_to_index(int I, int J) const {
    long long index = Zsave[0][J * BX + I];
    return index;
  }
  void index_to_IJ(long long index, int &I, int &J) const {
    I = i_inverse[0][index];
    J = j_inverse[0][index];
    return;
  }
  long long Encode(int level, int index[2]) {
    long long retval = 0;
    int ix = index[0];
    int iy = index[1];
    for (int l = level; l >= 0; l--) {
      long long Zp = forward(l, ix, iy);
      retval += Zp;
      ix /= 2;
      iy /= 2;
    }
    ix = 2 * index[0];
    iy = 2 * index[1];
    for (int l = level + 1; l < sim.levelMax; l++) {
      long long Zc = forward(l, ix, iy);
      Zc -= Zc % 4;
      retval += Zc;
      int ix1, iy1;
      inverse(Zc, l, ix1, iy1);
      ix = 2 * ix1;
      iy = 2 * iy1;
    }
    retval += level;
    return retval;
  }
};
static long long getZforward(int level, int i, int j) {
  return sim.space_curve->forward(level, i % (1 << level * sim.bpdx),
                                  j % (1 << level * sim.bpdy));
}
struct Value {
  std::string content;
  Value() = default;
  Value(const std::string &content_) : content(content_) {}
  Value(const Value &c) = default;
  Value &operator=(const Value &rhs) {
    if (this != &rhs)
      content = rhs.content;
    return *this;
  }
  double asDouble(double def = 0) {
    if (content == "") {
      std::ostringstream sbuf;
      sbuf << def;
      content = sbuf.str();
    }
    return (double)atof(content.c_str());
  }
  int asInt(int def = 0) {
    if (content == "") {
      std::ostringstream sbuf;
      sbuf << def;
      content = sbuf.str();
    }
    return atoi(content.c_str());
  }
  std::string asString(const std::string &def) {
    if (content == "")
      content = def;
    return content;
  }
};
struct CommandlineParser {
  bool bStrictMode;
  std::map<std::string, Value> mapArguments;
  void set_strict_mode() { bStrictMode = true; }
  void unset_strict_mode() { bStrictMode = false; }
  CommandlineParser(const int argc, char **argv) : bStrictMode(false) {
    for (int i = 1; i < argc; i++)
      if (argv[i][0] == '-') {
        std::string values = "";
        int itemCount = 0;
        for (int j = i + 1; j < argc; j++) {
          const bool leadingDash = (argv[j][0] == '-');
          char *end = NULL;
          strtod(argv[j], &end);
          const bool isNumeric = end != argv[j];
          if (leadingDash && !isNumeric)
            break;
          else {
            if (std::strcmp(values.c_str(), ""))
              values += ' ';
            values += argv[j];
            itemCount++;
          }
        }
        if (itemCount == 0)
          values = "true";
        std::string key(argv[i]);
        key.erase(0, 1);
        if (key[0] == '+') {
          key.erase(0, 1);
          mapArguments[key] = Value(values);
        } else {
          if (mapArguments.find(key) == mapArguments.end())
            mapArguments[key] = Value(values);
        }
        i += itemCount;
      }
  }
  Value &operator()(std::string key) {
    if (key[0] == '-')
      key.erase(0, 1);
    if (bStrictMode) {
      if (mapArguments.find(key) == mapArguments.end()) {
        printf("runtime %s is not set\n", key.data());
        abort();
      }
    }
    return mapArguments[key];
  }
};
enum State : signed char { Leave = 0, Refine = 1, Compress = -1 };
struct BlockCase;
struct BlockInfo {
  bool changed2;
  double h, origin[2];
  enum State state;
  int index[3], level;
  long long id, id2, halo_id, Z, Zchild[2][2], Znei[3][3], Zparent;
  void *block{nullptr};
  BlockCase *auxiliary;
  bool operator<(const BlockInfo &other) const { return id2 < other.id2; }
};
struct BlockCase {
  BlockCase(){};
  uint8_t *d[4];
  int level;
  long long Z;
};
struct StencilInfo {
  int sx;
  int sy;
  int ex;
  int ey;
  bool tensorial;
  StencilInfo() {}
  StencilInfo(int _sx, int _sy, int _ex, int _ey, bool _tensorial)
      : sx(_sx), sy(_sy), ex(_ex), ey(_ey), tensorial(_tensorial) {}
  StencilInfo(const StencilInfo &c) = default;
  std::vector<int> _all() const {
    int extra[] = {sx, sy, ex, ey, (int)tensorial};
    std::vector<int> all;
    all.insert(all.end(), extra, extra + sizeof(extra) / sizeof(int));
    return all;
  }
  bool operator<(StencilInfo s) const {
    std::vector<int> me = _all(), you = s._all();
    const int N = std::min(me.size(), you.size());
    for (int i = 0; i < N; ++i)
      if (me[i] < you[i])
        return true;
      else if (me[i] > you[i])
        return false;
    return me.size() < you.size();
  }
};
struct Interface {
  BlockInfo *infos[2];
  int icode[2];
  bool CoarseStencil;
  bool ToBeKept;
  int dis;
  Interface(BlockInfo &i0, BlockInfo &i1, const int a_icode0,
            const int a_icode1) {
    infos[0] = &i0;
    infos[1] = &i1;
    icode[0] = a_icode0;
    icode[1] = a_icode1;
    CoarseStencil = false;
    ToBeKept = true;
    dis = 0;
  }
  bool operator<(const Interface &other) const {
    if (infos[0]->id2 == other.infos[0]->id2) {
      if (icode[0] == other.icode[0]) {
        if (infos[1]->id2 == other.infos[1]->id2) {
          return (icode[1] < other.icode[1]);
        }
        return (infos[1]->id2 < other.infos[1]->id2);
      }
      return (icode[0] < other.icode[0]);
    }
    return (infos[0]->id2 < other.infos[0]->id2);
  }
};
struct Range {
  std::vector<int> removedIndices;
  int index;
  int sx;
  int sy;
  int sz;
  int ex;
  int ey;
  int ez;
  bool needed{true};
  bool avg_down{true};
  bool contains(Range &r) const {
    if (avg_down != r.avg_down)
      return false;
    int V = (ez - sz) * (ey - sy) * (ex - sx);
    int Vr = (r.ez - r.sz) * (r.ey - r.sy) * (r.ex - r.sx);
    return (sx <= r.sx && r.ex <= ex) && (sy <= r.sy && r.ey <= ey) &&
           (sz <= r.sz && r.ez <= ez) && (Vr < V);
  }
  void Remove(const Range &other) {
    size_t s = removedIndices.size();
    removedIndices.resize(s + other.removedIndices.size());
    for (size_t i = 0; i < other.removedIndices.size(); i++)
      removedIndices[s + i] = other.removedIndices[i];
  }
};
struct UnPackInfo {
  int offset;
  int lx;
  int ly;
  int lz;
  int srcxstart;
  int srcystart;
  int srczstart;
  int LX;
  int LY;
  int CoarseVersionOffset;
  int CoarseVersionLX;
  int CoarseVersionLY;
  int CoarseVersionsrcxstart;
  int CoarseVersionsrcystart;
  int CoarseVersionsrczstart;
  int level;
  int icode;
  int rank;
  int index_0;
  int index_1;
  int index_2;
  long long IDreceiver;
};
struct StencilManager {
  const StencilInfo stencil;
  const StencilInfo Cstencil;
  int nX;
  int nY;
  int nZ;
  int sLength[3 * 27 * 3];
  std::array<Range, 3 * 27> AllStencils;
  Range Coarse_Range;
  StencilManager(StencilInfo a_stencil, StencilInfo a_Cstencil, int a_nX,
                 int a_nY, int a_nZ)
      : stencil(a_stencil), Cstencil(a_Cstencil), nX(a_nX), nY(a_nY), nZ(a_nZ) {
    const int sC[3] = {(stencil.sx - 1) / 2 + Cstencil.sx,
                       (stencil.sy - 1) / 2 + Cstencil.sy, (0 - 1) / 2 + 0};
    const int eC[3] = {stencil.ex / 2 + Cstencil.ex,
                       stencil.ey / 2 + Cstencil.ey, 1 / 2 + 1};
    for (int icode = 0; icode < 27; icode++) {
      const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                           (icode / 9) % 3 - 1};
      Range &range0 = AllStencils[icode];
      range0.sx = code[0] < 1 ? (code[0] < 0 ? nX + stencil.sx : 0) : 0;
      range0.sy = code[1] < 1 ? (code[1] < 0 ? nY + stencil.sy : 0) : 0;
      range0.sz = code[2] < 1 ? (code[2] < 0 ? nZ : 0) : 0;
      range0.ex = code[0] < 1 ? nX : stencil.ex - 1;
      range0.ey = code[1] < 1 ? nY : stencil.ey - 1;
      range0.ez = code[2] < 1 ? nZ : 0;
      sLength[3 * icode + 0] = range0.ex - range0.sx;
      sLength[3 * icode + 1] = range0.ey - range0.sy;
      sLength[3 * icode + 2] = range0.ez - range0.sz;
      Range &range1 = AllStencils[icode + 27];
      range1.sx = code[0] < 1 ? (code[0] < 0 ? nX + 2 * stencil.sx : 0) : 0;
      range1.sy = code[1] < 1 ? (code[1] < 0 ? nY + 2 * stencil.sy : 0) : 0;
      range1.sz = code[2] < 1 ? (code[2] < 0 ? nZ : 0) : 0;
      range1.ex = code[0] < 1 ? nX : 2 * (stencil.ex - 1);
      range1.ey = code[1] < 1 ? nY : 2 * (stencil.ey - 1);
      range1.ez = code[2] < 1 ? nZ : 0;
      sLength[3 * (icode + 27) + 0] = (range1.ex - range1.sx) / 2;
      sLength[3 * (icode + 27) + 1] = (range1.ey - range1.sy) / 2;
      sLength[3 * (icode + 27) + 2] = 1;
      Range &range2 = AllStencils[icode + 2 * 27];
      range2.sx = code[0] < 1 ? (code[0] < 0 ? nX / 2 + sC[0] : 0) : 0;
      range2.sy = code[1] < 1 ? (code[1] < 0 ? nY / 2 + sC[1] : 0) : 0;
      range2.ex = code[0] < 1 ? nX / 2 : eC[0] - 1;
      range2.ey = code[1] < 1 ? nY / 2 : eC[1] - 1;
      range2.sz = 0;
      range2.ez = 1;
      sLength[3 * (icode + 2 * 27) + 0] = range2.ex - range2.sx;
      sLength[3 * (icode + 2 * 27) + 1] = range2.ey - range2.sy;
      sLength[3 * (icode + 2 * 27) + 2] = range2.ez - range2.sz;
    }
  }
  void CoarseStencilLength(const int icode, int *L) const {
    L[0] = sLength[3 * (icode + 2 * 27) + 0];
    L[1] = sLength[3 * (icode + 2 * 27) + 1];
    L[2] = sLength[3 * (icode + 2 * 27) + 2];
  }
  void DetermineStencilLength(const int level_sender, const int level_receiver,
                              const int icode, int *L) {
    if (level_sender == level_receiver) {
      L[0] = sLength[3 * icode + 0];
      L[1] = sLength[3 * icode + 1];
      L[2] = sLength[3 * icode + 2];
    } else if (level_sender > level_receiver) {
      L[0] = sLength[3 * (icode + 27) + 0];
      L[1] = sLength[3 * (icode + 27) + 1];
      L[2] = sLength[3 * (icode + 27) + 2];
    } else {
      L[0] = sLength[3 * (icode + 2 * 27) + 0];
      L[1] = sLength[3 * (icode + 2 * 27) + 1];
      L[2] = sLength[3 * (icode + 2 * 27) + 2];
    }
  }
  Range &DetermineStencil(const Interface &f, bool CoarseVersion = false) {
    if (CoarseVersion) {
      AllStencils[f.icode[1] + 2 * 27].needed = true;
      return AllStencils[f.icode[1] + 2 * 27];
    } else {
      if (f.infos[0]->level == f.infos[1]->level) {
        AllStencils[f.icode[1]].needed = true;
        return AllStencils[f.icode[1]];
      } else if (f.infos[0]->level > f.infos[1]->level) {
        AllStencils[f.icode[1] + 27].needed = true;
        return AllStencils[f.icode[1] + 27];
      } else {
        Coarse_Range.needed = true;
        const int code[3] = {f.icode[1] % 3 - 1, (f.icode[1] / 3) % 3 - 1,
                             (f.icode[1] / 9) % 3 - 1};
        const int s[3] = {
            code[0] < 1
                ? (code[0] < 0 ? ((stencil.sx - 1) / 2 + Cstencil.sx) : 0)
                : nX / 2,
            code[1] < 1
                ? (code[1] < 0 ? ((stencil.sy - 1) / 2 + Cstencil.sy) : 0)
                : nY / 2,
            code[2] < 1 ? (code[2] < 0 ? ((0 - 1) / 2) : 0) : nZ / 2};
        const int e[3] = {
            code[0] < 1 ? (code[0] < 0 ? 0 : nX / 2)
                        : nX / 2 + stencil.ex / 2 + Cstencil.ex - 1,
            code[1] < 1 ? (code[1] < 0 ? 0 : nY / 2)
                        : nY / 2 + stencil.ey / 2 + Cstencil.ey - 1,
            code[2] < 1 ? (code[2] < 0 ? 0 : nZ / 2) : nZ / 2};
        const int base[3] = {(f.infos[1]->index[0] + code[0]) % 2,
                             (f.infos[1]->index[1] + code[1]) % 2,
                             (f.infos[1]->index[2] + code[2]) % 2};
        int Cindex_true[3];
        for (int d = 0; d < 3; d++)
          Cindex_true[d] = f.infos[1]->index[d] + code[d];
        int CoarseEdge[3];
        CoarseEdge[0] = (code[0] == 0) ? 0
                        : (((f.infos[1]->index[0] % 2 == 0) &&
                            (Cindex_true[0] > f.infos[1]->index[0])) ||
                           ((f.infos[1]->index[0] % 2 == 1) &&
                            (Cindex_true[0] < f.infos[1]->index[0])))
                            ? 1
                            : 0;
        CoarseEdge[1] = (code[1] == 0) ? 0
                        : (((f.infos[1]->index[1] % 2 == 0) &&
                            (Cindex_true[1] > f.infos[1]->index[1])) ||
                           ((f.infos[1]->index[1] % 2 == 1) &&
                            (Cindex_true[1] < f.infos[1]->index[1])))
                            ? 1
                            : 0;
        CoarseEdge[2] = 0;
        Coarse_Range.sx = s[0] + std::max(code[0], 0) * nX / 2 +
                          (1 - abs(code[0])) * base[0] * nX / 2 - code[0] * nX +
                          CoarseEdge[0] * code[0] * nX / 2;
        Coarse_Range.sy = s[1] + std::max(code[1], 0) * nY / 2 +
                          (1 - abs(code[1])) * base[1] * nY / 2 - code[1] * nY +
                          CoarseEdge[1] * code[1] * nY / 2;
        Coarse_Range.sz = 0;
        Coarse_Range.ex = e[0] + std::max(code[0], 0) * nX / 2 +
                          (1 - abs(code[0])) * base[0] * nX / 2 - code[0] * nX +
                          CoarseEdge[0] * code[0] * nX / 2;
        Coarse_Range.ey = e[1] + std::max(code[1], 0) * nY / 2 +
                          (1 - abs(code[1])) * base[1] * nY / 2 - code[1] * nY +
                          CoarseEdge[1] * code[1] * nY / 2;
        Coarse_Range.ez = 1;
        return Coarse_Range;
      }
    }
  }
  void __FixDuplicates(const Interface &f, const Interface &f_dup, int lx,
                       int ly, int lz, int lx_dup, int ly_dup, int lz_dup,
                       int &sx, int &sy, int &sz) {
    const BlockInfo &receiver = *f.infos[1];
    const BlockInfo &receiver_dup = *f_dup.infos[1];
    if (receiver.level >= receiver_dup.level) {
      int icode_dup = f_dup.icode[1];
      const int code_dup[3] = {icode_dup % 3 - 1, (icode_dup / 3) % 3 - 1,
                               (icode_dup / 9) % 3 - 1};
      sx = (lx == lx_dup || code_dup[0] != -1) ? 0 : lx - lx_dup;
      sy = (ly == ly_dup || code_dup[1] != -1) ? 0 : ly - ly_dup;
      sz = (lz == lz_dup || code_dup[2] != -1) ? 0 : lz - lz_dup;
    } else {
      Range &range = DetermineStencil(f);
      Range &range_dup = DetermineStencil(f_dup);
      sx = range_dup.sx - range.sx;
      sy = range_dup.sy - range.sy;
      sz = range_dup.sz - range.sz;
    }
  }
  void __FixDuplicates2(const Interface &f, const Interface &f_dup, int &sx,
                        int &sy, int &sz) {
    if (f.infos[0]->level != f.infos[1]->level ||
        f_dup.infos[0]->level != f_dup.infos[1]->level)
      return;
    Range &range = DetermineStencil(f, true);
    Range &range_dup = DetermineStencil(f_dup, true);
    sx = range_dup.sx - range.sx;
    sy = range_dup.sy - range.sy;
    sz = range_dup.sz - range.sz;
  }
};
struct HaloBlockGroup {
  std::vector<BlockInfo *> myblocks;
  std::set<int> myranks;
  bool ready = false;
};
struct PackInfo {
  Real *block;
  Real *pack;
  int sx;
  int sy;
  int sz;
  int ex;
  int ey;
  int ez;
};
static std::vector<Range *> keepEl(std::vector<Range> compass[27]) {
  std::vector<Range *> retval;
  for (int i = 0; i < 27; i++)
    for (size_t j = 0; j < compass[i].size(); j++)
      if (compass[i][j].needed)
        retval.push_back(&compass[i][j]);
  return retval;
}
static void needed0(std::vector<Range> compass[27], std::vector<int> &v) {
  static constexpr std::array<int, 3> faces_and_edges[18] = {
      {0, 1, 1}, {2, 1, 1}, {1, 0, 1}, {1, 2, 1}, {1, 1, 0}, {1, 1, 2},
      {0, 0, 1}, {0, 2, 1}, {2, 0, 1}, {2, 2, 1}, {1, 0, 0}, {1, 0, 2},
      {1, 2, 0}, {1, 2, 2}, {0, 1, 0}, {0, 1, 2}, {2, 1, 0}, {2, 1, 2}};
  for (auto &f : faces_and_edges)
    if (compass[f[0] + f[1] * 3 + f[2] * 9].size() != 0) {
      bool needme = false;
      auto &me = compass[f[0] + f[1] * 3 + f[2] * 9];
      for (size_t j1 = 0; j1 < me.size(); j1++)
        if (me[j1].needed) {
          needme = true;
          for (size_t j2 = 0; j2 < me.size(); j2++)
            if (me[j2].needed && me[j2].contains(me[j1])) {
              me[j1].needed = false;
              me[j2].removedIndices.push_back(me[j1].index);
              me[j2].Remove(me[j1]);
              v.push_back(me[j1].index);
              break;
            }
        }
      if (!needme)
        continue;
      int imax = (f[0] == 1) ? 2 : f[0];
      int imin = (f[0] == 1) ? 0 : f[0];
      int jmax = (f[1] == 1) ? 2 : f[1];
      int jmin = (f[1] == 1) ? 0 : f[1];
      int kmax = (f[2] == 1) ? 2 : f[2];
      int kmin = (f[2] == 1) ? 0 : f[2];
      for (int k = kmin; k <= kmax; k++)
        for (int j = jmin; j <= jmax; j++)
          for (int i = imin; i <= imax; i++) {
            if (i == f[0] && j == f[1] && k == f[2])
              continue;
            auto &other = compass[i + j * 3 + k * 9];
            for (size_t j1 = 0; j1 < other.size(); j1++) {
              auto &o = other[j1];
              if (o.needed)
                for (size_t k1 = 0; k1 < me.size(); k1++) {
                  auto &m = me[k1];
                  if (m.needed && m.contains(o)) {
                    o.needed = false;
                    m.removedIndices.push_back(o.index);
                    m.Remove(o);
                    v.push_back(o.index);
                    break;
                  }
                }
            }
          }
    }
}
struct DuplicatesManager {
  std::vector<int> positions;
  std::vector<size_t> sizes;
  void Add(int r, int index) {
    if (sizes[r] == 0)
      positions[r] = index;
    sizes[r]++;
  }
};
template <typename TGrid> struct Synchronizer {
  bool use_averages;
  const int dim;
  const int NC;
  std::set<int> Neighbors;
  std::unordered_map<int, MPI_Request *> mapofrequests;
  std::unordered_map<std::string, HaloBlockGroup> mapofHaloBlockGroups;
  std::vector<BlockInfo *> halo_blocks;
  std::vector<BlockInfo *> inner_blocks;
  std::vector<int> recv_buffer_size;
  std::vector<int> send_buffer_size;
  std::vector<MPI_Request> requests;
  std::vector<std::vector<Interface>> recv_interfaces;
  std::vector<std::vector<Interface>> send_interfaces;
  std::vector<std::vector<int>> ToBeAveragedDown;
  std::vector<std::vector<PackInfo>> send_packinfos;
  std::vector<std::vector<Real>> recv_buffer;
  std::vector<std::vector<Real>> send_buffer;
  std::vector<std::vector<UnPackInfo>> myunpacks;
  StencilInfo Cstencil;
  StencilInfo stencil;
  StencilManager SM;
  TGrid *grid;
  std::vector<BlockInfo *> dummy_vector;
  Synchronizer(StencilInfo a_stencil, StencilInfo a_Cstencil, TGrid *_grid,
               int dim)
      : dim(dim), stencil(a_stencil), Cstencil(a_Cstencil),
        SM(a_stencil, a_Cstencil, _BS_, _BS_, 1), NC(dim) {
    grid = _grid;
    use_averages = (stencil.tensorial || stencil.sx < -2 || stencil.sy < -2 ||
                    0 < -2 || stencil.ex > 3 || stencil.ey > 3);
    send_interfaces.resize(sim.size);
    recv_interfaces.resize(sim.size);
    send_packinfos.resize(sim.size);
    send_buffer_size.resize(sim.size);
    recv_buffer_size.resize(sim.size);
    send_buffer.resize(sim.size);
    recv_buffer.resize(sim.size);
    ToBeAveragedDown.resize(sim.size);
  }
  std::vector<BlockInfo *> &avail_next() {
    bool done = false;
    auto it = mapofHaloBlockGroups.begin();
    while (done == false) {
      done = true;
      it = mapofHaloBlockGroups.begin();
      while (it != mapofHaloBlockGroups.end()) {
        if ((it->second).ready == false) {
          std::set<int> ranks = (it->second).myranks;
          int flag = 0;
          for (auto r : ranks) {
            const auto retval = mapofrequests.find(r);
            MPI_Test(retval->second, &flag, MPI_STATUS_IGNORE);
            if (flag == false)
              break;
          }
          if (flag == 1) {
            (it->second).ready = true;
            return (it->second).myblocks;
          }
        }
        done = done && (it->second).ready;
        it++;
      }
    }
    return dummy_vector;
  }
  bool UseCoarseStencil(const Interface &f) {
    BlockInfo &a = *f.infos[0];
    BlockInfo &b = *f.infos[1];
    if (a.level == 0 || (!use_averages))
      return false;
    int imin[2];
    int imax[2];
    const int aux = 1 << a.level;
    const bool periodic0[2] = {sim.bcx == periodic, sim.bcy == periodic};
    const int blocks[3] = {sim.bpdx * aux - 1, sim.bpdy * aux - 1};
    for (int d = 0; d < 2; d++) {
      imin[d] = (a.index[d] < b.index[d]) ? 0 : -1;
      imax[d] = (a.index[d] > b.index[d]) ? 0 : +1;
      if (periodic0[d]) {
        if (a.index[d] == 0 && b.index[d] == blocks[d])
          imin[d] = -1;
        if (b.index[d] == 0 && a.index[d] == blocks[d])
          imax[d] = +1;
      } else {
        if (a.index[d] == 0 && b.index[d] == 0)
          imin[d] = 0;
        if (a.index[d] == blocks[d] && b.index[d] == blocks[d])
          imax[d] = 0;
      }
    }
    bool retval = false;
    for (int i1 = imin[1]; i1 <= imax[1]; i1++)
      for (int i0 = imin[0]; i0 <= imax[0]; i0++) {
        if ((grid->Tree0(a.level, a.Znei[1 + i0][1 + i1])) == -2) {
          retval = true;
          break;
        }
      }
    return retval;
  }
  void _Setup() {
    DuplicatesManager DM;
    std::vector<int> offsets(sim.size, 0);
    std::vector<int> offsets_recv(sim.size, 0);
    DM.positions.resize(sim.size);
    DM.sizes.resize(sim.size);
    Neighbors.clear();
    inner_blocks.clear();
    halo_blocks.clear();
    for (int r = 0; r < sim.size; r++) {
      send_interfaces[r].clear();
      recv_interfaces[r].clear();
      send_buffer_size[r] = 0;
    }
    for (size_t i = 0; i < myunpacks.size(); i++)
      myunpacks[i].clear();
    myunpacks.clear();
    std::vector<Range> compass[27];
    for (BlockInfo &info : grid->infos) {
      info.halo_id = -1;
      bool xskin =
          info.index[0] == 0 || info.index[0] == ((sim.bpdx << info.level) - 1);
      bool yskin =
          info.index[1] == 0 || info.index[1] == ((sim.bpdy << info.level) - 1);
      int xskip = info.index[0] == 0 ? -1 : 1;
      int yskip = info.index[1] == 0 ? -1 : 1;

      bool isInner = true;
      std::vector<int> ToBeChecked;
      bool Coarsened = false;
      for (int icode = 0; icode < 27; icode++) {
        if (icode == 1 * 1 + 3 * 1 + 9 * 1)
          continue;
        int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, (icode / 9) % 3 - 1};
        if (code[2] != 0)
          continue;
        if (!(sim.bcx == periodic) && code[0] == xskip && xskin)
          continue;
        if (!(sim.bcy == periodic) && code[1] == yskip && yskin)
          continue;
        int &infoNeiTree =
            grid->Tree0(info.level, info.Znei[1 + code[0]][1 + code[1]]);
        if (infoNeiTree >= 0 && infoNeiTree != sim.rank) {
          isInner = false;
          Neighbors.insert(infoNeiTree);
          BlockInfo &infoNei =
              grid->get(info.level, info.Znei[1 + code[0]][1 + code[1]]);
          int icode2 = (-code[0] + 1) + (-code[1] + 1) * 3 + (-code[2] + 1) * 9;
          send_interfaces[infoNeiTree].push_back(
              {info, infoNei, icode, icode2});
          recv_interfaces[infoNeiTree].push_back(
              {infoNei, info, icode2, icode});
          ToBeChecked.push_back(infoNeiTree);
          ToBeChecked.push_back((int)send_interfaces[infoNeiTree].size() - 1);
          ToBeChecked.push_back((int)recv_interfaces[infoNeiTree].size() - 1);
          DM.Add(infoNeiTree, (int)send_interfaces[infoNeiTree].size() - 1);
        } else if (infoNeiTree == -2) {
          Coarsened = true;
          BlockInfo &infoNei =
              grid->get(info.level, info.Znei[1 + code[0]][1 + code[1]]);
          int infoNeiCoarserrank = grid->Tree0(info.level - 1, infoNei.Zparent);
          if (infoNeiCoarserrank != sim.rank) {
            isInner = false;
            Neighbors.insert(infoNeiCoarserrank);
            BlockInfo &infoNeiCoarser =
                grid->get(infoNei.level - 1, infoNei.Zparent);
            int icode2 =
                (-code[0] + 1) + (-code[1] + 1) * 3 + (-code[2] + 1) * 9;
            int Bmax[3] = {sim.bpdx << (info.level - 1),
                           sim.bpdy << (info.level - 1), 1 << (info.level - 1)};
            int test_idx[3] = {
                (infoNeiCoarser.index[0] - code[0] + Bmax[0]) % Bmax[0],
                (infoNeiCoarser.index[1] - code[1] + Bmax[1]) % Bmax[1],
                (infoNeiCoarser.index[2] - code[2] + Bmax[2]) % Bmax[2]};
            if (info.index[0] / 2 == test_idx[0] &&
                info.index[1] / 2 == test_idx[1] &&
                info.index[2] / 2 == test_idx[2]) {
              send_interfaces[infoNeiCoarserrank].push_back(
                  {info, infoNeiCoarser, icode, icode2});
              recv_interfaces[infoNeiCoarserrank].push_back(
                  {infoNeiCoarser, info, icode2, icode});
              DM.Add(infoNeiCoarserrank,
                     (int)send_interfaces[infoNeiCoarserrank].size() - 1);
              if (abs(code[0]) + abs(code[1]) + abs(code[2]) == 1) {
                int d0 = abs(code[1] + 2 * code[2]);
                int d1 = (d0 + 1) % 3;
                int d2 = (d0 + 2) % 3;
                int code3[3];
                code3[d0] = code[d0];
                code3[d1] = -2 * (info.index[d1] % 2) + 1;
                code3[d2] = -2 * (info.index[d2] % 2) + 1;
                int icode3 =
                    (code3[0] + 1) + (code3[1] + 1) * 3 + (code3[2] + 1) * 9;
                int code4[3];
                code4[d0] = code[d0];
                code4[d1] = code3[d1];
                code4[d2] = 0;
                int icode4 =
                    (code4[0] + 1) + (code4[1] + 1) * 3 + (code4[2] + 1) * 9;
                int code5[3];
                code5[d0] = code[d0];
                code5[d1] = 0;
                code5[d2] = code3[d2];
                int icode5 =
                    (code5[0] + 1) + (code5[1] + 1) * 3 + (code5[2] + 1) * 9;
                if (code3[2] == 0)
                  recv_interfaces[infoNeiCoarserrank].push_back(
                      {infoNeiCoarser, info, icode2, icode3});
                if (code4[2] == 0)
                  recv_interfaces[infoNeiCoarserrank].push_back(
                      {infoNeiCoarser, info, icode2, icode4});
                if (code5[2] == 0)
                  recv_interfaces[infoNeiCoarserrank].push_back(
                      {infoNeiCoarser, info, icode2, icode5});
              }
            }
          }
        } else if (infoNeiTree == -1) {
          BlockInfo &infoNei =
              grid->get(info.level, info.Znei[1 + code[0]][1 + code[1]]);
          int Bstep = 1;
          if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 2))
            Bstep = 3;
          else if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 3))
            Bstep = 4;
          for (int B = 0; B <= 3; B += Bstep) {
            if (Bstep == 1 && B >= 2)
              continue;
            if (Bstep > 1 && B >= 1)
              continue;
            int temp = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
            long long nFine =
                infoNei.Zchild[std::max(-code[0], 0) +
                               (B % 2) * std::max(0, 1 - abs(code[0]))]
                              [std::max(-code[1], 0) +
                               temp * std::max(0, 1 - abs(code[1]))];
            int infoNeiFinerrank = grid->Tree0(info.level + 1, nFine);
            if (infoNeiFinerrank != sim.rank) {
              isInner = false;
              Neighbors.insert(infoNeiFinerrank);
              BlockInfo &infoNeiFiner = grid->get(info.level + 1, nFine);
              int icode2 =
                  (-code[0] + 1) + (-code[1] + 1) * 3 + (-code[2] + 1) * 9;
              send_interfaces[infoNeiFinerrank].push_back(
                  {info, infoNeiFiner, icode, icode2});
              recv_interfaces[infoNeiFinerrank].push_back(
                  {infoNeiFiner, info, icode2, icode});
              DM.Add(infoNeiFinerrank,
                     (int)send_interfaces[infoNeiFinerrank].size() - 1);
              if (Bstep == 1) {
                int d0 = abs(code[1] + 2 * code[2]);
                int d1 = (d0 + 1) % 3;
                int d2 = (d0 + 2) % 3;
                int code3[3];
                code3[d0] = -code[d0];
                code3[d1] = -2 * (infoNeiFiner.index[d1] % 2) + 1;
                code3[d2] = -2 * (infoNeiFiner.index[d2] % 2) + 1;
                int icode3 =
                    (code3[0] + 1) + (code3[1] + 1) * 3 + (code3[2] + 1) * 9;
                int code4[3];
                code4[d0] = -code[d0];
                code4[d1] = code3[d1];
                code4[d2] = 0;
                int icode4 =
                    (code4[0] + 1) + (code4[1] + 1) * 3 + (code4[2] + 1) * 9;
                int code5[3];
                code5[d0] = -code[d0];
                code5[d1] = 0;
                code5[d2] = code3[d2];
                int icode5 =
                    (code5[0] + 1) + (code5[1] + 1) * 3 + (code5[2] + 1) * 9;
                if (code3[2] == 0) {
                  send_interfaces[infoNeiFinerrank].push_back(
                      Interface(info, infoNeiFiner, icode, icode3));
                  DM.Add(infoNeiFinerrank,
                         (int)send_interfaces[infoNeiFinerrank].size() - 1);
                }
                if (code4[2] == 0) {
                  send_interfaces[infoNeiFinerrank].push_back(
                      Interface(info, infoNeiFiner, icode, icode4));
                  DM.Add(infoNeiFinerrank,
                         (int)send_interfaces[infoNeiFinerrank].size() - 1);
                }
                if (code5[2] == 0) {
                  send_interfaces[infoNeiFinerrank].push_back(
                      Interface(info, infoNeiFiner, icode, icode5));
                  DM.Add(infoNeiFinerrank,
                         (int)send_interfaces[infoNeiFinerrank].size() - 1);
                }
              }
            }
          }
        }
      }
      if (isInner) {
        info.halo_id = -1;
        inner_blocks.push_back(&info);
      } else {
        info.halo_id = halo_blocks.size();
        halo_blocks.push_back(&info);
        if (Coarsened) {
          for (size_t j = 0; j < ToBeChecked.size(); j += 3) {
            int r = ToBeChecked[j];
            int send = ToBeChecked[j + 1];
            int recv = ToBeChecked[j + 2];
            bool tmp = UseCoarseStencil(send_interfaces[r][send]);
            send_interfaces[r][send].CoarseStencil = tmp;
            recv_interfaces[r][recv].CoarseStencil = tmp;
          }
        }
        for (int r = 0; r < sim.size; r++)
          if (DM.sizes[r] > 0) {
            std::vector<Interface> &f = send_interfaces[r];
            int &total_size = send_buffer_size[r];
            bool skip_needed = false;
            const int nc = dim;
            std::sort(f.begin() + DM.positions[r],
                      f.begin() + DM.sizes[r] + DM.positions[r]);
            for (int i = 0; i < sizeof compass / sizeof *compass; i++)
              compass[i].clear();
            for (size_t i = 0; i < DM.sizes[r]; i++) {
              compass[f[i + DM.positions[r]].icode[0]].push_back(
                  SM.DetermineStencil(f[i + DM.positions[r]]));
              compass[f[i + DM.positions[r]].icode[0]].back().index =
                  i + DM.positions[r];
              compass[f[i + DM.positions[r]].icode[0]].back().avg_down =
                  (f[i + DM.positions[r]].infos[0]->level >
                   f[i + DM.positions[r]].infos[1]->level);
              if (skip_needed == false)
                skip_needed = f[i + DM.positions[r]].CoarseStencil;
            }
            if (skip_needed == false) {
              std::vector<int> remEl;
              needed0(compass, remEl);
              for (size_t k = 0; k < remEl.size(); k++)
                f[remEl[k]].ToBeKept = false;
            }
            int L[3] = {0, 0, 0};
            int Lc[3] = {0, 0, 0};
            for (auto &i : keepEl(compass)) {
              const int k = i->index;
              SM.DetermineStencilLength(f[k].infos[0]->level,
                                        f[k].infos[1]->level, f[k].icode[1], L);
              const int V = L[0] * L[1] * L[2];
              total_size += V;
              f[k].dis = offsets[r];
              if (f[k].CoarseStencil) {
                SM.CoarseStencilLength(f[k].icode[1], Lc);
                const int Vc = Lc[0] * Lc[1] * Lc[2];
                total_size += Vc;
                offsets[r] += Vc * nc;
              }
              offsets[r] += V * nc;
              for (size_t kk = 0; kk < (*i).removedIndices.size(); kk++)
                f[i->removedIndices[kk]].dis = f[k].dis;
            }
            DM.sizes[r] = 0;
          }
      }
      grid->get(info.level, info.Z).halo_id = info.halo_id;
    }
    myunpacks.resize(halo_blocks.size());
    for (int r = 0; r < sim.size; r++) {
      recv_buffer_size[r] = 0;
      std::sort(recv_interfaces[r].begin(), recv_interfaces[r].end());
      size_t counter = 0;
      while (counter < recv_interfaces[r].size()) {
        long long ID = recv_interfaces[r][counter].infos[0]->id2;
        size_t start = counter;
        size_t finish = start + 1;
        counter++;
        size_t j;
        for (j = counter; j < recv_interfaces[r].size(); j++) {
          if (recv_interfaces[r][j].infos[0]->id2 == ID)
            finish++;
          else
            break;
        }
        counter = j;
        std::vector<Interface> &f = recv_interfaces[r];
        int &total_size = recv_buffer_size[r];
        const int otherrank = r;
        bool skip_needed = false;
        const int nc = dim;
        for (int i = 0; i < sizeof compass / sizeof *compass; i++)
          compass[i].clear();
        for (size_t i = start; i < finish; i++) {
          compass[f[i].icode[0]].push_back(SM.DetermineStencil(f[i]));
          compass[f[i].icode[0]].back().index = i;
          compass[f[i].icode[0]].back().avg_down =
              (f[i].infos[0]->level > f[i].infos[1]->level);
          if (skip_needed == false)
            skip_needed = f[i].CoarseStencil;
        }
        if (skip_needed == false) {
          std::vector<int> remEl;
          needed0(compass, remEl);
          for (size_t k = 0; k < remEl.size(); k++)
            f[remEl[k]].ToBeKept = false;
        }
        for (auto &i : keepEl(compass)) {
          const int k = i->index;
          int L[3] = {0, 0, 0};
          int Lc[3] = {0, 0, 0};
          SM.DetermineStencilLength(f[k].infos[0]->level, f[k].infos[1]->level,
                                    f[k].icode[1], L);
          const int V = L[0] * L[1] * L[2];
          int Vc = 0;
          total_size += V;
          f[k].dis = offsets_recv[otherrank];
          UnPackInfo info = {f[k].dis,
                             L[0],
                             L[1],
                             L[2],
                             0,
                             0,
                             0,
                             L[0],
                             L[1],
                             -1,
                             0,
                             0,
                             0,
                             0,
                             0,
                             f[k].infos[0]->level,
                             f[k].icode[1],
                             otherrank,
                             f[k].infos[0]->index[0],
                             f[k].infos[0]->index[1],
                             f[k].infos[0]->index[2],
                             f[k].infos[1]->id2};
          if (f[k].CoarseStencil) {
            SM.CoarseStencilLength(f[k].icode[1], Lc);
            Vc = Lc[0] * Lc[1] * Lc[2];
            total_size += Vc;
            offsets_recv[otherrank] += Vc * nc;
            info.CoarseVersionOffset = V * nc;
            info.CoarseVersionLX = Lc[0];
            info.CoarseVersionLY = Lc[1];
          }
          offsets_recv[otherrank] += V * nc;
          myunpacks[f[k].infos[1]->halo_id].push_back(info);
          for (size_t kk = 0; kk < (*i).removedIndices.size(); kk++) {
            const int remEl1 = i->removedIndices[kk];
            SM.DetermineStencilLength(f[remEl1].infos[0]->level,
                                      f[remEl1].infos[1]->level,
                                      f[remEl1].icode[1], &L[0]);
            int srcx, srcy, srcz;
            SM.__FixDuplicates(f[k], f[remEl1], info.lx, info.ly, info.lz, L[0],
                               L[1], L[2], srcx, srcy, srcz);
            int Csrcx = 0;
            int Csrcy = 0;
            int Csrcz = 0;
            if (f[k].CoarseStencil)
              SM.__FixDuplicates2(f[k], f[remEl1], Csrcx, Csrcy, Csrcz);
            myunpacks[f[remEl1].infos[1]->halo_id].push_back(
                {info.offset,
                 L[0],
                 L[1],
                 L[2],
                 srcx,
                 srcy,
                 srcz,
                 info.LX,
                 info.LY,
                 info.CoarseVersionOffset,
                 info.CoarseVersionLX,
                 info.CoarseVersionLY,
                 Csrcx,
                 Csrcy,
                 Csrcz,
                 f[remEl1].infos[0]->level,
                 f[remEl1].icode[1],
                 otherrank,
                 f[remEl1].infos[0]->index[0],
                 f[remEl1].infos[0]->index[1],
                 f[remEl1].infos[0]->index[2],
                 f[remEl1].infos[1]->id2});
            f[remEl1].dis = info.offset;
          }
        }
      }
      send_buffer[r].resize(send_buffer_size[r] * NC);
      recv_buffer[r].resize(recv_buffer_size[r] * NC);
      send_packinfos[r].clear();
      ToBeAveragedDown[r].clear();
      for (int i = 0; i < (int)send_interfaces[r].size(); i++) {
        Interface &f = send_interfaces[r][i];
        if (!f.ToBeKept)
          continue;
        if (f.infos[0]->level <= f.infos[1]->level) {
          Range &range = SM.DetermineStencil(f);
          send_packinfos[r].push_back(
              {(Real *)f.infos[0]->block, &send_buffer[r][f.dis], range.sx,
               range.sy, range.sz, range.ex, range.ey, range.ez});
          if (f.CoarseStencil) {
            int V = (range.ex - range.sx) * (range.ey - range.sy) *
                    (range.ez - range.sz);
            ToBeAveragedDown[r].push_back(i);
            ToBeAveragedDown[r].push_back(f.dis + V * NC);
          }
        } else {
          ToBeAveragedDown[r].push_back(i);
          ToBeAveragedDown[r].push_back(f.dis);
        }
      }
    }
    mapofHaloBlockGroups.clear();
    for (auto &info : halo_blocks) {
      int id = info->halo_id;
      UnPackInfo *unpacks = myunpacks[id].data();
      std::set<int> ranks;
      for (size_t jj = 0; jj < myunpacks[id].size(); jj++) {
        UnPackInfo &unpack = unpacks[jj];
        ranks.insert(unpack.rank);
      }
      std::string set_ID;
      for (auto r : ranks) {
        std::stringstream ss;
        ss << std::setw(sim.size) << std::setfill('0') << r;
        std::string s = ss.str();
        set_ID += s;
      }
      auto retval = mapofHaloBlockGroups.find(set_ID);
      if (retval == mapofHaloBlockGroups.end()) {
        HaloBlockGroup temporary;
        temporary.myranks = ranks;
        temporary.myblocks.push_back(info);
        mapofHaloBlockGroups[set_ID] = temporary;
      } else {
        (retval->second).myblocks.push_back(info);
      }
    }
  }
  void sync0() {
    auto it = mapofHaloBlockGroups.begin();
    while (it != mapofHaloBlockGroups.end()) {
      (it->second).ready = false;
      it++;
    }
    const int timestamp = grid->timestamp;
    mapofrequests.clear();
    requests.clear();
    requests.reserve(2 * sim.size);
    for (auto r : Neighbors)
      if (recv_buffer_size[r] > 0) {
        requests.resize(requests.size() + 1);
        mapofrequests[r] = &requests.back();
        MPI_Irecv(&recv_buffer[r][0], recv_buffer_size[r] * NC, MPI_Real, r,
                  timestamp, MPI_COMM_WORLD, &requests.back());
      }
    for (int r = 0; r < sim.size; r++)
      if (send_buffer_size[r] != 0) {
#pragma omp parallel
        {
#pragma omp for
          for (size_t j = 0; j < ToBeAveragedDown[r].size(); j += 2) {
            const int i = ToBeAveragedDown[r][j];
            const int d = ToBeAveragedDown[r][j + 1];
            const Interface &f = send_interfaces[r][i];
            const int code[3] = {-(f.icode[0] % 3 - 1),
                                 -((f.icode[0] / 3) % 3 - 1),
                                 -((f.icode[0] / 9) % 3 - 1)};
            if (f.CoarseStencil) {
              Real *dst = send_buffer[r].data() + d;
              const BlockInfo *const info = f.infos[0];
              const int eC[3] = {(stencil.ex) / 2 + Cstencil.ex,
                                 (stencil.ey) / 2 + Cstencil.ey, 1};
              const int sC[3] = {(stencil.sx - 1) / 2 + Cstencil.sx,
                                 (stencil.sy - 1) / 2 + Cstencil.sy,
                                 (0 - 1) / 2 + 0};
              const int s[3] = {
                  code[0] < 1 ? (code[0] < 0 ? sC[0] : 0) : _BS_ / 2,
                  code[1] < 1 ? (code[1] < 0 ? sC[1] : 0) : _BS_ / 2,
                  code[2] < 1 ? (code[2] < 0 ? sC[2] : 0) : 1 / 2};
              const int e[3] = {code[0] < 1 ? (code[0] < 0 ? 0 : _BS_ / 2)
                                            : _BS_ / 2 + eC[0] - 1,
                                code[1] < 1 ? (code[1] < 0 ? 0 : _BS_ / 2)
                                            : _BS_ / 2 + eC[1] - 1,
                                code[2] < 1 ? (code[2] < 0 ? 0 : 1 / 2)
                                            : 1 / 2 + eC[2] - 1};
              Real *src = (Real *)(*info).block;
              int pos = 0;
              for (int iy = s[1]; iy < e[1]; iy++) {
                const int YY =
                    2 * (iy - s[1]) + s[1] + std::max(code[1], 0) * _BS_ / 2 -
                    code[1] * _BS_ + std::min(0, code[1]) * (e[1] - s[1]);
                for (int ix = s[0]; ix < e[0]; ix++) {
                  const int XX =
                      2 * (ix - s[0]) + s[0] + std::max(code[0], 0) * _BS_ / 2 -
                      code[0] * _BS_ + std::min(0, code[0]) * (e[0] - s[0]);
                  for (int c = 0; c < NC; c++) {
                    int comp = c;
                    dst[pos] =
                        0.25 *
                        (((*(src + dim * (XX + (YY)*_BS_) + comp)) +
                          (*(src + dim * (XX + 1 + (YY + 1) * _BS_) + comp))) +
                         ((*(src + dim * (XX + (YY + 1) * _BS_) + comp)) +
                          (*(src + dim * (XX + 1 + (YY)*_BS_) + comp))));
                    pos++;
                  }
                }
              }
            } else {
              Real *dst = send_buffer[r].data() + d;
              const BlockInfo *const info = f.infos[0];
              const int s[3] = {
                  code[0] < 1 ? (code[0] < 0 ? stencil.sx : 0) : _BS_,
                  code[1] < 1 ? (code[1] < 0 ? stencil.sy : 0) : _BS_,
                  code[2] < 1 ? (code[2] < 0 ? 0 : 0) : 1};
              const int e[3] = {code[0] < 1 ? (code[0] < 0 ? 0 : _BS_)
                                            : _BS_ + stencil.ex - 1,
                                code[1] < 1 ? (code[1] < 0 ? 0 : _BS_)
                                            : _BS_ + stencil.ey - 1,
                                code[2] < 1 ? (code[2] < 0 ? 0 : 1) : -1};
              Real *src = (Real *)(*info).block;
              const int xStep = (code[0] == 0) ? 2 : 1;
              const int yStep = (code[1] == 0) ? 2 : 1;
              int pos = 0;
              for (int iy = s[1]; iy < e[1]; iy += yStep) {
                const int YY = (abs(code[1]) == 1)
                                   ? 2 * (iy - code[1] * _BS_) +
                                         std::min(0, code[1]) * _BS_
                                   : iy;
                for (int ix = s[0]; ix < e[0]; ix += xStep) {
                  const int XX = (abs(code[0]) == 1)
                                     ? 2 * (ix - code[0] * _BS_) +
                                           std::min(0, code[0]) * _BS_
                                     : ix;
                  for (int c = 0; c < NC; c++) {
                    int comp = c;
                    dst[pos] =
                        0.25 *
                        (((*(src + dim * (XX + (YY)*_BS_) + comp)) +
                          (*(src + dim * (XX + 1 + (YY + 1) * _BS_) + comp))) +
                         ((*(src + dim * (XX + (YY + 1) * _BS_) + comp)) +
                          (*(src + dim * (XX + 1 + (YY)*_BS_) + comp))));
                    pos++;
                  }
                }
              }
            }
          }
#pragma omp for
          for (size_t i = 0; i < send_packinfos[r].size(); i++) {
            const PackInfo &info = send_packinfos[r][i];
            pack(info.block, info.pack, dim, info.sx, info.sy, info.sz, info.ex,
                 info.ey, info.ez, _BS_, _BS_);
          }
        }
      }
    for (auto r : Neighbors)
      if (send_buffer_size[r] > 0) {
        requests.resize(requests.size() + 1);
        MPI_Isend(&send_buffer[r][0], send_buffer_size[r] * NC, MPI_Real, r,
                  timestamp, MPI_COMM_WORLD, &requests.back());
      }
  }
  void fetch(const BlockInfo &info, const unsigned int Length[3],
             const unsigned int CLength[3], Real *cacheBlock,
             Real *coarseBlock) {
    const int id = info.halo_id;
    if (id < 0)
      return;
    UnPackInfo *unpacks = myunpacks[id].data();
    for (size_t jj = 0; jj < myunpacks[id].size(); jj++) {
      const UnPackInfo &unpack = unpacks[jj];
      const int code[3] = {unpack.icode % 3 - 1, (unpack.icode / 3) % 3 - 1,
                           (unpack.icode / 9) % 3 - 1};
      const int otherrank = unpack.rank;
      const int s[3] = {code[0] < 1 ? (code[0] < 0 ? stencil.sx : 0) : _BS_,
                        code[1] < 1 ? (code[1] < 0 ? stencil.sy : 0) : _BS_,
                        code[2] < 1 ? (code[2] < 0 ? 0 : 0) : 1};
      const int e[3] = {
          code[0] < 1 ? (code[0] < 0 ? 0 : _BS_) : _BS_ + stencil.ex - 1,
          code[1] < 1 ? (code[1] < 0 ? 0 : _BS_) : _BS_ + stencil.ey - 1,
          code[2] < 1 ? (code[2] < 0 ? 0 : 1) : 1};
      if (unpack.level == info.level) {
        Real *dst =
            cacheBlock + ((s[2] - 0) * Length[0] * Length[1] +
                          (s[1] - stencil.sy) * Length[0] + s[0] - stencil.sx) *
                             dim;
        unpack_subregion(&recv_buffer[otherrank][unpack.offset], &dst[0], dim,
                         unpack.srcxstart, unpack.srcystart, unpack.srczstart,
                         unpack.LX, unpack.LY, 0, 0, 0, unpack.lx, unpack.ly,
                         unpack.lz, Length[0], Length[1]);
        if (unpack.CoarseVersionOffset >= 0) {
          const int offset[3] = {(stencil.sx - 1) / 2 + Cstencil.sx,
                                 (stencil.sy - 1) / 2 + Cstencil.sy,
                                 (0 - 1) / 2 + 0};
          const int sC[3] = {
              code[0] < 1 ? (code[0] < 0 ? offset[0] : 0) : _BS_ / 2,
              code[1] < 1 ? (code[1] < 0 ? offset[1] : 0) : _BS_ / 2,
              code[2] < 1 ? (code[2] < 0 ? offset[2] : 0) : 1 / 2};
          Real *dst1 = coarseBlock +
                       ((sC[2] - offset[2]) * CLength[0] * CLength[1] +
                        (sC[1] - offset[1]) * CLength[0] + sC[0] - offset[0]) *
                           dim;
          int L[3];
          SM.CoarseStencilLength(
              (-code[0] + 1) + 3 * (-code[1] + 1) + 9 * (-code[2] + 1), L);
          unpack_subregion(&recv_buffer[otherrank][unpack.offset +
                                                   unpack.CoarseVersionOffset],
                           &dst1[0], dim, unpack.CoarseVersionsrcxstart,
                           unpack.CoarseVersionsrcystart,
                           unpack.CoarseVersionsrczstart,
                           unpack.CoarseVersionLX, unpack.CoarseVersionLY, 0, 0,
                           0, L[0], L[1], L[2], CLength[0], CLength[1]);
        }
      } else if (unpack.level < info.level) {
        const int offset[3] = {(stencil.sx - 1) / 2 + Cstencil.sx,
                               (stencil.sy - 1) / 2 + Cstencil.sy,
                               (0 - 1) / 2 + 0};
        const int sC[3] = {
            code[0] < 1 ? (code[0] < 0 ? offset[0] : 0) : _BS_ / 2,
            code[1] < 1 ? (code[1] < 0 ? offset[1] : 0) : _BS_ / 2,
            code[2] < 1 ? (code[2] < 0 ? offset[2] : 0) : 1 / 2};
        Real *dst = coarseBlock +
                    ((sC[2] - offset[2]) * CLength[0] * CLength[1] + sC[0] -
                     offset[0] + (sC[1] - offset[1]) * CLength[0]) *
                        dim;
        unpack_subregion(&recv_buffer[otherrank][unpack.offset], &dst[0], dim,
                         unpack.srcxstart, unpack.srcystart, unpack.srczstart,
                         unpack.LX, unpack.LY, 0, 0, 0, unpack.lx, unpack.ly,
                         unpack.lz, CLength[0], CLength[1]);
      } else {
        int B;
        if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 3))
          B = 0;
        else if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 2)) {
          int t;
          if (code[0] == 0)
            t = unpack.index_0 - 2 * info.index[0];
          else if (code[1] == 0)
            t = unpack.index_1 - 2 * info.index[1];
          else
            t = unpack.index_2 - 2 * info.index[2];
          assert(t == 0 || t == 1);
          B = (t == 1) ? 3 : 0;
        } else {
          int Bmod, Bdiv;
          if (abs(code[0]) == 1) {
            Bmod = unpack.index_1 - 2 * info.index[1];
            Bdiv = unpack.index_2 - 2 * info.index[2];
          } else if (abs(code[1]) == 1) {
            Bmod = unpack.index_0 - 2 * info.index[0];
            Bdiv = unpack.index_2 - 2 * info.index[2];
          } else {
            Bmod = unpack.index_0 - 2 * info.index[0];
            Bdiv = unpack.index_1 - 2 * info.index[1];
          }
          B = 2 * Bdiv + Bmod;
        }
        const int aux1 = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
        Real *dst =
            cacheBlock +
            ((abs(code[2]) * (s[2] - 0) +
              (1 - abs(code[2])) * (0 + (B / 2) * (e[2] - s[2]) / 2)) *
                 Length[0] * Length[1] +
             (abs(code[1]) * (s[1] - stencil.sy) +
              (1 - abs(code[1])) * (-stencil.sy + aux1 * (e[1] - s[1]) / 2)) *
                 Length[0] +
             abs(code[0]) * (s[0] - stencil.sx) +
             (1 - abs(code[0])) * (-stencil.sx + (B % 2) * (e[0] - s[0]) / 2)) *
                dim;
        unpack_subregion(&recv_buffer[otherrank][unpack.offset], &dst[0], dim,
                         unpack.srcxstart, unpack.srcystart, unpack.srczstart,
                         unpack.LX, unpack.LY, 0, 0, 0, unpack.lx, unpack.ly,
                         unpack.lz, Length[0], Length[1]);
      }
    }
  }
};
static BlockInfo &getf(std::unordered_map<long long, BlockInfo *> *BlockInfoAll,
                       std::vector<long long> *level_base, int m, long long n) {
  const long long aux = (*level_base)[m] + n;
  const auto retval = BlockInfoAll->find(aux);
  if (retval != BlockInfoAll->end()) {
    return *retval->second;
  } else {
#pragma omp critical
    {
      const auto retval1 = BlockInfoAll->find(aux);
      if (retval1 == BlockInfoAll->end()) {
        BlockInfo *dumm = new BlockInfo;
        dumm->level = m;
        dumm->h = sim.h0 / (1 << dumm->level);
        int i, j;
        sim.space_curve->inverse(n, m, i, j);
        dumm->origin[0] = i * _BS_ * dumm->h;
        dumm->origin[1] = j * _BS_ * dumm->h;
        dumm->Z = n;
        dumm->state = Leave;
        dumm->changed2 = true;
        dumm->auxiliary = nullptr;
        const int TwoPower = 1 << dumm->level;
        sim.space_curve->inverse(dumm->Z, dumm->level, dumm->index[0],
                                 dumm->index[1]);
        dumm->index[2] = 0;
        const int Bmax[2] = {sim.bpdx * TwoPower, sim.bpdy * TwoPower};
        for (int i = -1; i < 2; i++)
          for (int j = -1; j < 2; j++)
            dumm->Znei[i + 1][j + 1] = sim.space_curve->forward(
                dumm->level, (dumm->index[0] + i) % Bmax[0],
                (dumm->index[1] + j) % Bmax[1]);
        for (int i = 0; i < 2; i++)
          for (int j = 0; j < 2; j++)
            dumm->Zchild[i][j] = sim.space_curve->forward(
                dumm->level + 1, 2 * dumm->index[0] + i,
                2 * dumm->index[1] + j);
        dumm->Zparent =
            (dumm->level == 0)
                ? 0
                : sim.space_curve->forward(dumm->level - 1,
                                           (dumm->index[0] / 2) % Bmax[0],
                                           (dumm->index[1] / 2) % Bmax[1]);
        dumm->id2 = sim.space_curve->Encode(dumm->level, dumm->index);
        dumm->id = dumm->id2;
        (*BlockInfoAll)[aux] = dumm;
      }
    }
    return getf(BlockInfoAll, level_base, m, n);
  }
}
struct Grid {
  struct face {
    BlockInfo *infos[2];
    int icode[2];
    int offset;
    face(BlockInfo &i0, BlockInfo &i1, int a_icode0, int a_icode1) {
      infos[0] = &i0;
      infos[1] = &i1;
      icode[0] = a_icode0;
      icode[1] = a_icode1;
    }
    bool operator<(const face &other) const {
      if (infos[0]->id2 == other.infos[0]->id2) {
        return (icode[0] < other.icode[0]);
      } else {
        return (infos[0]->id2 < other.infos[0]->id2);
      }
    }
  };
  bool UpdateFluxCorrection{true};
  const int dim;
  size_t timestamp;
  std::map<std::array<long long, 2>, BlockCase *> Map;
  std::map<StencilInfo, Synchronizer<Grid> *> Synchronizers;
  std::unordered_map<long long, BlockInfo *> BlockInfoAll;
  std::unordered_map<long long, int> Octree;
  std::vector<BlockCase *> Cases;
  std::vector<BlockInfo *> boundary;
  std::vector<BlockInfo> infos;
  std::vector<long long> level_base;
  std::vector<std::vector<face>> recv_faces;
  std::vector<std::vector<face>> send_faces;
  std::vector<std::vector<Real>> recv_buffer;
  std::vector<std::vector<Real>> send_buffer;
  Grid(int dim) : dim(dim) {}
  void FillCase(Grid *grid, face &F) {
    BlockInfo &info = *F.infos[1];
    const int icode = F.icode[1];
    const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                         (icode / 9) % 3 - 1};
    const int myFace = abs(code[0]) * std::max(0, code[0]) +
                       abs(code[1]) * (std::max(0, code[1]) + 2) +
                       abs(code[2]) * (std::max(0, code[2]) + 4);
    std::array<long long, 2> temp = {(long long)info.level, info.Z};
    auto search = Map.find(temp);
    assert(search != Map.end());
    BlockCase &CoarseCase = (*search->second);
    Real *CoarseFace = (Real *)CoarseCase.d[myFace];
    for (int B = 0; B <= 1; B++) {
      const int aux = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
      const long long Z =
          getZforward(info.level + 1,
                      2 * info.index[0] + std::max(code[0], 0) + code[0] +
                          (B % 2) * std::max(0, 1 - abs(code[0])),
                      2 * info.index[1] + std::max(code[1], 0) + code[1] +
                          aux * std::max(0, 1 - abs(code[1])));
      if (Z != F.infos[0]->Z)
        continue;
      const int d = myFace / 2;
      const int d1 = std::max((d + 1) % 3, (d + 2) % 3);
      const int d2 = std::min((d + 1) % 3, (d + 2) % 3);
      const int N1 = sizes[d1];
      const int N2 = sizes[d2];
      int base = 0;
      if (B == 1)
        base = (N2 / 2) + (0) * N2;
      else if (B == 2)
        base = (0) + (N1 / 2) * N2;
      else if (B == 3)
        base = (N2 / 2) + (N1 / 2) * N2;
      int r = (*grid).Tree0(F.infos[0]->level, F.infos[0]->Z);
      int dis = 0;
      for (int i2 = 0; i2 < N2; i2 += 2) {
        Real *s = &CoarseFace[dim * (base + (i2 / 2))];
        for (int j = 0; j < dim; j++)
          s[j] += recv_buffer[r][F.offset + dis + j];
        dis += dim;
      }
    }
  }
  void FillCase_2(face &F, int codex, int codey) {
    BlockInfo &info = *F.infos[1];
    const int icode = F.icode[1];
    const int code[2] = {icode % 3 - 1, (icode / 3) % 3 - 1};
    if (abs(code[0]) != codex)
      return;
    if (abs(code[1]) != codey)
      return;
    const int myFace = abs(code[0]) * std::max(0, code[0]) +
                       abs(code[1]) * (std::max(0, code[1]) + 2);
    std::array<long long, 2> temp = {(long long)info.level, info.Z};
    auto search = Map.find(temp);
    assert(search != Map.end());
    BlockCase &CoarseCase = (*search->second);
    Real *CoarseFace = (Real *)CoarseCase.d[myFace];
    Real *block = (Real *)info.block;
    const int d = myFace / 2;
    const int d2 = std::min((d + 1) % 3, (d + 2) % 3);
    const int N2 = sizes[d2];
    assert(d != 2);
    if (d == 0) {
      const int j = (myFace % 2 == 0) ? 0 : _BS_ - 1;
      for (int i2 = 0; i2 < N2; i2++) {
        int k = _BS_ * i2 + j;
        for (int d = 0; d < dim; d++)
          block[dim * k + d] += CoarseFace[dim * i2 + d];
        memset(&CoarseFace[i2], 0, dim * sizeof(Real));
      }
    } else {
      const int j = (myFace % 2 == 0) ? 0 : _BS_ - 1;
      for (int i2 = 0; i2 < N2; i2++) {
        int k = _BS_ * j + i2;
        for (int d = 0; d < dim; d++)
          block[dim * k + d] += CoarseFace[dim * i2 + d];
        memset(&CoarseFace[i2], 0, dim * sizeof(Real));
      }
    }
  }
  void prepare0(Grid *grid) {
    if (grid->UpdateFluxCorrection == false)
      return;
    grid->UpdateFluxCorrection = false;
    send_buffer.resize(sim.size);
    recv_buffer.resize(sim.size);
    send_faces.resize(sim.size);
    recv_faces.resize(sim.size);
    for (int r = 0; r < sim.size; r++) {
      send_faces[r].clear();
      recv_faces[r].clear();
    }
    std::vector<int> send_buffer_size(sim.size, 0);
    std::vector<int> recv_buffer_size(sim.size, 0);
    for (int i = 0; i < Cases.size(); i++)
      for (int j = 0; j < 4; j++)
        free(Cases[i]->d[j]);
    Cases.clear();
    Map.clear();
    std::vector<BlockInfo> &BB = grid->infos;
    std::array<int, 6> icode = {1 * 2 + 3 * 1 + 9 * 1, 1 * 0 + 3 * 1 + 9 * 1,
                                1 * 1 + 3 * 2 + 9 * 1, 1 * 1 + 3 * 0 + 9 * 1,
                                1 * 1 + 3 * 1 + 9 * 2, 1 * 1 + 3 * 1 + 9 * 0};
    for (auto &info : BB) {
      (*grid).get(info.level, info.Z).auxiliary = nullptr;
      info.auxiliary = nullptr;
      const int aux = 1 << info.level;
      const bool xskin =
          info.index[0] == 0 || info.index[0] == sim.bpdx * aux - 1;
      const bool yskin =
          info.index[1] == 0 || info.index[1] == sim.bpdy * aux - 1;
      const int xskip = info.index[0] == 0 ? -1 : 1;
      const int yskip = info.index[1] == 0 ? -1 : 1;

      bool storeFace[4] = {false, false, false, false};
      bool stored = false;
      for (int f = 0; f < 6; f++) {
        const int code[3] = {icode[f] % 3 - 1, (icode[f] / 3) % 3 - 1,
                             (icode[f] / 9) % 3 - 1};
        if (sim.bcx != periodic && code[0] == xskip && xskin)
          continue;
        if (sim.bcy != periodic && code[1] == yskip && yskin)
          continue;
        if (code[2] != 0)
          continue;
        if (!((*grid).Tree0(info.level, info.Znei[1 + code[0]][1 + code[1]]) >=
              0)) {
          storeFace[abs(code[0]) * std::max(0, code[0]) +
                    abs(code[1]) * (std::max(0, code[1]) + 2) +
                    abs(code[2]) * (std::max(0, code[2]) + 4)] = true;
          stored = true;
        }
        int L[3];
        L[0] = (code[0] == 0) ? _BS_ / 2 : 1;
        L[1] = (code[1] == 0) ? _BS_ / 2 : 1;
        int V = L[0] * L[1];
        if ((*grid).Tree0(info.level, info.Znei[1 + code[0]][1 + code[1]]) ==
            -2) {
          BlockInfo &infoNei =
              (*grid).get(info.level, info.Znei[1 + code[0]][1 + code[1]]);
          const long long nCoarse = infoNei.Zparent;
          BlockInfo &infoNeiCoarser = (*grid).get(info.level - 1, nCoarse);
          const int infoNeiCoarserrank = (*grid).Tree0(info.level - 1, nCoarse);
          {
            int code2[3] = {-code[0], -code[1], -code[2]};
            int icode2 =
                (code2[0] + 1) + (code2[1] + 1) * 3 + (code2[2] + 1) * 9;
            send_faces[infoNeiCoarserrank].push_back(
                face(info, infoNeiCoarser, icode[f], icode2));
            send_buffer_size[infoNeiCoarserrank] += V;
          }
        } else if ((*grid).Tree0(info.level,
                                 info.Znei[1 + code[0]][1 + code[1]]) == -1) {
          BlockInfo &infoNei =
              (*grid).get(info.level, info.Znei[1 + code[0]][1 + code[1]]);
          int Bstep = 1;
          for (int B = 0; B <= 1; B += Bstep) {
            const int temp = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
            const long long nFine =
                infoNei.Zchild[std::max(-code[0], 0) +
                               (B % 2) * std::max(0, 1 - abs(code[0]))]
                              [std::max(-code[1], 0) +
                               temp * std::max(0, 1 - abs(code[1]))];
            const int infoNeiFinerrank =
                (*grid).Tree0(infoNei.level + 1, nFine);
            {
              BlockInfo &infoNeiFiner = (*grid).get(infoNei.level + 1, nFine);
              int icode2 =
                  (-code[0] + 1) + (-code[1] + 1) * 3 + (-code[2] + 1) * 9;
              recv_faces[infoNeiFinerrank].push_back(
                  face(infoNeiFiner, info, icode2, icode[f]));
              recv_buffer_size[infoNeiFinerrank] += V;
            }
          }
        }
      }
      if (stored) {
        BlockCase *c = new BlockCase;
        c->level = info.level;
        c->Z = info.Z;
        for (int i = 0; i < 4; i++)
          c->d[i] = storeFace[i] ? (uint8_t *)malloc(_BS_ * dim * sizeof(Real))
                                 : nullptr;
        Cases.push_back(c);
      }
    }
    size_t Cases_index = 0;
    if (Cases.size() > 0)
      for (auto &info : BB) {
        if (Cases_index == Cases.size())
          break;
        if (Cases[Cases_index]->level == info.level &&
            Cases[Cases_index]->Z == info.Z) {
          Map.insert(std::pair<std::array<long long, 2>, BlockCase *>(
              {Cases[Cases_index]->level, Cases[Cases_index]->Z},
              Cases[Cases_index]));
          grid->get(Cases[Cases_index]->level, Cases[Cases_index]->Z)
              .auxiliary = Cases[Cases_index];
          info.auxiliary = Cases[Cases_index];
          Cases_index++;
        }
      }
    for (int r = 0; r < sim.size; r++) {
      std::sort(send_faces[r].begin(), send_faces[r].end());
      std::sort(recv_faces[r].begin(), recv_faces[r].end());
    }
    for (int r = 0; r < sim.size; r++) {
      send_buffer[r].resize(send_buffer_size[r] * dim);
      recv_buffer[r].resize(recv_buffer_size[r] * dim);
      int offset = 0;
      for (int k = 0; k < (int)recv_faces[r].size(); k++) {
        face &f = recv_faces[r][k];
        const int code[3] = {f.icode[1] % 3 - 1, (f.icode[1] / 3) % 3 - 1,
                             (f.icode[1] / 9) % 3 - 1};
        int V =
            ((code[0] == 0) ? _BS_ / 2 : 1) * ((code[1] == 0) ? _BS_ / 2 : 1);
        f.offset = offset;
        offset += V * dim;
      }
    }
  }
  void FillBlockCases(Grid *grid) {
    for (int r = 0; r < sim.size; r++) {
      int displacement = 0;
      for (int k = 0; k < (int)send_faces[r].size(); k++) {
        face &f = send_faces[r][k];
        BlockInfo &info = *(f.infos[0]);
        auto search = Map.find({(long long)info.level, info.Z});
        assert(search != Map.end());
        BlockCase &FineCase = (*search->second);
        int icode = f.icode[0];
        assert((icode / 9) % 3 - 1 == 0);
        int code[2] = {icode % 3 - 1, (icode / 3) % 3 - 1};
        int myFace = abs(code[0]) * std::max(0, code[0]) +
                     abs(code[1]) * (std::max(0, code[1]) + 2);
        Real *FineFace = (Real *)FineCase.d[myFace];
        int d = myFace / 2;
        assert(d == 0 || d == 1);
        int d2 = std::min((d + 1) % 3, (d + 2) % 3);
        int N2 = sizes[d2];
        for (int i2 = 0; i2 < N2; i2 += 2) {
          Real *a = &FineFace[dim * i2];
          Real *b = &FineFace[dim * (i2 + 1)];
          for (d = 0; d < dim; d++) {
            Real avg = a[d] + b[d];
            memcpy(&send_buffer[r][displacement], &avg, sizeof(Real));
            displacement++;
          }
          memset(&FineFace[dim * i2], 0, dim * sizeof(Real));
          memset(&FineFace[dim * (i2 + 1)], 0, dim * sizeof(Real));
        }
      }
    }
    std::vector<MPI_Request> send_requests;
    std::vector<MPI_Request> recv_requests;
    for (int r = 0; r < sim.size; r++)
      if (r != sim.rank) {
        if (recv_buffer[r].size() != 0) {
          MPI_Request req{};
          recv_requests.push_back(req);
          MPI_Irecv(&recv_buffer[r][0], recv_buffer[r].size(), MPI_Real, r,
                    123456, MPI_COMM_WORLD, &recv_requests.back());
        }
        if (send_buffer[r].size() != 0) {
          MPI_Request req{};
          send_requests.push_back(req);
          MPI_Isend(&send_buffer[r][0], send_buffer[r].size(), MPI_Real, r,
                    123456, MPI_COMM_WORLD, &send_requests.back());
        }
      }
    MPI_Request me_send_request;
    MPI_Request me_recv_request;
    if (recv_buffer[sim.rank].size() != 0) {
      MPI_Irecv(&recv_buffer[sim.rank][0], recv_buffer[sim.rank].size(),
                MPI_Real, sim.rank, 123456, MPI_COMM_WORLD, &me_recv_request);
    }
    if (send_buffer[sim.rank].size() != 0) {
      MPI_Isend(&send_buffer[sim.rank][0], send_buffer[sim.rank].size(),
                MPI_Real, sim.rank, 123456, MPI_COMM_WORLD, &me_send_request);
    }
    if (recv_buffer[sim.rank].size() > 0)
      MPI_Waitall(1, &me_recv_request, MPI_STATUSES_IGNORE);
    if (send_buffer[sim.rank].size() > 0)
      MPI_Waitall(1, &me_send_request, MPI_STATUSES_IGNORE);
    for (int index = 0; index < (int)recv_faces[sim.rank].size(); index++)
      FillCase(grid, recv_faces[sim.rank][index]);
    if (recv_requests.size() > 0)
      MPI_Waitall(recv_requests.size(), &recv_requests[0], MPI_STATUSES_IGNORE);
    for (int r = 0; r < sim.size; r++)
      if (r != sim.rank)
        for (int index = 0; index < (int)recv_faces[r].size(); index++)
          FillCase(grid, recv_faces[r][index]);
    for (int r = 0; r < sim.size; r++)
      for (int index = 0; index < (int)recv_faces[r].size(); index++)
        FillCase_2(recv_faces[r][index], 1, 0);
    for (int r = 0; r < sim.size; r++)
      for (int index = 0; index < (int)recv_faces[r].size(); index++)
        FillCase_2(recv_faces[r][index], 0, 1);
    if (send_requests.size() > 0)
      MPI_Waitall(send_requests.size(), &send_requests[0], MPI_STATUSES_IGNORE);
  }
  void *avail(const int m, const long long n) {
    return (Tree0(m, n) == sim.rank) ? get(m, n).block : nullptr;
  }
  void UpdateBoundary(bool clean = false) {
    std::vector<std::vector<long long>> send_buffer(sim.size);
    std::vector<BlockInfo *> &bbb = boundary;
    std::set<int> Neighbors;
    for (size_t jjj = 0; jjj < bbb.size(); jjj++) {
      BlockInfo &info = *bbb[jjj];
      std::set<int> receivers;
      const int aux = 1 << info.level;
      const bool xskin =
          info.index[0] == 0 || info.index[0] == sim.bpdx * aux - 1;
      const bool yskin =
          info.index[1] == 0 || info.index[1] == sim.bpdy * aux - 1;
      const int xskip = info.index[0] == 0 ? -1 : 1;
      const int yskip = info.index[1] == 0 ? -1 : 1;

      for (int icode = 0; icode < 27; icode++) {
        if (icode == 1 * 1 + 3 * 1 + 9 * 1)
          continue;
        const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                             (icode / 9) % 3 - 1};
        if (sim.bcx != periodic && code[0] == xskip && xskin)
          continue;
        if (sim.bcy != periodic && code[1] == yskip && yskin)
          continue;
        if (code[2] != 0)
          continue;
        BlockInfo &infoNei =
            get(info.level, info.Znei[1 + code[0]][1 + code[1]]);
        const int &infoNeiTree = Tree0(infoNei.level, infoNei.Z);
        if (infoNeiTree >= 0 && infoNeiTree != sim.rank) {
          if (infoNei.state != Refine || clean)
            infoNei.state = Leave;
          receivers.insert(infoNeiTree);
          Neighbors.insert(infoNeiTree);
        } else if (infoNeiTree == -2) {
          const long long nCoarse = infoNei.Zparent;
          BlockInfo &infoNeiCoarser = get(infoNei.level - 1, nCoarse);
          const int infoNeiCoarserrank = Tree0(infoNei.level - 1, nCoarse);
          if (infoNeiCoarserrank != sim.rank) {
            assert(infoNeiCoarserrank >= 0);
            if (infoNeiCoarser.state != Refine || clean)
              infoNeiCoarser.state = Leave;
            receivers.insert(infoNeiCoarserrank);
            Neighbors.insert(infoNeiCoarserrank);
          }
        } else if (infoNeiTree == -1) {
          int Bstep = 1;
          if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 2))
            Bstep = 3;
          else if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 3))
            Bstep = 4;
          for (int B = 0; B <= 1; B += Bstep) {
            const int temp = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
            const long long nFine =
                infoNei.Zchild[std::max(-code[0], 0) +
                               (B % 2) * std::max(0, 1 - abs(code[0]))]
                              [std::max(-code[1], 0) +
                               temp * std::max(0, 1 - abs(code[1]))];
            BlockInfo &infoNeiFiner = get(infoNei.level + 1, nFine);
            const int infoNeiFinerrank = Tree0(infoNei.level + 1, nFine);
            if (infoNeiFinerrank != sim.rank) {
              if (infoNeiFiner.state != Refine || clean)
                infoNeiFiner.state = Leave;
              receivers.insert(infoNeiFinerrank);
              Neighbors.insert(infoNeiFinerrank);
            }
          }
        }
      }
      if (info.changed2 && info.state != Leave) {
        if (info.state == Refine)
          info.changed2 = false;
        std::set<int>::iterator it = receivers.begin();
        while (it != receivers.end()) {
          int temp = (info.state == Compress) ? 1 : 2;
          send_buffer[*it].push_back(info.level);
          send_buffer[*it].push_back(info.Z);
          send_buffer[*it].push_back(temp);
          it++;
        }
      }
    }
    std::vector<MPI_Request> requests;
    long long dummy = 0;
    for (int r : Neighbors)
      if (r != sim.rank) {
        requests.resize(requests.size() + 1);
        if (send_buffer[r].size() != 0)
          MPI_Isend(&send_buffer[r][0], send_buffer[r].size(), MPI_LONG_LONG, r,
                    123, MPI_COMM_WORLD, &requests[requests.size() - 1]);
        else {
          MPI_Isend(&dummy, 1, MPI_LONG_LONG, r, 123, MPI_COMM_WORLD,
                    &requests[requests.size() - 1]);
        }
      }
    std::vector<std::vector<long long>> recv_buffer(sim.size);
    for (int r : Neighbors)
      if (r != sim.rank) {
        int recv_size;
        MPI_Status status;
        MPI_Probe(r, 123, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_LONG_LONG, &recv_size);
        if (recv_size > 0) {
          recv_buffer[r].resize(recv_size);
          requests.resize(requests.size() + 1);
          MPI_Irecv(&recv_buffer[r][0], recv_buffer[r].size(), MPI_LONG_LONG, r,
                    123, MPI_COMM_WORLD, &requests[requests.size() - 1]);
        }
      }
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    for (int r = 0; r < sim.size; r++)
      if (recv_buffer[r].size() > 1)
        for (int index = 0; index < (int)recv_buffer[r].size(); index += 3) {
          int level = recv_buffer[r][index];
          long long Z = recv_buffer[r][index + 1];
          get(level, Z).state =
              (recv_buffer[r][index + 2] == 1) ? Compress : Refine;
        }
  };
  void UpdateBlockInfoAll_States(bool UpdateIDs) {
    std::vector<int> myNeighbors;
    double low[3] = {+1e20, +1e20, +1e20};
    double high[3] = {-1e20, -1e20, -1e20};
    double p_low[3];
    double p_high[3];
    for (auto &info : infos) {
      const double h2 = 2 * info.h;
      p_low[0] = info.origin[0] + info.h * 0.5;
      p_low[1] = info.origin[1] + info.h * 0.5;
      p_high[0] = info.origin[0] + info.h * (_BS_ - 0.5);
      p_high[1] = info.origin[1] + info.h * (_BS_ - 0.5);
      p_low[0] -= h2;
      p_low[1] -= h2;
      p_low[2] = 0;
      p_high[0] += h2;
      p_high[1] += h2;
      p_high[2] = 0;
      low[0] = std::min(low[0], p_low[0]);
      low[1] = std::min(low[1], p_low[1]);
      low[2] = 0;
      high[0] = std::max(high[0], p_high[0]);
      high[1] = std::max(high[1], p_high[1]);
      high[2] = 0;
    }
    std::vector<double> all_boxes(sim.size * 6);
    double my_box[6] = {low[0], low[1], low[2], high[0], high[1], high[2]};
    MPI_Allgather(my_box, 6, MPI_DOUBLE, all_boxes.data(), 6, MPI_DOUBLE,
                  MPI_COMM_WORLD);
    for (int i = 0; i < sim.size; i++) {
      if (i == sim.rank)
        continue;
      if (Intersect0(low, high, &all_boxes[i * 6], &all_boxes[i * 6 + 3]))
        myNeighbors.push_back(i);
    }
    std::vector<long long> myData;
    for (auto &info : infos) {
      bool myflag = false;
      int aux = 1 << info.level;
      bool xskin = info.index[0] == 0 || info.index[0] == sim.bpdx * aux - 1;
      bool yskin = info.index[1] == 0 || info.index[1] == sim.bpdy * aux - 1;
      int xskip = info.index[0] == 0 ? -1 : 1;
      int yskip = info.index[1] == 0 ? -1 : 1;

      for (int icode = 0; icode < 27; icode++) {
        if (icode == 1 * 1 + 3 * 1 + 9 * 1)
          continue;
        int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, (icode / 9) % 3 - 1};
        if (sim.bcx != periodic && code[0] == xskip && xskin)
          continue;
        if (sim.bcy != periodic && code[1] == yskip && yskin)
          continue;
        if (code[2] != 0)
          continue;
        BlockInfo &infoNei =
            get(info.level, info.Znei[1 + code[0]][1 + code[1]]);
        int &infoNeiTree = Tree0(infoNei.level, infoNei.Z);
        if (infoNeiTree >= 0 && infoNeiTree != sim.rank) {
          myflag = true;
          break;
        } else if (infoNeiTree == -2) {
          long long nCoarse = infoNei.Zparent;
          int infoNeiCoarserrank = Tree0(infoNei.level - 1, nCoarse);
          if (infoNeiCoarserrank != sim.rank) {
            myflag = true;
            break;
          }
        } else if (infoNeiTree == -1) {
          int Bstep = 1;
          if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 2))
            Bstep = 3;
          else if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 3))
            Bstep = 4;
          for (int B = 0; B <= 3; B += Bstep) {
            int temp = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
            long long nFine =
                infoNei.Zchild[std::max(-code[0], 0) +
                               (B % 2) * std::max(0, 1 - abs(code[0]))]
                              [std::max(-code[1], 0) +
                               temp * std::max(0, 1 - abs(code[1]))];
            int infoNeiFinerrank = Tree0(infoNei.level + 1, nFine);
            if (infoNeiFinerrank != sim.rank) {
              myflag = true;
              break;
            }
          }
        } else if (infoNeiTree < 0) {
          myflag = true;
          break;
        }
      }
      if (myflag) {
        myData.push_back(info.level);
        myData.push_back(info.Z);
        if (UpdateIDs)
          myData.push_back(info.id);
      }
    }
    std::vector<std::vector<long long>> recv_buffer(myNeighbors.size());
    std::vector<std::vector<long long>> send_buffer(myNeighbors.size());
    std::vector<int> recv_size(myNeighbors.size());
    std::vector<MPI_Request> size_requests(2 * myNeighbors.size());
    int mysize = (int)myData.size();
    int kk = 0;
    for (auto r : myNeighbors) {
      MPI_Irecv(&recv_size[kk], 1, MPI_INT, r, timestamp, MPI_COMM_WORLD,
                &size_requests[2 * kk]);
      MPI_Isend(&mysize, 1, MPI_INT, r, timestamp, MPI_COMM_WORLD,
                &size_requests[2 * kk + 1]);
      kk++;
    }
    kk = 0;
    for (size_t j = 0; j < myNeighbors.size(); j++) {
      send_buffer[kk].resize(myData.size());
      for (size_t i = 0; i < myData.size(); i++)
        send_buffer[kk][i] = myData[i];
      kk++;
    }
    MPI_Waitall(size_requests.size(), size_requests.data(),
                MPI_STATUSES_IGNORE);
    std::vector<MPI_Request> requests(2 * myNeighbors.size());
    kk = 0;
    for (auto r : myNeighbors) {
      recv_buffer[kk].resize(recv_size[kk]);
      MPI_Irecv(recv_buffer[kk].data(), recv_buffer[kk].size(), MPI_LONG_LONG,
                r, timestamp, MPI_COMM_WORLD, &requests[2 * kk]);
      MPI_Isend(send_buffer[kk].data(), send_buffer[kk].size(), MPI_LONG_LONG,
                r, timestamp, MPI_COMM_WORLD, &requests[2 * kk + 1]);
      kk++;
    }
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    kk = -1;
    const int increment = UpdateIDs ? 3 : 2;
    for (auto r : myNeighbors) {
      kk++;
      for (size_t index__ = 0; index__ < recv_buffer[kk].size();
           index__ += increment) {
        const int level = (int)recv_buffer[kk][index__];
        const long long Z = recv_buffer[kk][index__ + 1];
        Tree0(level, Z) = r;
        if (UpdateIDs)
          get(level, Z).id = recv_buffer[kk][index__ + 2];
        int p[2];
        sim.space_curve->inverse(Z, level, p[0], p[1]);
        if (level < sim.levelMax - 1)
          for (int j = 0; j < 2; j++)
            for (int i = 0; i < 2; i++) {
              const long long nc =
                  getZforward(level + 1, 2 * p[0] + i, 2 * p[1] + j);
              Tree0(level + 1, nc) = -2;
            }
        if (level > 0) {
          const long long nf = getZforward(level - 1, p[0] / 2, p[1] / 2);
          Tree0(level - 1, nf) = -1;
        }
      }
    }
  }
  bool Intersect0(double *l1, double *h1, double *l2, double *h2) {
    const Real intersect[3][2] = {
        {std::max(l1[0], l2[0]), std::min(h1[0], h2[0])},
        {std::max(l1[1], l2[1]), std::min(h1[1], h2[1])},
        {std::max(l1[2], l2[2]), std::min(h1[2], h2[2])}};
    bool intersection[3];
    intersection[0] = intersect[0][1] - intersect[0][0] > 0.0;
    intersection[1] = intersect[1][1] - intersect[1][0] > 0.0;
    intersection[2] = true;
    const bool isperiodic[2] = {sim.bcx == periodic, sim.bcy == periodic};
    for (int d = 0; d < 2; d++) {
      if (isperiodic[d]) {
        if (h2[d] > sim.extents[d])
          intersection[d] = std::min(h1[d], h2[d] - sim.extents[d]) -
                            std::max(l1[d], l2[d] - sim.extents[d]);
        else if (h1[d] > sim.extents[d])
          intersection[d] = std::min(h2[d], h1[d] - sim.extents[d]) -
                            std::max(l2[d], l1[d] - sim.extents[d]);
      }
      if (!intersection[d])
        return false;
    }
    return true;
  }
  Synchronizer<Grid> *sync1(const StencilInfo &stencil) {
    StencilInfo Cstencil(-1, -1, 2, 2, true);
    Synchronizer<Grid> *queryresult = nullptr;
    typename std::map<StencilInfo, Synchronizer<Grid> *>::iterator
        itSynchronizerMPI = Synchronizers.find(stencil);
    if (itSynchronizerMPI == Synchronizers.end()) {
      queryresult = new Synchronizer<Grid>(stencil, Cstencil, this, dim);
      queryresult->_Setup();
      Synchronizers[stencil] = queryresult;
    } else {
      queryresult = itSynchronizerMPI->second;
    }
    queryresult->sync0();
    timestamp = (timestamp + 1) % 32768;
    return queryresult;
  }
  int &Tree0(const int m, const long long n) {
    const long long aux = level_base[m] + n;
    const auto retval = Octree.find(aux);
    if (retval == Octree.end()) {
#pragma omp critical
      {
        const auto retval1 = Octree.find(aux);
        if (retval1 == Octree.end()) {
          Octree[aux] = -3;
        }
      }
      return Tree0(m, n);
    } else {
      return retval->second;
    }
  }
  int &Tree1(const BlockInfo &info) { return Tree0(info.level, info.Z); }
  void _alloc(int level, long long Z) {
    BlockInfo &new_info = get(level, Z);
    new_info.block = malloc(dim * _BS_ * _BS_ * sizeof(Real));
#pragma omp critical
    { infos.push_back(new_info); }
    Tree0(level, Z) = sim.rank;
  }
  void _dealloc(const int m, const long long n) {
    free(get0(m, n).block);
    for (size_t j = 0; j < infos.size(); j++) {
      if (infos[j].level == m && infos[j].Z == n) {
        infos.erase(infos.begin() + j);
        return;
      }
    }
  }
  void dealloc_many(const std::vector<long long> &dealloc_IDs) {
    for (size_t j = 0; j < infos.size(); j++)
      infos[j].changed2 = false;
    for (size_t i = 0; i < dealloc_IDs.size(); i++)
      for (size_t j = 0; j < infos.size(); j++) {
        if (infos[j].id2 == dealloc_IDs[i]) {
          const int m = infos[j].level;
          const long long n = infos[j].Z;
          infos[j].changed2 = true;
          free(get0(m, n).block);
          break;
        }
      }
    infos.erase(std::remove_if(infos.begin(), infos.end(),
                               [](const BlockInfo &x) { return x.changed2; }),
                infos.end());
  }
  void FillPos() {
    std::sort(infos.begin(), infos.end());
    for (size_t j = 0; j < infos.size(); j++) {
      const int m = infos[j].level;
      const long long n = infos[j].Z;
      BlockInfo &correct_info = get(m, n);
      correct_info.id = j;
      infos[j] = correct_info;
      assert(Tree0(m, n) >= 0);
    }
  }
  void *avail1(const int ix, const int iy, const int m) {
    const long long n = getZforward(m, ix, iy);
    return avail(m, n);
  }
  BlockInfo &get0(int m, long long n) {
    const auto retval = BlockInfoAll.find(level_base[m] + n);
    assert(retval != BlockInfoAll.end());
    return *retval->second;
  }
  BlockInfo &get(int m, long long n) {
    return getf(&BlockInfoAll, &level_base, m, n);
  }
};

static void LI(Real *a0, Real *b0, Real *c0) {
  Real a = *a0;
  Real b = *b0;
  Real c = *c0;
  Real kappa = ((4.0 / 15.0) * a + (6.0 / 15.0) * c) + (-10.0 / 15.0) * b;
  Real lambda = (b - c) - kappa;
  *a0 = (4.0 * kappa + 2.0 * lambda) + c;
}
static void LE(Real *a0, Real *b0, Real *c0) {
  Real a = *a0;
  Real b = *b0;
  Real c = *c0;
  Real kappa = ((4.0 / 15.0) * a + (6.0 / 15.0) * c) + (-10.0 / 15.0) * b;
  Real lambda = (b - c) - kappa;
  *a0 = (9.0 * kappa + 3.0 * lambda) + c;
}
static void TestInterp(Real *C[3][3], Real *R, int x, int y) {
  double dx = 0.25 * (2 * x - 1);
  double dy = 0.25 * (2 * y - 1);
  Real dudx = 0.5 * ((*C[2][1]) - (*C[0][1]));
  Real dudy = 0.5 * ((*C[1][2]) - (*C[1][0]));
  Real dudxdy = 0.25 * (((*C[0][0]) + (*C[2][2])) - ((*C[2][0]) + (*C[0][2])));
  Real dudx2 = ((*C[0][1]) + (*C[2][1])) - 2.0 * (*C[1][1]);
  Real dudy2 = ((*C[1][0]) + (*C[1][2])) - 2.0 * (*C[1][1]);
  *R = (*C[1][1] + (dx * dudx + dy * dudy)) +
       (((0.5 * dx * dx) * dudx2 + (0.5 * dy * dy) * dudy2) +
        (dx * dy) * dudxdy);
}
template <typename Element> struct BlockLab {
  Synchronizer<Grid> *refSynchronizerMPI;
  bool coarsened, istensorial, use_averages;
  int coarsened_nei_codes_size, end[3], NX, NY, NZ, offset[3], start[3];
  unsigned int nm[2], nc[2];
  Element *m, *c;
  std::array<void *, 27> myblocks;
  std::array<int, 27> coarsened_nei_codes;
  const int dim;
  BlockLab(int dim) : dim(dim) {
    m = NULL;
    c = NULL;
  }
  ~BlockLab() {
    free(m);
    free(c);
  }
  const Element &operator()(int x, int y) const {
    x -= start[0];
    y -= start[1];
    return m[nm[0] * y + x];
  }
  void prepare(Grid *grid, const StencilInfo &stencil) {
    refSynchronizerMPI = grid->Synchronizers.find(stencil)->second;
    istensorial = stencil.tensorial;
    coarsened = false;
    start[0] = stencil.sx;
    start[1] = stencil.sy;
    start[2] = 0;
    end[0] = stencil.ex;
    end[1] = stencil.ey;
    end[2] = 1;
    nm[0] = _BS_ + end[0] - start[0] - 1;
    nm[1] = _BS_ + end[1] - start[1] - 1;
    free(m);
    m = (Element *)malloc(nm[0] * nm[1] * dim * sizeof(Real));
    offset[0] = (start[0] - 1) / 2 - 1;
    offset[1] = (start[1] - 1) / 2 - 1;
    offset[2] = (start[2] - 1) / 2;
    nc[0] = _BS_ / 2 + end[0] / 2 + 1 - offset[0];
    nc[1] = _BS_ / 2 + end[1] / 2 + 1 - offset[1];
    free(c);
    c = (Element *)malloc(nc[0] * nc[1] * dim * sizeof(Real));
    use_averages = istensorial || start[0] < -2 || start[1] < -2 ||
                   end[0] > 3 || end[1] > 3;
  }
  void load(Grid *grid, BlockInfo &info, bool applybc) {
    const int aux = 1 << info.level;
    NX = sim.bpdx * aux;
    NY = sim.bpdy * aux;
    NZ = 1 * aux;
    assert(m != NULL);
    Real *p = (Real *)info.block;
    Real *u = (Real *)m;
    for (int iy = -start[1]; iy < -start[1] + _BS_; iy += 4) {
      Real *q = u + dim * iy * nm[0] - dim * start[0];
      memcpy(q, p, sizeof(Real) * dim * _BS_), q += dim * nm[0],
          p += dim * _BS_;
      memcpy(q, p, sizeof(Real) * dim * _BS_), q += dim * nm[0],
          p += dim * _BS_;
      memcpy(q, p, sizeof(Real) * dim * _BS_), q += dim * nm[0],
          p += dim * _BS_;
      memcpy(q, p, sizeof(Real) * dim * _BS_), q += dim * nm[0],
          p += dim * _BS_;
    }
    coarsened = false;
    bool xskin = info.index[0] == 0 || info.index[0] == NX - 1;
    bool yskin = info.index[1] == 0 || info.index[1] == NY - 1;
    int xskip = info.index[0] == 0 ? -1 : 1;
    int yskip = info.index[1] == 0 ? -1 : 1;
    int icodes[8];
    int k = 0;
    coarsened_nei_codes_size = 0;
    for (int icode = 9; icode < 18; icode++) {
      myblocks[icode] = nullptr;
      if (icode == 1 * 1 + 3 * 1 + 9 * 1)
        continue;
      const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, icode / 9 - 1};
      if (sim.bcx != periodic && code[0] == xskip && xskin)
        continue;
      if (sim.bcy != periodic && code[1] == yskip && yskin)
        continue;
      const auto &TreeNei =
          grid->Tree0(info.level, info.Znei[1 + code[0]][1 + code[1]]);
      if (TreeNei >= 0) {
        icodes[k++] = icode;
      } else if (TreeNei == -2) {
        coarsened_nei_codes[coarsened_nei_codes_size++] = icode;
        CoarseFineExchange(grid, info, code);
      }
      if (!istensorial && !use_averages &&
          abs(code[0]) + abs(code[1]) + abs(code[2]) > 1)
        continue;
      const int s[3] = {code[0] < 1 ? (code[0] < 0 ? start[0] : 0) : _BS_,
                        code[1] < 1 ? (code[1] < 0 ? start[1] : 0) : _BS_,
                        code[2] < 1 ? (code[2] < 0 ? start[2] : 0) : 1};
      const int e[3] = {
          code[0] < 1 ? (code[0] < 0 ? 0 : _BS_) : _BS_ + end[0] - 1,
          code[1] < 1 ? (code[1] < 0 ? 0 : _BS_) : _BS_ + end[1] - 1,
          code[2] < 1 ? (code[2] < 0 ? 0 : 1) : 1 + end[2] - 1};
      if (TreeNei >= 0)
        SameLevelExchange(grid, info, code, s, e);
      else if (TreeNei == -1)
        FineToCoarseExchange(grid, info, code, s, e);
    }
    if (coarsened_nei_codes_size > 0)
      for (int i = 0; i < k; ++i) {
        int icode = icodes[i];
        int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, icode / 9 - 1};
        int infoNei_index[3] = {(info.index[0] + code[0] + NX) % NX,
                                (info.index[1] + code[1] + NY) % NY,
                                (info.index[2] + code[2] + NZ) % NZ};
        if (UseCoarseStencil(info, infoNei_index)) {
          FillCoarseVersion(code);
          coarsened = true;
        }
      }
    if (sim.size == 1)
      post_load(grid, info, applybc);
    refSynchronizerMPI->fetch(info, nm, nc, (Real *)m, (Real *)c);
    if (sim.size > 1)
      post_load(grid, info, applybc);
  }

  void post_load(Grid *grid, BlockInfo &info, bool applybc) {
    if (coarsened) {
      for (int j = 0; j < _BS_ / 2; j++) {
        for (int i = 0; i < _BS_ / 2; i++) {
          if (i > 1 && i < _BS_ / 2 - 2 && j > 2 && j < _BS_ / 2 - 2)
            continue;
          int ix = 2 * i - start[0];
          int iy = 2 * j - start[1];
          int i00 = ix + nm[0] * iy;
          int i10 = ix + 1 + nm[0] * iy;
          int i01 = ix + nm[0] * (iy + 1);
          int i11 = ix + 1 + nm[0] * (iy + 1);
          int j00 = i - offset[0] + nc[0] * (j - offset[1]);
          for (int d = 0; d < dim; d++) {
            Real *uc = (Real *)c;
            Real *um = (Real *)m;
            uc[dim * j00 + d] = (um[dim * i01 + d] + um[dim * i00 + d] +
                                 um[dim * i10 + d] + um[dim * i11 + d]) /
                                4;
          }
        }
      }
    }
    if (applybc)
      _apply_bc(info, true);
    CoarseFineInterpolation(grid, info);
    if (applybc)
      _apply_bc(info, false);
  }
  bool UseCoarseStencil(const BlockInfo &a, const int *b_index) {
    if (a.level == 0 || (!use_averages))
      return false;
    int imin[3];
    int imax[3];
    const int aux = 1 << a.level;
    const bool periodic0[3] = {sim.bcx == periodic, sim.bcy == periodic, false};
    const int blocks[3] = {sim.bpdx * aux - 1, sim.bpdy * aux - 1, 1 * aux - 1};
    for (int d = 0; d < 3; d++) {
      imin[d] = (a.index[d] < b_index[d]) ? 0 : -1;
      imax[d] = (a.index[d] > b_index[d]) ? 0 : +1;
      if (periodic0[d]) {
        if (a.index[d] == 0 && b_index[d] == blocks[d])
          imin[d] = -1;
        if (b_index[d] == 0 && a.index[d] == blocks[d])
          imax[d] = +1;
      } else {
        if (a.index[d] == 0 && b_index[d] == 0)
          imin[d] = 0;
        if (a.index[d] == blocks[d] && b_index[d] == blocks[d])
          imax[d] = 0;
      }
    }
    for (int itest = 0; itest < coarsened_nei_codes_size; itest++)
      for (int i2 = imin[2]; i2 <= imax[2]; i2++)
        for (int i1 = imin[1]; i1 <= imax[1]; i1++)
          for (int i0 = imin[0]; i0 <= imax[0]; i0++) {
            const int icode_test = (i0 + 1) + 3 * (i1 + 1) + 9 * (i2 + 1);
            if (coarsened_nei_codes[itest] == icode_test)
              return true;
          }
    return false;
  }
  void SameLevelExchange(Grid *grid, const BlockInfo &info,
                         const int *const code, const int *const s,
                         const int *const e) {
    int bytes = (e[0] - s[0]) * dim * sizeof(Real);
    if (!bytes)
      return;
    int icode = (code[0] + 1) + 3 * (code[1] + 1) + 9;
    myblocks[icode] =
        grid->avail(info.level, info.Znei[1 + code[0]][1 + code[1]]);
    if (myblocks[icode] == nullptr)
      return;
    Real *b = (Real *)myblocks[icode];
    Real *um = (Real *)m;
    int i = s[0] - start[0];
    int mod = (e[1] - s[1]) % 4;
    for (int iy = s[1]; iy < e[1] - mod; iy += 4) {
      int i0 = i + (iy - start[1]) * nm[0];
      int i1 = i + (iy + 1 - start[1]) * nm[0];
      int i2 = i + (iy + 2 - start[1]) * nm[0];
      int i3 = i + (iy + 3 - start[1]) * nm[0];
      int x0 = s[0] - code[0] * _BS_;
      int y0 = iy - code[1] * _BS_;
      int y1 = iy + 1 - code[1] * _BS_;
      int y2 = iy + 2 - code[1] * _BS_;
      int y3 = iy + 3 - code[1] * _BS_;
      Real *p0 = &um[dim * i0];
      Real *p1 = &um[dim * i1];
      Real *p2 = &um[dim * i2];
      Real *p3 = &um[dim * i3];
      Real *q0 = &b[dim * (_BS_ * y0 + x0)];
      Real *q1 = &b[dim * (_BS_ * y1 + x0)];
      Real *q2 = &b[dim * (_BS_ * y2 + x0)];
      Real *q3 = &b[dim * (_BS_ * y3 + x0)];
      memcpy(p0, q0, bytes);
      memcpy(p1, q1, bytes);
      memcpy(p2, q2, bytes);
      memcpy(p3, q3, bytes);
    }
    for (int iy = e[1] - mod; iy < e[1]; iy++) {
      int i0 = i + (iy - start[1]) * nm[0];
      int x0 = s[0] - code[0] * _BS_;
      int y0 = iy - code[1] * _BS_;
      Real *p = &um[dim * i0];
      Real *q = &b[dim * (_BS_ * y0 + x0)];
      memcpy(p, q, bytes);
    }
  }
  void FineToCoarseExchange(Grid *grid, const BlockInfo &info,
                            const int *const code, const int *const s,
                            const int *const e) {
    Real *um = (Real *)m;
    const int bytes = (abs(code[0]) * (e[0] - s[0]) +
                       (1 - abs(code[0])) * ((e[0] - s[0]) / 2)) *
                      dim * sizeof(Real);
    if (!bytes)
      return;
    int ys = (code[1] == 0) ? 2 : 1;
    int mod = ((e[1] - s[1]) / ys) % 4;
    int Bstep = 1;
    if ((abs(code[0]) + abs(code[1]) == 2))
      Bstep = 3;
    else if ((abs(code[0]) + abs(code[1]) == 3))
      Bstep = 4;
    for (int B = 0; B <= 3; B += Bstep) {
      const int aux = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
      Real *b = (Real *)grid->avail1(
          2 * info.index[0] + std::max(code[0], 0) + code[0] +
              (B % 2) * std::max(0, 1 - abs(code[0])),
          2 * info.index[1] + std::max(code[1], 0) + code[1] +
              aux * std::max(0, 1 - abs(code[1])),
          info.level + 1);
      if (b == nullptr)
        continue;
      const int i =
          abs(code[0]) * (s[0] - start[0]) +
          (1 - abs(code[0])) * (s[0] - start[0] + (B % 2) * (e[0] - s[0]) / 2);
      const int x =
          s[0] - code[0] * _BS_ + std::min(0, code[0]) * (e[0] - s[0]);
      for (int iy = s[1]; iy < e[1] - mod; iy += 4 * ys) {
        int k0 = i + (abs(code[1]) * (iy + 0 * ys - start[1]) +
                      (1 - abs(code[1])) * ((iy + 0 * ys) / 2 - start[1] +
                                            aux * (e[1] - s[1]) / 2)) *
                         nm[0];
        int k1 = i + (abs(code[1]) * (iy + 1 * ys - start[1]) +
                      (1 - abs(code[1])) * ((iy + 1 * ys) / 2 - start[1] +
                                            aux * (e[1] - s[1]) / 2)) *
                         nm[0];
        int k2 = i + (abs(code[1]) * (iy + 2 * ys - start[1]) +
                      (1 - abs(code[1])) * ((iy + 2 * ys) / 2 - start[1] +
                                            aux * (e[1] - s[1]) / 2)) *
                         nm[0];
        int k3 = i + (abs(code[1]) * (iy + 3 * ys - start[1]) +
                      (1 - abs(code[1])) * ((iy + 3 * ys) / 2 - start[1] +
                                            aux * (e[1] - s[1]) / 2)) *
                         nm[0];
        int y0 = (abs(code[1]) == 1) ? 2 * (iy + 0 * ys - code[1] * _BS_) +
                                           std::min(0, code[1]) * _BS_
                                     : iy + 0 * ys;
        int y1 = (abs(code[1]) == 1) ? 2 * (iy + 1 * ys - code[1] * _BS_) +
                                           std::min(0, code[1]) * _BS_
                                     : iy + 1 * ys;
        int y2 = (abs(code[1]) == 1) ? 2 * (iy + 2 * ys - code[1] * _BS_) +
                                           std::min(0, code[1]) * _BS_
                                     : iy + 2 * ys;
        int y3 = (abs(code[1]) == 1) ? 2 * (iy + 3 * ys - code[1] * _BS_) +
                                           std::min(0, code[1]) * _BS_
                                     : iy + 3 * ys;
        int z0 = y0 + 1;
        int z1 = y1 + 1;
        int z2 = y2 + 1;
        int z3 = y3 + 1;
        Real *p0 = um + dim * k0;
        Real *p1 = um + dim * k1;
        Real *p2 = um + dim * k2;
        Real *p3 = um + dim * k3;
        Real *q00 = b + dim * (_BS_ * y0 + x);
        Real *q10 = b + dim * (_BS_ * z0 + x);
        Real *q01 = b + dim * (_BS_ * y1 + x);
        Real *q11 = b + dim * (_BS_ * z1 + x);
        Real *q02 = b + dim * (_BS_ * y2 + x);
        Real *q12 = b + dim * (_BS_ * z2 + x);
        Real *q03 = b + dim * (_BS_ * y3 + x);
        Real *q13 = b + dim * (_BS_ * z3 + x);
        for (int ee = 0; ee < (abs(code[0]) * (e[0] - s[0]) +
                               (1 - abs(code[0])) * ((e[0] - s[0]) / 2));
             ee++) {
          Real *q000 = q00 + dim * 2 * ee;
          Real *q001 = q00 + dim * (2 * ee + 1);
          Real *q010 = q01 + dim * 2 * ee;
          Real *q011 = q01 + dim * (2 * ee + 1);
          Real *q020 = q02 + dim * 2 * ee;
          Real *q021 = q02 + dim * (2 * ee + 1);
          Real *q030 = q03 + dim * 2 * ee;
          Real *q031 = q03 + dim * (2 * ee + 1);
          Real *q110 = q11 + dim * 2 * ee;
          Real *q111 = q11 + dim * (2 * ee + 1);
          Real *q120 = q12 + dim * 2 * ee;
          Real *q121 = q12 + dim * (2 * ee + 1);
          Real *q130 = q13 + dim * 2 * ee;
          Real *q131 = q13 + dim * (2 * ee + 1);
          for (int d = 0; d < dim; d++) {
            *(p0 + dim * ee + d) =
                (*(q000 + d) + *(q010 + d) + *(q001 + d) + *(q011 + d)) / 4;
            *(p1 + dim * ee + d) =
                (*(q010 + d) + *(q110 + d) + *(q011 + d) + *(q111 + d)) / 4;
            *(p2 + dim * ee + d) =
                (*(q020 + d) + *(q120 + d) + *(q021 + d) + *(q121 + d)) / 4;
            *(p3 + dim * ee + d) =
                (*(q030 + d) + *(q130 + d) + *(q031 + d) + *(q131 + d)) / 4;
          }
        }
      }
      for (int iy = e[1] - mod; iy < e[1]; iy += ys) {
        int k = i + (abs(code[1]) * (iy - start[1]) +
                     (1 - abs(code[1])) *
                         (iy / 2 - start[1] + aux * (e[1] - s[1]) / 2)) *
                        nm[0];
        int y = (abs(code[1]) == 1)
                    ? 2 * (iy - code[1] * _BS_) + std::min(0, code[1]) * _BS_
                    : iy;
        int z = y + 1;
        Real *p = um + dim * k;
        Real *q0 = b + dim * (_BS_ * y + x);
        Real *q1 = b + dim * (_BS_ * z + x);
        for (int ee = 0; ee < (abs(code[0]) * (e[0] - s[0]) +
                               (1 - abs(code[0])) * ((e[0] - s[0]) / 2));
             ee++) {
          Real *q00 = q0 + dim * 2 * ee;
          Real *q01 = q0 + dim * (2 * ee + 1);
          Real *q10 = q1 + dim * 2 * ee;
          Real *q11 = q1 + dim * (2 * ee + 1);
          for (int d = 0; d < dim; d++)
            *(p + dim * ee + d) =
                (*(q00 + d) + *(q10 + d) + *(q01 + d) + *(q11 + d)) / 4;
        }
      }
    }
  }
  void CoarseFineExchange(Grid *grid, const BlockInfo &info,
                          const int *const code) {
    Real *uc = (Real *)c;
    int infoNei_index[2] = {(info.index[0] + code[0] + NX) % NX,
                            (info.index[1] + code[1] + NY) % NY};
    int infoNei_index_true[2] = {(info.index[0] + code[0]),
                                 (info.index[1] + code[1])};
    Real *b = (Real *)grid->avail1((infoNei_index[0]) / 2,
                                   (infoNei_index[1]) / 2, info.level - 1);
    if (b == nullptr)
      return;
    int s[2] = {code[0] < 1 ? (code[0] < 0 ? offset[0] : 0) : (_BS_ / 2),
                code[1] < 1 ? (code[1] < 0 ? offset[1] : 0) : (_BS_ / 2)};
    int e[2] = {code[0] < 1 ? (code[0] < 0 ? 0 : (_BS_ / 2))
                            : (_BS_ / 2) + (end[0]) / 2 + (2) - 1,
                code[1] < 1 ? (code[1] < 0 ? 0 : (_BS_ / 2))
                            : (_BS_ / 2) + (end[1]) / 2 + (2) - 1};
    int bytes = (e[0] - s[0]) * dim * sizeof(Real);
    if (!bytes)
      return;
    int base[2] = {(info.index[0] + code[0]) % 2,
                   (info.index[1] + code[1]) % 2};
    int CoarseEdge[2];
    CoarseEdge[0] = (code[0] == 0) ? 0
                    : (((info.index[0] % 2 == 0) &&
                        (infoNei_index_true[0] > info.index[0])) ||
                       ((info.index[0] % 2 == 1) &&
                        (infoNei_index_true[0] < info.index[0])))
                        ? 1
                        : 0;
    CoarseEdge[1] = (code[1] == 0) ? 0
                    : (((info.index[1] % 2 == 0) &&
                        (infoNei_index_true[1] > info.index[1])) ||
                       ((info.index[1] % 2 == 1) &&
                        (infoNei_index_true[1] < info.index[1])))
                        ? 1
                        : 0;
    const int start[2] = {
        std::max(code[0], 0) * _BS_ / 2 +
            (1 - abs(code[0])) * base[0] * _BS_ / 2 - code[0] * _BS_ +
            CoarseEdge[0] * code[0] * _BS_ / 2,
        std::max(code[1], 0) * _BS_ / 2 +
            (1 - abs(code[1])) * base[1] * _BS_ / 2 - code[1] * _BS_ +
            CoarseEdge[1] * code[1] * _BS_ / 2};
    const int i = s[0] - offset[0];
    const int mod = (e[1] - s[1]) % 4;
    for (int iy = s[1]; iy < e[1] - mod; iy += 4) {
      int i0 = i + (iy + 0 - offset[1]) * nc[0];
      int i1 = i + (iy + 1 - offset[1]) * nc[0];
      int i2 = i + (iy + 2 - offset[1]) * nc[0];
      int i3 = i + (iy + 3 - offset[1]) * nc[0];
      int y0 = iy + 0 + start[1];
      int y1 = iy + 1 + start[1];
      int y2 = iy + 2 + start[1];
      int y3 = iy + 3 + start[1];
      int x = s[0] + start[0];
      Real *p0 = uc + dim * i0;
      Real *p1 = uc + dim * i1;
      Real *p2 = uc + dim * i2;
      Real *p3 = uc + dim * i3;
      Real *q0 = b + dim * (_BS_ * y0 + x);
      Real *q1 = b + dim * (_BS_ * y1 + x);
      Real *q2 = b + dim * (_BS_ * y2 + x);
      Real *q3 = b + dim * (_BS_ * y3 + x);
      memcpy(p0, q0, bytes);
      memcpy(p1, q1, bytes);
      memcpy(p2, q2, bytes);
      memcpy(p3, q3, bytes);
    }
    for (int iy = e[1] - mod; iy < e[1]; iy++) {
      int i0 = i + (iy - offset[1]) * nc[0];
      int y0 = iy + start[1];
      int x = s[0] + start[0];
      Real *p = uc + dim * i0;
      Real *q = b + dim * (_BS_ * y0 + x);
      memcpy(p, q, bytes);
    }
  }
  void FillCoarseVersion(const int *const code) {
    Real *uc = (Real *)c;
    const int icode = (code[0] + 1) + 3 * (code[1] + 1) + 9;
    if (myblocks[icode] == nullptr)
      return;
    Real *b = (Real *)myblocks[icode];
    int eC[2] = {(end[0]) / 2 + (2), (end[1]) / 2 + (2)};
    int s[2] = {code[0] < 1 ? (code[0] < 0 ? offset[0] : 0) : (_BS_ / 2),
                code[1] < 1 ? (code[1] < 0 ? offset[1] : 0) : (_BS_ / 2)};
    int e[2] = {
        code[0] < 1 ? (code[0] < 0 ? 0 : (_BS_ / 2)) : (_BS_ / 2) + eC[0] - 1,
        code[1] < 1 ? (code[1] < 0 ? 0 : (_BS_ / 2)) : (_BS_ / 2) + eC[1] - 1};
    int bytes = (e[0] - s[0]) * dim * sizeof(Real);
    if (!bytes)
      return;
    int start[2] = {s[0] + std::max(code[0], 0) * (_BS_ / 2) - code[0] * _BS_ +
                        std::min(0, code[0]) * (e[0] - s[0]),
                    s[1] + std::max(code[1], 0) * (_BS_ / 2) - code[1] * _BS_ +
                        std::min(0, code[1]) * (e[1] - s[1])};
    int i = s[0] - offset[0];
    int x = start[0];
    for (int iy = s[1]; iy < e[1]; iy++) {
      int i0 = i + (iy - offset[1]) * nc[0];
      Real *p1 = uc + dim * i0;
      int y0 = 2 * (iy - s[1]) + start[1];
      int y1 = y0 + 1;
      Real *q0 = b + dim * (_BS_ * y0 + x);
      Real *q1 = b + dim * (_BS_ * y1 + x);
      for (int ee = 0; ee < e[0] - s[0]; ee++) {
        Real *q00 = q0 + dim * 2 * ee;
        Real *q01 = q0 + dim * (2 * ee + 1);
        Real *q10 = q1 + dim * 2 * ee;
        Real *q11 = q1 + dim * (2 * ee + 1);
        for (int d = 0; d < dim; d++)
          *(p1 + dim * ee + d) =
              (*(q00 + d) + *(q10 + d) + *(q01 + d) + *(q11 + d)) / 4;
      }
    }
  }
  void CoarseFineInterpolation(Grid *grid, const BlockInfo &info) {
    Real *um = (Real *)m;
    Real *uc = (Real *)c;
    int aux = 1 << info.level;
    bool xskin = info.index[0] == 0 || info.index[0] == sim.bpdx * aux - 1;
    bool yskin = info.index[1] == 0 || info.index[1] == sim.bpdy * aux - 1;
    int xskip = info.index[0] == 0 ? -1 : 1;
    int yskip = info.index[1] == 0 ? -1 : 1;

    for (int ii = 0; ii < coarsened_nei_codes_size; ++ii) {
      int icode = coarsened_nei_codes[ii];
      if (icode == 1 * 1 + 3 * 1 + 9 * 1)
        continue;
      int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, (icode / 9) % 3 - 1};
      if (code[2] != 0)
        continue;
      if (sim.bcx != periodic && code[0] == xskip && xskin)
        continue;
      if (sim.bcy != periodic && code[1] == yskip && yskin)
        continue;
      if (!istensorial && !use_averages &&
          abs(code[0]) + abs(code[1]) + abs(code[2]) > 1)
        continue;
      int s[3] = {code[0] < 1 ? (code[0] < 0 ? start[0] : 0) : _BS_,
                  code[1] < 1 ? (code[1] < 0 ? start[1] : 0) : _BS_,
                  code[2] < 1 ? (code[2] < 0 ? start[2] : 0) : 1};
      int e[3] = {code[0] < 1 ? (code[0] < 0 ? 0 : _BS_) : _BS_ + end[0] - 1,
                  code[1] < 1 ? (code[1] < 0 ? 0 : _BS_) : _BS_ + end[1] - 1,
                  code[2] < 1 ? (code[2] < 0 ? 0 : 1) : 1 + end[2] - 1};
      int sC[3] = {
          code[0] < 1 ? (code[0] < 0 ? ((start[0] - 1) / 2) : 0) : (_BS_ / 2),
          code[1] < 1 ? (code[1] < 0 ? ((start[1] - 1) / 2) : 0) : (_BS_ / 2),
          code[2] < 1 ? (code[2] < 0 ? ((start[2] - 1) / 2) : 0) : 1};
      int bytes = (e[0] - s[0]) * dim * sizeof(Real);
      if (!bytes)
        continue;
      if (use_averages) {
        for (int iy = s[1]; iy < e[1]; iy += 1) {
          int YY =
              (iy - s[1] - std::min(0, code[1]) * ((e[1] - s[1]) % 2)) / 2 +
              sC[1];
          for (int ix = s[0]; ix < e[0]; ix += 1) {
            int XX =
                (ix - s[0] - std::min(0, code[0]) * ((e[0] - s[0]) % 2)) / 2 +
                sC[0];
            Real *Test[3][3];
            for (int i = 0; i < 3; i++)
              for (int j = 0; j < 3; j++) {
                int i0 =
                    XX - 1 + i - offset[0] + nc[0] * (YY - 1 + j - offset[1]);
                Test[i][j] = uc + dim * i0;
              }
            int i1 = ix - start[0] + nm[0] * (iy - start[1]);
            for (int d = 0; d < dim; d++)
              TestInterp(
                  Test, um + dim * i1 + d,
                  abs(ix - s[0] - std::min(0, code[0]) * ((e[0] - s[0]) % 2)) %
                      2,
                  abs(iy - s[1] - std::min(0, code[1]) * ((e[1] - s[1]) % 2)) %
                      2);
          }
        }
      }
      if (abs(code[0]) + abs(code[1]) == 1) {
        for (int iy = s[1]; iy < e[1]; iy += 2) {
          int YY =
              (iy - s[1] - std::min(0, code[1]) * ((e[1] - s[1]) % 2)) / 2 +
              sC[1] - offset[1];
          int y =
              abs(iy - s[1] - std::min(0, code[1]) * ((e[1] - s[1]) % 2)) % 2;
          int iyp = (abs(iy) % 2 == 1) ? -1 : 1;
          double dy = 0.25 * (2 * y - 1);
          for (int ix = s[0]; ix < e[0]; ix += 2) {
            int XX =
                (ix - s[0] - std::min(0, code[0]) * ((e[0] - s[0]) % 2)) / 2 +
                sC[0] - offset[0];
            int x =
                abs(ix - s[0] - std::min(0, code[0]) * ((e[0] - s[0]) % 2)) % 2;
            int ixp = (abs(ix) % 2 == 1) ? -1 : 1;
            double dx = 0.25 * (2 * x - 1);
            if (ix < -2 || iy < -2 || ix > _BS_ + 1 || iy > _BS_ + 1)
              continue;
            int i0 = XX + nc[0] * (YY + 2);
            int i1 = XX + nc[0] * (YY);
            int i2 = XX + nc[0] * (YY + 1);
            int i3 = XX + nc[0] * (YY - 2);
            int i4 = XX + nc[0] * (YY - 1);
            int i5 = XX + 2 + nc[0] * (YY);
            int i6 = XX + 1 + nc[0] * (YY);
            int i7 = XX - 1 + nc[0] * (YY);
            int i8 = XX - 2 + nc[0] * (YY);
            int j0 = ix - start[0] + nm[0] * (iy - start[1]);
            int j1 = ix - start[0] + nm[0] * (iy - start[1] + iyp);
            int j2 = ix - start[0] + ixp + nm[0] * (iy - start[1]);
            int j3 = ix - start[0] + ixp + nm[0] * (iy - start[1] + iyp);
            for (int d = 0; d < dim; d++) {
              if (code[0] != 0) {
                Real dudy, dudy2;
                if (YY + offset[1] == 0) {
                  dudy = (-0.5 * uc[dim * i0 + d] - 1.5 * uc[dim * i1 + d]) +
                         2.0 * uc[dim * i2 + d];
                  dudy2 = (uc[dim * i0 + d] + uc[dim * i1 + d]) -
                          2.0 * uc[dim * i2 + d];
                } else if (YY + offset[1] == (_BS_ / 2) - 1) {
                  dudy = (0.5 * uc[dim * i3 + d] + 1.5 * uc[dim * i1 + d]) -
                         2.0 * uc[dim * i4 + d];
                  dudy2 = (uc[dim * i3 + d] + uc[dim * i1 + d]) -
                          2.0 * uc[dim * i4 + d];
                } else {
                  dudy = 0.5 * (uc[dim * i2 + d] - uc[dim * i4 + d]);
                  dudy2 = (uc[dim * i2 + d] + uc[dim * i4 + d]) -
                          2.0 * uc[dim * i1 + d];
                }
                um[dim * j0 + d] =
                    uc[dim * i1 + d] + dy * dudy + (0.5 * dy * dy) * dudy2;
                if (iy + iyp >= s[1] && iy + iyp < e[1])
                  um[dim * j1 + d] =
                      uc[dim * i1 + d] - dy * dudy + (0.5 * dy * dy) * dudy2;
                if (ix + ixp >= s[0] && ix + ixp < e[0])
                  um[dim * j2 + d] =
                      uc[dim * i1 + d] + dy * dudy + (0.5 * dy * dy) * dudy2;
                if (ix + ixp >= s[0] && ix + ixp < e[0] && iy + iyp >= s[1] &&
                    iy + iyp < e[1])
                  um[dim * j3 + d] =
                      uc[dim * i1 + d] - dy * dudy + (0.5 * dy * dy) * dudy2;
              } else {
                Real dudx, dudx2;
                if (XX + offset[0] == 0) {
                  dudx = (-0.5 * uc[dim * i5 + d] - 1.5 * uc[dim * i1 + d]) +
                         2.0 * uc[dim * i6 + d];
                  dudx2 = (uc[dim * i5 + d] + uc[dim * i1 + d]) -
                          2.0 * uc[dim * i6 + d];
                } else if (XX + offset[0] == (_BS_ / 2) - 1) {
                  dudx = (0.5 * uc[dim * i8 + d] + 1.5 * uc[dim * i1 + d]) -
                         2.0 * uc[dim * i7 + d];
                  dudx2 = (uc[dim * i8 + d] + uc[dim * i1 + d]) -
                          2.0 * uc[dim * i7 + d];
                } else {
                  dudx = 0.5 * (uc[dim * i6 + d] - uc[dim * i7 + d]);
                  dudx2 = (uc[dim * i6 + d] + uc[dim * i7 + d]) -
                          2.0 * uc[dim * i1 + d];
                }
                um[dim * j0 + d] =
                    uc[dim * i1 + d] + dx * dudx + (0.5 * dx * dx) * dudx2;
                if (iy + iyp >= s[1] && iy + iyp < e[1])
                  um[dim * j1 + d] =
                      uc[dim * i1 + d] + dx * dudx + (0.5 * dx * dx) * dudx2;
                if (ix + ixp >= s[0] && ix + ixp < e[0])
                  um[dim * j2 + d] =
                      uc[dim * i1 + d] - dx * dudx + (0.5 * dx * dx) * dudx2;
                if (ix + ixp >= s[0] && ix + ixp < e[0] && iy + iyp >= s[1] &&
                    iy + iyp < e[1])
                  um[dim * j3 + d] =
                      uc[dim * i1 + d] - dx * dudx + (0.5 * dx * dx) * dudx2;
              }
            }
          }
        }
        for (int iy = s[1]; iy < e[1]; iy += 1) {
          for (int ix = s[0]; ix < e[0]; ix += 1) {
            if (ix < -2 || iy < -2 || ix > _BS_ + 1 || iy > _BS_ + 1)
              continue;
            int k0 = ix - start[0] + nm[0] * (iy - start[1] - 1);
            int k1 = ix - start[0] + nm[0] * (iy - start[1] - 2);
            int k2 = ix - start[0] + nm[0] * (iy - start[1] + 1);
            int k3 = ix - start[0] + nm[0] * (iy - start[1] + 2);
            int k4 = ix - start[0] + nm[0] * (iy - start[1] + 3);
            int k5 = ix - start[0] - 1 + nm[0] * (iy - start[1]);
            int k6 = ix - start[0] - 2 + nm[0] * (iy - start[1]);
            int k7 = ix - start[0] - 3 + nm[0] * (iy - start[1]);
            int k8 = ix - start[0] + 1 + nm[0] * (iy - start[1]);
            int k9 = ix - start[0] + 2 + nm[0] * (iy - start[1]);
            int k10 = ix - start[0] + 3 + nm[0] * (iy - start[1]);
            int k11 = ix - start[0] + nm[0] * (iy - start[1] - 3);
            int k12 = ix - start[0] + nm[0] * (iy - start[1]);
            int x =
                abs(ix - s[0] - std::min(0, code[0]) * ((e[0] - s[0]) % 2)) % 2;
            int y =
                abs(iy - s[1] - std::min(0, code[1]) * ((e[1] - s[1]) % 2)) % 2;
            for (int d = 0; d < dim; d++) {
              Real *a = um + dim * k12 + d;
              if (code[0] == 0 && code[1] == 1) {
                if (y == 0) {
                  Real *b = um + dim * k0 + d;
                  Real *c = um + dim * k1 + d;
                  LI(a, b, c);
                } else if (y == 1) {
                  Real *b = um + dim * k1 + d;
                  Real *c = um + dim * k11 + d;
                  LE(a, b, c);
                }
              } else if (code[0] == 0 && code[1] == -1) {
                if (y == 1) {
                  Real *b = um + dim * k2 + d;
                  Real *c = um + dim * k3 + d;
                  LI(a, b, c);
                } else if (y == 0) {
                  Real *b = um + dim * k3 + d;
                  Real *c = um + dim * k4 + d;
                  LE(a, b, c);
                }
              } else if (code[1] == 0 && code[0] == 1) {
                if (x == 0) {
                  Real *b = um + dim * k5 + d;
                  Real *c = um + dim * k6 + d;
                  LI(a, b, c);
                } else if (x == 1) {
                  Real *b = um + dim * k6 + d;
                  Real *c = um + dim * k7 + d;
                  LE(a, b, c);
                }
              } else if (code[1] == 0 && code[0] == -1) {
                if (x == 1) {
                  Real *b = um + dim * k8 + d;
                  Real *c = um + dim * k9 + d;
                  LI(a, b, c);
                } else if (x == 0) {
                  Real *b = um + dim * k9 + d;
                  Real *c = um + dim * k10 + d;
                  LE(a, b, c);
                }
              }
            }
          }
        }
      }
    }
  }
  virtual void _apply_bc(BlockInfo &info, bool coarse) {}
  BlockLab(const BlockLab &) = delete;
  BlockLab &operator=(const BlockLab &) = delete;
};
struct Adaptation {
  bool movedBlocks;
  const int dim;
  struct MPI_Block {
    long long mn[2];
    uint8_t data[_BS_ * _BS_ * max_dim * sizeof(Real)];
  };
  StencilInfo stencil;
  bool CallValidStates;
  bool boundary_needed;
  bool basic_refinement;
  std::vector<long long> dealloc_IDs;
  Adaptation(int dim) : dim(dim) { movedBlocks = false; }
  void AddBlock(Grid *grid, const int level, const long long Z, uint8_t *data) {
    grid->_alloc(level, Z);
    BlockInfo &info = grid->get(level, Z);
    memcpy(info.block, data, _BS_ * _BS_ * dim * sizeof(Real));
    int p[2];
    sim.space_curve->inverse(Z, level, p[0], p[1]);
    if (level < sim.levelMax - 1)
      for (int j1 = 0; j1 < 2; j1++)
        for (int i1 = 0; i1 < 2; i1++) {
          const long long nc =
              getZforward(level + 1, 2 * p[0] + i1, 2 * p[1] + j1);
          grid->Tree0(level + 1, nc) = -2;
        }
    if (level > 0) {
      const long long nf = getZforward(level - 1, p[0] / 2, p[1] / 2);
      grid->Tree0(level - 1, nf) = -1;
    }
  }
  void Balance_Diffusion(Grid *grid,
                         std::vector<long long> &block_distribution) {
    movedBlocks = false;
    {
      long long max_b = block_distribution[0];
      long long min_b = block_distribution[0];
      for (auto &b : block_distribution) {
        max_b = std::max(max_b, b);
        min_b = std::min(min_b, b);
      }
      const double ratio = static_cast<double>(max_b) / min_b;
      if (ratio > 1.01 || min_b == 0) {
        Balance_Global(grid, block_distribution);
        return;
      }
    }
    const int right = (sim.rank == sim.size - 1) ? MPI_PROC_NULL : sim.rank + 1;
    const int left = (sim.rank == 0) ? MPI_PROC_NULL : sim.rank - 1;
    const int my_blocks = grid->infos.size();
    int right_blocks, left_blocks;
    MPI_Request reqs[4];
    MPI_Irecv(&left_blocks, 1, MPI_INT, left, 123, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(&right_blocks, 1, MPI_INT, right, 456, MPI_COMM_WORLD, &reqs[1]);
    MPI_Isend(&my_blocks, 1, MPI_INT, left, 456, MPI_COMM_WORLD, &reqs[2]);
    MPI_Isend(&my_blocks, 1, MPI_INT, right, 123, MPI_COMM_WORLD, &reqs[3]);
    MPI_Waitall(4, &reqs[0], MPI_STATUSES_IGNORE);
    const int nu = 4;
    const int flux_left = (sim.rank == 0) ? 0 : (my_blocks - left_blocks) / nu;
    const int flux_right =
        (sim.rank == sim.size - 1) ? 0 : (my_blocks - right_blocks) / nu;
    std::vector<BlockInfo> SortedInfos = grid->infos;
    if (flux_right != 0 || flux_left != 0)
      std::sort(SortedInfos.begin(), SortedInfos.end());
    std::vector<MPI_Block> send_left;
    std::vector<MPI_Block> recv_left;
    std::vector<MPI_Block> send_right;
    std::vector<MPI_Block> recv_right;
    std::vector<MPI_Request> request;
    if (flux_left > 0) {
      send_left.resize(flux_left);
#pragma omp parallel for schedule(runtime)
      for (int i = 0; i < flux_left; i++) {
        BlockInfo *info = &SortedInfos[i];
        MPI_Block *x = &send_left[i];
        x->mn[0] = info->level;
        x->mn[1] = info->Z;
        std::memcpy(x->data, info->block, _BS_ * _BS_ * dim * sizeof(Real));
      }
      MPI_Request req{};
      request.push_back(req);
      MPI_Isend(&send_left[0], send_left.size() * sizeof(send_left[0]),
                MPI_UINT8_T, left, 7890, MPI_COMM_WORLD, &request.back());
    } else if (flux_left < 0) {
      recv_left.resize(abs(flux_left));
      MPI_Request req{};
      request.push_back(req);
      MPI_Irecv(&recv_left[0], recv_left.size() * sizeof(recv_left[0]),
                MPI_UINT8_T, left, 4560, MPI_COMM_WORLD, &request.back());
    }
    if (flux_right > 0) {
      send_right.resize(flux_right);
#pragma omp parallel for schedule(runtime)
      for (int i = 0; i < flux_right; i++) {
        BlockInfo *info = &SortedInfos[my_blocks - i - 1];
        MPI_Block *x = &send_right[i];
        x->mn[0] = info->level;
        x->mn[1] = info->Z;
        std::memcpy(x->data, info->block, _BS_ * _BS_ * dim * sizeof(Real));
      }
      MPI_Request req{};
      request.push_back(req);
      MPI_Isend(&send_right[0], send_right.size() * sizeof(send_right[0]),
                MPI_UINT8_T, right, 4560, MPI_COMM_WORLD, &request.back());
    } else if (flux_right < 0) {
      recv_right.resize(abs(flux_right));
      MPI_Request req{};
      request.push_back(req);
      MPI_Irecv(&recv_right[0], recv_right.size() * sizeof(recv_right[0]),
                MPI_UINT8_T, right, 7890, MPI_COMM_WORLD, &request.back());
    }
    for (int i = 0; i < flux_right; i++) {
      BlockInfo &info = SortedInfos[my_blocks - i - 1];
      grid->_dealloc(info.level, info.Z);
      grid->Tree0(info.level, info.Z) = right;
    }
    for (int i = 0; i < flux_left; i++) {
      BlockInfo &info = SortedInfos[i];
      grid->_dealloc(info.level, info.Z);
      grid->Tree0(info.level, info.Z) = left;
    }
    if (request.size() != 0) {
      movedBlocks = true;
      MPI_Waitall(request.size(), &request[0], MPI_STATUSES_IGNORE);
    }
    int temp = movedBlocks ? 1 : 0;
    MPI_Request request_reduction;
    MPI_Iallreduce(MPI_IN_PLACE, &temp, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD,
                   &request_reduction);
    for (int i = 0; i < -flux_left; i++)
      AddBlock(grid, recv_left[i].mn[0], recv_left[i].mn[1], recv_left[i].data);
    for (int i = 0; i < -flux_right; i++)
      AddBlock(grid, recv_right[i].mn[0], recv_right[i].mn[1],
               recv_right[i].data);
    MPI_Wait(&request_reduction, MPI_STATUS_IGNORE);
    movedBlocks = (temp >= 1);
    grid->FillPos();
  }
  void Balance_Global(Grid *grid, std::vector<long long> &all_b) {
    std::vector<BlockInfo> SortedInfos = grid->infos;
    std::sort(SortedInfos.begin(), SortedInfos.end());
    long long total_load = 0;
    for (int r = 0; r < sim.size; r++)
      total_load += all_b[r];
    long long my_load = total_load / sim.size;
    if (sim.rank < (total_load % sim.size))
      my_load += 1;
    std::vector<long long> index_start(sim.size);
    index_start[0] = 0;
    for (int r = 1; r < sim.size; r++)
      index_start[r] = index_start[r - 1] + all_b[r - 1];
    long long ideal_index = (total_load / sim.size) * sim.rank;
    ideal_index += (sim.rank < (total_load % sim.size))
                       ? sim.rank
                       : (total_load % sim.size);
    std::vector<std::vector<MPI_Block>> send_blocks(sim.size);
    std::vector<std::vector<MPI_Block>> recv_blocks(sim.size);
    for (int r = 0; r < sim.size; r++)
      if (sim.rank != r) {
        {
          long long a1 = ideal_index;
          long long a2 = ideal_index + my_load - 1;
          long long b1 = index_start[r];
          long long b2 = index_start[r] + all_b[r] - 1;
          long long c1 = std::max(a1, b1);
          long long c2 = std::min(a2, b2);
          if (c2 - c1 + 1 > 0)
            recv_blocks[r].resize(c2 - c1 + 1);
        }
        {
          long long other_ideal_index = (total_load / sim.size) * r;
          other_ideal_index +=
              (r < (total_load % sim.size)) ? r : (total_load % sim.size);
          long long other_load = total_load / sim.size;
          if (r < (total_load % sim.size))
            other_load += 1;
          long long a1 = other_ideal_index;
          long long a2 = other_ideal_index + other_load - 1;
          long long b1 = index_start[sim.rank];
          long long b2 = index_start[sim.rank] + all_b[sim.rank] - 1;
          long long c1 = std::max(a1, b1);
          long long c2 = std::min(a2, b2);
          if (c2 - c1 + 1 > 0)
            send_blocks[r].resize(c2 - c1 + 1);
        }
      }
    int tag = 12345;
    std::vector<MPI_Request> requests;
    for (int r = 0; r < sim.size; r++)
      if (recv_blocks[r].size() != 0) {
        MPI_Request req{};
        requests.push_back(req);
        MPI_Irecv(recv_blocks[r].data(),
                  recv_blocks[r].size() * sizeof(recv_blocks[r][0]),
                  MPI_UINT8_T, r, tag, MPI_COMM_WORLD, &requests.back());
      }
    long long counter_S = 0;
    long long counter_E = 0;
    for (int r = 0; r < sim.rank; r++)
      if (send_blocks[r].size() != 0) {
        for (size_t i = 0; i < send_blocks[r].size(); i++) {
          BlockInfo *info = &SortedInfos[counter_S + i];
          MPI_Block *x = &send_blocks[r][i];
          x->mn[0] = info->level;
          x->mn[1] = info->Z;
          std::memcpy(x->data, info->block, _BS_ * _BS_ * dim * sizeof(Real));
        }
        counter_S += send_blocks[r].size();
        MPI_Request req{};
        requests.push_back(req);
        MPI_Isend(send_blocks[r].data(),
                  send_blocks[r].size() * sizeof(send_blocks[r][0]),
                  MPI_UINT8_T, r, tag, MPI_COMM_WORLD, &requests.back());
      }
    for (int r = sim.size - 1; r > sim.rank; r--)
      if (send_blocks[r].size() != 0) {
        for (size_t i = 0; i < send_blocks[r].size(); i++) {
          BlockInfo *info =
              &SortedInfos[SortedInfos.size() - 1 - (counter_E + i)];
          MPI_Block *x = &send_blocks[r][i];
          x->mn[0] = info->level;
          x->mn[1] = info->Z;
          std::memcpy(x->data, info->block, _BS_ * _BS_ * dim * sizeof(Real));
        }
        counter_E += send_blocks[r].size();
        MPI_Request req{};
        requests.push_back(req);
        MPI_Isend(send_blocks[r].data(),
                  send_blocks[r].size() * sizeof(send_blocks[r][0]),
                  MPI_UINT8_T, r, tag, MPI_COMM_WORLD, &requests.back());
      }
    movedBlocks = true;
    std::vector<long long> deallocIDs;
    counter_S = 0;
    counter_E = 0;
    for (int r = 0; r < sim.size; r++)
      if (send_blocks[r].size() != 0) {
        if (r < sim.rank) {
          for (size_t i = 0; i < send_blocks[r].size(); i++) {
            BlockInfo &info = SortedInfos[counter_S + i];
            deallocIDs.push_back(info.id2);
            grid->Tree0(info.level, info.Z) = r;
          }
          counter_S += send_blocks[r].size();
        } else {
          for (size_t i = 0; i < send_blocks[r].size(); i++) {
            BlockInfo &info =
                SortedInfos[SortedInfos.size() - 1 - (counter_E + i)];
            deallocIDs.push_back(info.id2);
            grid->Tree0(info.level, info.Z) = r;
          }
          counter_E += send_blocks[r].size();
        }
      }
    grid->dealloc_many(deallocIDs);
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
#pragma omp parallel
    {
      for (int r = 0; r < sim.size; r++)
        if (recv_blocks[r].size() != 0) {
#pragma omp for
          for (size_t i = 0; i < recv_blocks[r].size(); i++)
            AddBlock(grid, recv_blocks[r][i].mn[0], recv_blocks[r][i].mn[1],
                     recv_blocks[r][i].data);
        }
    }
    grid->FillPos();
  }
  template <typename TLab> void Adapt(Grid *grid, bool basic) {
    basic_refinement = basic;
    Synchronizer<Grid> *Synch = nullptr;
    if (basic == false) {
      Synch = grid->sync1(stencil);
      MPI_Waitall(Synch->requests.size(), Synch->requests.data(),
                  MPI_STATUSES_IGNORE);
      grid->boundary = Synch->halo_blocks;
      if (boundary_needed)
        grid->UpdateBoundary();
    }
    int r = 0;
    int c = 0;
    std::vector<int> m_com;
    std::vector<int> m_ref;
    std::vector<long long> n_com;
    std::vector<long long> n_ref;
    std::vector<BlockInfo> &I = grid->infos;
    long long blocks_after = I.size();
    for (auto &info : I) {
      if (info.state == Refine) {
        m_ref.push_back(info.level);
        n_ref.push_back(info.Z);
        blocks_after += (1 << 2) - 1;
        r++;
      } else if (info.state == Compress && info.index[0] % 2 == 0 &&
                 info.index[1] % 2 == 0 && info.index[2] % 2 == 0) {
        m_com.push_back(info.level);
        n_com.push_back(info.Z);
        c++;
      } else if (info.state == Compress) {
        blocks_after--;
      }
    }
    MPI_Request requests[2];
    int temp[2] = {r, c};
    int result[2];
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::vector<long long> block_distribution(size);
    MPI_Iallreduce(&temp, &result, 2, MPI_INT, MPI_SUM, MPI_COMM_WORLD,
                   &requests[0]);
    MPI_Iallgather(&blocks_after, 1, MPI_LONG_LONG, block_distribution.data(),
                   1, MPI_LONG_LONG, MPI_COMM_WORLD, &requests[1]);
    dealloc_IDs.clear();
    {
      TLab lab(dim);
      if (Synch != nullptr)
        lab.prepare(grid, Synch->stencil);
      for (size_t i = 0; i < m_ref.size(); i++) {
        const int level = m_ref[i];
        const long long Z = n_ref[i];
        BlockInfo &parent = grid->get(level, Z);
        parent.state = Leave;
        if (basic_refinement == false)
          lab.load(grid, parent, true);
        const int p[3] = {parent.index[0], parent.index[1], parent.index[2]};
        assert(parent.block != NULL);
        assert(level <= sim.levelMax - 1);
        void *Blocks[4];
        for (int j = 0; j < 2; j++)
          for (int i = 0; i < 2; i++) {
            const long long nc =
                getZforward(level + 1, 2 * p[0] + i, 2 * p[1] + j);
            BlockInfo &Child = grid->get(level + 1, nc);
            Child.state = Leave;
            grid->_alloc(level + 1, nc);
            grid->Tree0(level + 1, nc) = -2;
            Blocks[j * 2 + i] = Child.block;
          }
        if (basic_refinement == false) {
          int nm = _BS_ + Synch->stencil.ex - Synch->stencil.sx - 1;
          int offsetX[2] = {0, _BS_ / 2};
          int offsetY[2] = {0, _BS_ / 2};
          Real *um = (Real *)lab.m;
          for (int J = 0; J < 2; J++)
            for (int I = 0; I < 2; I++) {
              void *bb = Blocks[J * 2 + I];
              Real *b = (Real *)bb;
              memset(bb, 0, dim * _BS_ * _BS_ * sizeof(Real));
              for (int j = 0; j < _BS_; j += 2)
                for (int i = 0; i < _BS_; i += 2) {
                  int i0 = i / 2 + offsetX[I] - Synch->stencil.sx;
                  int j0 = j / 2 + offsetY[J] - Synch->stencil.sy;
                  int im = i0 - 1;
                  int ip = i0 + 1;
                  int jm = j0 - 1;
                  int jp = j0 + 1;
                  int o0 = _BS_ * j + i;
                  int o1 = _BS_ * j + i + 1;
                  int o2 = _BS_ * (j + 1) + i;
                  int o3 = _BS_ * (j + 1) + i + 1;
                  for (int d = 0; d < dim; d++) {
                    Real l00 = um[dim * (nm * j0 + i0) + d];
                    Real l0p = um[dim * (nm * jp + i0) + d];
                    Real lm0 = um[dim * (nm * j0 + im) + d];
                    Real lmm = um[dim * (nm * jm + im) + d];
                    Real lmp = um[dim * (nm * jp + im) + d];
                    Real lp0 = um[dim * (nm * j0 + ip) + d];
                    Real lpm = um[dim * (nm * jm + ip) + d];
                    Real lpp = um[dim * (nm * jp + ip) + d];
                    Real l0m = um[dim * (nm * jm + i0) + d];
                    Real x = 0.5 * (lp0 - lm0);
                    Real y = 0.5 * (l0p - l0m);
                    Real x2 = (lp0 + lm0) - 2.0 * l00;
                    Real y2 = (l0p + l0m) - 2.0 * l00;
                    Real xy = 0.25 * ((lpp + lmm) - (lpm + lmp));
                    b[dim * o0 + d] =
                        (l00 + (-0.25 * x - 0.25 * y)) +
                        ((0.03125 * x2 + 0.03125 * y2) + 0.0625 * xy);
                    b[dim * o1 + d] =
                        (l00 + (+0.25 * x - 0.25 * y)) +
                        ((0.03125 * x2 + 0.03125 * y2) - 0.0625 * xy);
                    b[dim * o2 + d] =
                        (l00 + (-0.25 * x + 0.25 * y)) +
                        ((0.03125 * x2 + 0.03125 * y2) - 0.0625 * xy);
                    b[dim * o3 + d] =
                        (l00 + (+0.25 * x + 0.25 * y)) +
                        ((0.03125 * x2 + 0.03125 * y2) + 0.0625 * xy);
                  }
                }
            }
        }
      }
      for (size_t i = 0; i < m_ref.size(); i++) {
        const int level = m_ref[i];
        const long long Z = n_ref[i];
#pragma omp critical
        { dealloc_IDs.push_back(grid->get(level, Z).id2); }
        BlockInfo &parent = grid->get(level, Z);
        grid->Tree1(parent) = -1;
        parent.state = Leave;
        int p[3] = {parent.index[0], parent.index[1], parent.index[2]};
        for (int j = 0; j < 2; j++)
          for (int i = 0; i < 2; i++) {
            const long long nc =
                getZforward(level + 1, 2 * p[0] + i, 2 * p[1] + j);
            BlockInfo &Child = grid->get(level + 1, nc);
            grid->Tree1(Child) = sim.rank;
            if (level + 2 < sim.levelMax)
              for (int i0 = 0; i0 < 2; i0++)
                for (int i1 = 0; i1 < 2; i1++)
                  grid->Tree0(level + 2, Child.Zchild[i0][i1]) = -2;
          }
      }
    }
    grid->dealloc_many(dealloc_IDs);
    std::vector<std::vector<MPI_Block>> send_blocks(sim.size);
    std::vector<std::vector<MPI_Block>> recv_blocks(sim.size);
    for (auto &b : I) {
      const long long nBlock =
          getZforward(b.level, 2 * (b.index[0] / 2), 2 * (b.index[1] / 2));
      const BlockInfo &base = grid->get(b.level, nBlock);
      if (!(grid->Tree1(base) >= 0) || base.state != Compress)
        continue;
      const BlockInfo &bCopy = grid->get(b.level, b.Z);
      const int baserank = grid->Tree0(b.level, nBlock);
      const int brank = grid->Tree0(b.level, b.Z);
      if (b.Z != nBlock) {
        if (baserank != sim.rank && brank == sim.rank) {
          MPI_Block x;
          x.mn[0] = bCopy.level;
          x.mn[1] = bCopy.Z;
          std::memcpy(&x.data[0], bCopy.block,
                      _BS_ * _BS_ * dim * sizeof(Real));
          send_blocks[baserank].push_back(x);
          grid->Tree0(b.level, b.Z) = baserank;
        }
      } else {
        for (int j = 0; j < 2; j++)
          for (int i = 0; i < 2; i++) {
            const long long n =
                getZforward(b.level, b.index[0] + i, b.index[1] + j);
            if (n == nBlock)
              continue;
            BlockInfo &temp = grid->get(b.level, n);
            const int temprank = grid->Tree0(b.level, n);
            if (temprank != sim.rank) {
              MPI_Block x;
              x.mn[0] = bCopy.level;
              x.mn[1] = bCopy.Z;
              recv_blocks[temprank].push_back(x);
              grid->Tree0(b.level, n) = baserank;
            }
          }
      }
    }
    std::vector<MPI_Request> requests0;
    for (int r = 0; r < sim.size; r++)
      if (r != sim.rank) {
        if (recv_blocks[r].size() != 0) {
          MPI_Request req{};
          requests0.push_back(req);
          MPI_Irecv(&recv_blocks[r][0],
                    recv_blocks[r].size() * sizeof(recv_blocks[r][0]),
                    MPI_UINT8_T, r, 2468, MPI_COMM_WORLD, &requests0.back());
        }
        if (send_blocks[r].size() != 0) {
          MPI_Request req{};
          requests0.push_back(req);
          MPI_Isend(&send_blocks[r][0],
                    send_blocks[r].size() * sizeof(send_blocks[r][0]),
                    MPI_UINT8_T, r, 2468, MPI_COMM_WORLD, &requests0.back());
        }
      }
    for (int r = 0; r < sim.size; r++)
      for (int i = 0; i < (int)send_blocks[r].size(); i++) {
        grid->_dealloc(send_blocks[r][i].mn[0], send_blocks[r][i].mn[1]);
        grid->Tree0(send_blocks[r][i].mn[0], send_blocks[r][i].mn[1]) = -2;
      }
    if (requests0.size() != 0) {
      movedBlocks = true;
      MPI_Waitall(requests0.size(), &requests0[0], MPI_STATUSES_IGNORE);
    }
    for (int r = 0; r < sim.size; r++)
      for (int i = 0; i < (int)recv_blocks[r].size(); i++) {
        const int level = (int)recv_blocks[r][i].mn[0];
        const long long Z = recv_blocks[r][i].mn[1];
        grid->_alloc(level, Z);
        BlockInfo &info = grid->get(level, Z);
        std::memcpy(info.block, recv_blocks[r][i].data,
                    _BS_ * _BS_ * dim * sizeof(Real));
      }

    dealloc_IDs.clear();
    for (size_t i = 0; i < m_com.size(); i++) {
      const int level = m_com[i];
      const long long Z = n_com[i];
      assert(level > 0);
      BlockInfo &info = grid->get(level, Z);
      assert(info.state == Compress);
      void *Blocks[4];
      for (int J = 0; J < 2; J++)
        for (int I = 0; I < 2; I++) {
          const int blk = J * 2 + I;
          const long long n =
              getZforward(level, info.index[0] + I, info.index[1] + J);
          Blocks[blk] = (grid->get(level, n)).block;
        }
      const int offsetX[2] = {0, _BS_ / 2};
      const int offsetY[2] = {0, _BS_ / 2};
      if (basic_refinement == false)
        for (int J = 0; J < 2; J++)
          for (int I = 0; I < 2; I++) {
            Real *b = (Real *)Blocks[J * 2 + I];
            for (int j = 0; j < _BS_; j += 2)
              for (int i = 0; i < _BS_; i += 2) {
                int i00 = _BS_ * j + i;
                int i01 = _BS_ * (j + 1) + i;
                int i10 = _BS_ * j + i + 1;
                int i11 = _BS_ * (j + 1) + i + 1;
                int o = _BS_ * (j / 2 + offsetY[J]) + i / 2 + offsetX[I];
                for (int d = 0; d < dim; d++)
                  ((Real *)Blocks[0])[dim * o + d] =
                      (b[dim * i00 + d] + b[dim * i01 + d] + b[dim * i10 + d] +
                       b[dim * i11 + d]) /
                      4;
              }
          }
      const long long np =
          getZforward(level - 1, info.index[0] / 2, info.index[1] / 2);
      BlockInfo &parent = grid->get(level - 1, np);
      grid->Tree0(parent.level, parent.Z) = sim.rank;
      parent.block = info.block;
      parent.state = Leave;
      if (level - 2 >= 0)
        grid->Tree0(level - 2, parent.Zparent) = -1;
      for (int J = 0; J < 2; J++)
        for (int I = 0; I < 2; I++) {
          const long long n =
              getZforward(level, info.index[0] + I, info.index[1] + J);
          if (I + J == 0) {
            for (size_t j = 0; j < grid->infos.size(); j++)
              if (level == grid->infos[j].level && n == grid->infos[j].Z) {
                BlockInfo &correct_info = grid->get(level - 1, np);
                correct_info.state = Leave;
                grid->infos[j] = correct_info;
                break;
              }
          } else {
#pragma omp critical
            { dealloc_IDs.push_back(grid->get(level, n).id2); }
          }
          grid->Tree0(level, n) = -2;
          grid->get(level, n).state = Leave;
        }
    }
    grid->dealloc_many(dealloc_IDs);
    MPI_Waitall(2, requests, MPI_STATUS_IGNORE);
    Balance_Diffusion(grid, block_distribution);
    if (result[0] > 0 || result[1] > 0 || movedBlocks) {
      grid->UpdateFluxCorrection = true;
      grid->UpdateBlockInfoAll_States(false);
      auto it = grid->Synchronizers.begin();
      while (it != grid->Synchronizers.end()) {
        (*it->second)._Setup();
        it++;
      }
    }
  }
};
template <typename Lab, typename Kernel>
static void computeA(Kernel &&kernel, Grid *g, int dim) {
  Synchronizer<Grid> &Synch = *(g->sync1(kernel.stencil));
  std::vector<BlockInfo *> *inner = &Synch.inner_blocks;
  std::vector<BlockInfo *> *halo_next;
  bool done = false;
#pragma omp parallel
  {
    Lab lab(dim);
    lab.prepare(g, kernel.stencil);
#pragma omp for nowait
    for (const auto &I : *inner) {
      lab.load(g, *I, true);
      kernel(lab, *I);
    }
    while (done == false) {
#pragma omp master
      halo_next = &Synch.avail_next();
#pragma omp barrier
#pragma omp for nowait
      for (const auto &I : *halo_next) {
        lab.load(g, *I, true);
        kernel(lab, *I);
      }
#pragma omp single
      {
        if (halo_next->size() == 0)
          done = true;
      }
    }
  }
  MPI_Waitall(Synch.requests.size(), Synch.requests.data(),
              MPI_STATUSES_IGNORE);
}
template <typename Kernel, typename Element1, typename LabMPI,
          typename Element2, typename LabMPI2>
static void computeB(const Kernel &kernel, Grid &grid, int dim1, Grid &grid2,
                     int dim2) {
  Synchronizer<Grid> &Synch = *grid.sync1(kernel.stencil);
  Kernel kernel2 = kernel;
  kernel2.stencil.sx = kernel2.stencil2.sx;
  kernel2.stencil.sy = kernel2.stencil2.sy;
  kernel2.stencil.ex = kernel2.stencil2.ex;
  kernel2.stencil.ey = kernel2.stencil2.ey;
  kernel2.stencil.tensorial = kernel2.stencil2.tensorial;
  Synchronizer<Grid> &Synch2 = *grid2.sync1(kernel2.stencil);
  const StencilInfo &stencil = Synch.stencil;
  const StencilInfo &stencil2 = Synch2.stencil;
  std::vector<BlockInfo> &blk = grid.infos;
  std::vector<bool> ready(blk.size(), false);
  std::vector<BlockInfo *> &avail0 = Synch.inner_blocks;
  std::vector<BlockInfo *> &avail02 = Synch2.inner_blocks;
  const int Ninner = avail0.size();
  std::vector<BlockInfo *> avail1;
  std::vector<BlockInfo *> avail12;
#pragma omp parallel
  {
    LabMPI lab(dim1);
    LabMPI2 lab2(dim2);
    lab.prepare(&grid, stencil);
    lab2.prepare(&grid2, stencil2);
#pragma omp for
    for (int i = 0; i < Ninner; i++) {
      BlockInfo &I = *avail0[i];
      BlockInfo &I2 = *avail02[i];
      lab.load(&grid, I, true);
      lab2.load(&grid2, I2, true);
      kernel(lab, lab2, I, I2);
      ready[I.id] = true;
    }
#pragma omp master
    {
      MPI_Waitall(Synch.requests.size(), Synch.requests.data(),
                  MPI_STATUSES_IGNORE);
      avail1 = Synch.halo_blocks;

      MPI_Waitall(Synch2.requests.size(), Synch2.requests.data(),
                  MPI_STATUSES_IGNORE);
      avail12 = Synch2.halo_blocks;
    }
#pragma omp barrier
    const int Nhalo = avail1.size();
#pragma omp for
    for (int i = 0; i < Nhalo; i++) {
      BlockInfo &I = *avail1[i];
      BlockInfo &I2 = *avail12[i];
      lab.load(&grid, I, true);
      lab2.load(&grid2, I2, true);
      kernel(lab, lab2, I, I2);
    }
  }
}
struct Vector {
  Real u[2];
  Vector() { u[0] = u[1] = 0; }
  Vector &operator*=(const Real a) {
    u[0] *= a;
    u[1] *= a;
    return *this;
  }
  Vector &operator+=(const Vector &rhs) {
    u[0] += rhs.u[0];
    u[1] += rhs.u[1];
    return *this;
  }
  Vector &operator-=(const Vector &rhs) {
    u[0] -= rhs.u[0];
    u[1] -= rhs.u[1];
    return *this;
  }
  friend Vector operator*(const Real a, Vector el) { return (el *= a); }
  friend Vector operator+(Vector lhs, const Vector &rhs) {
    return (lhs += rhs);
  }
  friend Vector operator-(Vector lhs, const Vector &rhs) {
    return (lhs -= rhs);
  }
};
typedef Real ScalarBlock[_BS_][_BS_];
typedef Vector VectorBlock[_BS_][_BS_];
struct VectorLab : public BlockLab<Vector> {
  VectorLab(int dim) : BlockLab<Vector>(dim) {}
  VectorLab(const VectorLab &) = delete;
  VectorLab &operator=(const VectorLab &) = delete;
  template <int dir, int side>
  void applyBCface(bool wall, bool coarse = false) {
    const int A = 1 - dir;
    if (!coarse) {
      int s[3] = {0, 0, 0}, e[3] = {0, 0, 0};
      const int *const stenBeg = this->start;
      const int *const stenEnd = this->end;
      s[0] = dir == 0 ? (side == 0 ? stenBeg[0] : _BS_) : stenBeg[0];
      s[1] = dir == 1 ? (side == 0 ? stenBeg[1] : _BS_) : stenBeg[1];
      e[0] = dir == 0 ? (side == 0 ? 0 : _BS_ + stenEnd[0] - 1)
                      : _BS_ + stenEnd[0] - 1;
      e[1] = dir == 1 ? (side == 0 ? 0 : _BS_ + stenEnd[1] - 1)
                      : _BS_ + stenEnd[1] - 1;
      if (!wall)
        for (int iy = s[1]; iy < e[1]; iy++)
          for (int ix = s[0]; ix < e[0]; ix++) {
            const int x =
                (dir == 0 ? (side == 0 ? 0 : _BS_ - 1) : ix) - stenBeg[0];
            const int y =
                (dir == 1 ? (side == 0 ? 0 : _BS_ - 1) : iy) - stenBeg[1];
            m[ix - stenBeg[0] + nm[0] * (iy - stenBeg[1])].u[1 - A] =
                (-1.0) * m[x + nm[0] * (y)].u[1 - A];
            m[ix - stenBeg[0] + nm[0] * (iy - stenBeg[1])].u[A] =
                m[x + nm[0] * (y)].u[A];
          }
      else
        for (int iy = s[1]; iy < e[1]; iy++)
          for (int ix = s[0]; ix < e[0]; ix++) {
            const int x =
                (dir == 0 ? (side == 0 ? 0 : _BS_ - 1) : ix) - stenBeg[0];
            const int y =
                (dir == 1 ? (side == 0 ? 0 : _BS_ - 1) : iy) - stenBeg[1];
            m[ix - stenBeg[0] + nm[0] * (iy - stenBeg[1])] =
                (-1.0) * m[x + nm[0] * (y)];
          }
    } else {
      const int eI[3] = {(this->end[0]) / 2 + 1 + (2) - 1,
                         (this->end[1]) / 2 + 1 + (2) - 1,
                         (this->end[2]) / 2 + 1 + (1) - 1};
      const int sI[3] = {(this->start[0] - 1) / 2 + (-1),
                         (this->start[1] - 1) / 2 + (-1),
                         (this->start[2] - 1) / 2 + (0)};
      const int *const stenBeg = sI;
      const int *const stenEnd = eI;
      int s[3] = {0, 0, 0}, e[3] = {0, 0, 0};
      s[0] = dir == 0 ? (side == 0 ? stenBeg[0] : _BS_ / 2) : stenBeg[0];
      s[1] = dir == 1 ? (side == 0 ? stenBeg[1] : _BS_ / 2) : stenBeg[1];
      e[0] = dir == 0 ? (side == 0 ? 0 : _BS_ / 2 + stenEnd[0] - 1)
                      : _BS_ / 2 + stenEnd[0] - 1;
      e[1] = dir == 1 ? (side == 0 ? 0 : _BS_ / 2 + stenEnd[1] - 1)
                      : _BS_ / 2 + stenEnd[1] - 1;
      if (!wall)
        for (int iy = s[1]; iy < e[1]; iy++)
          for (int ix = s[0]; ix < e[0]; ix++) {
            const int x =
                (dir == 0 ? (side == 0 ? 0 : _BS_ / 2 - 1) : ix) - stenBeg[0];
            const int y =
                (dir == 1 ? (side == 0 ? 0 : _BS_ / 2 - 1) : iy) - stenBeg[1];
            c[ix - stenBeg[0] + nc[0] * (iy - stenBeg[1])].u[1 - A] =
                (-1.0) * c[x + nc[0] * (y)].u[1 - A];
            c[ix - stenBeg[0] + nc[0] * (iy - stenBeg[1])].u[A] =
                c[x + nc[0] * (y)].u[A];
          }
      else
        for (int iy = s[1]; iy < e[1]; iy++)
          for (int ix = s[0]; ix < e[0]; ix++) {
            const int x =
                (dir == 0 ? (side == 0 ? 0 : _BS_ / 2 - 1) : ix) - stenBeg[0];
            const int y =
                (dir == 1 ? (side == 0 ? 0 : _BS_ / 2 - 1) : iy) - stenBeg[1];
            c[ix - stenBeg[0] + nc[0] * (iy - stenBeg[1])] =
                (-1.0) * c[x + nc[0] * (y)];
          }
    }
  }
  void _apply_bc(BlockInfo &info, bool coarse) override {
    if (!coarse) {
      if (sim.bcx != periodic) {
        if (info.index[0] == 0)
          this->template applyBCface<0, 0>(sim.bcx == wall);
        if (info.index[0] == this->NX - 1)
          this->template applyBCface<0, 1>(sim.bcx == wall);
      }
      if (sim.bcy != periodic) {
        if (info.index[1] == 0)
          this->template applyBCface<1, 0>(sim.bcy == wall);
        if (info.index[1] == this->NY - 1)
          this->template applyBCface<1, 1>(sim.bcy == wall);
      }
    } else {
      if (sim.bcx != periodic) {
        if (info.index[0] == 0)
          this->template applyBCface<0, 0>(sim.bcx == wall, coarse);
        if (info.index[0] == this->NX - 1)
          this->template applyBCface<0, 1>(sim.bcx == wall, coarse);
      }
      if (sim.bcy != periodic) {
        if (info.index[1] == 0)
          this->template applyBCface<1, 0>(sim.bcy == wall, coarse);
        if (info.index[1] == this->NY - 1)
          this->template applyBCface<1, 1>(sim.bcy == wall, coarse);
      }
    }
  }
};
struct ScalarLab : public BlockLab<Real> {
  ScalarLab(int dim) : BlockLab(dim){};
  ScalarLab(const ScalarLab &) = delete;
  ScalarLab &operator=(const ScalarLab &) = delete;
  template <int dir, int side> void Neumann2D(bool coarse) {
    int stenBeg[2];
    int stenEnd[2];
    int bsize[2];
    if (!coarse) {
      stenEnd[0] = this->end[0];
      stenEnd[1] = this->end[1];
      stenBeg[0] = this->start[0];
      stenBeg[1] = this->start[1];
      bsize[0] = _BS_;
      bsize[1] = _BS_;
    } else {
      stenEnd[0] = (this->end[0]) / 2 + 1 + (2) - 1;
      stenEnd[1] = (this->end[1]) / 2 + 1 + (2) - 1;
      stenBeg[0] = (this->start[0] - 1) / 2 + (-1);
      stenBeg[1] = (this->start[1] - 1) / 2 + (-1);
      bsize[0] = _BS_ / 2;
      bsize[1] = _BS_ / 2;
    }
    auto *const cb = coarse ? this->c : this->m;
    const unsigned int *n = coarse ? this->nc : this->nm;
    int s[2];
    int e[2];
    s[0] = dir == 0 ? (side == 0 ? stenBeg[0] : bsize[0]) : stenBeg[0];
    s[1] = dir == 1 ? (side == 0 ? stenBeg[1] : bsize[1]) : stenBeg[1];
    e[0] = dir == 0 ? (side == 0 ? 0 : bsize[0] + stenEnd[0] - 1)
                    : bsize[0] + stenEnd[0] - 1;
    e[1] = dir == 1 ? (side == 0 ? 0 : bsize[1] + stenEnd[1] - 1)
                    : bsize[1] + stenEnd[1] - 1;
    for (int iy = s[1]; iy < e[1]; iy++)
      for (int ix = s[0]; ix < e[0]; ix++)
        cb[ix - stenBeg[0] + n[0] * (iy - stenBeg[1])] =
            cb[(dir == 0 ? (side == 0 ? 0 : bsize[0] - 1) : ix) - stenBeg[0] +
               n[0] * ((dir == 1 ? (side == 0 ? 0 : bsize[1] - 1) : iy) -
                       stenBeg[1])];
  }
  virtual void _apply_bc(BlockInfo &info, bool coarse) override {
    if (sim.bcx != periodic) {
      if (info.index[0] == 0)
        Neumann2D<0, 0>(coarse);
      if (info.index[0] == this->NX - 1)
        Neumann2D<0, 1>(coarse);
    }
    if (sim.bcy != periodic) {
      if (info.index[1] == 0)
        Neumann2D<1, 0>(coarse);
      if (info.index[1] == this->NY - 1)
        Neumann2D<1, 1>(coarse);
    }
  }
};
struct Skin {
  size_t n;
  std::vector<Real> xSurf, ySurf, normXSurf, normYSurf, midX, midY;
  Skin(size_t n)
      : n(n), xSurf(n), ySurf(n), normXSurf(n), normYSurf(n), midX(n), midY(n) {
  }
};
static struct {
  Grid *chi, *vel, *vold, *pres, *tmpV, *tmp, *pold;
  Adaptation *tmp_amr, *chi_amr, *pres_amr, *pold_amr, *vel_amr, *vold_amr,
      *tmpV_amr;
  struct {
    Grid **g;
    Adaptation **a;
    int dim;
  } F[7] = {
      {&chi, &chi_amr, 1},   {&vel, &vel_amr, 2},   {&vold, &vold_amr, 2},
      {&pres, &pres_amr, 1}, {&tmpV, &tmpV_amr, 2}, {&tmp, &tmp_amr, 1},
      {&pold, &pold_amr, 1},
  };
} var;
typedef Real CHI_MAT[_BS_][_BS_];
typedef Real UDEFMAT[_BS_][_BS_][2];
struct surface_data {
  int ix, iy;
  Real dchidx, dchidy, delta;
};
struct ObstacleBlock {
  Real chi[_BS_][_BS_];
  Real dist[_BS_][_BS_];
  Real udef[_BS_][_BS_][2];
  size_t n_surfPoints = 0;
  bool filled = false;
  std::vector<surface_data> surface;
  Real *x_s = nullptr;
  Real *y_s = nullptr;
  Real *p_s = nullptr;
  Real *u_s = nullptr;
  Real *v_s = nullptr;
  Real *nx_s = nullptr;
  Real *ny_s = nullptr;
  Real *omega_s = nullptr;
  Real *uDef_s = nullptr;
  Real *vDef_s = nullptr;
  Real *fX_s = nullptr;
  Real *fY_s = nullptr;
  Real *fXv_s = nullptr;
  Real *fYv_s = nullptr;
  Real perimeter = 0, forcex = 0, forcey = 0, forcex_P = 0, forcey_P = 0;
  Real forcex_V = 0, forcey_V = 0, torque = 0, torque_P = 0, torque_V = 0;
  Real drag = 0, thrust = 0, lift = 0, Pout = 0, PoutNew = 0, PoutBnd = 0,
       defPower = 0, defPowerBnd = 0;
  Real circulation = 0;
  Real COM_x = 0;
  Real COM_y = 0;
  Real Mass = 0;
  ObstacleBlock() {
    clear_surface();
    std::fill(dist[0], dist[0] + _BS_ * _BS_, -1);
    std::fill(chi[0], chi[0] + _BS_ * _BS_, 0);
    memset(udef, 0, sizeof(Real) * _BS_ * _BS_ * 2);
    surface.reserve(4 * _BS_);
  }
  void clear_surface() {
    filled = false;
    n_surfPoints = 0;
    perimeter = forcex = forcey = forcex_P = forcey_P = 0;
    forcex_V = forcey_V = torque = torque_P = torque_V = drag = thrust = lift =
        0;
    Pout = PoutBnd = defPower = defPowerBnd = circulation = 0;
    surface.clear();
    free(x_s);
    free(y_s);
    free(p_s);
    free(u_s);
    free(v_s);
    free(nx_s);
    free(ny_s);
    free(omega_s);
    free(uDef_s);
    free(vDef_s);
    free(fX_s);
    free(fY_s);
    free(fXv_s);
    free(fYv_s);
  }
};
struct KernelVorticity {
  const std::vector<BlockInfo> &tmpInfo = var.tmp->infos;
  const StencilInfo stencil{-1, -1, 2, 2, false};
  void operator()(VectorLab &lab, const BlockInfo &info) const {
    Real *um = (Real*)lab.m;
    const Real i2h = 0.5 / info.h;
    Real *TMP = (Real*)tmpInfo[info.id].block;
    int nm = _BS_ + stencil.ex - stencil.sx - 1;
    for (int j = 0; j < _BS_; ++j)
      for (int i = 0; i < _BS_; ++i) {
        int x0 = i - stencil.sx;
        int y0 = j - stencil.sy;
        int xp = x0 + 1;
        int yp = y0 + 1;
        int xm = x0 - 1;
        int ym = y0 - 1;
	Real e0 = lab(x0, ym).u[0];
	Real e1 = lab(x0, yp).u[0];
	Real e2 = lab(xp, y0).u[1];
	Real e3 = lab(xm, y0).u[1];
        TMP[j][i] = i2h * (e0 - e1 + e2 - e3);
      }
  }
};
static void dump(Real time, long nblock, BlockInfo *infos, char *path) {
  long i, j, k, l, x, y, ncell, ncell_total, offset;
  char xyz_path[FILENAME_MAX], attr_path[FILENAME_MAX], xdmf_path[FILENAME_MAX],
      *xyz_base, *attr_base;
  MPI_File mpi_file;
  FILE *xmf;
  float *xyz, *attr;
  Real sum;
  snprintf(xyz_path, sizeof xyz_path, "%s.xyz.raw", path);
  snprintf(attr_path, sizeof attr_path, "%s.attr.raw", path);
  snprintf(xdmf_path, sizeof xdmf_path, "%s.xdmf2", path);
  xyz_base = xyz_path;
  attr_base = attr_path;
  for (j = 0; xyz_path[j] != '\0'; j++) {
    if (xyz_path[j] == '/' && xyz_path[j + 1] != '\0') {
      xyz_base = &xyz_path[j + 1];
      attr_base = &attr_path[j + 1];
    }
  }
  ncell = nblock * _BS_ * _BS_;
  MPI_Exscan(&ncell, &offset, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  if (sim.rank == 0)
    offset = 0;
  if (sim.rank == sim.size - 1) {
    ncell_total = ncell + offset;
    xmf = fopen(xdmf_path, "w");
    fprintf(xmf,
            "<Xdmf\n"
            "    Version=\"2.0\">\n"
            "  <Domain>\n"
            "    <Grid>\n"
            "      <Time Value=\"%.16e\"/>\n"
            "      <Topology\n"
            "	  Dimensions=\"%ld\"\n"
            "	  TopologyType=\"Quadrilateral\"/>\n"
            "      <Geometry\n"
            "	  GeometryType=\"XY\">\n"
            "	<DataItem\n"
            "	    Dimensions=\"%ld 2\"\n"
            "	    Format=\"Binary\">\n"
            "	  %s\n"
            "	</DataItem>\n"
            "      </Geometry>\n"
            "      <Attribute\n"
            "	  Name=\"vort\"\n"
            "	  Center=\"Cell\">\n"
            "	<DataItem\n"
            "	    Dimensions=\"%ld\"\n"
            "	    Format=\"Binary\">\n"
            "	  %s\n"
            "	</DataItem>\n"
            "      </Attribute>\n"
            "    </Grid>\n"
            "  </Domain>\n"
            "</Xdmf>\n",
            time, ncell_total, 4 * ncell_total, xyz_base, ncell_total,
            attr_base);
    fclose(xmf);
  }
  xyz = (float *)malloc(8 * ncell * sizeof *xyz);
  attr = (float *)malloc(ncell * sizeof *xyz);
  k = 0;
  l = 0;
  sum = 0;
  for (i = 0; i < nblock; i++) {
    const BlockInfo &info = infos[i];
    ScalarBlock &b = *(ScalarBlock *)info.block;
    for (y = 0; y < _BS_; y++)
      for (x = 0; x < _BS_; x++) {
        double u0, v0, u1, v1, h;
        h = sim.h0 / (1 << info.level);
        u0 = info.origin[0] + h * x;
        v0 = info.origin[1] + h * y;
        u1 = u0 + h;
        v1 = v0 + h;
        xyz[k++] = u0;
        xyz[k++] = v0;
        xyz[k++] = u0;
        xyz[k++] = v1;
        xyz[k++] = u1;
        xyz[k++] = v1;
        xyz[k++] = u1;
        xyz[k++] = v0;
        attr[l++] = b[y][x];
        sum += b[y][x];
      }
  }
  printf("main.cpp: %d: %8.3e\n", sim.rank, sum / (_BS_ * _BS_ * nblock));
  MPI_File_open(MPI_COMM_WORLD, xyz_path, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &mpi_file);
  MPI_File_write_at_all(mpi_file, 8 * offset * sizeof *xyz, xyz,
                        8 * ncell * sizeof *xyz, MPI_BYTE, MPI_STATUS_IGNORE);
  MPI_File_close(&mpi_file);
  free(xyz);
  MPI_File_open(MPI_COMM_WORLD, attr_path, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &mpi_file);
  MPI_File_write_at_all(mpi_file, offset * sizeof *attr, attr,
                        ncell * sizeof *attr, MPI_BYTE, MPI_STATUS_IGNORE);
  MPI_File_close(&mpi_file);
  free(attr);
}
struct Integrals {
  const Real x, y, m, j, u, v, a;
  Integrals(Real _x, Real _y, Real _m, Real _j, Real _u, Real _v, Real _a)
      : x(_x), y(_y), m(_m), j(_j), u(_u), v(_v), a(_a) {}
  Integrals(const Integrals &c)
      : x(c.x), y(c.y), m(c.m), j(c.j), u(c.u), v(c.v), a(c.a) {}
};
struct IF2D_Interpolation1D {
  static void naturalCubicSpline(const Real *x, const Real *y, const unsigned n,
                                 const Real *xx, Real *yy, const unsigned nn) {
    return naturalCubicSpline(x, y, n, xx, yy, nn, 0);
  }
  static void naturalCubicSpline(const Real *x, const Real *y, const unsigned n,
                                 const Real *xx, Real *yy, const unsigned nn,
                                 const Real offset) {
    std::vector<Real> y2(n), u(n - 1);
    y2[0] = 0;
    u[0] = 0;
    for (unsigned i = 1; i < n - 1; i++) {
      const Real sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
      const Real p = sig * y2[i - 1] + 2;
      y2[i] = (sig - 1) / p;
      u[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) -
             (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
      u[i] = (6 * u[i] / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p;
    }
    const Real qn = 0;
    const Real un = 0;
    y2[n - 1] = (un - qn * u[n - 2]) / (qn * y2[n - 2] + 1);
    for (unsigned k = n - 2; k > 0; k--)
      y2[k] = y2[k] * y2[k + 1] + u[k];
    for (unsigned j = 0; j < nn; j++) {
      unsigned int klo = 0;
      unsigned int khi = n - 1;
      unsigned int k = 0;
      while (khi - klo > 1) {
        k = (khi + klo) >> 1;
        if (x[k] > (xx[j] + offset))
          khi = k;
        else
          klo = k;
      }
      const Real h = x[khi] - x[klo];
      if (h <= 0.0) {
        std::cout << "Interpolation points must be distinct!" << std::endl;
        abort();
      }
      const Real a = (x[khi] - (xx[j] + offset)) / h;
      const Real b = ((xx[j] + offset) - x[klo]) / h;
      yy[j] =
          a * y[klo] + b * y[khi] +
          ((a * a * a - a) * y2[klo] + (b * b * b - b) * y2[khi]) * (h * h) / 6;
    }
  }
  static void cubicInterpolation(const Real x0, const Real x1, const Real x,
                                 const Real y0, const Real y1, const Real dy0,
                                 const Real dy1, Real &y, Real &dy) {
    const Real xrel = (x - x0);
    const Real deltax = (x1 - x0);
    const Real a = (dy0 + dy1) / (deltax * deltax) -
                   2 * (y1 - y0) / (deltax * deltax * deltax);
    const Real b =
        (-2 * dy0 - dy1) / deltax + 3 * (y1 - y0) / (deltax * deltax);
    const Real c = dy0;
    const Real d = y0;
    y = a * xrel * xrel * xrel + b * xrel * xrel + c * xrel + d;
    dy = 3 * a * xrel * xrel + 2 * b * xrel + c;
  }
  static void cubicInterpolation(const Real x0, const Real x1, const Real x,
                                 const Real y0, const Real y1, Real &y,
                                 Real &dy) {
    return cubicInterpolation(x0, x1, x, y0, y1, 0, 0, y, dy);
  }
  static void linearInterpolation(const Real x0, const Real x1, const Real x,
                                  const Real y0, const Real y1, Real &y,
                                  Real &dy) {
    y = (y1 - y0) / (x1 - x0) * (x - x0) + y0;
    dy = (y1 - y0) / (x1 - x0);
  }
};
template <int Npoints> struct Scheduler {
  static constexpr int npoints = Npoints;
  std::array<Real, Npoints> parameters_t0;
  std::array<Real, Npoints> parameters_t1;
  std::array<Real, Npoints> dparameters_t0;
  Real t0, t1;
  Scheduler() {
    t0 = -1;
    t1 = 0;
    parameters_t0 = std::array<Real, Npoints>();
    parameters_t1 = std::array<Real, Npoints>();
    dparameters_t0 = std::array<Real, Npoints>();
  }
  void transition(const Real t, const Real tstart, const Real tend,
                  const std::array<Real, Npoints> parameters_tstart,
                  const std::array<Real, Npoints> parameters_tend) {
    if (t < tstart or t > tend)
      return;
    if (tstart < t0)
      return;
    t0 = tstart;
    t1 = tend;
    parameters_t0 = parameters_tstart;
    parameters_t1 = parameters_tend;
  }
  void gimmeValues(const Real t, std::array<Real, Npoints> &parameters,
                   std::array<Real, Npoints> &dparameters) {
    if (t < t0 or t0 < 0) {
      parameters = parameters_t0;
      dparameters = std::array<Real, Npoints>();
    } else if (t > t1) {
      parameters = parameters_t1;
      dparameters = std::array<Real, Npoints>();
    } else {
      for (int i = 0; i < Npoints; ++i)
        IF2D_Interpolation1D::cubicInterpolation(
            t0, t1, t, parameters_t0[i], parameters_t1[i], dparameters_t0[i],
            0.0, parameters[i], dparameters[i]);
    }
  }
  void gimmeValuesLinear(const Real t, std::array<Real, Npoints> &parameters,
                         std::array<Real, Npoints> &dparameters) {
    if (t < t0 or t0 < 0) {
      parameters = parameters_t0;
      dparameters = std::array<Real, Npoints>();
    } else if (t > t1) {
      parameters = parameters_t1;
      dparameters = std::array<Real, Npoints>();
    } else {
      for (int i = 0; i < Npoints; ++i)
        IF2D_Interpolation1D::linearInterpolation(
            t0, t1, t, parameters_t0[i], parameters_t1[i], parameters[i],
            dparameters[i]);
    }
  }
  void gimmeValues(const Real t, std::array<Real, Npoints> &parameters) {
    std::array<Real, Npoints> dparameters_whocares;
    return gimmeValues(t, parameters, dparameters_whocares);
  }
};
struct SchedulerScalar : Scheduler<1> {
  void transition(const Real t, const Real tstart, const Real tend,
                  const Real parameter_tstart, const Real parameter_tend) {
    const std::array<Real, 1> myParameterStart = {parameter_tstart};
    const std::array<Real, 1> myParameterEnd = {parameter_tend};
    return Scheduler<1>::transition(t, tstart, tend, myParameterStart,
                                    myParameterEnd);
  }
  void gimmeValues(const Real t, Real &parameter, Real &dparameter) {
    std::array<Real, 1> myParameter, mydParameter;
    Scheduler<1>::gimmeValues(t, myParameter, mydParameter);
    parameter = myParameter[0];
    dparameter = mydParameter[0];
  }
};
template <int Npoints> struct SchedulerVector : Scheduler<Npoints> {
  void gimmeValues(const Real t, const std::array<Real, Npoints> &positions,
                   const int Nfine, const Real *const positions_fine,
                   Real *const parameters_fine, Real *const dparameters_fine) {
    Real *parameters_t0_fine = new Real[Nfine];
    Real *parameters_t1_fine = new Real[Nfine];
    Real *dparameters_t0_fine = new Real[Nfine];
    IF2D_Interpolation1D::naturalCubicSpline(
        positions.data(), this->parameters_t0.data(), Npoints, positions_fine,
        parameters_t0_fine, Nfine);
    IF2D_Interpolation1D::naturalCubicSpline(
        positions.data(), this->parameters_t1.data(), Npoints, positions_fine,
        parameters_t1_fine, Nfine);
    IF2D_Interpolation1D::naturalCubicSpline(
        positions.data(), this->dparameters_t0.data(), Npoints, positions_fine,
        dparameters_t0_fine, Nfine);
    if (t < this->t0 or this->t0 < 0) {
      memcpy(parameters_fine, parameters_t0_fine, Nfine * sizeof(Real));
      memset(dparameters_fine, 0, Nfine * sizeof(Real));
    } else if (t > this->t1) {
      memcpy(parameters_fine, parameters_t1_fine, Nfine * sizeof(Real));
      memset(dparameters_fine, 0, Nfine * sizeof(Real));
    } else {
      for (int i = 0; i < Nfine; ++i)
        IF2D_Interpolation1D::cubicInterpolation(
            this->t0, this->t1, t, parameters_t0_fine[i], parameters_t1_fine[i],
            dparameters_t0_fine[i], 0, parameters_fine[i], dparameters_fine[i]);
    }
    delete[] parameters_t0_fine;
    delete[] parameters_t1_fine;
    delete[] dparameters_t0_fine;
  }
  void gimmeValues(const Real t, std::array<Real, Npoints> &parameters) {
    Scheduler<Npoints>::gimmeValues(t, parameters);
  }
  void gimmeValues(const Real t, std::array<Real, Npoints> &parameters,
                   std::array<Real, Npoints> &dparameters) {
    Scheduler<Npoints>::gimmeValues(t, parameters, dparameters);
  }
};
template <int Npoints> struct SchedulerLearnWave : Scheduler<Npoints> {
  template <typename T>
  void gimmeValues(const Real t, const Real Twave, const Real Length,
                   const std::array<Real, Npoints> &positions, const int Nfine,
                   const T *const positions_fine, T *const parameters_fine,
                   Real *const dparameters_fine) {
    const Real _1oL = 1. / Length;
    const Real _1oT = 1. / Twave;
    for (int i = 0; i < Nfine; ++i) {
      const Real c = positions_fine[i] * _1oL - (t - this->t0) * _1oT;
      bool bCheck = true;
      if (c < positions[0]) {
        IF2D_Interpolation1D::cubicInterpolation(
            c, positions[0], c, this->parameters_t0[0], this->parameters_t0[0],
            parameters_fine[i], dparameters_fine[i]);
        bCheck = false;
      } else if (c > positions[Npoints - 1]) {
        IF2D_Interpolation1D::cubicInterpolation(
            positions[Npoints - 1], c, c, this->parameters_t0[Npoints - 1],
            this->parameters_t0[Npoints - 1], parameters_fine[i],
            dparameters_fine[i]);
        bCheck = false;
      } else {
        for (int j = 1; j < Npoints; ++j) {
          if ((c >= positions[j - 1]) && (c <= positions[j])) {
            IF2D_Interpolation1D::cubicInterpolation(
                positions[j - 1], positions[j], c, this->parameters_t0[j - 1],
                this->parameters_t0[j], parameters_fine[i],
                dparameters_fine[i]);
            dparameters_fine[i] = -dparameters_fine[i] * _1oT;
            bCheck = false;
          }
        }
      }
      if (bCheck) {
        std::cout << "Ciaone2!" << std::endl;
        abort();
      }
    }
  }
  void Turn(const Real b, const Real t_turn) {
    this->t0 = t_turn;
    for (int i = Npoints - 1; i > 1; --i)
      this->parameters_t0[i] = this->parameters_t0[i - 2];
    this->parameters_t0[1] = b;
    this->parameters_t0[0] = 0;
  }
};
struct Shape {
  Shape(CommandlineParser &p, Real C[2])
      : center{C[0], C[1]}, centerOfMass{C[0], C[1]},
        orientation(p("-angle").asDouble(0) * M_PI / 180),
        forcedu(-p("-xvel").asDouble(0)), forcedv(-p("-yvel").asDouble(0)),
        forcedomega(-p("-angvel").asDouble(0)), length(p("-L").asDouble(0.1)),
        Tperiod(p("-T").asDouble(1)), phaseShift(p("-phi").asDouble(0)),
        rK(new Real[Nm]), vK(new Real[Nm]), rC(new Real[Nm]), vC(new Real[Nm]),
        rB(new Real[Nm]), vB(new Real[Nm]), rS(new Real[Nm]), rX(new Real[Nm]),
        rY(new Real[Nm]), vX(new Real[Nm]), vY(new Real[Nm]),
        norX(new Real[Nm]), norY(new Real[Nm]), vNorX(new Real[Nm]),
        vNorY(new Real[Nm]), width(new Real[Nm]) {}
  std::vector<ObstacleBlock *> obstacleBlocks;
  Real center[2];
  Real centerOfMass[2];
  Real orientation;
  Real d_gm[2] = {0, 0};
  const Real forcedu;
  const Real forcedv;
  const Real forcedomega;
  Real M = 0;
  Real J = 0;
  Real u = forcedu;
  Real v = forcedv;
  Real omega = forcedomega;
  Real appliedForceX = 0;
  Real appliedForceY = 0;
  Real appliedTorque = 0;
  Real perimeter = 0, forcex = 0, forcey = 0, forcex_P = 0, forcey_P = 0;
  Real forcex_V = 0, forcey_V = 0, torque = 0, torque_P = 0, torque_V = 0;
  Real drag = 0, thrust = 0, lift = 0, circulation = 0, Pout = 0, PoutNew = 0,
       PoutBnd = 0, defPower = 0;
  Real defPowerBnd = 0, Pthrust = 0, Pdrag = 0, EffPDef = 0, EffPDefBnd = 0;
  const Real Tperiod, phaseShift;
  Real area_internal = 0, J_internal = 0;
  Real CoM_internal[2] = {0, 0}, vCoM_internal[2] = {0, 0};
  Real theta_internal = 0, angvel_internal = 0;
  Real length, h;
  Real fracRefined = 0.1, fracMid = 1 - 2 * fracRefined;
  Real dSmid_tgt = sim.minH / std::sqrt(2);
  Real dSrefine_tgt = 0.125 * sim.minH;
  int Nmid = (int)std::ceil(length * fracMid / dSmid_tgt / 8) * 8;
  Real dSmid = length * fracMid / Nmid;
  int Nend =
      (int)std::ceil(fracRefined * length * 2 / (dSmid + dSrefine_tgt) / 4) * 4;
  Real dSref = fracRefined * length * 2 / Nend - dSmid;
  int Nm = Nmid + 2 * Nend + 1;
  Real *rS;
  Real *rX;
  Real *rY;
  Real *vX;
  Real *vY;
  Real *norX;
  Real *norY;
  Real *vNorX;
  Real *vNorY;
  Real *width;
  Real linMom[2], area, angMom;
  Skin upperSkin = Skin(Nm);
  Skin lowerSkin = Skin(Nm);
  Real amplitudeFactor;
  Real curv_PID_fac = 0;
  Real curv_PID_dif = 0;
  Real avgDeltaY = 0;
  Real avgDangle = 0;
  Real avgAngVel = 0;
  Real lastTact = 0;
  Real lastCurv = 0;
  Real oldrCurv = 0;
  Real periodPIDval = Tperiod;
  Real periodPIDdif = 0;
  bool TperiodPID = false;
  Real time0 = 0;
  Real timeshift = 0;
  Real lastTime = 0;
  Real lastAvel = 0;
  SchedulerVector<6> curvatureScheduler;
  SchedulerLearnWave<7> rlBendingScheduler;
  SchedulerScalar periodScheduler;
  Real current_period = Tperiod;
  Real next_period = Tperiod;
  Real transition_start = 0.0;
  Real transition_duration = 0.1 * Tperiod;
  Real *rK;
  Real *vK;
  Real *rC;
  Real *vC;
  Real *rB;
  Real *vB;
};
struct ComputeSurfaceNormals {
  ComputeSurfaceNormals(){};
  StencilInfo stencil{-1, -1, 2, 2, false};
  StencilInfo stencil2{-1, -1, 2, 2, false};
  void operator()(ScalarLab &labChi, ScalarLab &labSDF,
                  const BlockInfo &infoChi, const BlockInfo &infoSDF) const {
    for (const auto &shape : sim.shapes) {
      const std::vector<ObstacleBlock *> &OBLOCK = shape->obstacleBlocks;
      if (OBLOCK[infoChi.id] == nullptr)
        continue;
      const Real h = infoChi.h;
      ObstacleBlock &o = *OBLOCK[infoChi.id];
      const Real i2h = 0.5 / h;
      const Real fac = 0.5 * h;
      for (int iy = 0; iy < _BS_; iy++)
        for (int ix = 0; ix < _BS_; ix++) {
          const Real gradHX = labChi(ix + 1, iy) - labChi(ix - 1, iy);
          const Real gradHY = labChi(ix, iy + 1) - labChi(ix, iy - 1);
          if (gradHX * gradHX + gradHY * gradHY < 1e-12)
            continue;
          const Real gradUX = i2h * (labSDF(ix + 1, iy) - labSDF(ix - 1, iy));
          const Real gradUY = i2h * (labSDF(ix, iy + 1) - labSDF(ix, iy - 1));
          const Real gradUSq = (gradUX * gradUX + gradUY * gradUY) + EPS;
          const Real D = fac * (gradHX * gradUX + gradHY * gradUY) / gradUSq;
          if (std::fabs(D) > EPS) {
            o.n_surfPoints++;
            const Real dchidx = -D * gradUX, dchidy = -D * gradUY;
            struct surface_data s {
              ix, iy, dchidx, dchidy, D
            };
            o.surface.push_back(s);
          }
        }
      o.filled = true;
      o.x_s = (Real *)calloc(o.n_surfPoints, sizeof(Real));
      o.y_s = (Real *)calloc(o.n_surfPoints, sizeof(Real));
      o.p_s = (Real *)calloc(o.n_surfPoints, sizeof(Real));
      o.u_s = (Real *)calloc(o.n_surfPoints, sizeof(Real));
      o.v_s = (Real *)calloc(o.n_surfPoints, sizeof(Real));
      o.nx_s = (Real *)calloc(o.n_surfPoints, sizeof(Real));
      o.ny_s = (Real *)calloc(o.n_surfPoints, sizeof(Real));
      o.omega_s = (Real *)calloc(o.n_surfPoints, sizeof(Real));
      o.uDef_s = (Real *)calloc(o.n_surfPoints, sizeof(Real));
      o.vDef_s = (Real *)calloc(o.n_surfPoints, sizeof(Real));
      o.fX_s = (Real *)calloc(o.n_surfPoints, sizeof(Real));
      o.fY_s = (Real *)calloc(o.n_surfPoints, sizeof(Real));
      o.fXv_s = (Real *)calloc(o.n_surfPoints, sizeof(Real));
      o.fYv_s = (Real *)calloc(o.n_surfPoints, sizeof(Real));
    }
  }
};
struct AreaSegment {
  const Real safe_distance;
  const std::pair<int, int> s_range;
  Real w[2], c[2];
  Real normalI[2] = {(Real)1, (Real)0};
  Real normalJ[2] = {(Real)0, (Real)1};
  Real objBoxLabFr[2][2] = {{0, 0}, {0, 0}};
  Real objBoxObjFr[2][2] = {{0, 0}, {0, 0}};
  AreaSegment(std::pair<int, int> sr, const Real bb[2][2], const Real safe)
      : safe_distance(safe), s_range(sr), w{(bb[0][1] - bb[0][0]) / 2 + safe,
                                            (bb[1][1] - bb[1][0]) / 2 + safe},
        c{(bb[0][1] + bb[0][0]) / 2, (bb[1][1] + bb[1][0]) / 2} {
    assert(w[0] > 0);
    assert(w[1] > 0);
  }
  void changeToComputationalFrame(const Real pos[2], const Real angle) {
    Real Rmatrix2D[2][2] = {{std::cos(angle), -std::sin(angle)},
                            {std::sin(angle), std::cos(angle)}};
    Real p[2] = {c[0], c[1]};
    Real nx[2] = {normalI[0], normalI[1]};
    Real ny[2] = {normalJ[0], normalJ[1]};
    for (int i = 0; i < 2; ++i) {
      c[i] = Rmatrix2D[i][0] * p[0] + Rmatrix2D[i][1] * p[1];
      normalI[i] = Rmatrix2D[i][0] * nx[0] + Rmatrix2D[i][1] * nx[1];
      normalJ[i] = Rmatrix2D[i][0] * ny[0] + Rmatrix2D[i][1] * ny[1];
    }
    c[0] += pos[0];
    c[1] += pos[1];
    Real magI = std::sqrt(normalI[0] * normalI[0] + normalI[1] * normalI[1]);
    Real magJ = std::sqrt(normalJ[0] * normalJ[0] + normalJ[1] * normalJ[1]);
    assert(magI > std::numeric_limits<Real>::epsilon());
    assert(magJ > std::numeric_limits<Real>::epsilon());
    Real invMagI = 1 / magI, invMagJ = 1 / magJ;
    for (int i = 0; i < 2; ++i) {
      normalI[i] = std::fabs(normalI[i]) * invMagI;
      normalJ[i] = std::fabs(normalJ[i]) * invMagJ;
    }
    assert(normalI[0] >= 0 && normalI[1] >= 0);
    assert(normalJ[0] >= 0 && normalJ[1] >= 0);
    Real widthXvec[] = {w[0] * normalI[0], w[0] * normalI[1]};
    Real widthYvec[] = {w[1] * normalJ[0], w[1] * normalJ[1]};
    for (int i = 0; i < 2; ++i) {
      objBoxLabFr[i][0] = c[i] - widthXvec[i] - widthYvec[i];
      objBoxLabFr[i][1] = c[i] + widthXvec[i] + widthYvec[i];
      objBoxObjFr[i][0] = c[i] - w[i];
      objBoxObjFr[i][1] = c[i] + w[i];
    }
  }
  bool isIntersectingWithAABB(const Real start[2], const Real end[2]) const {
    Real AABB_w[2] = {(end[0] - start[0]) / 2 + safe_distance,
                      (end[1] - start[1]) / 2 + safe_distance};
    Real AABB_c[2] = {(end[0] + start[0]) / 2, (end[1] + start[1]) / 2};
    Real AABB_box[2][2] = {{AABB_c[0] - AABB_w[0], AABB_c[0] + AABB_w[0]},
                           {AABB_c[1] - AABB_w[1], AABB_c[1] + AABB_w[1]}};
    assert(AABB_w[0] > 0 && AABB_w[1] > 0);
    Real intersectionLabFrame[2][2] = {
        {std::max(objBoxLabFr[0][0], AABB_box[0][0]),
         std::min(objBoxLabFr[0][1], AABB_box[0][1])},
        {std::max(objBoxLabFr[1][0], AABB_box[1][0]),
         std::min(objBoxLabFr[1][1], AABB_box[1][1])}};
    if (intersectionLabFrame[0][1] - intersectionLabFrame[0][0] < 0 ||
        intersectionLabFrame[1][1] - intersectionLabFrame[1][0] < 0)
      return false;
    Real widthXbox[2] = {AABB_w[0] * normalI[0], AABB_w[0] * normalJ[0]};
    Real widthYbox[2] = {AABB_w[1] * normalI[1], AABB_w[1] * normalJ[1]};
    Real boxBox[2][2] = {{AABB_c[0] - widthXbox[0] - widthYbox[0],
                          AABB_c[0] + widthXbox[0] + widthYbox[0]},
                         {AABB_c[1] - widthXbox[1] - widthYbox[1],
                          AABB_c[1] + widthXbox[1] + widthYbox[1]}};
    Real intersectionFishFrame[2][2] = {
        {std::max(boxBox[0][0], objBoxObjFr[0][0]),
         std::min(boxBox[0][1], objBoxObjFr[0][1])},
        {std::max(boxBox[1][0], objBoxObjFr[1][0]),
         std::min(boxBox[1][1], objBoxObjFr[1][1])}};
    if (intersectionFishFrame[0][1] - intersectionFishFrame[0][0] < 0 ||
        intersectionFishFrame[1][1] - intersectionFishFrame[1][0] < 0)
      return false;
    return true;
  }
};
struct PutChiOnGrid {
  PutChiOnGrid(){};
  StencilInfo stencil{-1, -1, 2, 2, false};
  std::vector<BlockInfo> &chiInfo = var.chi->infos;
  void operator()(ScalarLab &lab, const BlockInfo &info) const {
    for (auto &shape : sim.shapes) {
      std::vector<ObstacleBlock *> &OBLOCK = shape->obstacleBlocks;
      if (OBLOCK[info.id] == nullptr)
        continue;
      Real h = info.h;
      Real h2 = h * h;
      ObstacleBlock &o = *OBLOCK[info.id];
      CHI_MAT &X = o.chi;
      const CHI_MAT &sdf = o.dist;
      o.COM_x = 0;
      o.COM_y = 0;
      o.Mass = 0;
      auto &CHI = *(ScalarBlock *)chiInfo[info.id].block;
      for (int iy = 0; iy < _BS_; iy++)
        for (int ix = 0; ix < _BS_; ix++) {
          if (sdf[iy][ix] > +h || sdf[iy][ix] < -h) {
            X[iy][ix] = sdf[iy][ix] > 0 ? 1 : 0;
          } else {
            Real distPx = lab(ix + 1, iy);
            Real distMx = lab(ix - 1, iy);
            Real distPy = lab(ix, iy + 1);
            Real distMy = lab(ix, iy - 1);
            Real IplusX = std::max((Real)0.0, distPx);
            Real IminuX = std::max((Real)0.0, distMx);
            Real IplusY = std::max((Real)0.0, distPy);
            Real IminuY = std::max((Real)0.0, distMy);
            Real gradIX = IplusX - IminuX;
            Real gradIY = IplusY - IminuY;
            Real gradUX = distPx - distMx;
            Real gradUY = distPy - distMy;
            Real gradUSq = (gradUX * gradUX + gradUY * gradUY) + EPS;
            X[iy][ix] = (gradIX * gradUX + gradIY * gradUY) / gradUSq;
          }
          CHI[iy][ix] = std::max(CHI[iy][ix], X[iy][ix]);
          if (X[iy][ix] > 0) {
            Real p[2];
            p[0] = info.origin[0] + info.h * (ix + 0.5);
            p[1] = info.origin[1] + info.h * (iy + 0.5);
            o.COM_x += X[iy][ix] * h2 * (p[0] - shape->centerOfMass[0]);
            o.COM_y += X[iy][ix] * h2 * (p[1] - shape->centerOfMass[1]);
            o.Mass += X[iy][ix] * h2;
          }
        }
    }
  }
};
struct PutFishOnBlocks {
  Real position[2];
  Real Rmatrix2D[2][2];
  void changeVelocityToComputationalFrame(Real x[2]) const {
    const Real p[2] = {x[0], x[1]};
    x[0] = Rmatrix2D[0][0] * p[0] + Rmatrix2D[0][1] * p[1];
    x[1] = Rmatrix2D[1][0] * p[0] + Rmatrix2D[1][1] * p[1];
  }
  void changeToComputationalFrame(Real x[2]) const {
    const Real p[2] = {x[0], x[1]};
    x[0] = Rmatrix2D[0][0] * p[0] + Rmatrix2D[0][1] * p[1];
    x[1] = Rmatrix2D[1][0] * p[0] + Rmatrix2D[1][1] * p[1];
    x[0] += position[0];
    x[1] += position[1];
  }
  void changeFromComputationalFrame(Real x[2]) const {
    Real p[2] = {x[0] - position[0], x[1] - position[1]};
    x[0] = Rmatrix2D[0][0] * p[0] + Rmatrix2D[1][0] * p[1];
    x[1] = Rmatrix2D[0][1] * p[0] + Rmatrix2D[1][1] * p[1];
  }
};
static void ongrid(Real dt) {
  std::vector<BlockInfo> &velInfo = var.vel->infos;
  std::vector<BlockInfo> &tmpInfo = var.tmp->infos;
  std::vector<BlockInfo> &chiInfo = var.chi->infos;
  int nSum[2] = {0, 0};
  Real uSum[2] = {0, 0};
  if (nSum[0] > 0) {
    sim.uinfx = uSum[0] / nSum[0];
  }
  if (nSum[1] > 0) {
    sim.uinfy = uSum[1] / nSum[1];
  }
  for (const auto &shape : sim.shapes) {
    shape->centerOfMass[0] += dt * (shape->u + sim.uinfx);
    shape->centerOfMass[1] += dt * (shape->v + sim.uinfy);
    shape->orientation += dt * shape->omega;
    shape->orientation = shape->orientation > M_PI
                             ? shape->orientation - 2 * M_PI
                             : shape->orientation;
    shape->orientation = shape->orientation < -M_PI
                             ? shape->orientation + 2 * M_PI
                             : shape->orientation;
    Real cosang = std::cos(shape->orientation);
    Real sinang = std::sin(shape->orientation);
    shape->center[0] = shape->centerOfMass[0] + cosang * shape->d_gm[0] -
                       sinang * shape->d_gm[1];
    shape->center[1] = shape->centerOfMass[1] + sinang * shape->d_gm[0] +
                       cosang * shape->d_gm[1];
    shape->theta_internal -= dt * shape->angvel_internal;
    if (shape->center[0] < 0 || shape->center[0] > sim.extents[0] ||
        shape->center[1] < 0 || shape->center[1] > sim.extents[1]) {
      printf("[CUP2D] ABORT: Body out of domain\n");
      fflush(0);
      abort();
    }
  }
  const size_t Nblocks = velInfo.size();
#pragma omp parallel for
  for (size_t i = 0; i < Nblocks; i++)
    for (int x = 0; x < _BS_; x++)
      for (int y = 0; y < _BS_; y++) {
        (*(ScalarBlock *)chiInfo[i].block)[x][y] = 0;
        (*(ScalarBlock *)tmpInfo[i].block)[x][y] = -1;
      }
  for (const auto &shape : sim.shapes) {
    for (auto &entry : shape->obstacleBlocks)
      delete entry;
    shape->obstacleBlocks.clear();
    shape->periodScheduler.transition(
        sim.time, shape->transition_start,
        shape->transition_start + shape->transition_duration,
        shape->current_period, shape->next_period);
    shape->periodScheduler.gimmeValues(sim.time, shape->periodPIDval,
                                       shape->periodPIDdif);
    if (shape->transition_start < sim.time &&
        sim.time < shape->transition_start + shape->transition_duration) {
      shape->timeshift =
          (sim.time - shape->time0) / shape->periodPIDval + shape->timeshift;
      shape->time0 = sim.time;
    }
    const std::array<Real, 6> curvaturePoints = {(Real)0,
                                                 (Real).15 * shape->length,
                                                 (Real).4 * shape->length,
                                                 (Real).65 * shape->length,
                                                 (Real).9 * shape->length,
                                                 shape->length};
    const std::array<Real, 6> curvatureValues = {
        (Real)0.82014 / shape->length, (Real)1.46515 / shape->length,
        (Real)2.57136 / shape->length, (Real)3.75425 / shape->length,
        (Real)5.09147 / shape->length, (Real)5.70449 / shape->length};
    const std::array<Real, 7> bendPoints = {(Real)-.5, (Real)-.25, (Real)0,
                                            (Real).25, (Real).5,   (Real).75,
                                            (Real)1};
    const std::array<Real, 6> curvatureZeros = {
        0.01 * curvatureValues[0], 0.01 * curvatureValues[1],
        0.01 * curvatureValues[2], 0.01 * curvatureValues[3],
        0.01 * curvatureValues[4], 0.01 * curvatureValues[5],
    };
    shape->curvatureScheduler.transition(0, 0, shape->Tperiod, curvatureZeros,
                                         curvatureValues);
    shape->curvatureScheduler.gimmeValues(sim.time, curvaturePoints, shape->Nm,
                                          shape->rS, shape->rC, shape->vC);
    shape->rlBendingScheduler.gimmeValues(sim.time, shape->periodPIDval,
                                          shape->length, bendPoints, shape->Nm,
                                          shape->rS, shape->rB, shape->vB);
    const Real diffT = 1 - (sim.time - shape->time0) * shape->periodPIDdif /
                               shape->periodPIDval;
    const Real darg = 2 * M_PI / shape->periodPIDval * diffT;
    const Real arg0 = 2 * M_PI *
                          ((sim.time - shape->time0) / shape->periodPIDval +
                           shape->timeshift) +
                      M_PI * shape->phaseShift;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < shape->Nm; ++i) {
      const Real arg = arg0 - 2 * M_PI * shape->rS[i] / shape->length;
      shape->rK[i] = shape->amplitudeFactor * shape->rC[i] *
                     (std::sin(arg) + shape->rB[i] + shape->curv_PID_fac);
      shape->vK[i] =
          shape->amplitudeFactor *
          (shape->vC[i] * (std::sin(arg) + shape->rB[i] + shape->curv_PID_fac) +
           shape->rC[i] *
               (std::cos(arg) * darg + shape->vB[i] + shape->curv_PID_dif));
    }
    if2d_solve(shape->Nm, shape->rS, shape->rK, shape->vK, shape->rX, shape->rY,
               shape->vX, shape->vY, shape->norX, shape->norY, shape->vNorX,
               shape->vNorY);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < shape->lowerSkin.n; ++i) {
      Real norm[2] = {shape->norX[i], shape->norY[i]};
      Real const norm_mod1 = std::sqrt(norm[0] * norm[0] + norm[1] * norm[1]);
      norm[0] /= norm_mod1;
      norm[1] /= norm_mod1;
      shape->lowerSkin.xSurf[i] = shape->rX[i] - shape->width[i] * norm[0];
      shape->lowerSkin.ySurf[i] = shape->rY[i] - shape->width[i] * norm[1];
      shape->upperSkin.xSurf[i] = shape->rX[i] + shape->width[i] * norm[0];
      shape->upperSkin.ySurf[i] = shape->rY[i] + shape->width[i] * norm[1];
    }
    Real _area = 0, _cmx = 0, _cmy = 0, _lmx = 0, _lmy = 0;
#pragma omp parallel for schedule(static)                                      \
    reduction(+ : _area, _cmx, _cmy, _lmx, _lmy)
    for (int i = 0; i < shape->Nm; ++i) {
      const Real ds =
          (i == 0) ? shape->rS[1] - shape->rS[0]
                   : ((i == shape->Nm - 1)
                          ? shape->rS[shape->Nm - 1] - shape->rS[shape->Nm - 2]
                          : shape->rS[i + 1] - shape->rS[i - 1]);
      const Real fac1 = 2 * shape->width[i];
      const Real fac2 =
          2 * std::pow(shape->width[i], 3) *
          (dds(i, shape->Nm, shape->norX, shape->rS) * shape->norY[i] -
           dds(i, shape->Nm, shape->norY, shape->rS) * shape->norX[i]) /
          3;
      _area += fac1 * ds / 2;
      _cmx += (shape->rX[i] * fac1 + shape->norX[i] * fac2) * ds / 2;
      _cmy += (shape->rY[i] * fac1 + shape->norY[i] * fac2) * ds / 2;
      _lmx += (shape->vX[i] * fac1 + shape->vNorX[i] * fac2) * ds / 2;
      _lmy += (shape->vY[i] * fac1 + shape->vNorY[i] * fac2) * ds / 2;
    }
    shape->area = _area;
    shape->CoM_internal[0] = _cmx;
    shape->CoM_internal[1] = _cmy;
    shape->linMom[0] = _lmx;
    shape->linMom[1] = _lmy;
    shape->CoM_internal[0] /= shape->area;
    shape->CoM_internal[1] /= shape->area;
    shape->vCoM_internal[0] = shape->linMom[0] / shape->area;
    shape->vCoM_internal[1] = shape->linMom[1] / shape->area;
    shape->area_internal = shape->area;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < shape->Nm; ++i) {
      shape->rX[i] -= shape->CoM_internal[0];
      shape->rY[i] -= shape->CoM_internal[1];
      shape->vX[i] -= shape->vCoM_internal[0];
      shape->vY[i] -= shape->vCoM_internal[1];
    }
    Real _J = 0, _am = 0;
#pragma omp parallel for reduction(+ : _J, _am) schedule(static)
    for (int i = 0; i < shape->Nm; ++i) {
      const Real ds =
          (i == 0) ? shape->rS[1] - shape->rS[0]
                   : ((i == shape->Nm - 1)
                          ? shape->rS[shape->Nm - 1] - shape->rS[shape->Nm - 2]
                          : shape->rS[i + 1] - shape->rS[i - 1]);
      const Real fac1 = 2 * shape->width[i];
      const Real fac2 =
          2 * std::pow(shape->width[i], 3) *
          (dds(i, shape->Nm, shape->norX, shape->rS) * shape->norY[i] -
           dds(i, shape->Nm, shape->norY, shape->rS) * shape->norX[i]) /
          3;
      const Real fac3 = 2 * std::pow(shape->width[i], 3) / 3;
      const Real tmp_M =
          (shape->rX[i] * shape->vY[i] - shape->rY[i] * shape->vX[i]) * fac1 +
          (shape->rX[i] * shape->vNorY[i] - shape->rY[i] * shape->vNorX[i] +
           shape->vY[i] * shape->norX[i] - shape->vX[i] * shape->norY[i]) *
              fac2 +
          (shape->norX[i] * shape->vNorY[i] -
           shape->norY[i] * shape->vNorX[i]) *
              fac3;
      const Real tmp_J =
          (shape->rX[i] * shape->rX[i] + shape->rY[i] * shape->rY[i]) * fac1 +
          2 * (shape->rX[i] * shape->norX[i] + shape->rY[i] * shape->norY[i]) *
              fac2 +
          fac3;
      _am += tmp_M * ds / 2;
      _J += tmp_J * ds / 2;
    }
    shape->J = _J;
    shape->angMom = _am;
    shape->angvel_internal = shape->angMom / shape->J;
    shape->J_internal = shape->J;
    const Real Rmatrix2D[2][2] = {
        {std::cos(shape->theta_internal), -std::sin(shape->theta_internal)},
        {std::sin(shape->theta_internal), std::cos(shape->theta_internal)}};
#pragma omp parallel for schedule(static)
    for (int i = 0; i < shape->Nm; ++i) {
      shape->vX[i] += shape->angvel_internal * shape->rY[i];
      shape->vY[i] -= shape->angvel_internal * shape->rX[i];
      rotate2D(Rmatrix2D, shape->rX[i], shape->rY[i]);
      rotate2D(Rmatrix2D, shape->vX[i], shape->vY[i]);
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < shape->Nm - 1; i++) {
      const auto ds = shape->rS[i + 1] - shape->rS[i];
      const auto tX = shape->rX[i + 1] - shape->rX[i];
      const auto tY = shape->rY[i + 1] - shape->rY[i];
      const auto tVX = shape->vX[i + 1] - shape->vX[i];
      const auto tVY = shape->vY[i + 1] - shape->vY[i];
      shape->norX[i] = -tY / ds;
      shape->norY[i] = tX / ds;
      shape->vNorX[i] = -tVY / ds;
      shape->vNorY[i] = tVX / ds;
    }
    shape->norX[shape->Nm - 1] = shape->norX[shape->Nm - 2];
    shape->norY[shape->Nm - 1] = shape->norY[shape->Nm - 2];
    shape->vNorX[shape->Nm - 1] = shape->vNorX[shape->Nm - 2];
    shape->vNorY[shape->Nm - 1] = shape->vNorY[shape->Nm - 2];
    {
      const Real Rmatrix2D[2][2] = {
          {std::cos(shape->theta_internal), -std::sin(shape->theta_internal)},
          {std::sin(shape->theta_internal), std::cos(shape->theta_internal)}};
#pragma omp parallel for schedule(static)
      for (size_t i = 0; i < shape->upperSkin.n; ++i) {
        shape->upperSkin.xSurf[i] -= shape->CoM_internal[0];
        shape->upperSkin.ySurf[i] -= shape->CoM_internal[1];
        rotate2D(Rmatrix2D, shape->upperSkin.xSurf[i],
                 shape->upperSkin.ySurf[i]);
        shape->lowerSkin.xSurf[i] -= shape->CoM_internal[0];
        shape->lowerSkin.ySurf[i] -= shape->CoM_internal[1];
        rotate2D(Rmatrix2D, shape->lowerSkin.xSurf[i],
                 shape->lowerSkin.ySurf[i]);
      }
    }
    const int Nsegments = (shape->Nm - 1) / 8;
    const int Nm = shape->Nm;
    assert((Nm - 1) % Nsegments == 0);
    std::vector<AreaSegment *> vSegments(Nsegments, nullptr);
    Real h = std::numeric_limits<Real>::infinity();
    for (size_t i = 0; i < var.vel->infos.size(); i++)
      h = std::min(var.vel->infos[i].h, h);
    MPI_Allreduce(MPI_IN_PLACE, &h, 1, MPI_Real, MPI_MIN, MPI_COMM_WORLD);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < Nsegments; ++i) {
      const int next_idx = (i + 1) * (Nm - 1) / Nsegments;
      const int idx = i * (Nm - 1) / Nsegments;
      Real bbox[2][2] = {{1e9, -1e9}, {1e9, -1e9}};
      for (int ss = idx; ss <= next_idx; ++ss) {
        const Real xBnd[2] = {
            shape->rX[ss] - shape->norX[ss] * shape->width[ss],
            shape->rX[ss] + shape->norX[ss] * shape->width[ss]};
        const Real yBnd[2] = {
            shape->rY[ss] - shape->norY[ss] * shape->width[ss],
            shape->rY[ss] + shape->norY[ss] * shape->width[ss]};
        const Real maxX = std::max(xBnd[0], xBnd[1]),
                   minX = std::min(xBnd[0], xBnd[1]);
        const Real maxY = std::max(yBnd[0], yBnd[1]),
                   minY = std::min(yBnd[0], yBnd[1]);
        bbox[0][0] = std::min(bbox[0][0], minX);
        bbox[0][1] = std::max(bbox[0][1], maxX);
        bbox[1][0] = std::min(bbox[1][0], minY);
        bbox[1][1] = std::max(bbox[1][1], maxY);
      }
      const Real DD = 4 * h;
      AreaSegment *const tAS =
          new AreaSegment(std::make_pair(idx, next_idx), bbox, DD);
      tAS->changeToComputationalFrame(shape->center, shape->orientation);
      vSegments[i] = tAS;
    }
    const auto N = tmpInfo.size();
    std::vector<std::vector<AreaSegment *> *> segmentsPerBlock(N, nullptr);
    shape->obstacleBlocks = std::vector<ObstacleBlock *>(N, nullptr);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < tmpInfo.size(); ++i) {
      const BlockInfo &info = tmpInfo[i];
      Real pStart[2], pEnd[2];
      pStart[0] = info.origin[0] + info.h * 0.5;
      pStart[1] = info.origin[1] + info.h * 0.5;
      pEnd[0] = info.origin[0] + info.h * (_BS_ - 0.5);
      pEnd[1] = info.origin[1] + info.h * (_BS_ - 0.5);
      for (size_t s = 0; s < vSegments.size(); ++s)
        if (vSegments[s]->isIntersectingWithAABB(pStart, pEnd)) {
          if (segmentsPerBlock[info.id] == nullptr)
            segmentsPerBlock[info.id] = new std::vector<AreaSegment *>(0);
          segmentsPerBlock[info.id]->push_back(vSegments[s]);
        }
      if (segmentsPerBlock[info.id] not_eq nullptr) {
        ObstacleBlock *const block = new ObstacleBlock();
        assert(block not_eq nullptr);
        shape->obstacleBlocks[info.id] = block;
        block->clear_surface();
        std::fill(block->dist[0], block->dist[0] + _BS_ * _BS_, -1);
        std::fill(block->chi[0], block->chi[0] + _BS_ * _BS_, 0);
        memset(block->udef, 0, sizeof(Real) * _BS_ * _BS_ * 2);
      }
    }
    assert(not segmentsPerBlock.empty());
#pragma omp parallel
    {
      PutFishOnBlocks putfish;
      putfish.position[0] = shape->center[0];
      putfish.position[1] = shape->center[1];

      putfish.Rmatrix2D[0][0] = std::cos(shape->orientation);
      putfish.Rmatrix2D[0][1] = -std::sin(shape->orientation);
      putfish.Rmatrix2D[1][0] = std::sin(shape->orientation);
      putfish.Rmatrix2D[1][1] = std::cos(shape->orientation);

#pragma omp for schedule(dynamic)
      for (size_t i = 0; i < tmpInfo.size(); i++) {
        const auto pos = segmentsPerBlock[tmpInfo[i].id];
        if (pos not_eq nullptr) {
          ObstacleBlock *const block = shape->obstacleBlocks[tmpInfo[i].id];
          assert(block not_eq nullptr);
          const BlockInfo &info = tmpInfo[i];
          ScalarBlock &b = *(ScalarBlock *)tmpInfo[i].block;
          ObstacleBlock *const o = block;
          const std::vector<AreaSegment *> &v = *pos;
          Real org[2];
          org[0] = info.origin[0] + info.h * 0.5;
          org[1] = info.origin[1] + info.h * 0.5;
          const Real h = info.h, invh = 1.0 / info.h;
          const Real *const rX = shape->rX, *const norX = shape->norX;
          const Real *const rY = shape->rY, *const norY = shape->norY;
          const Real *const vX = shape->vX, *const vNorX = shape->vNorX;
          const Real *const vY = shape->vY, *const vNorY = shape->vNorY;
          const Real *const width = shape->width;
          static constexpr int BS[2] = {_BS_, _BS_};
          std::fill(o->dist[0], o->dist[0] + BS[1] * BS[0], -1);
          std::fill(o->chi[0], o->chi[0] + BS[1] * BS[0], 0);
          for (int i = 0; i < (int)v.size(); ++i) {
            const int firstSegm = std::max(v[i]->s_range.first, 1);
            const int lastSegm = std::min(v[i]->s_range.second, shape->Nm - 2);
            for (int ss = firstSegm; ss <= lastSegm; ++ss) {
              assert(width[ss] > 0);
              for (int signp = -1; signp <= 1; signp += 2) {
                Real myP[2] = {
                    rX[ss + 0] + width[ss + 0] * signp * norX[ss + 0],
                    rY[ss + 0] + width[ss + 0] * signp * norY[ss + 0]};
                putfish.changeToComputationalFrame(myP);
                const int iap[2] = {(int)std::floor((myP[0] - org[0]) * invh),
                                    (int)std::floor((myP[1] - org[1]) * invh)};
                if (iap[0] + 3 <= 0 || iap[0] - 1 >= BS[0])
                  continue;
                if (iap[1] + 3 <= 0 || iap[1] - 1 >= BS[1])
                  continue;
                Real pP[2] = {rX[ss + 1] + width[ss + 1] * signp * norX[ss + 1],
                              rY[ss + 1] +
                                  width[ss + 1] * signp * norY[ss + 1]};
                putfish.changeToComputationalFrame(pP);
                Real pM[2] = {rX[ss - 1] + width[ss - 1] * signp * norX[ss - 1],
                              rY[ss - 1] +
                                  width[ss - 1] * signp * norY[ss - 1]};
                putfish.changeToComputationalFrame(pM);
                Real udef[2] = {
                    vX[ss + 0] + width[ss + 0] * signp * vNorX[ss + 0],
                    vY[ss + 0] + width[ss + 0] * signp * vNorY[ss + 0]};
                putfish.changeVelocityToComputationalFrame(udef);
                for (int sy = std::max(0, iap[1] - 2);
                     sy < std::min(iap[1] + 4, BS[1]); ++sy)
                  for (int sx = std::max(0, iap[0] - 2);
                       sx < std::min(iap[0] + 4, BS[0]); ++sx) {
                    Real p[2];
                    p[0] = info.origin[0] + info.h * (sx + 0.5);
                    p[1] = info.origin[1] + info.h * (sy + 0.5);
                    const Real dist0 = dist(p, myP);
                    const Real distP = dist(p, pP);
                    const Real distM = dist(p, pM);
                    if (std::fabs(o->dist[sy][sx]) <
                        std::min({dist0, distP, distM}))
                      continue;
                    putfish.changeFromComputationalFrame(p);
                    Real p0[2] = {rX[ss] + width[ss] * signp * norX[ss],
                                  rY[ss] + width[ss] * signp * norY[ss]};
                    Real distC = dist(p, p0);
                    assert(std::fabs(distC - dist0) < EPS);
                    int close_s = ss, secnd_s = ss + (distP < distM ? 1 : -1);
                    Real dist1 = dist0, dist2 = distP < distM ? distP : distM;
                    if (distP < dist0 || distM < dist0) {
                      dist1 = dist2;
                      dist2 = dist0;
                      close_s = secnd_s;
                      secnd_s = ss;
                    }
                    Real dSsq = std::pow(rX[close_s] - rX[secnd_s], 2) +
                                std::pow(rY[close_s] - rY[secnd_s], 2);
                    assert(dSsq > 2.2e-16);
                    Real cnt2ML = std::pow(width[close_s], 2);
                    Real nxt2ML = std::pow(width[secnd_s], 2);
                    Real safeW =
                        std::max(width[close_s], width[secnd_s]) + 2 * h;
                    Real xMidl[2] = {rX[close_s], rY[close_s]};
                    Real grd2ML = dist(p, xMidl);
                    Real diffH = std::fabs(width[close_s] - width[secnd_s]);
                    Real sign2d = 0;
                    if (dSsq > diffH * diffH || grd2ML > safeW * safeW) {
                      sign2d = grd2ML > cnt2ML ? -1 : 1;
                    } else {
                      Real corr = 2 * std::sqrt(cnt2ML * nxt2ML);
                      Real Rsq = (cnt2ML + nxt2ML - corr + dSsq) *
                                 (cnt2ML + nxt2ML + corr + dSsq) / 4 / dSsq;
                      Real maxAx = std::max(cnt2ML, nxt2ML);
                      int idAx1 = cnt2ML > nxt2ML ? close_s : secnd_s;
                      int idAx2 = idAx1 == close_s ? secnd_s : close_s;
                      Real d = std::sqrt((Rsq - maxAx) / dSsq);
                      Real xCentr[2] = {rX[idAx1] + (rX[idAx1] - rX[idAx2]) * d,
                                        rY[idAx1] +
                                            (rY[idAx1] - rY[idAx2]) * d};
                      Real grd2Core = dist(p, xCentr);
                      sign2d = grd2Core > Rsq ? -1 : 1;
                    }
                    if (std::fabs(o->dist[sy][sx]) > dist1) {
                      Real W =
                          1 - std::min((Real)1, std::sqrt(dist1) * (invh / 3));
                      assert(W >= 0);
                      o->udef[sy][sx][0] = W * udef[0];
                      o->udef[sy][sx][1] = W * udef[1];
                      o->dist[sy][sx] = sign2d * dist1;
                      o->chi[sy][sx] = W;
                    }
                  }
              }
            }
          }
          org[0] = info.origin[0] + info.h * 0.5;
          org[1] = info.origin[1] + info.h * 0.5;
          for (int i = 0; i < (int)v.size(); ++i) {
            const int firstSegm = std::max(v[i]->s_range.first, 1);
            const int lastSegm = std::min(v[i]->s_range.second, shape->Nm - 2);
            for (int ss = firstSegm; ss <= lastSegm; ++ss) {
              const Real myWidth = shape->width[ss];
              assert(myWidth > 0);
              const int Nw = std::floor(myWidth / h);
              for (int iw = -Nw + 1; iw < Nw; ++iw) {
                const Real offsetW = iw * h;
                Real xp[2] = {shape->rX[ss] + offsetW * shape->norX[ss],
                              shape->rY[ss] + offsetW * shape->norY[ss]};
                putfish.changeToComputationalFrame(xp);
                xp[0] = (xp[0] - org[0]) * invh;
                xp[1] = (xp[1] - org[1]) * invh;
                const Real ap[2] = {std::floor(xp[0]), std::floor(xp[1])};
                const int iap[2] = {(int)ap[0], (int)ap[1]};
                if (iap[0] + 2 <= 0 || iap[0] >= BS[0])
                  continue;
                if (iap[1] + 2 <= 0 || iap[1] >= BS[1])
                  continue;
                Real udef[2] = {shape->vX[ss] + offsetW * shape->vNorX[ss],
                                shape->vY[ss] + offsetW * shape->vNorY[ss]};
                putfish.changeVelocityToComputationalFrame(udef);
                Real wghts[2][2];
                for (int c = 0; c < 2; ++c) {
                  const Real t[2] = {std::fabs(xp[c] - ap[c]),
                                     std::fabs(xp[c] - (ap[c] + 1))};
                  wghts[c][0] = 1 - t[0];
                  wghts[c][1] = 1 - t[1];
                }
                for (int idy = std::max(0, iap[1]);
                     idy < std::min(iap[1] + 2, BS[1]); ++idy)
                  for (int idx = std::max(0, iap[0]);
                       idx < std::min(iap[0] + 2, BS[0]); ++idx) {
                    const int sx = idx - iap[0], sy = idy - iap[1];
                    const Real wxwy = wghts[1][sy] * wghts[0][sx];
                    assert(idx >= 0 && idx < _BS_ && wxwy >= 0);
                    assert(idy >= 0 && idy < _BS_ && wxwy <= 1);
                    o->udef[idy][idx][0] += wxwy * udef[0];
                    o->udef[idy][idx][1] += wxwy * udef[1];
                    o->chi[idy][idx] += wxwy;
                    static constexpr Real EPS =
                        std::numeric_limits<Real>::epsilon();
                    if (std::fabs(o->dist[idy][idx] + 1) < EPS)
                      o->dist[idy][idx] = 1;
                  }
              }
            }
          }
          static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
          for (int iy = 0; iy < _BS_; iy++)
            for (int ix = 0; ix < _BS_; ix++) {
              const Real normfac = o->chi[iy][ix] > EPS ? o->chi[iy][ix] : 1;
              o->udef[iy][ix][0] /= normfac;
              o->udef[iy][ix][1] /= normfac;
              o->dist[iy][ix] = o->dist[iy][ix] >= 0
                                    ? std::sqrt(o->dist[iy][ix])
                                    : -std::sqrt(-o->dist[iy][ix]);
              b[iy][ix] = std::max(b[iy][ix], o->dist[iy][ix]);
              ;
            }
          std::fill(o->chi[0], o->chi[0] + BS[1] * BS[0], 0);
        }
      }
    }
    for (auto &E : vSegments)
      delete E;
    for (auto &E : segmentsPerBlock)
      delete E;
  }
  computeA<ScalarLab>(PutChiOnGrid(), var.tmp, 1);
  computeB<ComputeSurfaceNormals, Real, ScalarLab, Real, ScalarLab>(
      ComputeSurfaceNormals(), *var.chi, 1, *var.tmp, 1);
  for (const auto &shape : sim.shapes) {
    Real com[3] = {0.0, 0.0, 0.0};
    const std::vector<ObstacleBlock *> &OBLOCK = shape->obstacleBlocks;
#pragma omp parallel for reduction(+ : com[:3])
    for (size_t i = 0; i < OBLOCK.size(); i++) {
      if (OBLOCK[i] == nullptr)
        continue;
      com[0] += OBLOCK[i]->Mass;
      com[1] += OBLOCK[i]->COM_x;
      com[2] += OBLOCK[i]->COM_y;
    }
    MPI_Allreduce(MPI_IN_PLACE, com, 3, MPI_Real, MPI_SUM, MPI_COMM_WORLD);
    shape->M = com[0];
    shape->centerOfMass[0] += com[1] / com[0];
    shape->centerOfMass[1] += com[2] / com[0];
  }
  for (const auto &shape : sim.shapes) {
    Real _x = 0, _y = 0, _m = 0, _j = 0, _u = 0, _v = 0, _a = 0;
#pragma omp parallel for schedule(dynamic, 1)                                  \
    reduction(+ : _x, _y, _m, _j, _u, _v, _a)
    for (size_t i = 0; i < chiInfo.size(); i++) {
      const Real hsq = std::pow(chiInfo[i].h, 2);
      const auto pos = shape->obstacleBlocks[chiInfo[i].id];
      if (pos == nullptr)
        continue;
      const CHI_MAT &CHI = pos->chi;
      const UDEFMAT &UDEF = pos->udef;
      for (int iy = 0; iy < _BS_; ++iy)
        for (int ix = 0; ix < _BS_; ++ix) {
          if (CHI[iy][ix] <= 0)
            continue;
          Real p[2];
          p[0] = chiInfo[i].origin[0] + chiInfo[i].h * (ix + 0.5);
          p[1] = chiInfo[i].origin[1] + chiInfo[i].h * (iy + 0.5);
          const Real chi = CHI[iy][ix] * hsq;
          p[0] -= shape->centerOfMass[0];
          p[1] -= shape->centerOfMass[1];
          _x += chi * p[0];
          _y += chi * p[1];
          _m += chi;
          _j += chi * (p[0] * p[0] + p[1] * p[1]);
          _u += chi * UDEF[iy][ix][0];
          _v += chi * UDEF[iy][ix][1];
          _a += chi * (p[0] * UDEF[iy][ix][1] - p[1] * UDEF[iy][ix][0]);
        }
    }
    Real quantities[7] = {_x, _y, _m, _j, _u, _v, _a};
    MPI_Allreduce(MPI_IN_PLACE, quantities, 7, MPI_Real, MPI_SUM,
                  MPI_COMM_WORLD);
    _x = quantities[0];
    _y = quantities[1];
    _m = quantities[2];
    _j = quantities[3];
    _u = quantities[4];
    _v = quantities[5];
    _a = quantities[6];
    _u /= _m;
    _v /= _m;
    _a /= _j;
    Integrals I = Integrals(_x, _y, _m, _j, _u, _v, _a);
    shape->M = I.m;
    shape->J = I.j;
    const Real dCx = shape->center[0] - shape->centerOfMass[0];
    const Real dCy = shape->center[1] - shape->centerOfMass[1];
    shape->d_gm[0] =
        dCx * std::cos(shape->orientation) + dCy * std::sin(shape->orientation);
    shape->d_gm[1] = -dCx * std::sin(shape->orientation) +
                     dCy * std::cos(shape->orientation);
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < chiInfo.size(); i++) {
      const auto pos = shape->obstacleBlocks[chiInfo[i].id];
      if (pos == nullptr)
        continue;
      for (int iy = 0; iy < _BS_; ++iy)
        for (int ix = 0; ix < _BS_; ++ix) {
          Real p[2];
          p[0] = chiInfo[i].origin[0] + chiInfo[i].h * (ix + 0.5);
          p[1] = chiInfo[i].origin[1] + chiInfo[i].h * (iy + 0.5);
          p[0] -= shape->centerOfMass[0];
          p[1] -= shape->centerOfMass[1];
          pos->udef[iy][ix][0] -= I.u - I.a * p[1];
          pos->udef[iy][ix][1] -= I.v + I.a * p[0];
        }
    }
    const Real Rmatrix2D[2][2] = {
        {std::cos(shape->orientation), -std::sin(shape->orientation)},
        {std::sin(shape->orientation), std::cos(shape->orientation)}};
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < shape->upperSkin.n; ++i) {
      rotate2D(Rmatrix2D, shape->upperSkin.xSurf[i], shape->upperSkin.ySurf[i]);
      shape->upperSkin.xSurf[i] += shape->centerOfMass[0];
      shape->upperSkin.ySurf[i] += shape->centerOfMass[1];
      rotate2D(Rmatrix2D, shape->lowerSkin.xSurf[i], shape->lowerSkin.ySurf[i]);
      shape->lowerSkin.xSurf[i] += shape->centerOfMass[0];
      shape->lowerSkin.ySurf[i] += shape->centerOfMass[1];
    }
    {
      const Real Rmatrix2D[2][2] = {
          {std::cos(shape->orientation), -std::sin(shape->orientation)},
          {std::sin(shape->orientation), std::cos(shape->orientation)}};
      for (int i = 0; i < shape->Nm; ++i) {
        rotate2D(Rmatrix2D, shape->rX[i], shape->rY[i]);
        rotate2D(Rmatrix2D, shape->norX[i], shape->norY[i]);
        shape->rX[i] += shape->centerOfMass[0];
        shape->rY[i] += shape->centerOfMass[1];
      }
#pragma omp parallel for
      for (size_t i = 0; i < shape->lowerSkin.n - 1; ++i) {
        shape->lowerSkin.midX[i] =
            (shape->lowerSkin.xSurf[i] + shape->lowerSkin.xSurf[i + 1]) / 2;
        shape->upperSkin.midX[i] =
            (shape->upperSkin.xSurf[i] + shape->upperSkin.xSurf[i + 1]) / 2;
        shape->lowerSkin.midY[i] =
            (shape->lowerSkin.ySurf[i] + shape->lowerSkin.ySurf[i + 1]) / 2;
        shape->upperSkin.midY[i] =
            (shape->upperSkin.ySurf[i] + shape->upperSkin.ySurf[i + 1]) / 2;
        shape->lowerSkin.normXSurf[i] =
            (shape->lowerSkin.ySurf[i + 1] - shape->lowerSkin.ySurf[i]);
        shape->upperSkin.normXSurf[i] =
            (shape->upperSkin.ySurf[i + 1] - shape->upperSkin.ySurf[i]);
        shape->lowerSkin.normYSurf[i] =
            -(shape->lowerSkin.xSurf[i + 1] - shape->lowerSkin.xSurf[i]);
        shape->upperSkin.normYSurf[i] =
            -(shape->upperSkin.xSurf[i + 1] - shape->upperSkin.xSurf[i]);
        Real normL = std::sqrt(std::pow(shape->lowerSkin.normXSurf[i], 2) +
                               std::pow(shape->lowerSkin.normYSurf[i], 2));
        Real normU = std::sqrt(std::pow(shape->upperSkin.normXSurf[i], 2) +
                               std::pow(shape->upperSkin.normYSurf[i], 2));
        shape->lowerSkin.normXSurf[i] /= normL;
        shape->upperSkin.normXSurf[i] /= normU;
        shape->lowerSkin.normYSurf[i] /= normL;
        shape->upperSkin.normYSurf[i] /= normU;
        const int ii =
            (i < 8)
                ? 8
                : ((i > shape->lowerSkin.n - 9) ? shape->lowerSkin.n - 9 : i);
        const Real dirL = shape->lowerSkin.normXSurf[i] *
                              (shape->lowerSkin.midX[i] - shape->rX[ii]) +
                          shape->lowerSkin.normYSurf[i] *
                              (shape->lowerSkin.midY[i] - shape->rY[ii]);
        const Real dirU = shape->upperSkin.normXSurf[i] *
                              (shape->upperSkin.midX[i] - shape->rX[ii]) +
                          shape->upperSkin.normYSurf[i] *
                              (shape->upperSkin.midY[i] - shape->rY[ii]);
        if (dirL < 0) {
          shape->lowerSkin.normXSurf[i] *= -1.0;
          shape->lowerSkin.normYSurf[i] *= -1.0;
        }
        if (dirU < 0) {
          shape->upperSkin.normXSurf[i] *= -1.0;
          shape->upperSkin.normYSurf[i] *= -1.0;
        }
      }
    }
  }
}
struct GradChiOnTmp {
  GradChiOnTmp() {}
  const StencilInfo stencil{-4, -4, 5, 5, true};
  const std::vector<BlockInfo> &tmpInfo = var.tmp->infos;
  void operator()(ScalarLab &lab, const BlockInfo &info) const {
    auto &TMP = *(ScalarBlock *)tmpInfo[info.id].block;
    int offset = (info.level == sim.levelMax - 1) ? 4 : 2;
    Real threshold = sim.bAdaptChiGradient ? 0.9 : 1e4;
    int nm = _BS_ + stencil.ex - stencil.sx - 1;
    Real *um = lab.m;
    for (int y = -offset; y < _BS_ + offset; ++y)
      for (int x = -offset; x < _BS_ + offset; ++x) {
        int k = nm * (y - stencil.sy) + x - stencil.sx;
        assert(k >= 0);
        um[k] = std::min(um[k], 1.0);
        um[k] = std::max(um[k], 0.0);
        if (um[k] > 0.0 && um[k] < threshold) {
          TMP[_BS_ / 2][_BS_ / 2 - 1] = 2 * sim.Rtol;
          TMP[_BS_ / 2 - 1][_BS_ / 2 - 1] = 2 * sim.Rtol;
          TMP[_BS_ / 2][_BS_ / 2] = 2 * sim.Rtol;
          TMP[_BS_ / 2 - 1][_BS_ / 2] = 2 * sim.Rtol;
          break;
        }
      }
  }
};
static void adapt() {
  computeA<VectorLab>(KernelVorticity(), var.vel, 2);
  computeA<ScalarLab>(GradChiOnTmp(), var.chi, 1);
  var.tmp_amr->boundary_needed = true;
  Synchronizer<Grid> *Synch = var.tmp->sync1(var.tmp_amr->stencil);
  var.tmp_amr->CallValidStates = false;
  bool Reduction = false;
  MPI_Request Reduction_req;
  int tmp;
  std::vector<BlockInfo *> *halo = &Synch->halo_blocks;
  std::vector<BlockInfo *> *infos[2] = {&Synch->inner_blocks, halo};
  typedef Real ScalarBlock[_BS_][_BS_];
  for (int iii = 0;; iii++) {
    std::vector<BlockInfo *> *I = infos[iii];
#pragma omp parallel
    {
#pragma omp for schedule(dynamic, 1)
      for (size_t i = 0; i < I->size(); i++) {
        BlockInfo &info = var.tmp->get((*I)[i]->level, (*I)[i]->Z);
        ScalarBlock &b = *(ScalarBlock *)info.block;
        double Linf = 0.0;
        for (int j = 0; j < _BS_; j++)
          for (int i = 0; i < _BS_; i++)
            Linf = std::max(Linf, std::fabs(b[j][i]));
        if (Linf > sim.Rtol)
          (*I)[i]->state = Refine;
        else if (Linf < sim.Ctol)
          (*I)[i]->state = Compress;
        else
          (*I)[i]->state = Leave;
        const bool maxLevel =
            ((*I)[i]->state == Refine) && ((*I)[i]->level == sim.levelMax - 1);
        const bool minLevel =
            ((*I)[i]->state == Compress) && ((*I)[i]->level == 0);
        if (maxLevel || minLevel)
          (*I)[i]->state = Leave;
        info.state = (*I)[i]->state;
        if (info.state != Leave) {
#pragma omp critical
          {
            var.tmp_amr->CallValidStates = true;
            if (!Reduction) {
              tmp = 1;
              Reduction = true;
              MPI_Iallreduce(MPI_IN_PLACE, &tmp, 1, MPI_INT, MPI_SUM,
                             MPI_COMM_WORLD, &Reduction_req);
            }
          }
        }
      }
    }
    if (iii == 1)
      break;
    MPI_Waitall(Synch->requests.size(), Synch->requests.data(),
                MPI_STATUSES_IGNORE);
  }
  if (!Reduction) {
    tmp = var.tmp_amr->CallValidStates ? 1 : 0;
    Reduction = true;
    MPI_Iallreduce(MPI_IN_PLACE, &tmp, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD,
                   &Reduction_req);
  }
  MPI_Wait(&Reduction_req, MPI_STATUS_IGNORE);
  var.tmp_amr->CallValidStates = (tmp > 0);
  var.tmp->boundary = *halo;
  if (var.tmp_amr->CallValidStates) {
    int levelMin = 0;
    std::vector<BlockInfo> &I = var.tmp->infos;
#pragma omp parallel for
    for (size_t j = 0; j < I.size(); j++) {
      BlockInfo &info = I[j];
      if ((info.state == Refine && info.level == sim.levelMax - 1) ||
          (info.state == Compress && info.level == levelMin)) {
        info.state = Leave;
        (var.tmp->get(info.level, info.Z)).state = Leave;
      }
      if (info.state != Leave) {
        info.changed2 = true;
        (var.tmp->get(info.level, info.Z)).changed2 = info.changed2;
      }
    }
    bool clean_boundary = true;
    for (int m = sim.levelMax - 1; m >= levelMin; m--) {
      for (size_t j = 0; j < I.size(); j++) {
        BlockInfo &info = I[j];
        if (info.level == m && info.state != Refine &&
            info.level != sim.levelMax - 1) {
          int TwoPower = 1 << info.level;
          bool xskin =
              info.index[0] == 0 || info.index[0] == sim.bpdx * TwoPower - 1;
          bool yskin =
              info.index[1] == 0 || info.index[1] == sim.bpdy * TwoPower - 1;
          int xskip = info.index[0] == 0 ? -1 : 1;
          int yskip = info.index[1] == 0 ? -1 : 1;

          for (int icode = 0; icode < 27; icode++) {
            if (info.state == Refine)
              break;
            if (icode == 1 * 1 + 3 * 1 + 9 * 1)
              continue;
            int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                           (icode / 9) % 3 - 1};
            if (sim.bcx != periodic && code[0] == xskip && xskin)
              continue;
            if (sim.bcy != periodic && code[1] == yskip && yskin)
              continue;
            if (code[2] != 0)
              continue;
            if (var.tmp->Tree0(info.level,
                               info.Znei[1 + code[0]][1 + code[1]]) == -1) {
              if (info.state == Compress) {
                info.state = Leave;
                (var.tmp->get(info.level, info.Z)).state = Leave;
              }
              int tmp = abs(code[0]) + abs(code[1]) + abs(code[2]);
              int Bstep = 1;
              if (tmp == 2)
                Bstep = 3;
              else if (tmp == 3)
                Bstep = 4;
              for (int B = 0; B <= 1; B += Bstep) {
                int aux = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
                int iNei = 2 * info.index[0] + std::max(code[0], 0) + code[0] +
                           (B % 2) * std::max(0, 1 - abs(code[0]));
                int jNei = 2 * info.index[1] + std::max(code[1], 0) + code[1] +
                           aux * std::max(0, 1 - abs(code[1]));
                long long zzz = getZforward(m + 1, iNei, jNei);
                BlockInfo &FinerNei = var.tmp->get(m + 1, zzz);
                State NeiState = FinerNei.state;
                if (NeiState == Refine) {
                  info.state = Refine;
                  (var.tmp->get(info.level, info.Z)).state = Refine;
                  info.changed2 = true;
                  (var.tmp->get(info.level, info.Z)).changed2 = true;
                  break;
                }
              }
            }
          }
        }
      }
      var.tmp->UpdateBoundary(clean_boundary);
      clean_boundary = false;
      if (m == levelMin)
        break;
      for (size_t j = 0; j < I.size(); j++) {
        BlockInfo &info = I[j];
        if (info.level == m && info.state == Compress) {
          int aux = 1 << info.level;
          bool xskin =
              info.index[0] == 0 || info.index[0] == sim.bpdx * aux - 1;
          bool yskin =
              info.index[1] == 0 || info.index[1] == sim.bpdy * aux - 1;
          int xskip = info.index[0] == 0 ? -1 : 1;
          int yskip = info.index[1] == 0 ? -1 : 1;

          for (int icode = 0; icode < 27; icode++) {
            if (icode == 1 * 1 + 3 * 1 + 9 * 1)
              continue;
            int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                           (icode / 9) % 3 - 1};
            if (sim.bcx != periodic && code[0] == xskip && xskin)
              continue;
            if (sim.bcy != periodic && code[1] == yskip && yskin)
              continue;
            if (code[2] != 0)
              continue;
            BlockInfo &infoNei =
                var.tmp->get(info.level, info.Znei[1 + code[0]][1 + code[1]]);
            if (var.tmp->Tree1(infoNei) >= 0 && infoNei.state == Refine) {
              info.state = Leave;
              (var.tmp->get(info.level, info.Z)).state = Leave;
              break;
            }
          }
        }
      }
    }
    for (size_t jjj = 0; jjj < I.size(); jjj++) {
      BlockInfo &info = I[jjj];
      int m = info.level;
      bool found = false;
      for (int i = 2 * (info.index[0] / 2); i <= 2 * (info.index[0] / 2) + 1;
           i++)
        for (int j = 2 * (info.index[1] / 2); j <= 2 * (info.index[1] / 2) + 1;
             j++)
          for (int k = 2 * (info.index[2] / 2);
               k <= 2 * (info.index[2] / 2) + 1; k++) {
            long long n = getZforward(m, i, j);
            BlockInfo &infoNei = var.tmp->get(m, n);
            if ((var.tmp->Tree1(infoNei) >= 0) == false ||
                infoNei.state != Compress) {
              found = true;
              if (info.state == Compress) {
                info.state = Leave;
                (var.tmp->get(info.level, info.Z)).state = Leave;
              }
              break;
            }
          }
      if (found)
        for (int i = 2 * (info.index[0] / 2); i <= 2 * (info.index[0] / 2) + 1;
             i++)
          for (int j = 2 * (info.index[1] / 2);
               j <= 2 * (info.index[1] / 2) + 1; j++)
            for (int k = 2 * (info.index[2] / 2);
                 k <= 2 * (info.index[2] / 2) + 1; k++) {
              long long n = getZforward(m, i, j);
              BlockInfo &infoNei = var.tmp->get(m, n);
              if (var.tmp->Tree1(infoNei) >= 0 && infoNei.state == Compress)
                infoNei.state = Leave;
            }
    }
  }
  struct {
    std::unordered_map<long long, BlockInfo *> *BlockInfoAll;
    std::vector<long long> *level_base;
    std::vector<BlockInfo> &I2;
  } args[] = {
      {&var.chi->BlockInfoAll, &var.chi->level_base, var.chi->infos},
      {&var.pres->BlockInfoAll, &var.pres->level_base, var.pres->infos},
      {&var.pold->BlockInfoAll, &var.pold->level_base, var.pold->infos},
      {&var.vel->BlockInfoAll, &var.vel->level_base, var.vel->infos},
      {&var.vold->BlockInfoAll, &var.vold->level_base, var.vold->infos},
      {&var.tmpV->BlockInfoAll, &var.tmpV->level_base, var.tmpV->infos},
  };
  for (int iarg = 0; iarg < sizeof args / sizeof *args; iarg++) {
    for (size_t i1 = 0; i1 < args[iarg].I2.size(); i1++) {
      BlockInfo &ary0 = args[iarg].I2[i1];
      BlockInfo &info = getf(args[iarg].BlockInfoAll, args[iarg].level_base,
                             ary0.level, ary0.Z);
      for (int i = 2 * (info.index[0] / 2); i <= 2 * (info.index[0] / 2) + 1;
           i++)
        for (int j = 2 * (info.index[1] / 2); j <= 2 * (info.index[1] / 2) + 1;
             j++) {
          const long long n = getZforward(info.level, i, j);
          BlockInfo &infoNei = getf(args[iarg].BlockInfoAll,
                                    args[iarg].level_base, info.level, n);
          infoNei.state = Leave;
        }
      info.state = Leave;
      ary0.state = Leave;
    }
#pragma omp parallel for
    for (size_t i = 0; i < var.tmp->infos.size(); i++) {
      const BlockInfo &info1 = var.tmp->infos[i];
      BlockInfo &info2 = args[iarg].I2[i];
      BlockInfo &info3 = getf(args[iarg].BlockInfoAll, args[iarg].level_base,
                              info2.level, info2.Z);
      info2.state = info1.state;
      info3.state = info1.state;
      if (info2.state == Compress) {
        const int i2 = 2 * (info2.index[0] / 2);
        const int j2 = 2 * (info2.index[1] / 2);
        const long long n = getZforward(info2.level, i2, j2);
        BlockInfo &infoNei = getf(args[iarg].BlockInfoAll,
                                  args[iarg].level_base, info2.level, n);
        infoNei.state = Compress;
      }
    }
  }
  var.tmp_amr->Adapt<ScalarLab>(var.tmp, false);
  var.chi_amr->Adapt<ScalarLab>(var.chi, false);
  var.vel_amr->Adapt<VectorLab>(var.vel, false);
  var.vold_amr->Adapt<VectorLab>(var.vold, false);
  var.pres_amr->Adapt<ScalarLab>(var.pres, false);
  var.pold_amr->Adapt<ScalarLab>(var.pold, false);
  var.tmpV_amr->Adapt<VectorLab>(var.tmpV, true);
}
static Real dU_adv_dif(VectorLab &V, Real uinf[2], Real advF, Real difF, int ix,
                       int iy) {
  Real u = V(ix, iy).u[0];
  Real v = V(ix, iy).u[1];
  Real UU = u + uinf[0];
  Real VV = v + uinf[1];
  Real up1x = V(ix + 1, iy).u[0];
  Real up2x = V(ix + 2, iy).u[0];
  Real up3x = V(ix + 3, iy).u[0];
  Real um1x = V(ix - 1, iy).u[0];
  Real um2x = V(ix - 2, iy).u[0];
  Real um3x = V(ix - 3, iy).u[0];
  Real up1y = V(ix, iy + 1).u[0];
  Real up2y = V(ix, iy + 2).u[0];
  Real up3y = V(ix, iy + 3).u[0];
  Real um1y = V(ix, iy - 1).u[0];
  Real um2y = V(ix, iy - 2).u[0];
  Real um3y = V(ix, iy - 3).u[0];
  Real dudx = derivative(UU, um3x, um2x, um1x, u, up1x, up2x, up3x);
  Real dudy = derivative(VV, um3y, um2y, um1y, u, up1y, up2y, up3y);
  return advF * (UU * dudx + VV * dudy) +
         difF * (((up1x + um1x) + (up1y + um1y)) - 4 * u);
}
static Real dV_adv_dif(VectorLab &V, Real uinf[2], Real advF, Real difF, int ix,
                       int iy) {
  Real u = V(ix, iy).u[0];
  Real v = V(ix, iy).u[1];
  Real UU = u + uinf[0];
  Real VV = v + uinf[1];
  Real vp1x = V(ix + 1, iy).u[1];
  Real vp2x = V(ix + 2, iy).u[1];
  Real vp3x = V(ix + 3, iy).u[1];
  Real vm1x = V(ix - 1, iy).u[1];
  Real vm2x = V(ix - 2, iy).u[1];
  Real vm3x = V(ix - 3, iy).u[1];
  Real vp1y = V(ix, iy + 1).u[1];
  Real vp2y = V(ix, iy + 2).u[1];
  Real vp3y = V(ix, iy + 3).u[1];
  Real vm1y = V(ix, iy - 1).u[1];
  Real vm2y = V(ix, iy - 2).u[1];
  Real vm3y = V(ix, iy - 3).u[1];
  Real dvdx = derivative(UU, vm3x, vm2x, vm1x, v, vp1x, vp2x, vp3x);
  Real dvdy = derivative(VV, vm3y, vm2y, vm1y, v, vp1y, vp2y, vp3y);
  return advF * (UU * dvdx + VV * dvdy) +
         difF * (((vp1x + vm1x) + (vp1y + vm1y)) - 4 * v);
}
struct KernelAdvectDiffuse {
  KernelAdvectDiffuse() {
    uinf[0] = sim.uinfx;
    uinf[1] = sim.uinfy;
  }
  Real uinf[2];
  StencilInfo stencil{-3, -3, 4, 4, true};
  std::vector<BlockInfo> &tmpVInfo = var.tmpV->infos;
  void operator()(VectorLab &lab, BlockInfo &info) {
    Real h = info.h;
    Real dfac = sim.nu * sim.dt;
    Real afac = -sim.dt * h;
    VectorBlock &TMP = *(VectorBlock *)tmpVInfo[info.id].block;
    for (int iy = 0; iy < _BS_; ++iy)
      for (int ix = 0; ix < _BS_; ++ix) {
        TMP[iy][ix].u[0] = dU_adv_dif(lab, uinf, afac, dfac, ix, iy);
        TMP[iy][ix].u[1] = dV_adv_dif(lab, uinf, afac, dfac, ix, iy);
      }
    BlockCase *tempCase = tmpVInfo[info.id].auxiliary;
    Vector *faceXm = nullptr;
    Vector *faceXp = nullptr;
    Vector *faceYm = nullptr;
    Vector *faceYp = nullptr;
    const Real aux_coef = dfac;
    if (tempCase != nullptr) {
      faceXm = (Vector *)tempCase->d[0];
      faceXp = (Vector *)tempCase->d[1];
      faceYm = (Vector *)tempCase->d[2];
      faceYp = (Vector *)tempCase->d[3];
    }
    if (faceXm != nullptr) {
      int ix = 0;
      for (int iy = 0; iy < _BS_; ++iy) {
        faceXm[iy].u[0] = aux_coef * (lab(ix, iy).u[0] - lab(ix - 1, iy).u[0]);
        faceXm[iy].u[1] = aux_coef * (lab(ix, iy).u[1] - lab(ix - 1, iy).u[1]);
      }
    }
    if (faceXp != nullptr) {
      int ix = _BS_ - 1;
      for (int iy = 0; iy < _BS_; ++iy) {
        faceXp[iy].u[0] = aux_coef * (lab(ix, iy).u[0] - lab(ix + 1, iy).u[0]);
        faceXp[iy].u[1] = aux_coef * (lab(ix, iy).u[1] - lab(ix + 1, iy).u[1]);
      }
    }
    if (faceYm != nullptr) {
      int iy = 0;
      for (int ix = 0; ix < _BS_; ++ix) {
        faceYm[ix].u[0] = aux_coef * (lab(ix, iy).u[0] - lab(ix, iy - 1).u[0]);
        faceYm[ix].u[1] = aux_coef * (lab(ix, iy).u[1] - lab(ix, iy - 1).u[1]);
      }
    }
    if (faceYp != nullptr) {
      int iy = _BS_ - 1;
      for (int ix = 0; ix < _BS_; ++ix) {
        faceYp[ix].u[0] = aux_coef * (lab(ix, iy).u[0] - lab(ix, iy + 1).u[0]);
        faceYp[ix].u[1] = aux_coef * (lab(ix, iy).u[1] - lab(ix, iy + 1).u[1]);
      }
    }
  }
};
struct KernelComputeForces {
  const int big = 5;
  const int small = -4;
  KernelComputeForces(){};
  StencilInfo stencil{small, small, big, big, true};
  StencilInfo stencil2{small, small, big, big, true};
  const int bigg = _BS_ + big - 1;
  const int stencil_start[3] = {small, small, small},
            stencil_end[3] = {big, big, big};
  const Real c0 = -137. / 60.;
  const Real c1 = 5.;
  const Real c2 = -5.;
  const Real c3 = 10. / 3.;
  const Real c4 = -5. / 4.;
  const Real c5 = 1. / 5.;
  bool inrange(const int i) const { return (i >= small && i < bigg); }
  const std::vector<BlockInfo> &presInfo = var.pres->infos;
  void operator()(VectorLab &lab, ScalarLab &chi, const BlockInfo &info,
                  const BlockInfo &info2) const {
    VectorLab &V = lab;
    ScalarBlock &P = *(ScalarBlock *)presInfo[info.id].block;
    for (const auto &shape : sim.shapes) {
      const std::vector<ObstacleBlock *> &OBLOCK = shape->obstacleBlocks;
      const Real Cx = shape->centerOfMass[0], Cy = shape->centerOfMass[1];
      const Real vel_norm =
          std::sqrt(shape->u * shape->u + shape->v * shape->v);
      const Real vel_unit[2] = {
          vel_norm > 0 ? (Real)shape->u / vel_norm : (Real)0,
          vel_norm > 0 ? (Real)shape->v / vel_norm : (Real)0};
      const Real NUoH = sim.nu / info.h;
      ObstacleBlock *const O = OBLOCK[info.id];
      if (O == nullptr)
        continue;
      assert(O->filled);
      for (size_t k = 0; k < O->n_surfPoints; ++k) {
        const int ix = O->surface[k].ix, iy = O->surface[k].iy;
        Real p[2];
        p[0] = info.origin[0] + info.h * (ix + 0.5);
        p[1] = info.origin[1] + info.h * (iy + 0.5);
        const Real normX = O->surface[k].dchidx;
        const Real normY = O->surface[k].dchidy;
        const Real norm = 1.0 / std::sqrt(normX * normX + normY * normY);
        const Real dx = normX * norm;
        const Real dy = normY * norm;
        Real DuDx;
        Real DuDy;
        Real DvDx;
        Real DvDy;
        {
          int x = ix;
          int y = iy;
          for (int kk = 0; kk < 5; kk++) {
            const int dxi = round(kk * dx);
            const int dyi = round(kk * dy);
            if (ix + dxi + 1 >= _BS_ + big - 1 || ix + dxi - 1 < small)
              continue;
            if (iy + dyi + 1 >= _BS_ + big - 1 || iy + dyi - 1 < small)
              continue;
            x = ix + dxi;
            y = iy + dyi;
            if (chi(x, y) < 0.01)
              break;
          }
          const auto &l = lab;
          const int sx = normX > 0 ? +1 : -1;
          const int sy = normY > 0 ? +1 : -1;
          Vector dveldx;
          if (inrange(x + 5 * sx))
            dveldx = sx * (c0 * l(x, y) + c1 * l(x + sx, y) +
                           c2 * l(x + 2 * sx, y) + c3 * l(x + 3 * sx, y) +
                           c4 * l(x + 4 * sx, y) + c5 * l(x + 5 * sx, y));
          else if (inrange(x + 2 * sx))
            dveldx = sx * (-1.5 * l(x, y) + 2.0 * l(x + sx, y) -
                           0.5 * l(x + 2 * sx, y));
          else
            dveldx = sx * (l(x + sx, y) - l(x, y));
          Vector dveldy;
          if (inrange(y + 5 * sy))
            dveldy = sy * (c0 * l(x, y) + c1 * l(x, y + sy) +
                           c2 * l(x, y + 2 * sy) + c3 * l(x, y + 3 * sy) +
                           c4 * l(x, y + 4 * sy) + c5 * l(x, y + 5 * sy));
          else if (inrange(y + 2 * sy))
            dveldy = sy * (-1.5 * l(x, y) + 2.0 * l(x, y + sy) -
                           0.5 * l(x, y + 2 * sy));
          else
            dveldy = sx * (l(x, y + sy) - l(x, y));
          const Vector dveldx2 = l(x - 1, y) - 2.0 * l(x, y) + l(x + 1, y);
          const Vector dveldy2 = l(x, y - 1) - 2.0 * l(x, y) + l(x, y + 1);
          Vector dveldxdy;
          if (inrange(x + 2 * sx) && inrange(y + 2 * sy))
            dveldxdy =
                sx * sy *
                (-0.5 * (-1.5 * l(x + 2 * sx, y) + 2 * l(x + 2 * sx, y + sy) -
                         0.5 * l(x + 2 * sx, y + 2 * sy)) +
                 2 * (-1.5 * l(x + sx, y) + 2 * l(x + sx, y + sy) -
                      0.5 * l(x + sx, y + 2 * sy)) -
                 1.5 * (-1.5 * l(x, y) + 2 * l(x, y + sy) -
                        0.5 * l(x, y + 2 * sy)));
          else
            dveldxdy = sx * sy * (l(x + sx, y + sy) - l(x + sx, y)) -
                       (l(x, y + sy) - l(x, y));
          DuDx =
              dveldx.u[0] + dveldx2.u[0] * (ix - x) + dveldxdy.u[0] * (iy - y);
          DvDx =
              dveldx.u[1] + dveldx2.u[1] * (ix - x) + dveldxdy.u[1] * (iy - y);
          DuDy =
              dveldy.u[0] + dveldy2.u[0] * (iy - y) + dveldxdy.u[0] * (ix - x);
          DvDy =
              dveldy.u[1] + dveldy2.u[1] * (iy - y) + dveldxdy.u[1] * (ix - x);
        }
        const Real fXV = NUoH * DuDx * normX + NUoH * DuDy * normY,
                   fXP = -P[iy][ix] * normX;
        const Real fYV = NUoH * DvDx * normX + NUoH * DvDy * normY,
                   fYP = -P[iy][ix] * normY;
        const Real fXT = fXV + fXP, fYT = fYV + fYP;
        O->x_s[k] = p[0];
        O->y_s[k] = p[1];
        O->p_s[k] = P[iy][ix];
        O->u_s[k] = V(ix, iy).u[0];
        O->v_s[k] = V(ix, iy).u[1];
        O->nx_s[k] = dx;
        O->ny_s[k] = dy;
        O->omega_s[k] = (DvDx - DuDy) / info.h;
        O->uDef_s[k] = O->udef[iy][ix][0];
        O->vDef_s[k] = O->udef[iy][ix][1];
        O->fX_s[k] = -P[iy][ix] * dx + NUoH * DuDx * dx + NUoH * DuDy * dy;
        O->fY_s[k] = -P[iy][ix] * dy + NUoH * DvDx * dx + NUoH * DvDy * dy;
        O->fXv_s[k] = NUoH * DuDx * dx + NUoH * DuDy * dy;
        O->fYv_s[k] = NUoH * DvDx * dx + NUoH * DvDy * dy;
        O->perimeter += std::sqrt(normX * normX + normY * normY);
        O->circulation += normX * O->v_s[k] - normY * O->u_s[k];
        O->forcex += fXT;
        O->forcey += fYT;
        O->forcex_V += fXV;
        O->forcey_V += fYV;
        O->forcex_P += fXP;
        O->forcey_P += fYP;
        O->torque += (p[0] - Cx) * fYT - (p[1] - Cy) * fXT;
        O->torque_P += (p[0] - Cx) * fYP - (p[1] - Cy) * fXP;
        O->torque_V += (p[0] - Cx) * fYV - (p[1] - Cy) * fXV;
        const Real forcePar = fXT * vel_unit[0] + fYT * vel_unit[1];
        O->thrust += .5 * (forcePar + std::fabs(forcePar));
        O->drag -= .5 * (forcePar - std::fabs(forcePar));
        const Real forcePerp = fXT * vel_unit[1] - fYT * vel_unit[0];
        O->lift += forcePerp;
        const Real powOut = fXT * O->u_s[k] + fYT * O->v_s[k];
        const Real powDef = fXT * O->uDef_s[k] + fYT * O->vDef_s[k];
        O->Pout += powOut;
        O->defPower += powDef;
        O->PoutBnd += std::min((Real)0, powOut);
        O->defPowerBnd += std::min((Real)0, powDef);
      }
      O->PoutNew = O->forcex * shape->u + O->forcey * shape->v;
    }
  }
};
struct PoissonSolver {
  struct EdgeCellIndexer;
  std::unique_ptr<LocalSpMatDnVec> LocalLS_;
  std::vector<long long> Nblocks_xcumsum_;
  std::vector<long long> Nrows_xcumsum_;
  PoissonSolver()
      : GenericCell(*this), XminCell(*this), XmaxCell(*this), YminCell(*this),
        YmaxCell(*this), edgeIndexers{&XminCell, &XmaxCell, &YminCell,
                                      &YmaxCell} {
    Nblocks_xcumsum_.resize(sim.size + 1);
    Nrows_xcumsum_.resize(sim.size + 1);
    std::vector<std::vector<double>> L;
    std::vector<std::vector<double>> L_inv;
    L.resize((_BS_ * _BS_));
    L_inv.resize((_BS_ * _BS_));
    for (int i(0); i < (_BS_ * _BS_); i++) {
      L[i].resize(i + 1);
      L_inv[i].resize(i + 1);
      for (int j(0); j <= i; j++) {
        L_inv[i][j] = (i == j) ? 1. : 0.;
      }
    }
    for (int i(0); i < (_BS_ * _BS_); i++) {
      double s1 = 0;
      for (int k(0); k <= i - 1; k++)
        s1 += L[i][k] * L[i][k];
      L[i][i] = sqrt(getA_local(i, i) - s1);
      for (int j(i + 1); j < (_BS_ * _BS_); j++) {
        double s2 = 0;
        for (int k(0); k <= i - 1; k++)
          s2 += L[i][k] * L[j][k];
        L[j][i] = (getA_local(j, i) - s2) / L[i][i];
      }
    }
    for (int br(0); br < (_BS_ * _BS_); br++) {
      const double bsf = 1. / L[br][br];
      for (int c(0); c <= br; c++)
        L_inv[br][c] *= bsf;
      for (int wr(br + 1); wr < (_BS_ * _BS_); wr++) {
        const double wsf = L[wr][br];
        for (int c(0); c <= br; c++)
          L_inv[wr][c] -= (wsf * L_inv[br][c]);
      }
    }
    std::vector<double> P_inv((_BS_ * _BS_) * (_BS_ * _BS_));
    for (int i(0); i < (_BS_ * _BS_); i++)
      for (int j(0); j < (_BS_ * _BS_); j++) {
        double aux = 0.;
        for (int k(0); k < (_BS_ * _BS_); k++)
          aux += (i <= k && j <= k) ? L_inv[k][i] * L_inv[k][j] : 0.;
        P_inv[i * (_BS_ * _BS_) + j] = -aux;
      }
    LocalLS_ = std::make_unique<LocalSpMatDnVec>(MPI_COMM_WORLD, _BS_ * _BS_,
                                                 sim.bMeanConstraint, P_inv);
  }
  void solve(const Grid *input) {
    const double max_error = sim.step < 10 ? 0.0 : sim.PoissonTol;
    const double max_rel_error = sim.step < 10 ? 0.0 : sim.PoissonTolRel;
    const int max_restarts = sim.step < 10 ? 100 : sim.maxPoissonRestarts;
    if (var.pres->UpdateFluxCorrection) {
      var.pres->UpdateFluxCorrection = false;
      this->getMat();
      this->getVec();
      LocalLS_->solveWithUpdate(max_error, max_rel_error, max_restarts);
    } else {
      this->getVec();
      LocalLS_->solveNoUpdate(max_error, max_rel_error, max_restarts);
    }
    std::vector<BlockInfo> &zInfo = var.pres->infos;
    const int Nblocks = zInfo.size();
    const std::vector<double> &x = LocalLS_->get_x();
    double avg = 0;
    double avg1 = 0;
#pragma omp parallel for reduction(+ : avg, avg1)
    for (int i = 0; i < Nblocks; i++) {
      ScalarBlock &P = *(ScalarBlock *)zInfo[i].block;
      const double vv = zInfo[i].h * zInfo[i].h;
      for (int iy = 0; iy < _BS_; iy++)
        for (int ix = 0; ix < _BS_; ix++) {
          P[iy][ix] = x[i * _BS_ * _BS_ + iy * _BS_ + ix];
          avg += P[iy][ix] * vv;
          avg1 += vv;
        }
    }
    double quantities[2] = {avg, avg1};
    MPI_Allreduce(MPI_IN_PLACE, &quantities, 2, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    avg = quantities[0];
    avg1 = quantities[1];
    avg = avg / avg1;
#pragma omp parallel for
    for (int i = 0; i < Nblocks; i++) {
      ScalarBlock &P = *(ScalarBlock *)zInfo[i].block;
      for (int iy = 0; iy < _BS_; iy++)
        for (int ix = 0; ix < _BS_; ix++)
          P[iy][ix] += -avg;
    }
  }
  struct CellIndexer {
    CellIndexer(const PoissonSolver &pSolver) : ps(pSolver) {}
    ~CellIndexer() = default;
    long long This(const BlockInfo &info, const int ix, const int iy) const {
      return blockOffset(info) + (long long)(iy * _BS_ + ix);
    }
    static bool validXm(const int ix, const int iy) { return ix > 0; }
    static bool validXp(const int ix, const int iy) { return ix < _BS_ - 1; }
    static bool validYm(const int ix, const int iy) { return iy > 0; }
    static bool validYp(const int ix, const int iy) { return iy < _BS_ - 1; }
    long long Xmin(const BlockInfo &info, const int ix, const int iy,
                   const int offset = 0) const {
      return blockOffset(info) + (long long)(iy * _BS_ + offset);
    }
    long long Xmax(const BlockInfo &info, const int ix, const int iy,
                   const int offset = 0) const {
      return blockOffset(info) + (long long)(iy * _BS_ + (_BS_ - 1 - offset));
    }
    long long Ymin(const BlockInfo &info, const int ix, const int iy,
                   const int offset = 0) const {
      return blockOffset(info) + (long long)(offset * _BS_ + ix);
    }
    long long Ymax(const BlockInfo &info, const int ix, const int iy,
                   const int offset = 0) const {
      return blockOffset(info) + (long long)((_BS_ - 1 - offset) * _BS_ + ix);
    }
    long long blockOffset(const BlockInfo &info) const {
      return (info.id + ps.Nblocks_xcumsum_[var.tmp->Tree1(info)]) *
             (_BS_ * _BS_);
    }
    static int ix_f(const int ix) { return (ix % (_BS_ / 2)) * 2; }
    static int iy_f(const int iy) { return (iy % (_BS_ / 2)) * 2; }
    const PoissonSolver &ps;
  };
  struct EdgeCellIndexer : public CellIndexer {
    EdgeCellIndexer(const PoissonSolver &pSolver) : CellIndexer(pSolver) {}
    virtual long long neiUnif(const BlockInfo &nei_info, const int ix,
                              const int iy) const = 0;
    virtual long long neiInward(const BlockInfo &info, const int ix,
                                const int iy) const = 0;
    virtual double taylorSign(const int ix, const int iy) const = 0;
    virtual int ix_c(const BlockInfo &info, const int ix) const {
      return info.index[0] % 2 == 0 ? ix / 2 : ix / 2 + _BS_ / 2;
    }
    virtual int iy_c(const BlockInfo &info, const int iy) const {
      return info.index[1] % 2 == 0 ? iy / 2 : iy / 2 + _BS_ / 2;
    }
    virtual long long neiFine1(const BlockInfo &nei_info, const int ix,
                               const int iy, const int offset = 0) const = 0;
    virtual long long neiFine2(const BlockInfo &nei_info, const int ix,
                               const int iy, const int offset = 0) const = 0;
    virtual bool isBD(const int ix, const int iy) const = 0;
    virtual bool isFD(const int ix, const int iy) const = 0;
    virtual long long Nei(const BlockInfo &info, const int ix, const int iy,
                          const int dist) const = 0;
    virtual long long Zchild(const BlockInfo &nei_info, const int ix,
                             const int iy) const = 0;
  };
  struct XbaseIndexer : public EdgeCellIndexer {
    XbaseIndexer(const PoissonSolver &pSolver) : EdgeCellIndexer(pSolver) {}
    double taylorSign(const int ix, const int iy) const override {
      return iy % 2 == 0 ? -1. : 1.;
    }
    bool isBD(const int ix, const int iy) const override {
      return iy == _BS_ - 1 || iy == _BS_ / 2 - 1;
    }
    bool isFD(const int ix, const int iy) const override {
      return iy == 0 || iy == _BS_ / 2;
    }
    long long Nei(const BlockInfo &info, const int ix, const int iy,
                  const int dist) const override {
      return This(info, ix, iy + dist);
    }
  };
  struct XminIndexer : public XbaseIndexer {
    XminIndexer(const PoissonSolver &pSolver) : XbaseIndexer(pSolver) {}
    long long neiUnif(const BlockInfo &nei_info, const int ix,
                      const int iy) const override {
      return Xmax(nei_info, ix, iy);
    }
    long long neiInward(const BlockInfo &info, const int ix,
                        const int iy) const override {
      return This(info, ix + 1, iy);
    }
    int ix_c(const BlockInfo &info, const int ix) const override {
      return _BS_ - 1;
    }
    long long neiFine1(const BlockInfo &nei_info, const int ix, const int iy,
                       const int offset = 0) const override {
      return Xmax(nei_info, ix_f(ix), iy_f(iy), offset);
    }
    long long neiFine2(const BlockInfo &nei_info, const int ix, const int iy,
                       const int offset = 0) const override {
      return Xmax(nei_info, ix_f(ix), iy_f(iy) + 1, offset);
    }
    long long Zchild(const BlockInfo &nei_info, const int ix,
                     const int iy) const override {
      return nei_info.Zchild[1][int(iy >= _BS_ / 2)];
    }
  };
  struct XmaxIndexer : public XbaseIndexer {
    XmaxIndexer(const PoissonSolver &pSolver) : XbaseIndexer(pSolver) {}
    long long neiUnif(const BlockInfo &nei_info, const int ix,
                      const int iy) const override {
      return Xmin(nei_info, ix, iy);
    }
    long long neiInward(const BlockInfo &info, const int ix,
                        const int iy) const override {
      return This(info, ix - 1, iy);
    }
    int ix_c(const BlockInfo &info, const int ix) const override { return 0; }
    long long neiFine1(const BlockInfo &nei_info, const int ix, const int iy,
                       const int offset = 0) const override {
      return Xmin(nei_info, ix_f(ix), iy_f(iy), offset);
    }
    long long neiFine2(const BlockInfo &nei_info, const int ix, const int iy,
                       const int offset = 0) const override {
      return Xmin(nei_info, ix_f(ix), iy_f(iy) + 1, offset);
    }
    long long Zchild(const BlockInfo &nei_info, const int ix,
                     const int iy) const override {
      return nei_info.Zchild[0][int(iy >= _BS_ / 2)];
    }
  };
  struct YbaseIndexer : public EdgeCellIndexer {
    YbaseIndexer(const PoissonSolver &pSolver) : EdgeCellIndexer(pSolver) {}
    double taylorSign(const int ix, const int iy) const override {
      return ix % 2 == 0 ? -1. : 1.;
    }
    bool isBD(const int ix, const int iy) const override {
      return ix == _BS_ - 1 || ix == _BS_ / 2 - 1;
    }
    bool isFD(const int ix, const int iy) const override {
      return ix == 0 || ix == _BS_ / 2;
    }
    long long Nei(const BlockInfo &info, const int ix, const int iy,
                  const int dist) const override {
      return This(info, ix + dist, iy);
    }
  };
  struct YminIndexer : public YbaseIndexer {
    YminIndexer(const PoissonSolver &pSolver) : YbaseIndexer(pSolver) {}
    long long neiUnif(const BlockInfo &nei_info, const int ix,
                      const int iy) const override {
      return Ymax(nei_info, ix, iy);
    }
    long long neiInward(const BlockInfo &info, const int ix,
                        const int iy) const override {
      return This(info, ix, iy + 1);
    }
    int iy_c(const BlockInfo &info, const int iy) const override {
      return _BS_ - 1;
    }
    long long neiFine1(const BlockInfo &nei_info, const int ix, const int iy,
                       const int offset = 0) const override {
      return Ymax(nei_info, ix_f(ix), iy_f(iy), offset);
    }
    long long neiFine2(const BlockInfo &nei_info, const int ix, const int iy,
                       const int offset = 0) const override {
      return Ymax(nei_info, ix_f(ix) + 1, iy_f(iy), offset);
    }
    long long Zchild(const BlockInfo &nei_info, const int ix,
                     const int iy) const override {
      return nei_info.Zchild[int(ix >= _BS_ / 2)][1];
    }
  };
  struct YmaxIndexer : public YbaseIndexer {
    YmaxIndexer(const PoissonSolver &pSolver) : YbaseIndexer(pSolver) {}
    long long neiUnif(const BlockInfo &nei_info, const int ix,
                      const int iy) const override {
      return Ymin(nei_info, ix, iy);
    }
    long long neiInward(const BlockInfo &info, const int ix,
                        const int iy) const override {
      return This(info, ix, iy - 1);
    }
    int iy_c(const BlockInfo &info, const int iy) const override { return 0; }
    long long neiFine1(const BlockInfo &nei_info, const int ix, const int iy,
                       const int offset = 0) const override {
      return Ymin(nei_info, ix_f(ix), iy_f(iy), offset);
    }
    long long neiFine2(const BlockInfo &nei_info, const int ix, const int iy,
                       const int offset = 0) const override {
      return Ymin(nei_info, ix_f(ix) + 1, iy_f(iy), offset);
    }
    long long Zchild(const BlockInfo &nei_info, const int ix,
                     const int iy) const override {
      return nei_info.Zchild[int(ix >= _BS_ / 2)][0];
    }
  };
  CellIndexer GenericCell;
  XminIndexer XminCell;
  XmaxIndexer XmaxCell;
  YminIndexer YminCell;
  YmaxIndexer YmaxCell;
  std::array<const EdgeCellIndexer *, 4> edgeIndexers;
  std::array<std::pair<long long, double>, 3> D1(const BlockInfo &info,
                                                 const EdgeCellIndexer &indexer,
                                                 const int ix,
                                                 const int iy) const {
    if (indexer.isBD(ix, iy))
      return {{{indexer.Nei(info, ix, iy, -2), 1. / 8.},
               {indexer.Nei(info, ix, iy, -1), -1. / 2.},
               {indexer.This(info, ix, iy), 3. / 8.}}};
    else if (indexer.isFD(ix, iy))
      return {{{indexer.Nei(info, ix, iy, 2), -1. / 8.},
               {indexer.Nei(info, ix, iy, 1), 1. / 2.},
               {indexer.This(info, ix, iy), -3. / 8.}}};
    return {{{indexer.Nei(info, ix, iy, -1), -1. / 8.},
             {indexer.Nei(info, ix, iy, 1), 1. / 8.},
             {indexer.This(info, ix, iy), 0.}}};
  }
  std::array<std::pair<long long, double>, 3> D2(const BlockInfo &info,
                                                 const EdgeCellIndexer &indexer,
                                                 const int ix,
                                                 const int iy) const {
    if (indexer.isBD(ix, iy))
      return {{{indexer.Nei(info, ix, iy, -2), 1. / 32.},
               {indexer.Nei(info, ix, iy, -1), -1. / 16.},
               {indexer.This(info, ix, iy), 1. / 32.}}};
    else if (indexer.isFD(ix, iy))
      return {{{indexer.Nei(info, ix, iy, 2), 1. / 32.},
               {indexer.Nei(info, ix, iy, 1), -1. / 16.},
               {indexer.This(info, ix, iy), 1. / 32.}}};
    return {{{indexer.Nei(info, ix, iy, -1), 1. / 32.},
             {indexer.Nei(info, ix, iy, 1), 1. / 32.},
             {indexer.This(info, ix, iy), -1. / 16.}}};
  }
  void interpolate(const BlockInfo &info_c, const int ix_c, const int iy_c,
                   const BlockInfo &info_f, const long long fine_close_idx,
                   const long long fine_far_idx, const double signInt,
                   const double signTaylor, const EdgeCellIndexer &indexer,
                   SpRowInfo &row) const {
    const int rank_c = var.tmp->Tree1(info_c);
    const int rank_f = var.tmp->Tree1(info_f);
    row.mapColVal(rank_f, fine_close_idx, signInt * 2. / 3.);
    row.mapColVal(rank_f, fine_far_idx, -signInt * 1. / 5.);
    const double tf = signInt * 8. / 15.;
    row.mapColVal(rank_c, indexer.This(info_c, ix_c, iy_c), tf);
    std::array<std::pair<long long, double>, 3> D;
    D = D1(info_c, indexer, ix_c, iy_c);
    for (int i(0); i < 3; i++)
      row.mapColVal(rank_c, D[i].first, signTaylor * tf * D[i].second);
    D = D2(info_c, indexer, ix_c, iy_c);
    for (int i(0); i < 3; i++)
      row.mapColVal(rank_c, D[i].first, tf * D[i].second);
  }
  void makeFlux(const BlockInfo &rhs_info, const int ix, const int iy,
                const BlockInfo &rhsNei, const EdgeCellIndexer &indexer,
                SpRowInfo &row) const {
    const long long sfc_idx = indexer.This(rhs_info, ix, iy);
    if (var.tmp->Tree1(rhsNei) >= 0) {
      const int nei_rank = var.tmp->Tree1(rhsNei);
      const long long nei_idx = indexer.neiUnif(rhsNei, ix, iy);
      row.mapColVal(nei_rank, nei_idx, 1.);
      row.mapColVal(sfc_idx, -1.);
    } else if (var.tmp->Tree1(rhsNei) == -2) {
      const BlockInfo &rhsNei_c =
          var.tmp->get(rhs_info.level - 1, rhsNei.Zparent);
      const int ix_c = indexer.ix_c(rhs_info, ix);
      const int iy_c = indexer.iy_c(rhs_info, iy);
      const long long inward_idx = indexer.neiInward(rhs_info, ix, iy);
      const double signTaylor = indexer.taylorSign(ix, iy);
      interpolate(rhsNei_c, ix_c, iy_c, rhs_info, sfc_idx, inward_idx, 1.,
                  signTaylor, indexer, row);
      row.mapColVal(sfc_idx, -1.);
    } else if (var.tmp->Tree1(rhsNei) == -1) {
      const BlockInfo &rhsNei_f =
          var.tmp->get(rhs_info.level + 1, indexer.Zchild(rhsNei, ix, iy));
      const int nei_rank = var.tmp->Tree1(rhsNei_f);
      long long fine_close_idx = indexer.neiFine1(rhsNei_f, ix, iy, 0);
      long long fine_far_idx = indexer.neiFine1(rhsNei_f, ix, iy, 1);
      row.mapColVal(nei_rank, fine_close_idx, 1.);
      interpolate(rhs_info, ix, iy, rhsNei_f, fine_close_idx, fine_far_idx, -1.,
                  -1., indexer, row);
      fine_close_idx = indexer.neiFine2(rhsNei_f, ix, iy, 0);
      fine_far_idx = indexer.neiFine2(rhsNei_f, ix, iy, 1);
      row.mapColVal(nei_rank, fine_close_idx, 1.);
      interpolate(rhs_info, ix, iy, rhsNei_f, fine_close_idx, fine_far_idx, -1.,
                  1., indexer, row);
    } else {
      throw std::runtime_error(
          "Neighbour doesn't exist, isn't coarser, nor finer...");
    }
  }
  void getMat() {
    var.tmp->UpdateBlockInfoAll_States(true);
    std::vector<BlockInfo> &RhsInfo = var.tmp->infos;
    const int Nblocks = RhsInfo.size();
    const int N = _BS_ * _BS_ * Nblocks;
    LocalLS_->reserve(N);
    const long long Nblocks_long = Nblocks;
    MPI_Allgather(&Nblocks_long, 1, MPI_LONG_LONG, Nblocks_xcumsum_.data(), 1,
                  MPI_LONG_LONG, MPI_COMM_WORLD);
    for (int i(Nblocks_xcumsum_.size() - 1); i > 0; i--) {
      Nblocks_xcumsum_[i] = Nblocks_xcumsum_[i - 1];
    }
    Nblocks_xcumsum_[0] = 0;
    Nrows_xcumsum_[0] = 0;
    for (size_t i(1); i < Nblocks_xcumsum_.size(); i++) {
      Nblocks_xcumsum_[i] += Nblocks_xcumsum_[i - 1];
      Nrows_xcumsum_[i] = (_BS_ * _BS_) * Nblocks_xcumsum_[i];
    }
    for (int i = 0; i < Nblocks; i++) {
      const BlockInfo &rhs_info = RhsInfo[i];
      const int aux = 1 << rhs_info.level;
      const int MAX_X_BLOCKS = sim.bpdx * aux - 1;
      const int MAX_Y_BLOCKS = sim.bpdy * aux - 1;
      std::array<bool, 4> isBoundary;
      isBoundary[0] = (rhs_info.index[0] == 0);
      isBoundary[1] = (rhs_info.index[0] == MAX_X_BLOCKS);
      isBoundary[2] = (rhs_info.index[1] == 0);
      isBoundary[3] = (rhs_info.index[1] == MAX_Y_BLOCKS);
      std::array<bool, 2> isPeriodic;
      isPeriodic[0] = (sim.bcx == periodic);
      isPeriodic[1] = (sim.bcy == periodic);
      std::array<const BlockInfo *, 4> rhsNei;
      rhsNei[0] = &(var.tmp->get(rhs_info.level, rhs_info.Znei[1 - 1][1]));
      rhsNei[1] = &(var.tmp->get(rhs_info.level, rhs_info.Znei[1 + 1][1]));
      rhsNei[2] = &(var.tmp->get(rhs_info.level, rhs_info.Znei[1][1 - 1]));
      rhsNei[3] = &(var.tmp->get(rhs_info.level, rhs_info.Znei[1][1 + 1]));
      for (int iy = 0; iy < _BS_; iy++)
        for (int ix = 0; ix < _BS_; ix++) {
          const long long sfc_idx = GenericCell.This(rhs_info, ix, iy);
          if ((ix > 0 && ix < _BS_ - 1) && (iy > 0 && iy < _BS_ - 1)) {
            LocalLS_->cooPushBackVal(1, sfc_idx,
                                     GenericCell.This(rhs_info, ix, iy - 1));
            LocalLS_->cooPushBackVal(1, sfc_idx,
                                     GenericCell.This(rhs_info, ix - 1, iy));
            LocalLS_->cooPushBackVal(-4, sfc_idx, sfc_idx);
            LocalLS_->cooPushBackVal(1, sfc_idx,
                                     GenericCell.This(rhs_info, ix + 1, iy));
            LocalLS_->cooPushBackVal(1, sfc_idx,
                                     GenericCell.This(rhs_info, ix, iy + 1));
          } else {
            std::array<bool, 4> validNei;
            validNei[0] = GenericCell.validXm(ix, iy);
            validNei[1] = GenericCell.validXp(ix, iy);
            validNei[2] = GenericCell.validYm(ix, iy);
            validNei[3] = GenericCell.validYp(ix, iy);
            std::array<long long, 4> idxNei;
            idxNei[0] = GenericCell.This(rhs_info, ix - 1, iy);
            idxNei[1] = GenericCell.This(rhs_info, ix + 1, iy);
            idxNei[2] = GenericCell.This(rhs_info, ix, iy - 1);
            idxNei[3] = GenericCell.This(rhs_info, ix, iy + 1);
            SpRowInfo row(var.tmp->Tree1(rhs_info), sfc_idx, 8);
            for (int j(0); j < 4; j++) {
              if (validNei[j]) {
                row.mapColVal(idxNei[j], 1);
                row.mapColVal(sfc_idx, -1);
              } else if (!isBoundary[j] || (isBoundary[j] && isPeriodic[j / 2]))
                this->makeFlux(rhs_info, ix, iy, *rhsNei[j], *edgeIndexers[j],
                               row);
            }
            LocalLS_->cooPushBackRow(row);
          }
        }
    }
    LocalLS_->make(Nrows_xcumsum_);
  }
  void getVec() {
    std::vector<BlockInfo> &RhsInfo = var.tmp->infos;
    std::vector<BlockInfo> &zInfo = var.pres->infos;
    const int Nblocks = RhsInfo.size();
    std::vector<double> &x = LocalLS_->get_x();
    std::vector<double> &b = LocalLS_->get_b();
    std::vector<double> &h2 = LocalLS_->get_h2();
    const long long shift = -Nrows_xcumsum_[sim.rank];
#pragma omp parallel for
    for (int i = 0; i < Nblocks; i++) {
      const BlockInfo &rhs_info = RhsInfo[i];
      const ScalarBlock &rhs = *(ScalarBlock *)RhsInfo[i].block;
      const ScalarBlock &p = *(ScalarBlock *)zInfo[i].block;
      h2[i] = RhsInfo[i].h * RhsInfo[i].h;
      for (int iy = 0; iy < _BS_; iy++)
        for (int ix = 0; ix < _BS_; ix++) {
          const long long sfc_loc = GenericCell.This(rhs_info, ix, iy) + shift;
          if (sim.bMeanConstraint && rhs_info.index[0] == 0 &&
              rhs_info.index[1] == 0 && rhs_info.index[2] == 0 && ix == 0 &&
              iy == 0)
            b[sfc_loc] = 0.;
          else
            b[sfc_loc] = rhs[iy][ix];
          x[sfc_loc] = p[iy][ix];
        }
    }
  }
};
struct pressureCorrectionKernel {
  pressureCorrectionKernel(){};
  const StencilInfo stencil{-1, -1, 2, 2, false};
  const std::vector<BlockInfo> &tmpVInfo = var.tmpV->infos;
  void operator()(ScalarLab &P, const BlockInfo &info) const {
    const Real h = info.h, pFac = -0.5 * sim.dt * h;
    VectorBlock &tmpV = *(VectorBlock *)tmpVInfo[info.id].block;
    for (int iy = 0; iy < _BS_; ++iy)
      for (int ix = 0; ix < _BS_; ++ix) {
        tmpV[iy][ix].u[0] = pFac * (P(ix + 1, iy) - P(ix - 1, iy));
        tmpV[iy][ix].u[1] = pFac * (P(ix, iy + 1) - P(ix, iy - 1));
      }
    BlockCase *tempCase = tmpVInfo[info.id].auxiliary;
    Vector *faceXm = nullptr;
    Vector *faceXp = nullptr;
    Vector *faceYm = nullptr;
    Vector *faceYp = nullptr;
    if (tempCase != nullptr) {
      faceXm = (Vector *)tempCase->d[0];
      faceXp = (Vector *)tempCase->d[1];
      faceYm = (Vector *)tempCase->d[2];
      faceYp = (Vector *)tempCase->d[3];
    }
    if (faceXm != nullptr) {
      int ix = 0;
      for (int iy = 0; iy < _BS_; ++iy) {
        faceXm[iy].u[0] = pFac * (P(ix - 1, iy) + P(ix, iy));
        faceXm[iy].u[1] = 0;
      }
    }
    if (faceXp != nullptr) {
      int ix = _BS_ - 1;
      for (int iy = 0; iy < _BS_; ++iy) {
        faceXp[iy].u[0] = -pFac * (P(ix + 1, iy) + P(ix, iy));
        faceXp[iy].u[1] = 0;
      }
    }
    if (faceYm != nullptr) {
      int iy = 0;
      for (int ix = 0; ix < _BS_; ++ix) {
        faceYm[ix].u[0] = 0;
        faceYm[ix].u[1] = pFac * (P(ix, iy - 1) + P(ix, iy));
      }
    }
    if (faceYp != nullptr) {
      int iy = _BS_ - 1;
      for (int ix = 0; ix < _BS_; ++ix) {
        faceYp[ix].u[0] = 0;
        faceYp[ix].u[1] = -pFac * (P(ix, iy + 1) + P(ix, iy));
      }
    }
  }
};
struct pressure_rhs {
  pressure_rhs(){};
  StencilInfo stencil{-1, -1, 2, 2, false};
  StencilInfo stencil2{-1, -1, 2, 2, false};
  const std::vector<BlockInfo> &tmpInfo = var.tmp->infos;
  const std::vector<BlockInfo> &chiInfo = var.chi->infos;
  void operator()(VectorLab &velLab, VectorLab &uDefLab, const BlockInfo &info,
                  const BlockInfo &) const {
    const Real h = info.h;
    const Real facDiv = 0.5 * h / sim.dt;
    ScalarBlock &TMP = *(ScalarBlock *)tmpInfo[info.id].block;
    ScalarBlock &CHI = *(ScalarBlock *)chiInfo[info.id].block;
    for (int iy = 0; iy < _BS_; ++iy)
      for (int ix = 0; ix < _BS_; ++ix) {
        TMP[iy][ix] =
            facDiv * ((velLab(ix + 1, iy).u[0] - velLab(ix - 1, iy).u[0]) +
                      (velLab(ix, iy + 1).u[1] - velLab(ix, iy - 1).u[1]));
        TMP[iy][ix] += -facDiv * CHI[iy][ix] *
                       ((uDefLab(ix + 1, iy).u[0] - uDefLab(ix - 1, iy).u[0]) +
                        (uDefLab(ix, iy + 1).u[1] - uDefLab(ix, iy - 1).u[1]));
      }
    BlockCase *tempCase = (BlockCase *)(tmpInfo[info.id].auxiliary);
    Real *faceXm = nullptr;
    Real *faceXp = nullptr;
    Real *faceYm = nullptr;
    Real *faceYp = nullptr;
    if (tempCase != nullptr) {
      faceXm = (Real *)tempCase->d[0];
      faceXp = (Real *)tempCase->d[1];
      faceYm = (Real *)tempCase->d[2];
      faceYp = (Real *)tempCase->d[3];
    }
    if (faceXm != nullptr) {
      int ix = 0;
      for (int iy = 0; iy < _BS_; ++iy) {
        faceXm[iy] = facDiv * (velLab(ix - 1, iy).u[0] + velLab(ix, iy).u[0]);
        faceXm[iy] += -(facDiv * CHI[iy][ix]) *
                      (uDefLab(ix - 1, iy).u[0] + uDefLab(ix, iy).u[0]);
      }
    }
    if (faceXp != nullptr) {
      int ix = _BS_ - 1;
      for (int iy = 0; iy < _BS_; ++iy) {
        faceXp[iy] = -facDiv * (velLab(ix + 1, iy).u[0] + velLab(ix, iy).u[0]);
        faceXp[iy] -= -(facDiv * CHI[iy][ix]) *
                      (uDefLab(ix + 1, iy).u[0] + uDefLab(ix, iy).u[0]);
      }
    }
    if (faceYm != nullptr) {
      int iy = 0;
      for (int ix = 0; ix < _BS_; ++ix) {
        faceYm[ix] = facDiv * (velLab(ix, iy - 1).u[1] + velLab(ix, iy).u[1]);
        faceYm[ix] += -(facDiv * CHI[iy][ix]) *
                      (uDefLab(ix, iy - 1).u[1] + uDefLab(ix, iy).u[1]);
      }
    }
    if (faceYp != nullptr) {
      int iy = _BS_ - 1;
      for (int ix = 0; ix < _BS_; ++ix) {
        faceYp[ix] = -facDiv * (velLab(ix, iy + 1).u[1] + velLab(ix, iy).u[1]);
        faceYp[ix] -= -(facDiv * CHI[iy][ix]) *
                      (uDefLab(ix, iy + 1).u[1] + uDefLab(ix, iy).u[1]);
      }
    }
  }
};
struct pressure_rhs1 {
  pressure_rhs1() {}
  StencilInfo stencil{-1, -1, 2, 2, false};
  void operator()(ScalarLab &lab, const BlockInfo &info) const {
    ScalarBlock &TMP = *(ScalarBlock *)var.tmp->infos[info.id].block;
    for (int iy = 0; iy < _BS_; ++iy)
      for (int ix = 0; ix < _BS_; ++ix)
        TMP[iy][ix] -= (((lab(ix - 1, iy) + lab(ix + 1, iy)) +
                         (lab(ix, iy - 1) + lab(ix, iy + 1))) -
                        4.0 * lab(ix, iy));
    BlockCase *tempCase = (BlockCase *)(var.tmp->infos[info.id].auxiliary);
    Real *faceXm = nullptr;
    Real *faceXp = nullptr;
    Real *faceYm = nullptr;
    Real *faceYp = nullptr;
    if (tempCase != nullptr) {
      faceXm = (Real *)tempCase->d[0];
      faceXp = (Real *)tempCase->d[1];
      faceYm = (Real *)tempCase->d[2];
      faceYp = (Real *)tempCase->d[3];
    }
    if (faceXm != nullptr) {
      int ix = 0;
      for (int iy = 0; iy < _BS_; ++iy)
        faceXm[iy] = lab(ix - 1, iy) - lab(ix, iy);
    }
    if (faceXp != nullptr) {
      int ix = _BS_ - 1;
      for (int iy = 0; iy < _BS_; ++iy)
        faceXp[iy] = lab(ix + 1, iy) - lab(ix, iy);
    }
    if (faceYm != nullptr) {
      int iy = 0;
      for (int ix = 0; ix < _BS_; ++ix)
        faceYm[ix] = lab(ix, iy - 1) - lab(ix, iy);
    }
    if (faceYp != nullptr) {
      int iy = _BS_ - 1;
      for (int ix = 0; ix < _BS_; ++ix)
        faceYp[ix] = lab(ix, iy + 1) - lab(ix, iy);
    }
  }
};
struct FactoryFileLineParser : public CommandlineParser {
  std::string &ltrim(std::string &s) {
    s.erase(s.begin(),
            std::find_if(s.begin(), s.end(),
                         std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
  }
  std::string &rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(),
                         std::not1(std::ptr_fun<int, int>(std::isspace)))
                .base(),
            s.end());
    return s;
  }
  std::string &trim(std::string &s) { return ltrim(rtrim(s)); }
  FactoryFileLineParser(std::istringstream &is_line)
      : CommandlineParser(0, NULL) {
    std::string key, value;
    while (std::getline(is_line, key, '=')) {
      if (std::getline(is_line, value, ' ')) {
        mapArguments[trim(key)] = Value(trim(value));
      }
    }
  }
};
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  CommandlineParser parser(argc, argv);
  MPI_Comm_size(MPI_COMM_WORLD, &sim.size);
  MPI_Comm_rank(MPI_COMM_WORLD, &sim.rank);
  if (sim.rank == 0)
    fprintf(stderr, "main.cpp: %d ranks\n", sim.size);
#ifdef _OPENMP
#pragma omp parallel
  {
#pragma omp master
    if (sim.rank == 0)
      fprintf(stderr, "main.cpp: %d threads\n", omp_get_num_threads());
  }
#endif
  parser.set_strict_mode();
  sim.bpdx = parser("-bpdx").asInt();
  sim.bpdy = parser("-bpdy").asInt();
  sim.levelMax = parser("-levelMax").asInt();
  sim.Rtol = parser("-Rtol").asDouble();
  sim.Ctol = parser("-Ctol").asDouble();
  parser.unset_strict_mode();
  sim.AdaptSteps = parser("-AdaptSteps").asInt(20);
  sim.bAdaptChiGradient = parser("-bAdaptChiGradient").asInt(1);
  sim.levelStart = parser("-levelStart").asInt(-1);
  Real extent = parser("-extent").asDouble(1);
  sim.dt = parser("-dt").asDouble(0);
  sim.CFL = parser("-CFL").asDouble(0.2);
  sim.nsteps = parser("-nsteps").asInt(0);
  sim.endTime = parser("-tend").asDouble(0);
  sim.lambda = parser("-lambda").asDouble(1e7);
  sim.dlm = parser("-dlm").asDouble(0);
  sim.nu = parser("-nu").asDouble(1e-2);
  std::string BC_x = parser("-BC_x").asString("freespace");
  std::string BC_y = parser("-BC_y").asString("freespace");
  sim.bcx = bc_flag(BC_x.c_str());
  sim.bcy = bc_flag(BC_x.c_str());
  sim.PoissonTol = parser("-poissonTol").asDouble(1e-6);
  sim.PoissonTolRel = parser("-poissonTolRel").asDouble(0);
  sim.maxPoissonRestarts = parser("-maxPoissonRestarts").asInt(30);
  sim.maxPoissonIterations = parser("-maxPoissonIterations").asInt(1000);
  sim.bMeanConstraint = parser("-bMeanConstraint").asInt(0);
  sim.dumpFreq = parser("-fdump").asInt(0);
  sim.dumpTime = parser("-tdump").asDouble(0);
  sim.h0 = extent / std::max(sim.bpdx, sim.bpdy) / _BS_;
  sim.extents[0] = sim.bpdx * sim.h0 * _BS_;
  sim.extents[1] = sim.bpdy * sim.h0 * _BS_;
  sim.minH = sim.h0 / (1 << (sim.levelMax - 1));
  sim.space_curve = new SpaceCurve(sim.bpdx, sim.bpdy);
  for (int i = 0; i < sizeof var.F / sizeof *var.F; i++) {
    Grid *g = *var.F[i].g = new Grid(var.F[i].dim);
    g->level_base.push_back(sim.bpdx * sim.bpdy * 2);
    for (int m = 1; m < sim.levelMax; m++)
      g->level_base.push_back(g->level_base[m - 1] + sim.bpdx * sim.bpdy * 1
                              << (m + 1));
    const long long total_blocks =
        sim.bpdx * sim.bpdy * pow(pow(2, sim.levelStart), 2);
    long long my_blocks = total_blocks / sim.size;
    if ((long long)sim.rank < total_blocks % sim.size)
      my_blocks++;
    long long n_start = sim.rank * (total_blocks / sim.size);
    if (total_blocks % sim.size > 0) {
      if ((long long)sim.rank < total_blocks % sim.size)
        n_start += sim.rank;
      else
        n_start += total_blocks % sim.size;
    }
    for (size_t i = 0; i < my_blocks; i++) {
      BlockInfo &new_info = g->get(sim.levelStart, n_start + i);
      new_info.block = malloc(g->dim * _BS_ * _BS_ * sizeof(Real));
      g->infos.push_back(new_info);
      g->Octree[g->level_base[sim.levelStart] + n_start + i] = sim.rank;
      int p[2];
      sim.space_curve->inverse(n_start + i, sim.levelStart, p[0], p[1]);
      if (sim.levelStart < sim.levelMax - 1)
        for (int j1 = 0; j1 < 2; j1++)
          for (int i1 = 0; i1 < 2; i1++) {
            const long long nc =
                getZforward(sim.levelStart + 1, 2 * p[0] + i1, 2 * p[1] + j1);
            g->Octree[g->level_base[sim.levelStart + 1] + nc] = -2;
          }
      if (sim.levelStart > 0) {
        const long long nf =
            getZforward(sim.levelStart - 1, p[0] / 2, p[1] / 2);
        g->Octree[g->level_base[sim.levelStart - 1] + nf] = -1;
      }
    }
    g->FillPos();
    g->UpdateFluxCorrection = true;
    g->UpdateBlockInfoAll_States(false);
    for (auto it = g->Synchronizers.begin(); it != g->Synchronizers.end(); ++it)
      (*it->second)._Setup();
    MPI_Barrier(MPI_COMM_WORLD);
    g->timestamp = 0;
  }
  std::vector<BlockInfo> &velInfo = var.vel->infos;
  std::string shapeArg = parser("-shapes").asString("");
  std::stringstream descriptors(shapeArg);
  std::string lines;
  while (std::getline(descriptors, lines)) {
    std::replace(lines.begin(), lines.end(), '_', ' ');
    std::stringstream ss(lines);
    std::string line;
    while (std::getline(ss, line, ',')) {
      std::istringstream line_stream(line);
      std::string objectName;
      line_stream >> objectName;
      if (objectName.empty() or objectName[0] == '#')
        continue;
      FactoryFileLineParser p(line_stream);
      Real center[2] = {p("-xpos").asDouble(.5 * sim.extents[0]),
                        p("-ypos").asDouble(.5 * sim.extents[1])};
      Shape *shape = new Shape(p, center);
      shape->amplitudeFactor = p("-amplitudeFactor").asDouble(1.0);
      shape->rS[0] = 0;
      int k = 0;
      for (int i = 0; i < shape->Nend; ++i, k++)
        shape->rS[k + 1] =
            shape->rS[k] + shape->dSref +
            (shape->dSmid - shape->dSref) * i / ((Real)shape->Nend - 1.);
      for (int i = 0; i < shape->Nmid; ++i, k++)
        shape->rS[k + 1] = shape->rS[k] + shape->dSmid;
      for (int i = 0; i < shape->Nend; ++i, k++)
        shape->rS[k + 1] = shape->rS[k] + shape->dSref +
                           (shape->dSmid - shape->dSref) *
                               (shape->Nend - i - 1) / ((Real)shape->Nend - 1.);
      shape->rS[k] = std::min(shape->rS[k], (Real)shape->length);
      std::fill(shape->rX, shape->rX + shape->Nm, 0);
      std::fill(shape->rY, shape->rY + shape->Nm, 0);
      std::fill(shape->vX, shape->vX + shape->Nm, 0);
      std::fill(shape->vY, shape->vY + shape->Nm, 0);
      for (int i = 0; i < shape->Nm; ++i) {
        const Real sb = .04 * shape->length, st = .95 * shape->length,
                   wt = .01 * shape->length, wh = .04 * shape->length;
        if (shape->rS[i] < 0 or shape->rS[i] > shape->length)
          shape->width[i] = 0;
        else
          shape->width[i] =
              (shape->rS[i] < sb
                   ? std::sqrt(2 * wh * shape->rS[i] -
                               shape->rS[i] * shape->rS[i])
                   : (shape->rS[i] < st
                          ? wh -
                                (wh - wt) *
                                    std::pow((shape->rS[i] - sb) / (st - sb), 1)
                          : (wt * (shape->length - shape->rS[i]) /
                             (shape->length - st))));
      }
      sim.shapes.push_back(shape);
    }
  }
  PoissonSolver pressureSolver;
#pragma omp parallel for
  for (size_t j = 0; j < velInfo.size(); j++)
    for (int i = 0; i < sizeof var.F / sizeof *var.F; i++)
      memset((*var.F[i].g)->infos[j].block, 0,
             var.F[i].dim * _BS_ * _BS_ * sizeof(Real));
  for (int i = 0; i < sizeof var.F / sizeof *var.F; i++) {
    Adaptation *a = *var.F[i].a = new Adaptation(var.F[i].dim);
    a->boundary_needed = false;
    a->stencil.sx = -1;
    a->stencil.sy = -1;
    a->stencil.ex = 2;
    a->stencil.ey = 2;
    a->stencil.tensorial = true;
  }
  for (int i = 0; i < sim.levelMax; i++) {
    ongrid(0.0);
    adapt();
  }
  ongrid(0.0);
  for (auto &shape : sim.shapes) {
    std::vector<ObstacleBlock *> &OBLOCK = shape->obstacleBlocks;
#pragma omp parallel for
    for (size_t i = 0; i < velInfo.size(); i++) {
      if (OBLOCK[var.tmpV->infos[i].id] == nullptr)
        continue;
      UDEFMAT &udef = OBLOCK[var.tmpV->infos[i].id]->udef;
      CHI_MAT &chi = OBLOCK[var.tmpV->infos[i].id]->chi;
      auto &UDEF = *(VectorBlock *)var.tmpV->infos[i].block;
      ScalarBlock &CHI = *(ScalarBlock *)var.chi->infos[i].block;
      for (int iy = 0; iy < _BS_; iy++)
        for (int ix = 0; ix < _BS_; ix++) {
          if (chi[iy][ix] < CHI[iy][ix])
            continue;
          UDEF[iy][ix].u[0] += udef[iy][ix][0];
          UDEF[iy][ix].u[1] += udef[iy][ix][1];
        }
    }
  }
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < velInfo.size(); i++) {
    VectorBlock &UF = *(VectorBlock *)velInfo[i].block;
    VectorBlock &US = *(VectorBlock *)var.tmpV->infos[i].block;
    ScalarBlock &X = *(ScalarBlock *)var.chi->infos[i].block;
    for (int iy = 0; iy < _BS_; ++iy)
      for (int ix = 0; ix < _BS_; ++ix) {
        UF[iy][ix].u[0] =
            UF[iy][ix].u[0] * (1 - X[iy][ix]) + US[iy][ix].u[0] * X[iy][ix];
        UF[iy][ix].u[1] =
            UF[iy][ix].u[1] * (1 - X[iy][ix]) + US[iy][ix].u[1] * X[iy][ix];
      }
  }
  while (1) {
    if (sim.rank == 0 && sim.step % 5 == 0)
      fprintf(stderr, "main.cpp: %08d\n", sim.step);
    Real CFL = sim.CFL;
    Real h = std::numeric_limits<Real>::infinity();
    for (size_t i = 0; i < var.vel->infos.size(); i++)
      h = std::min(var.vel->infos[i].h, h);
    MPI_Allreduce(MPI_IN_PLACE, &h, 1, MPI_Real, MPI_MIN, MPI_COMM_WORLD);
    size_t Nblocks = velInfo.size();
    Real umax = 0;
#pragma omp parallel for schedule(static) reduction(max : umax)
    for (size_t i = 0; i < Nblocks; i++) {
      VectorBlock &VEL = *(VectorBlock *)velInfo[i].block;
      for (int iy = 0; iy < _BS_; ++iy)
        for (int ix = 0; ix < _BS_; ++ix) {
          umax = std::max(umax, std::fabs(VEL[iy][ix].u[0] + sim.uinfx));
          umax = std::max(umax, std::fabs(VEL[iy][ix].u[1] + sim.uinfy));
          umax = std::max(umax, std::fabs(VEL[iy][ix].u[0]));
          umax = std::max(umax, std::fabs(VEL[iy][ix].u[1]));
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &umax, 1, MPI_Real, MPI_MAX, MPI_COMM_WORLD);
    if (CFL > 0) {
      Real dtDiffusion = 0.25 * h * h / (sim.nu + 0.25 * h * umax);
      Real dtAdvection = h / (umax + 1e-8);
      sim.dt = std::min({dtDiffusion, CFL * dtAdvection});
    }
    if (sim.dt <= 0) {
      std::cout << "[CUP2D] dt <= 0. Aborting..." << std::endl;
      fflush(0);
      abort();
    }
    if (sim.dlm > 0)
      sim.lambda = sim.dlm / sim.dt;
    bool done = false;
    if (!done || sim.dt > 2e-16) {
      bool timeDump = sim.dumpTime > 0 && sim.time >= sim.nextDumpTime;
      bool stepDump = sim.dumpFreq > 0 && (sim.step % sim.dumpFreq) == 0;
      bool bDump = stepDump || timeDump;
      if (bDump) {
        sim.nextDumpTime += sim.dumpTime;
        computeA<VectorLab>(KernelVorticity(), var.vel, 2);
        char path[FILENAME_MAX];
        snprintf(path, sizeof path, "vort.%08d", sim.step);
        dump(sim.time, var.tmp->infos.size(), var.tmp->infos.data(), path);
      }
      if (sim.step <= 10 || sim.step % sim.AdaptSteps == 0)
        adapt();
      ongrid(sim.dt);
      size_t Nblocks = velInfo.size();
      KernelAdvectDiffuse Step1;
#pragma omp parallel for
      for (size_t i = 0; i < velInfo.size(); i++) {
        VectorBlock &Vold = *(VectorBlock *)var.vold->infos[i].block;
        const VectorBlock &V = *(VectorBlock *)velInfo[i].block;
        for (int iy = 0; iy < _BS_; ++iy)
          for (int ix = 0; ix < _BS_; ++ix) {
            Vold[iy][ix].u[0] = V[iy][ix].u[0];
            Vold[iy][ix].u[1] = V[iy][ix].u[1];
          }
      }
      var.tmpV->prepare0(var.tmpV);
      computeA<VectorLab>(Step1, var.vel, 2);
      var.tmpV->FillBlockCases(var.tmpV);
#pragma omp parallel for
      for (size_t i = 0; i < velInfo.size(); i++) {
        VectorBlock &V = *(VectorBlock *)velInfo[i].block;
        const VectorBlock &Vold = *(VectorBlock *)var.vold->infos[i].block;
        const VectorBlock &tmpV = *(VectorBlock *)var.tmpV->infos[i].block;
        const Real ih2 = 1.0 / (velInfo[i].h * velInfo[i].h);
        for (int iy = 0; iy < _BS_; ++iy)
          for (int ix = 0; ix < _BS_; ++ix) {
            V[iy][ix].u[0] =
                Vold[iy][ix].u[0] + (0.5 * tmpV[iy][ix].u[0]) * ih2;
            V[iy][ix].u[1] =
                Vold[iy][ix].u[1] + (0.5 * tmpV[iy][ix].u[1]) * ih2;
          }
      }
      var.tmpV->prepare0(var.tmpV);
      computeA<VectorLab>(Step1, var.vel, 2);
      var.tmpV->FillBlockCases(var.tmpV);
#pragma omp parallel for
      for (size_t i = 0; i < velInfo.size(); i++) {
        VectorBlock &V = *(VectorBlock *)velInfo[i].block;
        const VectorBlock &Vold = *(VectorBlock *)var.vold->infos[i].block;
        const VectorBlock &tmpV = *(VectorBlock *)var.tmpV->infos[i].block;
        const Real ih2 = 1.0 / (velInfo[i].h * velInfo[i].h);
        for (int iy = 0; iy < _BS_; ++iy)
          for (int ix = 0; ix < _BS_; ++ix) {
            V[iy][ix].u[0] = Vold[iy][ix].u[0] + tmpV[iy][ix].u[0] * ih2;
            V[iy][ix].u[1] = Vold[iy][ix].u[1] + tmpV[iy][ix].u[1] * ih2;
          }
      }
#pragma omp parallel for
      for (size_t i = 0; i < velInfo.size(); i++) {
        VectorBlock &V = *(VectorBlock *)velInfo[i].block;
        const VectorBlock &Vold = *(VectorBlock *)var.vold->infos[i].block;
        const VectorBlock &tmpV = *(VectorBlock *)var.tmpV->infos[i].block;
        const Real ih2 = 1.0 / (velInfo[i].h * velInfo[i].h);
        for (int iy = 0; iy < _BS_; ++iy)
          for (int ix = 0; ix < _BS_; ++ix) {
            V[iy][ix].u[0] = Vold[iy][ix].u[0] + tmpV[iy][ix].u[0] * ih2;
            V[iy][ix].u[1] = Vold[iy][ix].u[1] + tmpV[iy][ix].u[1] * ih2;
          }
      }
      for (const auto &shape : sim.shapes) {
        const std::vector<ObstacleBlock *> &OBLOCK = shape->obstacleBlocks;
        const Real Cx = shape->centerOfMass[0];
        const Real Cy = shape->centerOfMass[1];
        Real PM = 0, PJ = 0, PX = 0, PY = 0, UM = 0, VM = 0, AM = 0;
#pragma omp parallel for reduction(+ : PM, PJ, PX, PY, UM, VM, AM)
        for (size_t i = 0; i < velInfo.size(); i++) {
          const VectorBlock &VEL = *(VectorBlock *)velInfo[i].block;
          const Real hsq = velInfo[i].h * velInfo[i].h;
          if (OBLOCK[velInfo[i].id] == nullptr)
            continue;
          const CHI_MAT &chi = OBLOCK[velInfo[i].id]->chi;
          const UDEFMAT &udef = OBLOCK[velInfo[i].id]->udef;
          const Real lambdt = sim.lambda * sim.dt;
          for (int iy = 0; iy < _BS_; ++iy)
            for (int ix = 0; ix < _BS_; ++ix) {
              if (chi[iy][ix] <= 0)
                continue;
              const Real udiff[2] = {VEL[iy][ix].u[0] - udef[iy][ix][0],
                                     VEL[iy][ix].u[1] - udef[iy][ix][1]};
              const Real Xlamdt = chi[iy][ix] >= 0.5 ? lambdt : 0.0;
              const Real F = hsq * Xlamdt / (1 + Xlamdt);
              Real p[2];
              p[0] = velInfo[i].origin[0] + velInfo[i].h * (ix + 0.5);
              p[1] = velInfo[i].origin[1] + velInfo[i].h * (iy + 0.5);
              p[0] -= Cx;
              p[1] -= Cy;
              PM += F;
              PJ += F * (p[0] * p[0] + p[1] * p[1]);
              PX += F * p[0];
              PY += F * p[1];
              UM += F * udiff[0];
              VM += F * udiff[1];
              AM += F * (p[0] * udiff[1] - p[1] * udiff[0]);
            }
        }
        Real quantities[7] = {PM, PJ, PX, PY, UM, VM, AM};
        MPI_Allreduce(MPI_IN_PLACE, quantities, 7, MPI_Real, MPI_SUM,
                      MPI_COMM_WORLD);
        PM = quantities[0];
        PJ = quantities[1];
        PX = quantities[2];
        PY = quantities[3];
        UM = quantities[4];
        VM = quantities[5];
        AM = quantities[6];
        double A[3][3] = {{PM, 0, -PY}, {0, PM, PX}, {-PY, PX, PJ}};
        double b[3] = {UM, VM, AM};
        gsl_matrix_view Agsl = gsl_matrix_view_array(&A[0][0], 3, 3);
        gsl_vector_view bgsl = gsl_vector_view_array(b, 3);
        gsl_vector *xgsl = gsl_vector_alloc(3);
        int sgsl;
        gsl_permutation *permgsl = gsl_permutation_alloc(3);
        gsl_linalg_LU_decomp(&Agsl.matrix, permgsl, &sgsl);
        gsl_linalg_LU_solve(&Agsl.matrix, permgsl, &bgsl.vector, xgsl);
        shape->u = gsl_vector_get(xgsl, 0);
        shape->v = gsl_vector_get(xgsl, 1);
        shape->omega = gsl_vector_get(xgsl, 2);
        gsl_permutation_free(permgsl);
        gsl_vector_free(xgsl);
      }
      const auto &shapes = sim.shapes;
      const auto &infos = var.chi->infos;
      const size_t N = shapes.size();
      sim.bCollisionID.clear();
      struct CollisionInfo {
        Real iM = 0;
        Real iPosX = 0;
        Real iPosY = 0;
        Real iPosZ = 0;
        Real iMomX = 0;
        Real iMomY = 0;
        Real iMomZ = 0;
        Real ivecX = 0;
        Real ivecY = 0;
        Real ivecZ = 0;
        Real jM = 0;
        Real jPosX = 0;
        Real jPosY = 0;
        Real jPosZ = 0;
        Real jMomX = 0;
        Real jMomY = 0;
        Real jMomZ = 0;
        Real jvecX = 0;
        Real jvecY = 0;
        Real jvecZ = 0;
      };
      std::vector<CollisionInfo> collisions(N);
      std::vector<Real> n_vec(3 * N, 0.0);
#pragma omp parallel for schedule(static)
      for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j) {
          if (i == j)
            continue;
          auto &coll = collisions[i];
          const auto &iBlocks = shapes[i]->obstacleBlocks;
          const Real iU0 = shapes[i]->u;
          const Real iU1 = shapes[i]->v;
          const Real iomega2 = shapes[i]->omega;
          const Real iCx = shapes[i]->centerOfMass[0];
          const Real iCy = shapes[i]->centerOfMass[1];
          const auto &jBlocks = shapes[j]->obstacleBlocks;
          const Real jU0 = shapes[j]->u;
          const Real jU1 = shapes[j]->v;
          const Real jomega2 = shapes[j]->omega;
          const Real jCx = shapes[j]->centerOfMass[0];
          const Real jCy = shapes[j]->centerOfMass[1];
          assert(iBlocks.size() == jBlocks.size());
          const size_t nBlocks = iBlocks.size();
          for (size_t k = 0; k < nBlocks; ++k) {
            if (iBlocks[k] == nullptr || jBlocks[k] == nullptr)
              continue;
            const auto &iSDF = iBlocks[k]->dist;
            const auto &jSDF = jBlocks[k]->dist;
            const CHI_MAT &iChi = iBlocks[k]->chi;
            const CHI_MAT &jChi = jBlocks[k]->chi;
            const UDEFMAT &iUDEF = iBlocks[k]->udef;
            const UDEFMAT &jUDEF = jBlocks[k]->udef;
            for (int iy = 0; iy < _BS_; ++iy)
              for (int ix = 0; ix < _BS_; ++ix) {
                if (iChi[iy][ix] <= 0.0 || jChi[iy][ix] <= 0.0)
                  continue;
                Real pos[2];
                pos[0] = infos[k].origin[0] + infos[k].h * (ix + 0.5);
                pos[1] = infos[k].origin[1] + infos[k].h * (iy + 0.5);
                const Real iUr0 = -iomega2 * (pos[1] - iCy);
                const Real iUr1 = iomega2 * (pos[0] - iCx);
                coll.iM += iChi[iy][ix];
                coll.iPosX += iChi[iy][ix] * pos[0];
                coll.iPosY += iChi[iy][ix] * pos[1];
                coll.iMomX += iChi[iy][ix] * (iU0 + iUr0 + iUDEF[iy][ix][0]);
                coll.iMomY += iChi[iy][ix] * (iU1 + iUr1 + iUDEF[iy][ix][1]);
                const Real jUr0 = -jomega2 * (pos[1] - jCy);
                const Real jUr1 = jomega2 * (pos[0] - jCx);
                coll.jM += jChi[iy][ix];
                coll.jPosX += jChi[iy][ix] * pos[0];
                coll.jPosY += jChi[iy][ix] * pos[1];
                coll.jMomX += jChi[iy][ix] * (jU0 + jUr0 + jUDEF[iy][ix][0]);
                coll.jMomY += jChi[iy][ix] * (jU1 + jUr1 + jUDEF[iy][ix][1]);
                Real dSDFdx_i;
                Real dSDFdx_j;
                if (ix == 0) {
                  dSDFdx_i = iSDF[iy][ix + 1] - iSDF[iy][ix];
                  dSDFdx_j = jSDF[iy][ix + 1] - jSDF[iy][ix];
                } else if (ix == _BS_ - 1) {
                  dSDFdx_i = iSDF[iy][ix] - iSDF[iy][ix - 1];
                  dSDFdx_j = jSDF[iy][ix] - jSDF[iy][ix - 1];
                } else {
                  dSDFdx_i = 0.5 * (iSDF[iy][ix + 1] - iSDF[iy][ix - 1]);
                  dSDFdx_j = 0.5 * (jSDF[iy][ix + 1] - jSDF[iy][ix - 1]);
                }
                Real dSDFdy_i;
                Real dSDFdy_j;
                if (iy == 0) {
                  dSDFdy_i = iSDF[iy + 1][ix] - iSDF[iy][ix];
                  dSDFdy_j = jSDF[iy + 1][ix] - jSDF[iy][ix];
                } else if (iy == _BS_ - 1) {
                  dSDFdy_i = iSDF[iy][ix] - iSDF[iy - 1][ix];
                  dSDFdy_j = jSDF[iy][ix] - jSDF[iy - 1][ix];
                } else {
                  dSDFdy_i = 0.5 * (iSDF[iy + 1][ix] - iSDF[iy - 1][ix]);
                  dSDFdy_j = 0.5 * (jSDF[iy + 1][ix] - jSDF[iy - 1][ix]);
                }
                coll.ivecX += iChi[iy][ix] * dSDFdx_i;
                coll.ivecY += iChi[iy][ix] * dSDFdy_i;
                coll.jvecX += jChi[iy][ix] * dSDFdx_j;
                coll.jvecY += jChi[iy][ix] * dSDFdy_j;
              }
          }
        }
      std::vector<Real> buffer(20 * N);
      for (size_t i = 0; i < N; i++) {
        auto &coll = collisions[i];
        buffer[20 * i] = coll.iM;
        buffer[20 * i + 1] = coll.iPosX;
        buffer[20 * i + 2] = coll.iPosY;
        buffer[20 * i + 3] = coll.iPosZ;
        buffer[20 * i + 4] = coll.iMomX;
        buffer[20 * i + 5] = coll.iMomY;
        buffer[20 * i + 6] = coll.iMomZ;
        buffer[20 * i + 7] = coll.ivecX;
        buffer[20 * i + 8] = coll.ivecY;
        buffer[20 * i + 9] = coll.ivecZ;
        buffer[20 * i + 10] = coll.jM;
        buffer[20 * i + 11] = coll.jPosX;
        buffer[20 * i + 12] = coll.jPosY;
        buffer[20 * i + 13] = coll.jPosZ;
        buffer[20 * i + 14] = coll.jMomX;
        buffer[20 * i + 15] = coll.jMomY;
        buffer[20 * i + 16] = coll.jMomZ;
        buffer[20 * i + 17] = coll.jvecX;
        buffer[20 * i + 18] = coll.jvecY;
        buffer[20 * i + 19] = coll.jvecZ;
      }
      MPI_Allreduce(MPI_IN_PLACE, buffer.data(), buffer.size(), MPI_Real,
                    MPI_SUM, MPI_COMM_WORLD);
      for (size_t i = 0; i < N; i++) {
        auto &coll = collisions[i];
        coll.iM = buffer[20 * i];
        coll.iPosX = buffer[20 * i + 1];
        coll.iPosY = buffer[20 * i + 2];
        coll.iPosZ = buffer[20 * i + 3];
        coll.iMomX = buffer[20 * i + 4];
        coll.iMomY = buffer[20 * i + 5];
        coll.iMomZ = buffer[20 * i + 6];
        coll.ivecX = buffer[20 * i + 7];
        coll.ivecY = buffer[20 * i + 8];
        coll.ivecZ = buffer[20 * i + 9];
        coll.jM = buffer[20 * i + 10];
        coll.jPosX = buffer[20 * i + 11];
        coll.jPosY = buffer[20 * i + 12];
        coll.jPosZ = buffer[20 * i + 13];
        coll.jMomX = buffer[20 * i + 14];
        coll.jMomY = buffer[20 * i + 15];
        coll.jMomZ = buffer[20 * i + 16];
        coll.jvecX = buffer[20 * i + 17];
        coll.jvecY = buffer[20 * i + 18];
        coll.jvecZ = buffer[20 * i + 19];
      }
#pragma omp parallel for schedule(static)
      for (size_t i = 0; i < N; ++i)
        for (size_t j = i + 1; j < N; ++j) {
          if (i == j)
            continue;
          Real m1 = shapes[i]->M;
          Real m2 = shapes[j]->M;
          Real v1[3] = {shapes[i]->u, shapes[i]->v, 0.0};
          Real v2[3] = {shapes[j]->u, shapes[j]->v, 0.0};
          Real o1[3] = {0, 0, shapes[i]->omega};
          Real o2[3] = {0, 0, shapes[j]->omega};
          Real C1[3] = {shapes[i]->centerOfMass[0], shapes[i]->centerOfMass[1],
                        0};
          Real C2[3] = {shapes[j]->centerOfMass[0], shapes[j]->centerOfMass[1],
                        0};
          Real I1[6] = {1.0, 0, 0, 0, 0, shapes[i]->J};
          Real I2[6] = {1.0, 0, 0, 0, 0, shapes[j]->J};
          auto &coll = collisions[i];
          auto &coll_other = collisions[j];
          if (coll.iM < 2.0 || coll.jM < 2.0)
            continue;
          if (coll_other.iM < 2.0 || coll_other.jM < 2.0)
            continue;
          if (std::fabs(coll.iPosX / coll.iM -
                        coll_other.iPosX / coll_other.iM) > shapes[i]->length ||
              std::fabs(coll.iPosY / coll.iM -
                        coll_other.iPosY / coll_other.iM) > shapes[i]->length) {
            continue;
          }
#pragma omp critical
          {
            sim.bCollisionID.push_back(i);
            sim.bCollisionID.push_back(j);
          }
          Real ho1[3];
          Real ho2[3];
          Real hv1[3];
          Real hv2[3];
          Real norm_i =
              std::sqrt(coll.ivecX * coll.ivecX + coll.ivecY * coll.ivecY +
                        coll.ivecZ * coll.ivecZ);
          Real norm_j =
              std::sqrt(coll.jvecX * coll.jvecX + coll.jvecY * coll.jvecY +
                        coll.jvecZ * coll.jvecZ);
          Real mX = coll.ivecX / norm_i - coll.jvecX / norm_j;
          Real mY = coll.ivecY / norm_i - coll.jvecY / norm_j;
          Real mZ = coll.ivecZ / norm_i - coll.jvecZ / norm_j;
          Real inorm = 1.0 / std::sqrt(mX * mX + mY * mY + mZ * mZ);
          Real NX = mX * inorm;
          Real NY = mY * inorm;
          Real NZ = mZ * inorm;
          Real hitVelX = coll.jMomX / coll.jM - coll.iMomX / coll.iM;
          Real hitVelY = coll.jMomY / coll.jM - coll.iMomY / coll.iM;
          Real hitVelZ = coll.jMomZ / coll.jM - coll.iMomZ / coll.iM;
          Real projVel = hitVelX * NX + hitVelY * NY + hitVelZ * NZ;
          Real vc1[3] = {coll.iMomX / coll.iM, coll.iMomY / coll.iM,
                         coll.iMomZ / coll.iM};
          Real vc2[3] = {coll.jMomX / coll.jM, coll.jMomY / coll.jM,
                         coll.jMomZ / coll.jM};
          if (projVel <= 0)
            continue;
          Real inv_iM = 1.0 / coll.iM;
          Real inv_jM = 1.0 / coll.jM;
          Real iPX = coll.iPosX * inv_iM;
          Real iPY = coll.iPosY * inv_iM;
          Real iPZ = coll.iPosZ * inv_iM;
          Real jPX = coll.jPosX * inv_jM;
          Real jPY = coll.jPosY * inv_jM;
          Real jPZ = coll.jPosZ * inv_jM;
          Real CX = 0.5 * (iPX + jPX);
          Real CY = 0.5 * (iPY + jPY);
          Real CZ = 0.5 * (iPZ + jPZ);
          collision(m1, m2, I1, I2, v1, v2, o1, o2, hv1, hv2, ho1, ho2, C1, C2,
                    NX, NY, NZ, CX, CY, CZ, vc1, vc2);
          shapes[i]->u = hv1[0];
          shapes[i]->v = hv1[1];
          shapes[j]->u = hv2[0];
          shapes[j]->v = hv2[1];
          shapes[i]->omega = ho1[2];
          shapes[j]->omega = ho2[2];
        }
      std::vector<BlockInfo> &chiInfo = var.chi->infos;
#pragma omp parallel for
      for (size_t i = 0; i < Nblocks; i++)
        for (auto &shape : sim.shapes) {
          std::vector<ObstacleBlock *> &OBLOCK = shape->obstacleBlocks;
          ObstacleBlock *o = OBLOCK[velInfo[i].id];
          if (o == nullptr)
            continue;
          Real u_s = shape->u;
          Real v_s = shape->v;
          Real omega_s = shape->omega;
          Real Cx = shape->centerOfMass[0];
          Real Cy = shape->centerOfMass[1];
          CHI_MAT &X = o->chi;
          UDEFMAT &UDEF = o->udef;
          ScalarBlock &CHI = *(ScalarBlock *)chiInfo[i].block;
          VectorBlock &V = *(VectorBlock *)velInfo[i].block;
          for (int iy = 0; iy < _BS_; ++iy)
            for (int ix = 0; ix < _BS_; ++ix) {
              if (CHI[iy][ix] > X[iy][ix])
                continue;
              if (X[iy][ix] <= 0)
                continue;
              Real p[2];
              p[0] = velInfo[i].origin[0] + velInfo[i].h * (ix + 0.5);
              p[1] = velInfo[i].origin[1] + velInfo[i].h * (iy + 0.5);
              p[0] -= Cx;
              p[1] -= Cy;
              Real alpha = X[iy][ix] > 0.5 ? 1 / (1 + sim.lambda * sim.dt) : 1;
              Real US = u_s - omega_s * p[1] + UDEF[iy][ix][0];
              Real VS = v_s + omega_s * p[0] + UDEF[iy][ix][1];
              V[iy][ix].u[0] = alpha * V[iy][ix].u[0] + (1 - alpha) * US;
              V[iy][ix].u[1] = alpha * V[iy][ix].u[1] + (1 - alpha) * VS;
            }
        }
      std::vector<BlockInfo> &tmpVInfo = var.tmpV->infos;
#pragma omp parallel for
      for (size_t i = 0; i < Nblocks; i++) {
        for (size_t y = 0; y < _BS_; y++)
          for (size_t x = 0; x < _BS_; x++) {
            (*(VectorBlock *)tmpVInfo[i].block)[y][x].u[0] = 0;
            (*(VectorBlock *)tmpVInfo[i].block)[y][x].u[1] = 0;
          }
      }
      for (auto &shape : sim.shapes) {
        std::vector<ObstacleBlock *> &OBLOCK = shape->obstacleBlocks;
#pragma omp parallel for
        for (size_t i = 0; i < Nblocks; i++) {
          if (OBLOCK[tmpVInfo[i].id] == nullptr)
            continue;
          UDEFMAT &udef = OBLOCK[tmpVInfo[i].id]->udef;
          CHI_MAT &chi = OBLOCK[tmpVInfo[i].id]->chi;
          auto &UDEF = *(VectorBlock *)tmpVInfo[i].block;
          ScalarBlock &CHI = *(ScalarBlock *)chiInfo[i].block;
          for (int iy = 0; iy < _BS_; iy++)
            for (int ix = 0; ix < _BS_; ix++) {
              if (chi[iy][ix] < CHI[iy][ix])
                continue;
              Real p[2];
              p[0] = tmpVInfo[i].origin[0] + tmpVInfo[i].h * (ix + 0.5);
              p[1] = tmpVInfo[i].origin[1] + tmpVInfo[i].h * (iy + 0.5);
              UDEF[iy][ix].u[0] += udef[iy][ix][0];
              UDEF[iy][ix].u[1] += udef[iy][ix][1];
            }
        }
      }
      var.tmp->prepare0(var.tmp);
      computeB<pressure_rhs, Vector, VectorLab, Vector, VectorLab>(
          pressure_rhs(), *var.vel, 2, *var.tmpV, 2);
      var.tmp->FillBlockCases(var.tmpV);
      std::vector<BlockInfo> &presInfo = var.pres->infos;
      std::vector<BlockInfo> &poldInfo = var.pold->infos;
#pragma omp parallel for
      for (size_t i = 0; i < Nblocks; i++) {
        ScalarBlock &PRES = *(ScalarBlock *)presInfo[i].block;
        ScalarBlock &POLD = *(ScalarBlock *)poldInfo[i].block;
        for (int iy = 0; iy < _BS_; ++iy)
          for (int ix = 0; ix < _BS_; ++ix) {
            POLD[iy][ix] = PRES[iy][ix];
            PRES[iy][ix] = 0;
          }
      }
      var.tmp->prepare0(var.tmp);
      computeA<ScalarLab>(pressure_rhs1(), var.pold, 1);
      var.tmp->FillBlockCases(var.tmpV);
      pressureSolver.solve(var.tmp);
      Real avg = 0;
      Real avg1 = 0;
#pragma omp parallel for reduction(+ : avg, avg1)
      for (size_t i = 0; i < Nblocks; i++) {
        ScalarBlock &P = *(ScalarBlock *)presInfo[i].block;
        const Real vv = presInfo[i].h * presInfo[i].h;
        for (int iy = 0; iy < _BS_; iy++)
          for (int ix = 0; ix < _BS_; ix++) {
            avg += P[iy][ix] * vv;
            avg1 += vv;
          }
      }
      Real quantities[2] = {avg, avg1};
      MPI_Allreduce(MPI_IN_PLACE, &quantities, 2, MPI_Real, MPI_SUM,
                    MPI_COMM_WORLD);
      avg = quantities[0];
      avg1 = quantities[1];
      avg = avg / avg1;
#pragma omp parallel for
      for (size_t i = 0; i < Nblocks; i++) {
        Real *pres = (Real *)presInfo[i].block;
        Real *pold = (Real *)poldInfo[i].block;
        for (int j = 0; j < _BS_ * _BS_; j++)
          pres[j] += pold[j] - avg;
      }
      var.tmpV->prepare0(var.tmpV);
      computeA<ScalarLab>(pressureCorrectionKernel(), var.pres, 1);
      var.tmpV->FillBlockCases(var.tmpV);
#pragma omp parallel for
      for (size_t i = 0; i < velInfo.size(); i++) {
        Real ih2 = 1.0 / velInfo[i].h / velInfo[i].h;
        Real *V = (Real *)velInfo[i].block;
        Real *tmpV = (Real *)tmpVInfo[i].block;
        for (int j = 0; j < 2 * _BS_ * _BS_; j++)
          V[j] += tmpV[j] * ih2;
      }
      computeB<KernelComputeForces, Vector, VectorLab, Real, ScalarLab>(
          KernelComputeForces(), *var.vel, 2, *var.chi, 1);
      for (const auto &shape : sim.shapes) {
        shape->perimeter = 0;
        shape->forcex = 0;
        shape->forcey = 0;
        shape->forcex_P = 0;
        shape->forcey_P = 0;
        shape->forcex_V = 0;
        shape->forcey_V = 0;
        shape->torque = 0;
        shape->torque_P = 0;
        shape->torque_V = 0;
        shape->drag = 0;
        shape->thrust = 0;
        shape->lift = 0;
        shape->Pout = 0;
        shape->PoutNew = 0;
        shape->PoutBnd = 0;
        shape->defPower = 0;
        shape->defPowerBnd = 0;
        shape->circulation = 0;
        for (auto &block : shape->obstacleBlocks)
          if (block not_eq nullptr) {
            shape->circulation += block->circulation;
            shape->perimeter += block->perimeter;
            shape->torque += block->torque;
            shape->forcex += block->forcex;
            shape->forcey += block->forcey;
            shape->forcex_P += block->forcex_P;
            shape->forcey_P += block->forcey_P;
            shape->forcex_V += block->forcex_V;
            shape->forcey_V += block->forcey_V;
            shape->torque_P += block->torque_P;
            shape->torque_V += block->torque_V;
            shape->drag += block->drag;
            shape->thrust += block->thrust;
            shape->lift += block->lift;
            shape->Pout += block->Pout;
            shape->PoutNew += block->PoutNew;
            shape->defPowerBnd += block->defPowerBnd;
            shape->PoutBnd += block->PoutBnd;
            shape->defPower += block->defPower;
          }
        Real quantities[19];
        quantities[0] = shape->circulation;
        quantities[1] = shape->perimeter;
        quantities[2] = shape->forcex;
        quantities[3] = shape->forcex_P;
        quantities[4] = shape->forcex_V;
        quantities[5] = shape->torque_P;
        quantities[6] = shape->drag;
        quantities[7] = shape->lift;
        quantities[8] = shape->Pout;
        quantities[9] = shape->PoutNew;
        quantities[10] = shape->PoutBnd;
        quantities[11] = shape->torque;
        quantities[12] = shape->forcey;
        quantities[13] = shape->forcey_P;
        quantities[14] = shape->forcey_V;
        quantities[15] = shape->torque_V;
        quantities[16] = shape->thrust;
        quantities[17] = shape->defPowerBnd;
        quantities[18] = shape->defPower;
        MPI_Allreduce(MPI_IN_PLACE, quantities, 19, MPI_Real, MPI_SUM,
                      MPI_COMM_WORLD);
        shape->circulation = quantities[0];
        shape->perimeter = quantities[1];
        shape->forcex = quantities[2];
        shape->forcex_P = quantities[3];
        shape->forcex_V = quantities[4];
        shape->torque_P = quantities[5];
        shape->drag = quantities[6];
        shape->lift = quantities[7];
        shape->Pout = quantities[8];
        shape->PoutNew = quantities[9];
        shape->PoutBnd = quantities[10];
        shape->torque = quantities[11];
        shape->forcey = quantities[12];
        shape->forcey_P = quantities[13];
        shape->forcey_V = quantities[14];
        shape->torque_V = quantities[15];
        shape->thrust = quantities[16];
        shape->defPowerBnd = quantities[17];
        shape->defPower = quantities[18];
        shape->Pthrust = shape->thrust *
                         std::sqrt(shape->u * shape->u + shape->v * shape->v);
        shape->Pdrag =
            shape->drag * std::sqrt(shape->u * shape->u + shape->v * shape->v);
        const Real denUnb = shape->Pthrust - std::min(shape->defPower, (Real)0);
        const Real demBnd = shape->Pthrust - shape->defPowerBnd;
        shape->EffPDef = shape->Pthrust / std::max(denUnb, EPS);
        shape->EffPDefBnd = shape->Pthrust / std::max(demBnd, EPS);
        int tot_blocks = 0;
        int nb = (int)var.chi->infos.size();
        MPI_Reduce(&nb, &tot_blocks, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
      }
      sim.time += sim.dt;
      sim.step++;
    }
    if (!done) {
      bool timeEnd = sim.endTime > 0 && sim.time >= sim.endTime;
      bool stepEnd = sim.nsteps > 0 && sim.step >= sim.nsteps;
      done = timeEnd || stepEnd;
    }
    if (done)
      break;
  }
  MPI_Finalize();
}
