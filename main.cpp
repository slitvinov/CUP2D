#include <algorithm>
#include <array>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <fenv.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "cuda.h"
enum { max_dim = 2 };

typedef double Real;
static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
struct Stencil {
  int sx, sy, ex, ey;
  bool tensorial;
  bool operator<(Stencil s) const {
    int me[] = {sx, sy, ex, ey, tensorial};
    int you[] = {s.sx, s.sy, s.ex, s.ey, s.tensorial};
    for (size_t i = 0; i < sizeof me / sizeof *me; ++i)
      if (me[i] < you[i])
        return true;
      else if (me[i] > you[i])
        return false;
    return false;
  }
};
struct Shape;
struct Solver;
static struct {
  int AdaptSteps;
  int levelMax;
  int levelStart;
  int maxPoissonRestarts;
  const int rank = 0;
  int step = 0;
  int dump_count = 0;
  Real CFL;
  Real Ctol;
  Real dt;
  Real dumpTime;
  Real endTime;
  Real lambda;
  Real nextDumpTime = 0;
  Real nu;
  Real PoissonTol;
  Real PoissonTolRel;
  Real Rtol;
  Real time = 0;
  std::vector<long long> levels, nblocks, nrows;
  std::vector<Shape *> shapes;
  struct Solver *solver;
  struct LocalSpMatDnVec *mat;
} sim;
#include "utils.h"
enum State : signed char { Leave = 0, Refine = 1, Compress = -1 };
struct Info {
  bool changed2;
  double h, origin[2];
  enum State state;
  int index[3], level;
  long long id, id2, Z, Zchild[2][2], Znei[3][3], Zparent;
  Real *block = NULL;
};
struct CollisionInfo {
  Real iM = 0;
  Real iPosX = 0;
  Real iPosY = 0;
  Real ivecX = 0;
  Real ivecY = 0;
  Real jM = 0;
  Real jPosX = 0;
  Real jPosY = 0;
  Real jvecX = 0;
  Real jvecY = 0;
};
static int &treef(std::unordered_map<long long, int> *tree, int m,
                  long long n) {
  long long aux = sim.levels[m] + n;
  auto retval = tree->find(aux);
  if (retval == tree->end()) {
#pragma omp critical
    {
      auto retval1 = tree->find(aux);
      if (retval1 == tree->end()) {
        (*tree)[aux] = -3;
      }
    }
    return treef(tree, m, n);
  } else {
    return retval->second;
  }
}
static void fill(Info *b, int level, long long Z) {
  int i, j, Bmax[2];
  b->level = level;
  b->Z = Z;
  b->h = 1.0 / _BS_ / (1 << level);
  sfc_inverse(Z, level, &i, &j);
  b->origin[0] = (Real)i / (1 << level);
  b->origin[1] = (Real)j / (1 << level);
  b->state = Leave;
  b->changed2 = true;
  sfc_inverse(Z, level, &b->index[0], &b->index[1]);
  b->index[2] = 0;
  Bmax[0] = 1 << level;
  Bmax[1] = 1 << level;
  for (i = -1; i < 2; i++)
    for (j = -1; j < 2; j++)
      b->Znei[i + 1][j + 1] = sfc_forward(level, (b->index[0] + i) % Bmax[0],
                                          (b->index[1] + j) % Bmax[1]);
  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      b->Zchild[i][j] =
          sfc_forward(level + 1, 2 * b->index[0] + i, 2 * b->index[1] + j);
  b->Zparent = Z >> 2;
  b->id2 = sfc_encode(level, b->index);
  b->id = b->id2;
}
static Info *getf(std::unordered_map<long long, Info *> *all, int m,
                  long long Z) {
  long long aux = sim.levels[m] + Z;
  auto retval = all->find(aux);
  if (retval != all->end()) {
    return retval->second;
  } else {
#pragma omp critical
    {
      const auto retval1 = all->find(aux);
      if (retval1 == all->end()) {
        Info *dumm = new Info;
        fill(dumm, m, Z);
        (*all)[aux] = dumm;
      }
    }
    return getf(all, m, Z);
  }
}
struct Face {
  Info *infos[2];
  int icode[2];
  int offset;
  Face(Info *i0, Info *i1, int a_icode0, int a_icode1) {
    infos[0] = i0;
    infos[1] = i1;
    icode[0] = a_icode0;
    icode[1] = a_icode1;
  }
  bool operator<(const Face &other) const {
    if (infos[0]->id2 == other.infos[0]->id2) {
      return (icode[0] < other.icode[0]);
    } else {
      return (infos[0]->id2 < other.infos[0]->id2);
    }
  }
};
static void update_blocks(std::vector<Info *> *infos,
                          std::unordered_map<long long, Info *> *all,
                          std::unordered_map<long long, int> *tree) {
  std::vector<long long> myData;
  for (auto &info : *infos) {
    int aux = 1 << info->level;
    bool xskin = info->index[0] == 0 || info->index[0] == aux - 1;
    bool yskin = info->index[1] == 0 || info->index[1] == aux - 1;
    int xskip = info->index[0] == 0 ? -1 : 1;
    int yskip = info->index[1] == 0 ? -1 : 1;
    for (int x = -1; x < 2; x++)
      for (int y = -1; y < 2; y++)
        if (x != 0 || y != 0) {
          if (x == xskip && xskin)
            continue;
          if (y == yskip && yskin)
            continue;
          Info *infoNei = getf(all, info->level, info->Znei[1 + x][1 + y]);
          int &infoNeiTree = treef(tree, infoNei->level, infoNei->Z);
          if (infoNeiTree == -2) {
            long long nCoarse = infoNei->Zparent;
            treef(tree, infoNei->level - 1, nCoarse);
          } else if (infoNeiTree == -1) {
            int Bstep = 1;
            if ((abs(x) + abs(y) == 2))
              Bstep = 3;
            for (int B = 0; B <= 3; B += Bstep) {
              int temp = (abs(x) == 1) ? (B % 2) : (B / 2);
              long long nFine =
                  infoNei->Zchild[std::max(-x, 0) +
                                  (B % 2) * std::max(0, 1 - abs(x))]
                                 [std::max(-y, 0) +
                                  temp * std::max(0, 1 - abs(y))];
              treef(tree, infoNei->level + 1, nFine);
            }
          }
        }
  }
}

static bool info_cmp(Info *a, Info *b) { return a->id2 < b->id2; }
static void fill_pos(std::vector<Info *> *infos,
                     std::unordered_map<long long, Info *> *all) {
  std::sort(infos->begin(), infos->end(), info_cmp);
  for (size_t j = 0; j < infos->size(); j++) {
    int m = (*infos)[j]->level;
    long long Z = (*infos)[j]->Z;
    auto retval = all->find(sim.levels[m] + Z);
    assert(retval != all->end());
    Info *info = retval->second;
    info->id = j;
    (*infos)[j] = info;
  }
}
static Real *avail(int m, long long n,
                   std::unordered_map<long long, Info *> *all) {
  return getf(all, m, n)->block;
}
static Real *avail1(int ix, int iy, int m,
                    std::unordered_map<long long, Info *> *all) {
  const long long n = forward(m, ix, iy);
  return avail(m, n, all);
}

static void dealloc_many(std::vector<long long> &ids,
                         std::vector<Info *> *infos) {
  for (size_t j = 0; j < infos->size(); j++)
    (*infos)[j]->changed2 = false;
  for (size_t i = 0; i < ids.size(); i++)
    for (size_t j = 0; j < infos->size(); j++) {
      if ((*infos)[j]->id2 == ids[i]) {
        (*infos)[j]->changed2 = true;
        free((*infos)[j]->block);
        break;
      }
    }
  infos->erase(std::remove_if(infos->begin(), infos->end(),
                              [](const Info *x) { return x->changed2; }),
               infos->end());
}

static int &Tree1(const Info *info, std::unordered_map<long long, int> *tree) {
  return treef(tree, info->level, info->Z);
}
struct Grid {
  bool UpdateFluxCorrection{true};
  std::unordered_map<long long, Info *> all;
  std::unordered_map<long long, int> tree;
  std::vector<Info *> infos;
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
struct BlockLab;
static void bc_scalar(BlockLab *, Info *, bool coarse);
static void bc_vector(BlockLab *, Info *, bool coarse);

struct BlockLab {
  const int dim;

private:
  bool coarsened, istensorial, use_averages;
  int coarsened_nei_codes_size, offset[3];
  std::array<Real *, 27> myblocks;
  std::array<int, 27> coarsened_nei_codes;

public:
  unsigned int nm[2], nc[2];
  int end[3], start0[3];
  Real *m, *c;
  BlockLab(int dim) : dim(dim) {
    m = NULL;
    c = NULL;
  }
  ~BlockLab() {
    free(m);
    free(c);
  }
  void prepare(const Stencil &stencil) {
    istensorial = stencil.tensorial;
    coarsened = false;
    start0[0] = stencil.sx;
    start0[1] = stencil.sy;
    start0[2] = 0;
    end[0] = stencil.ex;
    end[1] = stencil.ey;
    end[2] = 1;
    nm[0] = _BS_ + end[0] - start0[0] - 1;
    nm[1] = _BS_ + end[1] - start0[1] - 1;
    free(m);
    m = (Real *)malloc(nm[0] * nm[1] * dim * sizeof(Real));
    offset[0] = (start0[0] - 1) / 2 - 1;
    offset[1] = (start0[1] - 1) / 2 - 1;
    offset[2] = (start0[2] - 1) / 2;
    nc[0] = _BS_ / 2 + end[0] / 2 + 1 - offset[0];
    nc[1] = _BS_ / 2 + end[1] / 2 + 1 - offset[1];
    free(c);
    c = (Real *)malloc(nc[0] * nc[1] * dim * sizeof(Real));
    use_averages = istensorial || start0[0] < -2 || start0[1] < -2 ||
                   end[0] > 3 || end[1] > 3;
  }
  void load(std::unordered_map<long long, int> *tree,
            std::unordered_map<long long, Info *> *all, const Stencil &stencil,
            Info *info, bool applybc) {
    int n = 1 << info->level;
    int xi, yi;
    sfc_inverse(info->Z, info->level, &xi, &yi);
    assert(m != NULL);
    Real *p = info->block;
    for (int iy = -stencil.sy; iy < -stencil.sy + _BS_; iy += 4) {
      Real *q = m + dim * iy * nm[0] - dim * start0[0];
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
    bool xskin = xi == 0 || xi == n - 1;
    bool yskin = yi == 0 || yi == n - 1;
    int xskip = xi == 0 ? -1 : 1;
    int yskip = yi == 0 ? -1 : 1;
    int icodes[8];
    int k = 0;
    coarsened_nei_codes_size = 0;
    for (int icode = 9; icode < 18; icode++) {
      myblocks[icode] = nullptr;
      if (icode == 1 * 1 + 3 * 1 + 9 * 1)
        continue;
      int cx = icode % 3 - 1;
      int cy = (icode / 3) % 3 - 1;
      if (cx == xskip && xskin)
        continue;
      if (cy == yskip && yskin)
        continue;
      const auto &TreeNei =
          treef(tree, info->level, info->Znei[1 + cx][1 + cy]);
      if (TreeNei >= 0) {
        icodes[k++] = icode;
      } else if (TreeNei == -2) {
        coarsened_nei_codes[coarsened_nei_codes_size++] = icode;
        int infoNei_index[2] = {(xi + cx + n) % n, (yi + cy + n) % n};
        int infoNei_index_true[2] = {(xi + cx), (yi + cy)};
        Real *b = avail1((infoNei_index[0]) / 2, (infoNei_index[1]) / 2,
                         info->level - 1, all);
        if (b == nullptr)
          continue;
        int s[2] = {cx < 1 ? (cx < 0 ? offset[0] : 0) : (_BS_ / 2),
                    cy < 1 ? (cy < 0 ? offset[1] : 0) : (_BS_ / 2)};
        int e[2] = {cx < 1 ? (cx < 0 ? 0 : (_BS_ / 2))
                           : (_BS_ / 2) + (end[0]) / 2 + (2) - 1,
                    cy < 1 ? (cy < 0 ? 0 : (_BS_ / 2))
                           : (_BS_ / 2) + (end[1]) / 2 + (2) - 1};
        int bytes = (e[0] - s[0]) * dim * sizeof(Real);
        if (!bytes)
          continue;
        int base[2] = {(xi + cx) % 2, (yi + cy) % 2};
        int CoarseEdge[2];
        CoarseEdge[0] = cx == 0 ? 0
                        : (((xi % 2 == 0) && (infoNei_index_true[0] > xi)) ||
                           ((xi % 2 == 1) && (infoNei_index_true[0] < xi)))
                            ? 1
                            : 0;
        CoarseEdge[1] = cy == 0 ? 0
                        : (((yi % 2 == 0) && (infoNei_index_true[1] > yi)) ||
                           ((yi % 2 == 1) && (infoNei_index_true[1] < yi)))
                            ? 1
                            : 0;
        int start[2] = {
            std::max(cx, 0) * _BS_ / 2 + (1 - abs(cx)) * base[0] * _BS_ / 2 -
                cx * _BS_ + CoarseEdge[0] * cx * _BS_ / 2,
            std::max(cy, 0) * _BS_ / 2 + (1 - abs(cy)) * base[1] * _BS_ / 2 -
                cy * _BS_ + CoarseEdge[1] * cy * _BS_ / 2};
        int i = s[0] - offset[0];
        int mod = (e[1] - s[1]) % 4;
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
          Real *p0 = c + dim * i0;
          Real *p1 = c + dim * i1;
          Real *p2 = c + dim * i2;
          Real *p3 = c + dim * i3;
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
          Real *p = c + dim * i0;
          Real *q = b + dim * (_BS_ * y0 + x);
          memcpy(p, q, bytes);
        }
      }
      if (!istensorial && !use_averages && abs(cx) + abs(cy) > 1)
        continue;
      int s[3] = {cx < 1 ? (cx < 0 ? start0[0] : 0) : _BS_,
                  cy < 1 ? (cy < 0 ? start0[1] : 0) : _BS_, 0};
      int e[3] = {cx < 1 ? (cx < 0 ? 0 : _BS_) : _BS_ + end[0] - 1,
                  cy < 1 ? (cy < 0 ? 0 : _BS_) : _BS_ + end[1] - 1, 1};
      if (TreeNei >= 0) {
        int bytes = (e[0] - s[0]) * dim * sizeof(Real);
        if (!bytes)
          continue;
        int icode = (cx + 1) + 3 * (cy + 1) + 9;
        myblocks[icode] = avail(info->level, info->Znei[1 + cx][1 + cy], all);
        if (myblocks[icode] == nullptr)
          continue;
        Real *b = myblocks[icode];
        int i = s[0] - start0[0];
        int mod = (e[1] - s[1]) % 4;
        for (int iy = s[1]; iy < e[1] - mod; iy += 4) {
          int i0 = i + (iy - start0[1]) * nm[0];
          int i1 = i + (iy + 1 - start0[1]) * nm[0];
          int i2 = i + (iy + 2 - start0[1]) * nm[0];
          int i3 = i + (iy + 3 - start0[1]) * nm[0];
          int x0 = s[0] - cx * _BS_;
          int y0 = iy - cy * _BS_;
          int y1 = iy + 1 - cy * _BS_;
          int y2 = iy + 2 - cy * _BS_;
          int y3 = iy + 3 - cy * _BS_;
          Real *p0 = &m[dim * i0];
          Real *p1 = &m[dim * i1];
          Real *p2 = &m[dim * i2];
          Real *p3 = &m[dim * i3];
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
          int i0 = i + (iy - start0[1]) * nm[0];
          int x0 = s[0] - cx * _BS_;
          int y0 = iy - cy * _BS_;
          Real *p = &m[dim * i0];
          Real *q = &b[dim * (_BS_ * y0 + x0)];
          memcpy(p, q, bytes);
        }
      } else if (TreeNei == -1) {
        int bytes =
            (abs(cx) * (e[0] - s[0]) + (1 - abs(cx)) * ((e[0] - s[0]) / 2)) *
            dim * sizeof(Real);
        if (!bytes)
          continue;
        int ys = (cy == 0) ? 2 : 1;
        int mod = ((e[1] - s[1]) / ys) % 4;
        int Bstep = 1;
        if ((abs(cx) + abs(cy) == 2))
          Bstep = 3;
        else if ((abs(cx) + abs(cy) == 3))
          Bstep = 4;
        for (int B = 0; B <= 3; B += Bstep) {
          int aux = (abs(cx) == 1) ? (B % 2) : (B / 2);
          Real *b = avail1(2 * xi + std::max(cx, 0) + cx +
                               (B % 2) * std::max(0, 1 - abs(cx)),
                           2 * yi + std::max(cy, 0) + cy +
                               aux * std::max(0, 1 - abs(cy)),
                           info->level + 1, all);
          if (b == nullptr)
            continue;
          int i =
              abs(cx) * (s[0] - start0[0]) +
              (1 - abs(cx)) * (s[0] - start0[0] + (B % 2) * (e[0] - s[0]) / 2);
          int x = s[0] - cx * _BS_ + std::min(0, cx) * (e[0] - s[0]);
          for (int iy = s[1]; iy < e[1] - mod; iy += 4 * ys) {
            int k0 = i + (abs(cy) * (iy + 0 * ys - start0[1]) +
                          (1 - abs(cy)) * ((iy + 0 * ys) / 2 - start0[1] +
                                           aux * (e[1] - s[1]) / 2)) *
                             nm[0];
            int k1 = i + (abs(cy) * (iy + 1 * ys - start0[1]) +
                          (1 - abs(cy)) * ((iy + 1 * ys) / 2 - start0[1] +
                                           aux * (e[1] - s[1]) / 2)) *
                             nm[0];
            int k2 = i + (abs(cy) * (iy + 2 * ys - start0[1]) +
                          (1 - abs(cy)) * ((iy + 2 * ys) / 2 - start0[1] +
                                           aux * (e[1] - s[1]) / 2)) *
                             nm[0];
            int k3 = i + (abs(cy) * (iy + 3 * ys - start0[1]) +
                          (1 - abs(cy)) * ((iy + 3 * ys) / 2 - start0[1] +
                                           aux * (e[1] - s[1]) / 2)) *
                             nm[0];
            int y0 = (abs(cy) == 1) ? 2 * (iy + 0 * ys - cy * _BS_) +
                                          std::min(0, cy) * _BS_
                                    : iy + 0 * ys;
            int y1 = (abs(cy) == 1) ? 2 * (iy + 1 * ys - cy * _BS_) +
                                          std::min(0, cy) * _BS_
                                    : iy + 1 * ys;
            int y2 = (abs(cy) == 1) ? 2 * (iy + 2 * ys - cy * _BS_) +
                                          std::min(0, cy) * _BS_
                                    : iy + 2 * ys;
            int y3 = (abs(cy) == 1) ? 2 * (iy + 3 * ys - cy * _BS_) +
                                          std::min(0, cy) * _BS_
                                    : iy + 3 * ys;
            /* int z0 = y0 + 1; */
            int z1 = y1 + 1;
            int z2 = y2 + 1;
            int z3 = y3 + 1;
            Real *p0 = m + dim * k0;
            Real *p1 = m + dim * k1;
            Real *p2 = m + dim * k2;
            Real *p3 = m + dim * k3;
            Real *q00 = b + dim * (_BS_ * y0 + x);
            // Real *q10 = b + dim * (_BS_ * z0 + x);
            Real *q01 = b + dim * (_BS_ * y1 + x);
            Real *q11 = b + dim * (_BS_ * z1 + x);
            Real *q02 = b + dim * (_BS_ * y2 + x);
            Real *q12 = b + dim * (_BS_ * z2 + x);
            Real *q03 = b + dim * (_BS_ * y3 + x);
            Real *q13 = b + dim * (_BS_ * z3 + x);
            for (int ee = 0; ee < (abs(cx) * (e[0] - s[0]) +
                                   (1 - abs(cx)) * ((e[0] - s[0]) / 2));
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
            int k = i + (abs(cy) * (iy - start0[1]) +
                         (1 - abs(cy)) *
                             (iy / 2 - start0[1] + aux * (e[1] - s[1]) / 2)) *
                            nm[0];
            int y = (abs(cy) == 1)
                        ? 2 * (iy - cy * _BS_) + std::min(0, cy) * _BS_
                        : iy;
            int z = y + 1;
            Real *p = m + dim * k;
            Real *q0 = b + dim * (_BS_ * y + x);
            Real *q1 = b + dim * (_BS_ * z + x);
            for (int ee = 0; ee < (abs(cx) * (e[0] - s[0]) +
                                   (1 - abs(cx)) * ((e[0] - s[0]) / 2));
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
    }
    if (coarsened_nei_codes_size > 0)
      for (int i = 0; i < k; ++i) {
        int icode = icodes[i];
        int cx = icode % 3 - 1;
        int cy = (icode / 3) % 3 - 1;
        int infoNei_index[3] = {(xi + cx + n) % n, (yi + cy + n) % n, 0};
        if (UseCoarseStencil0(info, infoNei_index)) {
          int icode = (cx + 1) + 3 * (cy + 1) + 9;
          if (myblocks[icode] != nullptr) {
            Real *b = myblocks[icode];
            int eC[2] = {(end[0]) / 2 + (2), (end[1]) / 2 + (2)};
            int s[2] = {cx < 1 ? (cx < 0 ? offset[0] : 0) : (_BS_ / 2),
                        cy < 1 ? (cy < 0 ? offset[1] : 0) : (_BS_ / 2)};
            int e[2] = {
                cx < 1 ? (cx < 0 ? 0 : (_BS_ / 2)) : (_BS_ / 2) + eC[0] - 1,
                cy < 1 ? (cy < 0 ? 0 : (_BS_ / 2)) : (_BS_ / 2) + eC[1] - 1};
            int bytes = (e[0] - s[0]) * dim * sizeof(Real);
            if (bytes) {
              int start[2] = {s[0] + std::max(cx, 0) * (_BS_ / 2) - cx * _BS_ +
                                  std::min(0, cx) * (e[0] - s[0]),
                              s[1] + std::max(cy, 0) * (_BS_ / 2) - cy * _BS_ +
                                  std::min(0, cy) * (e[1] - s[1])};
              int i = s[0] - offset[0];
              int x = start[0];
              for (int iy = s[1]; iy < e[1]; iy++) {
                int i0 = i + (iy - offset[1]) * nc[0];
                Real *p1 = c + dim * i0;
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
          }
          coarsened = true;
        }
      }
    /* was post load */
    if (coarsened) {
      for (int j = 0; j < _BS_ / 2; j++) {
        for (int i = 0; i < _BS_ / 2; i++) {
          if (i > 1 && i < _BS_ / 2 - 2 && j > 2 && j < _BS_ / 2 - 2)
            continue;
          int ix = 2 * i - start0[0];
          int iy = 2 * j - start0[1];
          int i00 = ix + nm[0] * iy;
          int i10 = ix + 1 + nm[0] * iy;
          int i01 = ix + nm[0] * (iy + 1);
          int i11 = ix + 1 + nm[0] * (iy + 1);
          int j00 = i - offset[0] + nc[0] * (j - offset[1]);
          for (int d = 0; d < dim; d++) {
            c[dim * j00 + d] = (m[dim * i01 + d] + m[dim * i00 + d] +
                                m[dim * i10 + d] + m[dim * i11 + d]) /
                               4;
          }
        }
      }
    }
    if (applybc) {
      if (dim == 1)
        bc_scalar(this, info, true);
      else
        bc_vector(this, info, true);
    }
    for (int ii = 0; ii < coarsened_nei_codes_size; ++ii) {
      int icode = coarsened_nei_codes[ii];
      if (icode == 1 * 1 + 3 * 1 + 9 * 1)
        continue;
      int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, (icode / 9) % 3 - 1};
      if (code[2] != 0)
        continue;
      if (code[0] == xskip && xskin)
        continue;
      if (code[1] == yskip && yskin)
        continue;
      if (!istensorial && !use_averages && abs(code[0]) + abs(code[1]) > 1)
        continue;
      int s[2] = {code[0] < 1 ? (code[0] < 0 ? start0[0] : 0) : _BS_,
                  code[1] < 1 ? (code[1] < 0 ? start0[1] : 0) : _BS_};
      int e[2] = {code[0] < 1 ? (code[0] < 0 ? 0 : _BS_) : _BS_ + end[0] - 1,
                  code[1] < 1 ? (code[1] < 0 ? 0 : _BS_) : _BS_ + end[1] - 1};
      int sC[2] = {
          code[0] < 1 ? (code[0] < 0 ? ((start0[0] - 1) / 2) : 0) : (_BS_ / 2),
          code[1] < 1 ? (code[1] < 0 ? ((start0[1] - 1) / 2) : 0) : (_BS_ / 2)};
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
                Test[i][j] = c + dim * i0;
              }
            int i1 = ix - start0[0] + nm[0] * (iy - start0[1]);
            for (int d = 0; d < dim; d++)
              TestInterp(
                  Test, m + dim * i1 + d,
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
            int j0 = ix - start0[0] + nm[0] * (iy - start0[1]);
            int j1 = ix - start0[0] + nm[0] * (iy - start0[1] + iyp);
            int j2 = ix - start0[0] + ixp + nm[0] * (iy - start0[1]);
            int j3 = ix - start0[0] + ixp + nm[0] * (iy - start0[1] + iyp);
            for (int d = 0; d < dim; d++) {
              if (code[0] != 0) {
                Real dudy, dudy2;
                if (YY + offset[1] == 0) {
                  dudy = (-0.5 * c[dim * i0 + d] - 1.5 * c[dim * i1 + d]) +
                         2.0 * c[dim * i2 + d];
                  dudy2 = (c[dim * i0 + d] + c[dim * i1 + d]) -
                          2.0 * c[dim * i2 + d];
                } else if (YY + offset[1] == (_BS_ / 2) - 1) {
                  dudy = (0.5 * c[dim * i3 + d] + 1.5 * c[dim * i1 + d]) -
                         2.0 * c[dim * i4 + d];
                  dudy2 = (c[dim * i3 + d] + c[dim * i1 + d]) -
                          2.0 * c[dim * i4 + d];
                } else {
                  dudy = 0.5 * (c[dim * i2 + d] - c[dim * i4 + d]);
                  dudy2 = (c[dim * i2 + d] + c[dim * i4 + d]) -
                          2.0 * c[dim * i1 + d];
                }
                m[dim * j0 + d] =
                    c[dim * i1 + d] + dy * dudy + (0.5 * dy * dy) * dudy2;
                if (iy + iyp >= s[1] && iy + iyp < e[1])
                  m[dim * j1 + d] =
                      c[dim * i1 + d] - dy * dudy + (0.5 * dy * dy) * dudy2;
                if (ix + ixp >= s[0] && ix + ixp < e[0])
                  m[dim * j2 + d] =
                      c[dim * i1 + d] + dy * dudy + (0.5 * dy * dy) * dudy2;
                if (ix + ixp >= s[0] && ix + ixp < e[0] && iy + iyp >= s[1] &&
                    iy + iyp < e[1])
                  m[dim * j3 + d] =
                      c[dim * i1 + d] - dy * dudy + (0.5 * dy * dy) * dudy2;
              } else {
                Real dudx, dudx2;
                if (XX + offset[0] == 0) {
                  dudx = (-0.5 * c[dim * i5 + d] - 1.5 * c[dim * i1 + d]) +
                         2.0 * c[dim * i6 + d];
                  dudx2 = (c[dim * i5 + d] + c[dim * i1 + d]) -
                          2.0 * c[dim * i6 + d];
                } else if (XX + offset[0] == (_BS_ / 2) - 1) {
                  dudx = (0.5 * c[dim * i8 + d] + 1.5 * c[dim * i1 + d]) -
                         2.0 * c[dim * i7 + d];
                  dudx2 = (c[dim * i8 + d] + c[dim * i1 + d]) -
                          2.0 * c[dim * i7 + d];
                } else {
                  dudx = 0.5 * (c[dim * i6 + d] - c[dim * i7 + d]);
                  dudx2 = (c[dim * i6 + d] + c[dim * i7 + d]) -
                          2.0 * c[dim * i1 + d];
                }
                m[dim * j0 + d] =
                    c[dim * i1 + d] + dx * dudx + (0.5 * dx * dx) * dudx2;
                if (iy + iyp >= s[1] && iy + iyp < e[1])
                  m[dim * j1 + d] =
                      c[dim * i1 + d] + dx * dudx + (0.5 * dx * dx) * dudx2;
                if (ix + ixp >= s[0] && ix + ixp < e[0])
                  m[dim * j2 + d] =
                      c[dim * i1 + d] - dx * dudx + (0.5 * dx * dx) * dudx2;
                if (ix + ixp >= s[0] && ix + ixp < e[0] && iy + iyp >= s[1] &&
                    iy + iyp < e[1])
                  m[dim * j3 + d] =
                      c[dim * i1 + d] - dx * dudx + (0.5 * dx * dx) * dudx2;
              }
            }
          }
        }
        for (int iy = s[1]; iy < e[1]; iy += 1) {
          for (int ix = s[0]; ix < e[0]; ix += 1) {
            if (ix < -2 || iy < -2 || ix > _BS_ + 1 || iy > _BS_ + 1)
              continue;
            int k0 = ix - start0[0] + nm[0] * (iy - start0[1] - 1);
            int k1 = ix - start0[0] + nm[0] * (iy - start0[1] - 2);
            int k2 = ix - start0[0] + nm[0] * (iy - start0[1] + 1);
            int k3 = ix - start0[0] + nm[0] * (iy - start0[1] + 2);
            int k4 = ix - start0[0] + nm[0] * (iy - start0[1] + 3);
            int k5 = ix - start0[0] - 1 + nm[0] * (iy - start0[1]);
            int k6 = ix - start0[0] - 2 + nm[0] * (iy - start0[1]);
            int k7 = ix - start0[0] - 3 + nm[0] * (iy - start0[1]);
            int k8 = ix - start0[0] + 1 + nm[0] * (iy - start0[1]);
            int k9 = ix - start0[0] + 2 + nm[0] * (iy - start0[1]);
            int k10 = ix - start0[0] + 3 + nm[0] * (iy - start0[1]);
            int k11 = ix - start0[0] + nm[0] * (iy - start0[1] - 3);
            int k12 = ix - start0[0] + nm[0] * (iy - start0[1]);
            int x =
                abs(ix - s[0] - std::min(0, code[0]) * ((e[0] - s[0]) % 2)) % 2;
            int y =
                abs(iy - s[1] - std::min(0, code[1]) * ((e[1] - s[1]) % 2)) % 2;
            for (int d = 0; d < dim; d++) {
              Real *a = m + dim * k12 + d;
              if (code[0] == 0 && code[1] == 1) {
                if (y == 0) {
                  Real *b = m + dim * k0 + d;
                  Real *c = m + dim * k1 + d;
                  LI(a, b, c);
                } else if (y == 1) {
                  Real *b = m + dim * k1 + d;
                  Real *c = m + dim * k11 + d;
                  LE(a, b, c);
                }
              } else if (code[0] == 0 && code[1] == -1) {
                if (y == 1) {
                  Real *b = m + dim * k2 + d;
                  Real *c = m + dim * k3 + d;
                  LI(a, b, c);
                } else if (y == 0) {
                  Real *b = m + dim * k3 + d;
                  Real *c = m + dim * k4 + d;
                  LE(a, b, c);
                }
              } else if (code[1] == 0 && code[0] == 1) {
                if (x == 0) {
                  Real *b = m + dim * k5 + d;
                  Real *c = m + dim * k6 + d;
                  LI(a, b, c);
                } else if (x == 1) {
                  Real *b = m + dim * k6 + d;
                  Real *c = m + dim * k7 + d;
                  LE(a, b, c);
                }
              } else if (code[1] == 0 && code[0] == -1) {
                if (x == 1) {
                  Real *b = m + dim * k8 + d;
                  Real *c = m + dim * k9 + d;
                  LI(a, b, c);
                } else if (x == 0) {
                  Real *b = m + dim * k9 + d;
                  Real *c = m + dim * k10 + d;
                  LE(a, b, c);
                }
              }
            }
          }
        }
      }
    }
    if (applybc) {
      if (dim == 1)
        bc_scalar(this, info, false);
      else
        bc_vector(this, info, false);
    }
  }
  bool UseCoarseStencil0(Info *info, int *infoNei_index) {
    if (info->level == 0 || !use_averages)
      return false;
    int imin[3];
    int imax[3];
    int aux = 1 << info->level;
    int blocks[3] = {aux - 1, aux - 1, aux - 1};
    for (int d = 0; d < 3; d++) {
      imin[d] = (info->index[d] < infoNei_index[d]) ? 0 : -1;
      imax[d] = (info->index[d] > infoNei_index[d]) ? 0 : +1;
      if (info->index[d] == 0 && infoNei_index[d] == 0)
        imin[d] = 0;
      if (info->index[d] == blocks[d] && infoNei_index[d] == blocks[d])
        imax[d] = 0;
    }
    for (int itest = 0; itest < coarsened_nei_codes_size; itest++)
      for (int i2 = imin[2]; i2 <= imax[2]; i2++)
        for (int i1 = imin[1]; i1 <= imax[1]; i1++)
          for (int i0 = imin[0]; i0 <= imax[0]; i0++) {
            int icode_test = (i0 + 1) + 3 * (i1 + 1) + 9 * (i2 + 1);
            if (coarsened_nei_codes[itest] == icode_test)
              return true;
          }
    return false;
  }
};
template <typename Kernel>
static void computeA(Kernel &&kernel, Grid *g, int dim) {
  std::vector<Info *> *inner = &g->infos;
#pragma omp parallel
  {
    BlockLab lab(dim);
    lab.prepare(kernel.stencil);
#pragma omp for nowait
    for (std::size_t i = 0; i < inner->size(); ++i) {
      const auto &I = (*inner)[i];
      lab.load(&g->tree, &g->all, kernel.stencil, I, true);
      kernel(lab.m, I);
    }
  }
}
typedef Real ScalarBlock[_BS_][_BS_];
template <int dir, int side> void applyBCface(BlockLab *lab, bool coarse) {
  const int A = 1 - dir;
  if (!coarse) {
    int s[3] = {0, 0, 0}, e[3] = {0, 0, 0};
    const int *const stenBeg = lab->start0;
    const int *const stenEnd = lab->end;
    s[0] = dir == 0 ? (side == 0 ? stenBeg[0] : _BS_) : stenBeg[0];
    s[1] = dir == 1 ? (side == 0 ? stenBeg[1] : _BS_) : stenBeg[1];
    e[0] = dir == 0 ? (side == 0 ? 0 : _BS_ + stenEnd[0] - 1)
                    : _BS_ + stenEnd[0] - 1;
    e[1] = dir == 1 ? (side == 0 ? 0 : _BS_ + stenEnd[1] - 1)
                    : _BS_ + stenEnd[1] - 1;
    for (int iy = s[1]; iy < e[1]; iy++)
      for (int ix = s[0]; ix < e[0]; ix++) {
        const int x = (dir == 0 ? (side == 0 ? 0 : _BS_ - 1) : ix) - stenBeg[0];
        const int y = (dir == 1 ? (side == 0 ? 0 : _BS_ - 1) : iy) - stenBeg[1];
        int i0 = ix - stenBeg[0] + lab->nm[0] * (iy - stenBeg[1]);
        int i1 = x + lab->nm[0] * (y);
        lab->m[2 * i0 + 1 - A] = -lab->m[2 * i1 + 1 - A];
        lab->m[2 * i0 + A] = lab->m[2 * i1 + A];
      }
  } else {
    const int eI[3] = {(lab->end[0]) / 2 + 1 + (2) - 1,
                       (lab->end[1]) / 2 + 1 + (2) - 1,
                       (lab->end[2]) / 2 + 1 + (1) - 1};
    const int sI[3] = {(lab->start0[0] - 1) / 2 + (-1),
                       (lab->start0[1] - 1) / 2 + (-1),
                       (lab->start0[2] - 1) / 2};
    const int *const stenBeg = sI;
    const int *const stenEnd = eI;
    int s[3] = {0, 0, 0}, e[3] = {0, 0, 0};
    s[0] = dir == 0 ? (side == 0 ? stenBeg[0] : _BS_ / 2) : stenBeg[0];
    s[1] = dir == 1 ? (side == 0 ? stenBeg[1] : _BS_ / 2) : stenBeg[1];
    e[0] = dir == 0 ? (side == 0 ? 0 : _BS_ / 2 + stenEnd[0] - 1)
                    : _BS_ / 2 + stenEnd[0] - 1;
    e[1] = dir == 1 ? (side == 0 ? 0 : _BS_ / 2 + stenEnd[1] - 1)
                    : _BS_ / 2 + stenEnd[1] - 1;
    for (int iy = s[1]; iy < e[1]; iy++)
      for (int ix = s[0]; ix < e[0]; ix++) {
        const int x =
            (dir == 0 ? (side == 0 ? 0 : _BS_ / 2 - 1) : ix) - stenBeg[0];
        const int y =
            (dir == 1 ? (side == 0 ? 0 : _BS_ / 2 - 1) : iy) - stenBeg[1];
        int i0 = ix - stenBeg[0] + lab->nc[0] * (iy - stenBeg[1]);
        int i1 = x + lab->nc[0] * (y);
        lab->c[2 * i0 + 1 - A] = -lab->c[2 * i1 + 1 - A];
        lab->c[2 * i0 + A] = lab->c[2 * i1 + A];
      }
  }
}
static void bc_vector(BlockLab *lab, Info *info, bool coarse) {
  assert(lab->dim == 2);
  int n = 1 << info->level;
  if (!coarse) {
    if (info->index[0] == 0)
      applyBCface<0, 0>(lab, false);
    if (info->index[0] == n - 1)
      applyBCface<0, 1>(lab, false);
    if (info->index[1] == 0)
      applyBCface<1, 0>(lab, false);
    if (info->index[1] == n - 1)
      applyBCface<1, 1>(lab, false);
  } else {
    if (info->index[0] == 0)
      applyBCface<0, 0>(lab, coarse);
    if (info->index[0] == n - 1)
      applyBCface<0, 1>(lab, coarse);
    if (info->index[1] == 0)
      applyBCface<1, 0>(lab, coarse);
    if (info->index[1] == n - 1)
      applyBCface<1, 1>(lab, coarse);
  }
}
template <int dir, int side> void Neumann2D(BlockLab *lab, bool coarse) {
  int stenBeg[2];
  int stenEnd[2];
  int bsize[2];
  if (!coarse) {
    stenEnd[0] = lab->end[0];
    stenEnd[1] = lab->end[1];
    stenBeg[0] = lab->start0[0];
    stenBeg[1] = lab->start0[1];
    bsize[0] = _BS_;
    bsize[1] = _BS_;
  } else {
    stenEnd[0] = (lab->end[0]) / 2 + 1 + (2) - 1;
    stenEnd[1] = (lab->end[1]) / 2 + 1 + (2) - 1;
    stenBeg[0] = (lab->start0[0] - 1) / 2 + (-1);
    stenBeg[1] = (lab->start0[1] - 1) / 2 + (-1);
    bsize[0] = _BS_ / 2;
    bsize[1] = _BS_ / 2;
  }
  Real *cb = coarse ? lab->c : lab->m;
  const unsigned int *n = coarse ? lab->nc : lab->nm;
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
};
template <int, int> void Neumann2D(BlockLab *, bool);
void bc_scalar(BlockLab *lab, Info *info, bool coarse) {
  int n = 1 << info->level;
  assert(lab->dim == 1);
  if (info->index[0] == 0)
    Neumann2D<0, 0>(lab, coarse);
  if (info->index[0] == n - 1)
    Neumann2D<0, 1>(lab, coarse);
  if (info->index[1] == 0)
    Neumann2D<1, 0>(lab, coarse);
  if (info->index[1] == n - 1)
    Neumann2D<1, 1>(lab, coarse);
}
static struct {
  Grid *chi, *vel, *vold, *pres, *tmpV, *tmp, *pold;
  struct {
    Grid **g;
    int dim;
    bool basic;
    bool boundary_needed;
    const char *prefix;
  } F[7] = {{&tmp, 1, false, true, "tmp"},    {&chi, 1, false, false, "chi"},
            {&vel, 2, false, false, "vel"},   {&vold, 2, false, false, NULL},
            {&pres, 1, false, false, "pres"}, {&pold, 1, false, false, NULL},
            {&tmpV, 2, true, false, NULL}};
} var;
static void pressure_rhs_fun(BlockLab &velLab, BlockLab &uDefLab,
                             const Info *info, const Info *) {
  Stencil stencil{-1, -1, 2, 2, false};
  const std::vector<Info *> &tmpInfo = var.tmp->infos;
  const std::vector<Info *> &chiInfo = var.chi->infos;
  Real *vm = velLab.m;
  Real *um = uDefLab.m;
  int nm = _BS_ + stencil.ex - stencil.sx - 1;
  const Real h = info->h;
  const Real facDiv = 0.5 * h / sim.dt;
  Real *TMP = tmpInfo[info->id]->block;
  Real *CHI = chiInfo[info->id]->block;
  for (int iy = 0; iy < _BS_; ++iy)
    for (int ix = 0; ix < _BS_; ++ix) {
      int ip0 = ix - stencil.sx;
      int jp0 = iy - stencil.sy;
      int ip1 = ip0 + 1;
      int im1 = ip0 - 1;
      int jp1 = jp0 + 1;
      int jm1 = jp0 - 1;
      Real *v0 = vm + 2 * (nm * jp0 + ip1) + 0;
      Real *v1 = vm + 2 * (nm * jp0 + im1) + 0;
      Real *v2 = vm + 2 * (nm * jp1 + ip0) + 1;
      Real *v3 = vm + 2 * (nm * jm1 + ip0) + 1;
      Real *u0 = um + 2 * (nm * jp0 + ip1) + 0;
      Real *u1 = um + 2 * (nm * jp0 + im1) + 0;
      Real *u2 = um + 2 * (nm * jp1 + ip0) + 1;
      Real *u3 = um + 2 * (nm * jm1 + ip0) + 1;
      TMP[_BS_ * iy + ix] =
          facDiv * (*v0 - *v1 + *v2 - *v3) -
          facDiv * CHI[_BS_ * iy + ix] * (*u0 - *u1 + *u2 - *u3);
    }
};
struct Obstacle {
  Real chi[_BS_][_BS_];
  Real dist[_BS_][_BS_];
  Real udef[_BS_][_BS_][2];
  Real COM_x = 0;
  Real COM_y = 0;
  Real Mass = 0;
  Obstacle() {
    std::fill(&dist[0][0], &dist[0][0] + _BS_ * _BS_, -1);
    memset(&chi[0][0], 0, sizeof(Real) * _BS_ * _BS_);
    memset(&udef[0][0][0], 0, sizeof(Real) * _BS_ * _BS_ * 2);
  }
};
struct KernelVorticity {
  const Stencil stencil{-1, -1, 2, 2, false};
  void operator()(Real *um, const Info *info) const {
    const std::vector<Info *> &tmpInfo = var.tmp->infos;
    const Real i2h = 0.5 * (1 << info->level) * _BS_;
    Real *TMP = tmpInfo[info->id]->block;
    int nm = _BS_ + stencil.ex - stencil.sx - 1;
    for (int j = 0; j < _BS_; ++j)
      for (int i = 0; i < _BS_; ++i) {
        int x0 = i - stencil.sx;
        int y0 = j - stencil.sy;
        int xp = x0 + 1;
        int yp = y0 + 1;
        int xm = x0 - 1;
        int ym = y0 - 1;
        Real *e0 = um + 2 * (nm * ym + x0) + 0;
        Real *e1 = um + 2 * (nm * yp + x0) + 0;
        Real *e2 = um + 2 * (nm * y0 + xp) + 1;
        Real *e3 = um + 2 * (nm * y0 + xm) + 1;
        TMP[j * _BS_ + i] = i2h * (*e0 - *e1 + *e2 - *e3);
      }
  }
};
static void dump(Real time, Info **infos, char *path) {
  long i, j, k, x, y, nblock;
  char xyz_path[FILENAME_MAX], attr_path[FILENAME_MAX];
  FILE *file;
  float xyz[8 * _BS_ * _BS_];
  snprintf(xyz_path, sizeof xyz_path, "%s.xyz.raw", path);
  nblock = var.vel->infos.size();
  char *xyz_base, xdmf_path[FILENAME_MAX];
  FILE *xdmf;
  snprintf(xdmf_path, sizeof xdmf_path, "%s.xdmf2", path);
  xyz_base = xyz_path;
  for (j = 0; xyz_path[j] != '\0'; j++)
    if (xyz_path[j] == '/' && xyz_path[j + 1] != '\0')
      xyz_base = &xyz_path[j + 1];
  xdmf = fopen(xdmf_path, "w");
  fprintf(xdmf,
          "<Xdmf\n"
          "    Version=\"2.0\">\n"
          "  <Domain>\n"
          "    <Grid>\n"
          "      <Time Value=\"%.16e\"/>\n"
          "      <Topology\n"
          "          Dimensions=\"%ld\"\n"
          "          TopologyType=\"Quadrilateral\"/>\n"
          "     <Geometry\n"
          "         GeometryType=\"XY\">\n"
          "       <DataItem\n"
          "           Dimensions=\"%ld 2\"\n"
          "           Format=\"Binary\">\n"
          "         %s\n"
          "       </DataItem>\n"
          "     </Geometry>\n",
          time, _BS_ * _BS_ * nblock, 4 * _BS_ * _BS_ * nblock, xyz_base);
  for (size_t i = 0; i < sizeof var.F / sizeof *var.F; i++)
    if (var.F[i].prefix != NULL) {
      snprintf(attr_path, sizeof attr_path, "%s.%s.raw", path, var.F[i].prefix);
      int dim = var.F[i].dim;
      fprintf(xdmf,
              "       <Attribute\n"
              "           AttributeType=\"%s\"\n"
              "           Name=\"%s\"\n"
              "           Center=\"Cell\">\n"
              "         <DataItem\n"
              "             Dimensions=\"%ld %d\"\n"
              "             Precision=\"%ld\"\n"
              "             Format=\"Binary\">\n"
              "           %s\n"
              "         </DataItem>\n"
              "       </Attribute>\n",
              dim == 2 ? "Vector" : "Scalar", var.F[i].prefix,
              _BS_ * _BS_ * nblock, dim, sizeof(Real),
              attr_path + (xyz_path - xyz_base));
    }
  fprintf(xdmf, "    </Grid>\n"
                "  </Domain>\n"
                "</Xdmf>\n");
  fclose(xdmf);
  file = fopen(xyz_path, "wb");
  for (i = 0; i < nblock; i++) {
    Info *info = infos[i];
    k = 0;
    for (y = 0; y < _BS_; y++)
      for (x = 0; x < _BS_; x++) {
        double u0, v0, u1, v1, h;
        h = 1.0 / _BS_ / (1 << info->level);
        u0 = info->origin[0] + h * x;
        v0 = info->origin[1] + h * y;
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
      }
    fwrite(xyz, sizeof xyz, 1, file);
  }
  fclose(file);

  for (size_t i = 0; i < sizeof var.F / sizeof *var.F; i++)
    if (var.F[i].prefix != NULL) {
      Grid *g = *var.F[i].g;
      int dim = var.F[i].dim;
      snprintf(attr_path, sizeof attr_path, "%s.%s.raw", path, var.F[i].prefix);
      file = fopen(attr_path, "wb");
      for (j = 0; j < nblock; j++)
        fwrite(g->infos[j]->block, sizeof(Real), dim * _BS_ * _BS_, file);
      fclose(file);
    }
}
struct Integrals {
  const Real x, y, m, j, u, v, a;
  Integrals(Real _x, Real _y, Real _m, Real _j, Real _u, Real _v, Real _a)
      : x(_x), y(_y), m(_m), j(_j), u(_u), v(_v), a(_a) {}
  Integrals(const Integrals &c)
      : x(c.x), y(c.y), m(c.m), j(c.j), u(c.u), v(c.v), a(c.a) {}
};
struct Shape {
  float rmax;
  float *sdf;
  int nr;
  int np;
  Real x;
  Real y;
  Real J;
  Real length;
  Real mass;
  Real omega;
  Real orientation;
  Real u;
  Real v;
  std::vector<Obstacle *> obstacleBlocks;
};
struct PutChiOnGrid {
  Stencil stencil{-1, -1, 2, 2, false};
  void operator()(Real *um, const Info *info) const {
    std::vector<Info *> &chiInfo = var.chi->infos;
    int nm = _BS_ + stencil.ex - stencil.sx - 1;
    for (Shape *shape : sim.shapes) {
      std::vector<Obstacle *> &oblock = shape->obstacleBlocks;
      if (oblock[info->id] == nullptr)
        continue;
      Real h = 1.0 / _BS_ / (1 << info->level);
      Real h2 = h * h;
      Obstacle &o = *oblock[info->id];
      o.COM_x = 0;
      o.COM_y = 0;
      o.Mass = 0;
      Real *CHI = chiInfo[info->id]->block;
      Real *chi = (Real *)o.chi;
      Real *dist = (Real *)o.dist;
      for (int iy = 0; iy < _BS_; iy++)
        for (int ix = 0; ix < _BS_; ix++) {
          int j = _BS_ * iy + ix;
          int x0 = ix - stencil.sx;
          int y0 = iy - stencil.sy;
          int xp = x0 + 1;
          int yp = y0 + 1;
          int xm = x0 - 1;
          int ym = y0 - 1;
          if (dist[j] > +h || dist[j] < -h) {
            chi[j] = dist[j] > 0 ? 1 : 0;
          } else {
            Real distPx = *(um + nm * y0 + xp);
            Real distMx = *(um + nm * y0 + xm);
            Real distPy = *(um + nm * yp + x0);
            Real distMy = *(um + nm * ym + x0);
            Real IplusX = std::max(0.0, distPx);
            Real IminuX = std::max(0.0, distMx);
            Real IplusY = std::max(0.0, distPy);
            Real IminuY = std::max(0.0, distMy);
            Real gradIX = IplusX - IminuX;
            Real gradIY = IplusY - IminuY;
            Real gradUX = distPx - distMx;
            Real gradUY = distPy - distMy;
            Real gradUSq = (gradUX * gradUX + gradUY * gradUY) + EPS;
            chi[j] = (gradIX * gradUX + gradIY * gradUY) / gradUSq;
          }
          CHI[j] = std::max(CHI[j], chi[j]);
          if (chi[j] > 0) {
            Real p[2];
            p[0] = info->origin[0] + info->h * (ix + 0.5);
            p[1] = info->origin[1] + info->h * (iy + 0.5);
            o.COM_x += chi[j] * h2 * (p[0] - shape->x);
            o.COM_y += chi[j] * h2 * (p[1] - shape->y);
            o.Mass += chi[j] * h2;
          }
        }
    }
  }
};
static void ongrid() {
  std::vector<Info *> &tmpInfo = var.tmp->infos;
  std::vector<Info *> &chiInfo = var.chi->infos;
  const size_t Nblocks = var.chi->infos.size();
#pragma omp parallel for
  for (size_t i = 0; i < Nblocks; i++) {
    memset(chiInfo[i]->block, 0, _BS_ * _BS_ * sizeof(Real));
    std::fill(tmpInfo[i]->block, tmpInfo[i]->block + _BS_ * _BS_, -1.0);
  }
  for (Shape *shape : sim.shapes) {
    for (auto &entry : shape->obstacleBlocks)
      delete entry;
    shape->obstacleBlocks.clear();
    const auto N = tmpInfo.size();
    shape->obstacleBlocks = std::vector<Obstacle *>(N, nullptr);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < tmpInfo.size(); ++i) {
      const Info *info = tmpInfo[i];
      Obstacle *const block = new Obstacle();
      shape->obstacleBlocks[info->id] = block;
      std::fill(&block->dist[0][0], &block->dist[0][0] + _BS_ * _BS_, -1);
      memset(&block->chi[0][0], 0, sizeof(Real) * _BS_ * _BS_);
      memset(&block->udef[0][0][0], 0, sizeof(Real) * _BS_ * _BS_ * 2);
    }
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < tmpInfo.size(); i++) {
      Obstacle *const block = shape->obstacleBlocks[tmpInfo[i]->id];
      assert(block not_eq nullptr);
      const Info *info = tmpInfo[i];
      Real *b = tmpInfo[i]->block;
      Obstacle *const o = block;
      const Real h = info->h;
      std::fill(&o->dist[0][0], &o->dist[0][0] + _BS_ * _BS_, -1);
      memset(&o->chi[0][0], 0, sizeof(Real) * _BS_ * _BS_);
      memset(&o->udef[0][0][0], 0, sizeof(Real) * _BS_ * _BS_ * 2);
      for (int iy = 0; iy < _BS_; ++iy) {
        for (int ix = 0; ix < _BS_; ++ix) {
          Real c = std::cos(shape->orientation);
          Real s = std::sin(shape->orientation);
          Real x = info->origin[0] + h * (ix + 0.5);
          Real y = info->origin[1] + h * (iy + 0.5);
          x -= shape->x;
          y -= shape->y;
          Real x0 = c * x + s * y;
          Real y0 = -s * x + c * y;
          Real r = sqrt(x0 * x0 + y0 * y0);
          Real p = atan2(y0, x0);
          if (p < 0)
            p += 2 * M_PI;
          int i = r * shape->nr / shape->rmax;
          if (i >= shape->nr)
            i = shape->nr - 1;
          int j = p * (shape->np - 2) / (2 * M_PI);
          if (j >= shape->np)
            j = shape->np - 1;
          Real dist = shape->sdf[i * shape->np + j];
          o->dist[iy][ix] = dist;
          b[iy * _BS_ + ix] = std::max(b[iy * _BS_ + ix], dist);
          o->udef[iy][ix][0] = 0;
          o->udef[iy][ix][1] = 0;
        }
      }
      memset(&o->chi[0][0], 0, sizeof(Real) * _BS_ * _BS_);
    }
  }

  computeA(PutChiOnGrid(), var.tmp, 1);
  for (Shape *shape : sim.shapes) {
    Real com[3] = {0.0, 0.0, 0.0};
    const std::vector<Obstacle *> &oblock = shape->obstacleBlocks;
#pragma omp parallel for reduction(+ : com[:3])
    for (size_t i = 0; i < oblock.size(); i++) {
      if (oblock[i] == nullptr)
        continue;
      com[0] += oblock[i]->Mass;
      com[1] += oblock[i]->COM_x;
      com[2] += oblock[i]->COM_y;
    }
    shape->x += com[1] / com[0];
    shape->y += com[2] / com[0];
  }
  for (Shape *shape : sim.shapes) {
    Real _x = 0, _y = 0, _m = 0, _j = 0, _u = 0, _v = 0, _a = 0;
#pragma omp parallel for schedule(dynamic, 1)                                  \
    reduction(+ : _x, _y, _m, _j, _u, _v, _a)
    for (size_t i = 0; i < chiInfo.size(); i++) {
      const Real hsq = std::pow(chiInfo[i]->h, 2);
      const auto pos = shape->obstacleBlocks[chiInfo[i]->id];
      if (pos == nullptr)
        continue;
      Real *CHI = (Real *)pos->chi;
      Real *UDEF = (Real *)pos->udef;
      for (int iy = 0; iy < _BS_; ++iy)
        for (int ix = 0; ix < _BS_; ++ix) {
          int j = _BS_ * iy + ix;
          if (CHI[j] <= 0)
            continue;
          Real p[2];
          p[0] = chiInfo[i]->origin[0] + chiInfo[i]->h * (ix + 0.5);
          p[1] = chiInfo[i]->origin[1] + chiInfo[i]->h * (iy + 0.5);
          const Real chi = CHI[j] * hsq;
          p[0] -= shape->x;
          p[1] -= shape->y;
          _x += chi * p[0];
          _y += chi * p[1];
          _m += chi;
          _j += chi * (p[0] * p[0] + p[1] * p[1]);
          _u += chi * UDEF[2 * j + 0];
          _v += chi * UDEF[2 * j + 1];
          _a += chi * (p[0] * UDEF[2 * j + 1] - p[1] * UDEF[2 * j + 0]);
        }
    }
    Real quantities[7] = {_x, _y, _m, _j, _u, _v, _a};
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
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < chiInfo.size(); i++) {
      const auto pos = shape->obstacleBlocks[chiInfo[i]->id];
      if (pos == nullptr)
        continue;
      for (int iy = 0; iy < _BS_; ++iy)
        for (int ix = 0; ix < _BS_; ++ix) {
          Real p[2];
          p[0] = chiInfo[i]->origin[0] + chiInfo[i]->h * (ix + 0.5);
          p[1] = chiInfo[i]->origin[1] + chiInfo[i]->h * (iy + 0.5);
          p[0] -= shape->x;
          p[1] -= shape->y;
          pos->udef[iy][ix][0] -= I.u - I.a * p[1];
          pos->udef[iy][ix][1] -= I.v + I.a * p[0];
        }
    }
  }
}
struct GradChiOnTmp {
  GradChiOnTmp() {}
  const Stencil stencil{-4, -4, 5, 5, true};
  void operator()(Real *um, const Info *info) const {
    const std::vector<Info *> &tmpInfo = var.tmp->infos;
    Real *TMP = tmpInfo[info->id]->block;
    int offset = (info->level == sim.levelMax - 1) ? 4 : 2;
    Real threshold = 1e4;
    int nm = _BS_ + stencil.ex - stencil.sx - 1;
    for (int y = -offset; y < _BS_ + offset; ++y)
      for (int x = -offset; x < _BS_ + offset; ++x) {
        int k = nm * (y - stencil.sy) + x - stencil.sx;
        assert(k >= 0);
        um[k] = std::min(um[k], 1.0);
        um[k] = std::max(um[k], 0.0);
        if (um[k] > 0.0 && um[k] < threshold) {
          int i = _BS_ / 2;
          int j = _BS_ / 2 - 1;
          TMP[_BS_ * i + j] = 2 * sim.Rtol;
          TMP[_BS_ * j + j] = 2 * sim.Rtol;
          TMP[_BS_ * i + i] = 2 * sim.Rtol;
          TMP[_BS_ * j + i] = 2 * sim.Rtol;
          break;
        }
      }
  }
};
static void adapt() {
  computeA(KernelVorticity(), var.vel, 2);
  computeA(GradChiOnTmp(), var.chi, 1);
  bool Reduction = false;
  int tmp;
  std::vector<Info *> *I = &var.tmp->infos;
#pragma omp parallel
  {
#pragma omp for schedule(dynamic, 1)
    for (size_t i = 0; i < I->size(); i++) {
      Info *info = getf(&var.tmp->all, (*I)[i]->level, (*I)[i]->Z);
      Real *b = info->block;
      double Linf = 0.0;
      for (int j = 0; j < _BS_ * _BS_; j++)
        Linf = std::max(Linf, std::fabs(b[j]));
      (*I)[i]->state = Linf > sim.Rtol   ? Refine
                       : Linf < sim.Ctol ? Compress
                                         : Leave;
      const bool maxLevel =
          (*I)[i]->state == Refine && (*I)[i]->level == sim.levelMax - 1;
      const bool minLevel = (*I)[i]->state == Compress && (*I)[i]->level == 0;
      if (maxLevel || minLevel)
        (*I)[i]->state = Leave;
      info->state = (*I)[i]->state;
      if (info->state != Leave) {
#pragma omp critical
        {
          if (!Reduction) {
            tmp = 1;
            Reduction = true;
          }
        }
      }
    }
  }
  if (tmp > 0) {
    int levelMin = 0;
    std::vector<Info *> &I = var.tmp->infos;
#pragma omp parallel for
    for (size_t j = 0; j < I.size(); j++) {
      Info *info = I[j];
      if (info->state != Leave) {
        info->changed2 = true;
        (getf(&var.tmp->all, info->level, info->Z))->changed2 = info->changed2;
      }
    }
    for (int m = sim.levelMax - 1; m >= levelMin; m--) {
      for (size_t j = 0; j < I.size(); j++) {
        Info *info = I[j];
        if (info->level == m && info->state != Refine &&
            info->level != sim.levelMax - 1) {
          int TwoPower = 1 << info->level;
          bool xskin = info->index[0] == 0 || info->index[0] == TwoPower - 1;
          bool yskin = info->index[1] == 0 || info->index[1] == TwoPower - 1;
          int xskip = info->index[0] == 0 ? -1 : 1;
          int yskip = info->index[1] == 0 ? -1 : 1;

          if (info->state != Refine)
            for (int x = -1; x < 2; x++)
              for (int y = -1; y < 2; y++)
                if (x != 0 || y != 0) {
                  if (x == xskip && xskin)
                    continue;
                  if (y == yskip && yskin)
                    continue;
                  if (treef(&var.tmp->tree, info->level,
                            info->Znei[1 + x][1 + y]) == -1) {
                    if (info->state == Compress) {
                      info->state = Leave;
                      getf(&var.tmp->all, info->level, info->Z)->state = Leave;
                    }
                    int Bstep = abs(x) + abs(y) == 2 ? 3 : 1;
                    for (int B = 0; B <= 1; B += Bstep) {
                      int aux = abs(x) == 1 ? B % 2 : B / 2;
                      int iNei = 2 * info->index[0] + std::max(x, 0) + x +
                                 (B % 2) * std::max(0, 1 - abs(x));
                      int jNei = 2 * info->index[1] + std::max(y, 0) + y +
                                 aux * std::max(0, 1 - abs(y));
                      long long zzz = forward(m + 1, iNei, jNei);
                      Info *FinerNei = getf(&var.tmp->all, m + 1, zzz);
                      State NeiState = FinerNei->state;
                      if (NeiState == Refine) {
                        info->state = Refine;
                        getf(&var.tmp->all, info->level, info->Z)->state =
                            Refine;
                        info->changed2 = true;
                        getf(&var.tmp->all, info->level, info->Z)->changed2 =
                            true;
                        goto end;
                      }
                    }
                  }
                }
        end:;
        }
      }
      if (m == levelMin)
        break;
      for (size_t j = 0; j < I.size(); j++) {
        Info *info = I[j];
        if (info->level == m && info->state == Compress) {
          int aux = 1 << info->level;
          bool xskin = info->index[0] == 0 || info->index[0] == aux - 1;
          bool yskin = info->index[1] == 0 || info->index[1] == aux - 1;
          int xskip = info->index[0] == 0 ? -1 : 1;
          int yskip = info->index[1] == 0 ? -1 : 1;

          for (int icode = 0; icode < 27; icode++) {
            if (icode == 1 * 1 + 3 * 1 + 9 * 1)
              continue;
            int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                           (icode / 9) % 3 - 1};
            if (code[0] == xskip && xskin)
              continue;
            if (code[1] == yskip && yskin)
              continue;
            if (code[2] != 0)
              continue;
            Info *infoNei = getf(&var.tmp->all, info->level,
                                 info->Znei[1 + code[0]][1 + code[1]]);
            if (Tree1(infoNei, &var.tmp->tree) >= 0 &&
                infoNei->state == Refine) {
              info->state = Leave;
              (getf(&var.tmp->all, info->level, info->Z))->state = Leave;
              break;
            }
          }
        }
      }
    }
    for (size_t jjj = 0; jjj < I.size(); jjj++) {
      Info *info = I[jjj];
      int m = info->level;
      bool found = false;
      for (int i = 2 * (info->index[0] / 2); i <= 2 * (info->index[0] / 2) + 1;
           i++)
        for (int j = 2 * (info->index[1] / 2);
             j <= 2 * (info->index[1] / 2) + 1; j++)
          for (int k = 2 * (info->index[2] / 2);
               k <= 2 * (info->index[2] / 2) + 1; k++) {
            long long n = forward(m, i, j);
            Info *infoNei = getf(&var.tmp->all, m, n);
            if ((Tree1(infoNei, &var.tmp->tree) >= 0) == false ||
                infoNei->state != Compress) {
              found = true;
              if (info->state == Compress) {
                info->state = Leave;
                (getf(&var.tmp->all, info->level, info->Z))->state = Leave;
              }
              break;
            }
          }
      if (found)
        for (int i = 2 * (info->index[0] / 2);
             i <= 2 * (info->index[0] / 2) + 1; i++)
          for (int j = 2 * (info->index[1] / 2);
               j <= 2 * (info->index[1] / 2) + 1; j++)
            for (int k = 2 * (info->index[2] / 2);
                 k <= 2 * (info->index[2] / 2) + 1; k++) {
              long long n = forward(m, i, j);
              Info *infoNei = getf(&var.tmp->all, m, n);
              if (Tree1(infoNei, &var.tmp->tree) >= 0 &&
                  infoNei->state == Compress)
                infoNei->state = Leave;
            }
    }
  }
  struct {
    std::unordered_map<long long, Info *> *all;
    std::vector<Info *> &I2;
  } args[] = {
      {&var.chi->all, var.chi->infos},   {&var.pres->all, var.pres->infos},
      {&var.pold->all, var.pold->infos}, {&var.vel->all, var.vel->infos},
      {&var.vold->all, var.vold->infos}, {&var.tmpV->all, var.tmpV->infos},
  };
  for (size_t iarg = 0; iarg < sizeof args / sizeof *args; iarg++) {
    for (size_t i1 = 0; i1 < args[iarg].I2.size(); i1++) {
      Info *ary0 = args[iarg].I2[i1];
      Info *info = getf(args[iarg].all, ary0->level, ary0->Z);
      for (int i = 2 * (info->index[0] / 2); i <= 2 * (info->index[0] / 2) + 1;
           i++)
        for (int j = 2 * (info->index[1] / 2);
             j <= 2 * (info->index[1] / 2) + 1; j++) {
          const long long n = forward(info->level, i, j);
          Info *infoNei = getf(args[iarg].all, info->level, n);
          infoNei->state = Leave;
        }
      info->state = Leave;
      ary0->state = Leave;
    }
#pragma omp parallel for
    for (size_t i = 0; i < var.tmp->infos.size(); i++) {
      const Info *info1 = var.tmp->infos[i];
      Info *info2 = args[iarg].I2[i];
      Info *info3 = getf(args[iarg].all, info2->level, info2->Z);
      info2->state = info1->state;
      info3->state = info1->state;
      if (info2->state == Compress) {
        const int i2 = 2 * (info2->index[0] / 2);
        const int j2 = 2 * (info2->index[1] / 2);
        const long long n = forward(info2->level, i2, j2);
        Info *infoNei = getf(args[iarg].all, info2->level, n);
        infoNei->state = Compress;
      }
    }
  }
  for (size_t i = 0; i < sizeof var.F / sizeof *var.F; i++) {
    Grid *g = (*var.F[i].g);
    bool basic = var.F[i].basic;
    int dim = var.F[i].dim;
    const Stencil stencil{-1, -1, 2, 2, true};
    int r = 0;
    int c = 0;
    std::vector<int> m_com;
    std::vector<int> m_ref;
    std::vector<long long> n_com;
    std::vector<long long> n_ref;
    std::vector<Info *> &I = g->infos;
    long long blocks_after = I.size();
    for (auto &info : I) {
      if (info->state == Refine) {
        m_ref.push_back(info->level);
        n_ref.push_back(info->Z);
        blocks_after += (1 << 2) - 1;
        r++;
      } else if (info->state == Compress && info->index[0] % 2 == 0 &&
                 info->index[1] % 2 == 0 && info->index[2] % 2 == 0) {
        m_com.push_back(info->level);
        n_com.push_back(info->Z);
        c++;
      } else if (info->state == Compress) {
        blocks_after--;
      }
    }
    int result[2] = {r, c};
    int size = 1;
    std::vector<long long> block_distribution(size);
    std::vector<long long> dealloc_IDs;
    BlockLab lab(dim);
    if (basic == false)
      lab.prepare(stencil);
    for (size_t i = 0; i < m_ref.size(); i++) {
      const int level = m_ref[i];
      const long long Z = n_ref[i];
      Info *parent = getf(&g->all, level, Z);
      parent->state = Leave;
      if (basic == false)
        lab.load(&g->tree, &g->all, stencil, parent, true);
      const int p[3] = {parent->index[0], parent->index[1], parent->index[2]};
      assert(parent->block != NULL);
      assert(level <= sim.levelMax - 1);
      Real *Blocks[4];
      for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++) {
          long long Z = forward(level + 1, 2 * p[0] + i, 2 * p[1] + j);
          Info *info = getf(&g->all, level + 1, Z);
          info->state = Leave;
          info->block = (Real *)calloc(dim * _BS_ * _BS_, sizeof(Real));
#pragma omp critical
          { g->infos.push_back(info); }
          treef(&g->tree, level + 1, Z) = -2;
          Blocks[j * 2 + i] = info->block;
        }
      if (basic == false) {
        int nm = _BS_ + stencil.ex - stencil.sx - 1;
        int offsetX[2] = {0, _BS_ / 2};
        int offsetY[2] = {0, _BS_ / 2};
        Real *um = lab.m;
        for (int J = 0; J < 2; J++)
          for (int I = 0; I < 2; I++) {
            Real *b = Blocks[J * 2 + I];
            memset(b, 0, dim * _BS_ * _BS_ * sizeof(Real));
            for (int j = 0; j < _BS_; j += 2)
              for (int i = 0; i < _BS_; i += 2) {
                int i0 = i / 2 + offsetX[I] - stencil.sx;
                int j0 = j / 2 + offsetY[J] - stencil.sy;
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
      { dealloc_IDs.push_back(getf(&g->all, level, Z)->id2); }
      Info *parent = getf(&g->all, level, Z);
      Tree1(parent, &g->tree) = -1;
      parent->state = Leave;
      int p[3] = {parent->index[0], parent->index[1], parent->index[2]};
      for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++) {
          const long long nc = forward(level + 1, 2 * p[0] + i, 2 * p[1] + j);
          Info *Child = getf(&g->all, level + 1, nc);
          Tree1(Child, &g->tree) = sim.rank;
          if (level + 2 < sim.levelMax)
            for (int i0 = 0; i0 < 2; i0++)
              for (int i1 = 0; i1 < 2; i1++)
                treef(&g->tree, level + 2, Child->Zchild[i0][i1]) = -2;
        }
    }
    dealloc_many(dealloc_IDs, &g->infos);
    for (auto &b : I) {
      const long long nBlock =
          forward(b->level, 2 * (b->index[0] / 2), 2 * (b->index[1] / 2));
      const Info *base = getf(&g->all, b->level, nBlock);
      if (!(Tree1(base, &g->tree) >= 0) || base->state != Compress)
        continue;
      if (b->Z != nBlock) {
        /**/
      } else {
        for (int j = 0; j < 2; j++)
          for (int i = 0; i < 2; i++) {
            const long long n =
                forward(b->level, b->index[0] + i, b->index[1] + j);
            if (n == nBlock)
              continue;
          }
      }
    }
    dealloc_IDs.clear();
    for (size_t i = 0; i < m_com.size(); i++) {
      const int level = m_com[i];
      const long long Z = n_com[i];
      assert(level > 0);
      Info *info = getf(&g->all, level, Z);
      assert(info->state == Compress);
      Real *Blocks[4];
      for (int J = 0; J < 2; J++)
        for (int I = 0; I < 2; I++) {
          const int blk = J * 2 + I;
          const long long n =
              forward(level, info->index[0] + I, info->index[1] + J);
          Blocks[blk] = (getf(&g->all, level, n))->block;
        }
      const int offsetX[2] = {0, _BS_ / 2};
      const int offsetY[2] = {0, _BS_ / 2};
      if (basic == false)
        for (int J = 0; J < 2; J++)
          for (int I = 0; I < 2; I++) {
            Real *b = Blocks[J * 2 + I];
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
          forward(level - 1, info->index[0] / 2, info->index[1] / 2);
      Info *parent = getf(&g->all, level - 1, np);
      treef(&g->tree, parent->level, parent->Z) = sim.rank;
      parent->block = info->block;
      parent->state = Leave;
      if (level - 2 >= 0)
        treef(&g->tree, level - 2, parent->Zparent) = -1;
      for (int J = 0; J < 2; J++)
        for (int I = 0; I < 2; I++) {
          const long long n =
              forward(level, info->index[0] + I, info->index[1] + J);
          if (I + J == 0) {
            for (size_t j = 0; j < g->infos.size(); j++)
              if (level == g->infos[j]->level && n == g->infos[j]->Z) {
                Info *correct_info = getf(&g->all, level - 1, np);
                correct_info->state = Leave;
                g->infos[j] = correct_info;
                break;
              }
          } else {
#pragma omp critical
            { dealloc_IDs.push_back(getf(&g->all, level, n)->id2); }
          }
          treef(&g->tree, level, n) = -2;
          getf(&g->all, level, n)->state = Leave;
        }
    }
    dealloc_many(dealloc_IDs, &g->infos);
    long long max_b = block_distribution[0];
    long long min_b = block_distribution[0];
    for (auto &b : block_distribution) {
      max_b = std::max(max_b, b);
      min_b = std::min(min_b, b);
    }
    fill_pos(&g->infos, &g->all);
    if (result[0] > 0 || result[1] > 0) {
      g->UpdateFluxCorrection = true;
      update_blocks(&g->infos, &g->all, &g->tree);
    }
    //    delete lab;
  }
}
struct KernelAdvectDiffuse {
  Stencil stencil{-3, -3, 4, 4, true};
  void operator()(Real *um, Info *info) {
    std::vector<Info *> &tmpVInfo = var.tmpV->infos;
    Real h = info->h;
    Real dfac = sim.nu * sim.dt;
    Real afac = -sim.dt * h;
    Real *TMP = tmpVInfo[info->id]->block;
    int nm = _BS_ + stencil.ex - stencil.sx - 1;
    for (int iy = 0; iy < _BS_; ++iy)
      for (int ix = 0; ix < _BS_; ++ix) {
        int ip0 = ix - stencil.sx;
        int jp0 = iy - stencil.sy;
        int ip1 = ip0 + 1;
        int ip2 = ip0 + 2;
        int ip3 = ip0 + 3;
        int im1 = ip0 - 1;
        int im2 = ip0 - 2;
        int im3 = ip0 - 3;
        int jp1 = jp0 + 1;
        int jp2 = jp0 + 2;
        int jp3 = jp0 + 3;
        int jm1 = jp0 - 1;
        int jm2 = jp0 - 2;
        int jm3 = jp0 - 3;
        Real u = *(um + 2 * (nm * jp0 + ip0) + 0);
        Real v = *(um + 2 * (nm * jp0 + ip0) + 1);
        Real up1x0 = *(um + 2 * (nm * jp0 + ip1) + 0);
        Real up2x0 = *(um + 2 * (nm * jp0 + ip2) + 0);
        Real up3x0 = *(um + 2 * (nm * jp0 + ip3) + 0);
        Real um1x0 = *(um + 2 * (nm * jp0 + im1) + 0);
        Real um2x0 = *(um + 2 * (nm * jp0 + im2) + 0);
        Real um3x0 = *(um + 2 * (nm * jp0 + im3) + 0);
        Real up1y0 = *(um + 2 * (nm * jp1 + ip0) + 0);
        Real up2y0 = *(um + 2 * (nm * jp2 + ip0) + 0);
        Real up3y0 = *(um + 2 * (nm * jp3 + ip0) + 0);
        Real um1y0 = *(um + 2 * (nm * jm1 + ip0) + 0);
        Real um2y0 = *(um + 2 * (nm * jm2 + ip0) + 0);
        Real um3y0 = *(um + 2 * (nm * jm3 + ip0) + 0);
        Real up1x1 = *(um + 2 * (nm * jp0 + ip1) + 1);
        Real up2x1 = *(um + 2 * (nm * jp0 + ip2) + 1);
        Real up3x1 = *(um + 2 * (nm * jp0 + ip3) + 1);
        Real um1x1 = *(um + 2 * (nm * jp0 + im1) + 1);
        Real um2x1 = *(um + 2 * (nm * jp0 + im2) + 1);
        Real um3x1 = *(um + 2 * (nm * jp0 + im3) + 1);
        Real up1y1 = *(um + 2 * (nm * jp1 + ip0) + 1);
        Real up2y1 = *(um + 2 * (nm * jp2 + ip0) + 1);
        Real up3y1 = *(um + 2 * (nm * jp3 + ip0) + 1);
        Real um1y1 = *(um + 2 * (nm * jm1 + ip0) + 1);
        Real um2y1 = *(um + 2 * (nm * jm2 + ip0) + 1);
        Real um3y1 = *(um + 2 * (nm * jm3 + ip0) + 1);
        Real dudx = derivative(u, um3x0, um2x0, um1x0, u, up1x0, up2x0, up3x0);
        Real dudy = derivative(v, um3y0, um2y0, um1y0, u, up1y0, up2y0, up3y0);
        Real dvdx = derivative(u, um3x1, um2x1, um1x1, v, up1x1, up2x1, up3x1);
        Real dvdy = derivative(v, um3y1, um2y1, um1y1, v, up1y1, up2y1, up3y1);
        TMP[2 * (_BS_ * iy + ix)] =
            afac * (u * dudx + v * dudy) +
            dfac * (up1x0 + um1x0 + up1y0 + um1y0 - 4 * u);
        TMP[2 * (_BS_ * iy + ix) + 1] =
            afac * (u * dvdx + v * dvdy) +
            dfac * (up1x1 + um1x1 + up1y1 + um1y1 - 4 * v);
      }
  }
};
struct Solver {
  Solver()
      : GenericCell(), XminCell(), XmaxCell(), YminCell(),
        YmaxCell(), edgeIndexers{&XminCell, &XmaxCell, &YminCell, &YmaxCell} {}
  struct CellIndexer {
    ~CellIndexer() = default;
    long long This(const Info *info, int ix, int iy) const {
      return blockOffset(info) + (long long)(iy * _BS_ + ix);
    }
    long long Xmin(const Info *info, int, int iy, int offset) const {
      return blockOffset(info) + (long long)(iy * _BS_ + offset);
    }
    long long Xmax(const Info *info, int, int iy, int offset = 0) const {
      return blockOffset(info) + (long long)(iy * _BS_ + (_BS_ - 1 - offset));
    }
    long long Ymin(const Info *info, int ix, int, int offset = 0) const {
      return blockOffset(info) + (long long)(offset * _BS_ + ix);
    }
    long long Ymax(const Info *info, int ix, int, int offset = 0) const {
      return blockOffset(info) + (long long)((_BS_ - 1 - offset) * _BS_ + ix);
    }
    long long blockOffset(const Info *info) const {
      return (info->id + sim.nblocks[Tree1(info, &var.tmp->tree)]) *
             (_BS_ * _BS_);
    }
    static int ix_f(int ix) { return (ix % (_BS_ / 2)) * 2; }
    static int iy_f(int iy) { return (iy % (_BS_ / 2)) * 2; }
  };
  struct EdgeCellIndexer : public CellIndexer {
    EdgeCellIndexer() : CellIndexer() {}
    virtual long long neiUnif(const Info *nei_info, int ix, int iy) const = 0;
    virtual long long neiInward(const Info *info, int ix, int iy) const = 0;
    virtual double taylorSign(int ix, int iy) const = 0;
    virtual int ix_c(const Info *info, int ix) const {
      return info->index[0] % 2 == 0 ? ix / 2 : ix / 2 + _BS_ / 2;
    }
    virtual int iy_c(const Info *info, int iy) const {
      return info->index[1] % 2 == 0 ? iy / 2 : iy / 2 + _BS_ / 2;
    }
    virtual long long neiFine1(const Info *nei_info, int ix, int iy,
                               int offset = 0) const = 0;
    virtual long long neiFine2(const Info *nei_info, int ix, int iy,
                               int offset = 0) const = 0;
    virtual bool isBD(int ix, int iy) const = 0;
    virtual bool isFD(int ix, int iy) const = 0;
    virtual long long Nei(const Info *info, int ix, int iy, int dist) const = 0;
    virtual long long Zchild(const Info *nei_info, int ix, int iy) const = 0;
  };
  struct XbaseIndexer : public EdgeCellIndexer {
    XbaseIndexer() : EdgeCellIndexer() {}
    double taylorSign(int, int iy) const override {
      return iy % 2 == 0 ? -1. : 1.;
    }
    bool isBD(int, int iy) const override {
      return iy == _BS_ - 1 || iy == _BS_ / 2 - 1;
    }
    bool isFD(int, int iy) const override { return iy == 0 || iy == _BS_ / 2; }
    long long Nei(const Info *info, int ix, int iy, int dist) const override {
      return This(info, ix, iy + dist);
    }
  };
  struct XminIndexer : public XbaseIndexer {
    XminIndexer() : XbaseIndexer() {}
    long long neiUnif(const Info *nei_info, int ix, int iy) const override {
      return Xmax(nei_info, ix, iy);
    }
    long long neiInward(const Info *info, int ix, int iy) const override {
      return This(info, ix + 1, iy);
    }
    int ix_c(const Info *, int) const override { return _BS_ - 1; }
    long long neiFine1(const Info *nei_info, int ix, int iy,
                       int offset = 0) const override {
      return Xmax(nei_info, ix_f(ix), iy_f(iy), offset);
    }
    long long neiFine2(const Info *nei_info, int ix, int iy,
                       int offset = 0) const override {
      return Xmax(nei_info, ix_f(ix), iy_f(iy) + 1, offset);
    }
    long long Zchild(const Info *nei_info, int, int iy) const override {
      return nei_info->Zchild[1][int(iy >= _BS_ / 2)];
    }
  };
  struct XmaxIndexer : public XbaseIndexer {
    XmaxIndexer() : XbaseIndexer() {}
    long long neiUnif(const Info *nei_info, int ix, int iy) const override {
      return Xmin(nei_info, ix, iy, 0);
    }
    long long neiInward(const Info *info, int ix, int iy) const override {
      return This(info, ix - 1, iy);
    }
    int ix_c(const Info *, int) const override { return 0; }
    long long neiFine1(const Info *nei_info, int ix, int iy,
                       int offset = 0) const override {
      return Xmin(nei_info, ix_f(ix), iy_f(iy), offset);
    }
    long long neiFine2(const Info *nei_info, int ix, int iy,
                       int offset = 0) const override {
      return Xmin(nei_info, ix_f(ix), iy_f(iy) + 1, offset);
    }
    long long Zchild(const Info *nei_info, int, int iy) const override {
      return nei_info->Zchild[0][int(iy >= _BS_ / 2)];
    }
  };
  struct YbaseIndexer : public EdgeCellIndexer {
    YbaseIndexer() : EdgeCellIndexer() {}
    double taylorSign(int ix, int) const override {
      return ix % 2 == 0 ? -1. : 1.;
    }
    bool isBD(int ix, int) const override {
      return ix == _BS_ - 1 || ix == _BS_ / 2 - 1;
    }
    bool isFD(int ix, int) const override { return ix == 0 || ix == _BS_ / 2; }
    long long Nei(const Info *info, int ix, int iy, int dist) const override {
      return This(info, ix + dist, iy);
    }
  };
  struct YminIndexer : public YbaseIndexer {
    YminIndexer() : YbaseIndexer() {}
    long long neiUnif(const Info *nei_info, int ix, int iy) const override {
      return Ymax(nei_info, ix, iy);
    }
    long long neiInward(const Info *info, int ix, int iy) const override {
      return This(info, ix, iy + 1);
    }
    int iy_c(const Info *, int) const override { return _BS_ - 1; }
    long long neiFine1(const Info *nei_info, int ix, int iy,
                       int offset = 0) const override {
      return Ymax(nei_info, ix_f(ix), iy_f(iy), offset);
    }
    long long neiFine2(const Info *nei_info, int ix, int iy,
                       int offset = 0) const override {
      return Ymax(nei_info, ix_f(ix) + 1, iy_f(iy), offset);
    }
    long long Zchild(const Info *nei_info, int ix, int) const override {
      return nei_info->Zchild[int(ix >= _BS_ / 2)][1];
    }
  };
  struct YmaxIndexer : public YbaseIndexer {
    YmaxIndexer() : YbaseIndexer() {}
    long long neiUnif(const Info *nei_info, int ix, int iy) const override {
      return Ymin(nei_info, ix, iy);
    }
    long long neiInward(const Info *info, int ix, int iy) const override {
      return This(info, ix, iy - 1);
    }
    int iy_c(const Info *, int) const override { return 0; }
    long long neiFine1(const Info *nei_info, int ix, int iy,
                       int offset = 0) const override {
      return Ymin(nei_info, ix_f(ix), iy_f(iy), offset);
    }
    long long neiFine2(const Info *nei_info, int ix, int iy,
                       int offset = 0) const override {
      return Ymin(nei_info, ix_f(ix) + 1, iy_f(iy), offset);
    }
    long long Zchild(const Info *nei_info, int ix, int) const override {
      return nei_info->Zchild[int(ix >= _BS_ / 2)][0];
    }
  };
  CellIndexer GenericCell;
  XminIndexer XminCell;
  XmaxIndexer XmaxCell;
  YminIndexer YminCell;
  YmaxIndexer YmaxCell;
  std::array<const EdgeCellIndexer *, 4> edgeIndexers;
  std::array<std::pair<long long, double>, 3>
  D1(const Info *info, const EdgeCellIndexer *indexer, int ix, int iy) const {
    if (indexer->isBD(ix, iy))
      return {{{indexer->Nei(info, ix, iy, -2), 1. / 8.},
               {indexer->Nei(info, ix, iy, -1), -1. / 2.},
               {indexer->This(info, ix, iy), 3. / 8.}}};
    else if (indexer->isFD(ix, iy))
      return {{{indexer->Nei(info, ix, iy, 2), -1. / 8.},
               {indexer->Nei(info, ix, iy, 1), 1. / 2.},
               {indexer->This(info, ix, iy), -3. / 8.}}};
    return {{{indexer->Nei(info, ix, iy, -1), -1. / 8.},
             {indexer->Nei(info, ix, iy, 1), 1. / 8.},
             {indexer->This(info, ix, iy), 0.}}};
  }
  std::array<std::pair<long long, double>, 3>
  D2(const Info *info, const EdgeCellIndexer *indexer, int ix, int iy) const {
    if (indexer->isBD(ix, iy))
      return {{{indexer->Nei(info, ix, iy, -2), 1. / 32.},
               {indexer->Nei(info, ix, iy, -1), -1. / 16.},
               {indexer->This(info, ix, iy), 1. / 32.}}};
    else if (indexer->isFD(ix, iy))
      return {{{indexer->Nei(info, ix, iy, 2), 1. / 32.},
               {indexer->Nei(info, ix, iy, 1), -1. / 16.},
               {indexer->This(info, ix, iy), 1. / 32.}}};
    return {{{indexer->Nei(info, ix, iy, -1), 1. / 32.},
             {indexer->Nei(info, ix, iy, 1), 1. / 32.},
             {indexer->This(info, ix, iy), -1. / 16.}}};
  }
  void interpolate(const Info *info_c, int ix_c, int iy_c, const Info *info_f,
                   long long fine_close_idx, long long fine_far_idx,
                   double signInt, double signTaylor,
                   const EdgeCellIndexer *indexer, SpRowInfo &row) const {
    int rank_c = Tree1(info_c, &var.tmp->tree);
    int rank_f = Tree1(info_f, &var.tmp->tree);
    row.mapColVal(rank_f, fine_close_idx, signInt * 2. / 3.);
    row.mapColVal(rank_f, fine_far_idx, -signInt * 1. / 5.);
    const double tf = signInt * 8. / 15.;
    row.mapColVal(rank_c, indexer->This(info_c, ix_c, iy_c), tf);
    std::array<std::pair<long long, double>, 3> D;
    D = D1(info_c, indexer, ix_c, iy_c);
    for (int i(0); i < 3; i++)
      row.mapColVal(rank_c, D[i].first, signTaylor * tf * D[i].second);
    D = D2(info_c, indexer, ix_c, iy_c);
    for (int i(0); i < 3; i++)
      row.mapColVal(rank_c, D[i].first, tf * D[i].second);
  }
  void makeFlux(const Info *rhs_info, int ix, int iy, const Info *rhsNei,
                const EdgeCellIndexer *indexer, SpRowInfo &row) const {
    long long sfc_idx = indexer->This(rhs_info, ix, iy);
    if (Tree1(rhsNei, &var.tmp->tree) >= 0) {
      int nei_rank = Tree1(rhsNei, &var.tmp->tree);
      long long nei_idx = indexer->neiUnif(rhsNei, ix, iy);
      row.mapColVal(nei_rank, nei_idx, 1.);
      row.mapColVal(sfc_idx, -1.);
    } else if (Tree1(rhsNei, &var.tmp->tree) == -2) {
      Info *rhsNei_c =
          getf(&var.tmp->all, rhs_info->level - 1, rhsNei->Zparent);
      int ix_c = indexer->ix_c(rhs_info, ix);
      int iy_c = indexer->iy_c(rhs_info, iy);
      long long inward_idx = indexer->neiInward(rhs_info, ix, iy);
      double signTaylor = indexer->taylorSign(ix, iy);
      interpolate(rhsNei_c, ix_c, iy_c, rhs_info, sfc_idx, inward_idx, 1.,
                  signTaylor, indexer, row);
      row.mapColVal(sfc_idx, -1.);
    } else if (Tree1(rhsNei, &var.tmp->tree) == -1) {
      Info *rhsNei_f = getf(&var.tmp->all, rhs_info->level + 1,
                            indexer->Zchild(rhsNei, ix, iy));
      int nei_rank = Tree1(rhsNei_f, &var.tmp->tree);
      long long fine_close_idx = indexer->neiFine1(rhsNei_f, ix, iy, 0);
      long long fine_far_idx = indexer->neiFine1(rhsNei_f, ix, iy, 1);
      row.mapColVal(nei_rank, fine_close_idx, 1.);
      interpolate(rhs_info, ix, iy, rhsNei_f, fine_close_idx, fine_far_idx, -1.,
                  -1., indexer, row);
      fine_close_idx = indexer->neiFine2(rhsNei_f, ix, iy, 0);
      fine_far_idx = indexer->neiFine2(rhsNei_f, ix, iy, 1);
      row.mapColVal(nei_rank, fine_close_idx, 1.);
      interpolate(rhs_info, ix, iy, rhsNei_f, fine_close_idx, fine_far_idx, -1.,
                  1., indexer, row);
    } else {
      throw std::runtime_error(
          "Neighbour doesn't exist, isn't coarser, nor finer...");
    }
  }
  void getVec() {
    std::vector<Info *> &RhsInfo = var.tmp->infos;
    std::vector<Info *> &zInfo = var.pres->infos;
    int Nblocks = RhsInfo.size();
    std::vector<double> &x = sim.mat->x_;
    std::vector<double> &b = sim.mat->b_;
    std::vector<double> &h2 = sim.mat->h2_;
    long long shift = -sim.nrows[sim.rank];
#pragma omp parallel for
    for (int i = 0; i < Nblocks; i++) {
      Real *rhs = RhsInfo[i]->block;
      Real *p = zInfo[i]->block;
      h2[i] = RhsInfo[i]->h * RhsInfo[i]->h;
      for (int iy = 0; iy < _BS_; iy++)
        for (int ix = 0; ix < _BS_; ix++) {
          int j = iy * _BS_ + ix;
          long long sfc_loc = GenericCell.This(RhsInfo[i], ix, iy) + shift;
          b[sfc_loc] = rhs[j];
          x[sfc_loc] = p[j];
        }
    }
  }
};
struct pressureCorrectionKernel {
  const Stencil stencil{-1, -1, 2, 2, false};
  void operator()(Real *um, const Info *info) const {
    const std::vector<Info *> &tmpVInfo = var.tmpV->infos;
    int nm = _BS_ + stencil.ex - stencil.sx - 1;
    const Real h = info->h, pFac = -0.5 * sim.dt * h;
    Real *tmpV = tmpVInfo[info->id]->block;
    for (int iy = 0; iy < _BS_; ++iy)
      for (int ix = 0; ix < _BS_; ++ix) {
        int ip0 = ix - stencil.sx;
        int jp0 = iy - stencil.sy;
        int ip1 = ip0 + 1;
        int jp1 = jp0 + 1;
        int im1 = ip0 - 1;
        int jm1 = jp0 - 1;
        Real *p0 = um + nm * jp0 + ip1;
        Real *p1 = um + nm * jp0 + im1;
        Real *p2 = um + nm * jp1 + ip0;
        Real *p3 = um + nm * jm1 + ip0;
        tmpV[2 * (_BS_ * iy + ix)] = pFac * (*p0 - *p1);
        tmpV[2 * (_BS_ * iy + ix) + 1] = pFac * (*p2 - *p3);
      }
  }
};
struct pressure_rhs1 {
  pressure_rhs1() {}
  Stencil stencil{-1, -1, 2, 2, false};
  void operator()(Real *um, const Info *info) const {
    Real *TMP = var.tmp->infos[info->id]->block;
    int nm = _BS_ + stencil.ex - stencil.sx - 1;
    for (int iy = 0; iy < _BS_; ++iy)
      for (int ix = 0; ix < _BS_; ++ix) {
        int ip0 = ix - stencil.sx;
        int jp0 = iy - stencil.sy;
        int ip1 = ip0 + 1;
        int jp1 = jp0 + 1;
        int im1 = ip0 - 1;
        int jm1 = jp0 - 1;
        Real *l0 = um + nm * jp0 + ip0;
        Real *l1 = um + nm * jp0 + im1;
        Real *l2 = um + nm * jp0 + ip1;
        Real *l3 = um + nm * jm1 + ip0;
        Real *l4 = um + nm * jp1 + ip0;
        TMP[_BS_ * iy + ix] -= *l1 + *l2 + *l3 + *l4 - 4 * (*l0);
      }
  }
};
static std::string trim(std::string str) {
  size_t i = 0, j = str.length();
  while (i < j && isspace(str[i]))
    i++;
  while (j > i && isspace(str[j - 1]))
    j--;
  return str.substr(i, j - i);
}
struct LineParser : public CommandlineParser {
  LineParser(std::istringstream &is_line) : CommandlineParser(0, NULL) {
    std::string key, value;
    while (std::getline(is_line, key, '=')) {
      if (std::getline(is_line, value, ' ')) {
        mapArguments[trim(key)] = Value(trim(value));
      }
    }
  }
};

int main(int argc, char **argv) {
  CommandlineParser parser(argc, argv);
  feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);

#ifdef _OPENMP
#pragma omp parallel
  {
#pragma omp master
    if (sim.rank == 0)
      fprintf(stderr, "main.cpp: %d threads\n", omp_get_num_threads());
  }
#endif
  sim.levelMax = parser("levelMax").asInt();
  sim.Rtol = parser("Rtol").asDouble();
  sim.Ctol = parser("Ctol").asDouble();
  sim.AdaptSteps = parser("AdaptSteps").asInt();
  sim.levelStart = parser("levelStart").asInt();
  sim.CFL = parser("CFL").asDouble();
  sim.endTime = parser("tend").asDouble();
  sim.lambda = parser("lambda").asDouble();
  sim.nu = parser("nu").asDouble();
  sim.PoissonTol = parser("poissonTol").asDouble();
  sim.PoissonTolRel = parser("poissonTolRel").asDouble();
  sim.maxPoissonRestarts = parser("maxPoissonRestarts").asInt();
  sim.dumpTime = parser("tdump").asDouble();

  std::string shapeArg = parser("shapes").asString();
  std::stringstream descriptors(shapeArg);
  std::string lines;
  while (std::getline(descriptors, lines)) {
    std::stringstream ss(lines);
    std::string line;
    while (std::getline(ss, line, ',')) {
      std::istringstream line_stream(line);
      LineParser p(line_stream);
      Shape *shape = new Shape;
      shape->x = p("xcenter").asDouble();
      shape->y = p("ycenter").asDouble();
      shape->orientation = p("orientation").asDouble() * M_PI / 180;
      shape->omega = p("omega").asDouble();
      Real scale = p("scale").asDouble();
      std::string path0 = p("sdf").asString();
      const char *path = path0.c_str();
      FILE *file = fopen(path, "r");
      if (file == NULL) {
        fprintf(stderr, "main.cpp: error: fail to open '%s'\n", path);
        exit(1);
      }
      char tag[3] = {0};
      float area, J, length, rmax;
      fread(tag, sizeof *tag, sizeof tag, file);
      if (tag[0] != 'S' || tag[1] != 'D' || tag[2] != 'F') {
        fprintf(stderr, "main.cpp: error: not and sdf file\n");
        exit(1);
      }
      if (fread(&length, sizeof(length), 1, file) != 1 ||
          fread(&area, sizeof(area), 1, file) != 1 ||
          fread(&J, sizeof(J), 1, file) != 1 ||
          fread(&rmax, sizeof(rmax), 1, file) != 1 ||
          fread(&shape->nr, sizeof(shape->nr), 1, file) != 1 ||
          fread(&shape->np, sizeof(shape->np), 1, file) != 1) {
        fprintf(stderr,
                "main.cpp: error: fail to read shape header from file.\n");
        exit(1);
      }
      size_t ncount = shape->nr * shape->np;
      if ((shape->sdf = (float *)malloc(ncount * sizeof(float))) == NULL) {
        fprintf(stderr, "main.cpp: error: malloc() failed\n");
        exit(1);
      }
      if (fread(shape->sdf, sizeof *shape->sdf, ncount, file) != ncount) {
        fprintf(stderr, "main.cpp: error: fail to read arrays from '%s'\n",
                path);
      }
      shape->J = scale * J;
      shape->length = scale * length;
      shape->mass = scale * area;
      shape->rmax = scale * rmax;
      for (size_t i = 0; i < ncount; i++)
        shape->sdf[i] *= scale;
      shape->u = 0;
      shape->v = 0;

      sim.shapes.push_back(shape);
    }
  }

  sim.nblocks.resize(1);
  sim.nrows.resize(1);
  sim.levels.resize(sim.levelMax);
  sim.levels[0] = 0;
  for (int m = 0; m < sim.levelMax - 1; m++)
    sim.levels[m + 1] = sim.levels[m] + (1 << (2 * m));
  long long my_blocks = 1LL << (2 * sim.levelStart);
  for (size_t i = 0; i < sizeof var.F / sizeof *var.F; i++) {
    int dim = var.F[i].dim;
    Grid *g = *var.F[i].g = new Grid;
    for (size_t i = 0; i < my_blocks; i++) {
      long long Z = i;
      long long aux = sim.levels[sim.levelStart] + Z;
      Info *info = g->all[aux] = new Info;
      fill(info, sim.levelStart, Z);
      info->block = (Real *)calloc(dim * _BS_ * _BS_, sizeof(Real));
      g->infos.push_back(info);
      g->tree[aux] = sim.rank;
      int p[2];
      sfc_inverse(Z, sim.levelStart, &p[0], &p[1]);
      if (sim.levelStart < sim.levelMax - 1)
        for (int j1 = 0; j1 < 2; j1++)
          for (int i1 = 0; i1 < 2; i1++) {
            long long n =
                forward(sim.levelStart + 1, 2 * p[0] + i1, 2 * p[1] + j1);
            g->tree[sim.levels[sim.levelStart + 1] + n] = -2;
          }
      if (sim.levelStart > 0) {
        long long n = forward(sim.levelStart - 1, p[0] / 2, p[1] / 2);
        g->tree[sim.levels[sim.levelStart - 1] + n] = -1;
      }
    }
    std::sort(std::begin(g->infos), std::end(g->infos), info_cmp);
    for (size_t j = 0; j < g->infos.size(); j++)
      g->infos[j]->id = j;
    g->UpdateFluxCorrection = true;
    update_blocks(&g->infos, &g->all, &g->tree);
  }
  for (int i = 0;; i++) {
    ongrid();
    if (i == sim.levelMax)
      break;
    adapt();
  }
  std::vector<Info *> &velInfo = var.vel->infos;
  for (auto &shape : sim.shapes) {
    std::vector<Obstacle *> &oblock = shape->obstacleBlocks;
#pragma omp parallel for
    for (size_t i = 0; i < velInfo.size(); i++) {
      if (oblock[var.tmpV->infos[i]->id] == nullptr)
        continue;
      Real *udef = (Real *)oblock[var.tmpV->infos[i]->id]->udef;
      Real *chi = (Real *)oblock[var.tmpV->infos[i]->id]->chi;
      Real *UDEF = var.tmpV->infos[i]->block;
      Real *CHI = var.chi->infos[i]->block;
      for (int j = 0; j < _BS_ * _BS_; j++) {
        if (chi[j] < CHI[j])
          continue;
        UDEF[2 * j] += udef[2 * j];
        UDEF[2 * j + 1] += udef[2 * j + 1];
      }
    }
  }
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < velInfo.size(); i++) {
    Real *UF = velInfo[i]->block;
    Real *US = var.tmpV->infos[i]->block;
    Real *X = var.chi->infos[i]->block;
    for (int j = 0; j < _BS_ * _BS_; j++) {
      UF[2 * j + 0] = UF[2 * j + 0] * (1 - X[j]) + US[2 * j + 0] * X[j];
      UF[2 * j + 1] = UF[2 * j + 1] * (1 - X[j]) + US[2 * j + 1] * X[j];
    }
  }
  std::vector<double> P_inv = precond();
  sim.mat = new LocalSpMatDnVec(_BS_ * _BS_, 0, P_inv);
  sim.solver = new Solver;
  while (1) {
    if (sim.rank == 0 && sim.step % 5 == 0)
      fprintf(stderr, "main.cpp: %08d\n", sim.step);
    Real CFL = sim.CFL;
    Real h = std::numeric_limits<Real>::infinity();
    for (size_t i = 0; i < var.vel->infos.size(); i++)
      h = std::min(var.vel->infos[i]->h, h);
    Real umax = 0;
#pragma omp parallel for schedule(static) reduction(max : umax)
    for (size_t i = 0; i < velInfo.size(); i++) {
      Real *vel = velInfo[i]->block;
      for (int j = 0; j < 2 * _BS_ * _BS_; j++)
        umax = std::max(umax, std::fabs(vel[j]));
    }
    Real dtDiffusion = 0.25 * h * h / (sim.nu + 0.25 * h * umax);
    Real dtAdvection = h / (umax + 1e-8);
    sim.dt = std::min({dtDiffusion, CFL * dtAdvection});
    if (sim.dumpTime > 0 && sim.time >= sim.nextDumpTime) {
      sim.nextDumpTime += sim.dumpTime;
      computeA(KernelVorticity(), var.vel, 2);
      char path[FILENAME_MAX];
      snprintf(path, sizeof path, "vel.%08d", sim.dump_count++);
      dump(sim.time, var.vel->infos.data(), path);
    }
    if (sim.step <= 10 || sim.step % sim.AdaptSteps == 0)
      adapt();
    for (const auto &shape : sim.shapes) {
      shape->x += sim.dt * shape->u;
      shape->y += sim.dt * shape->v;
      shape->orientation += sim.dt * shape->omega;
      shape->orientation = shape->orientation > M_PI
                               ? shape->orientation - 2 * M_PI
                               : shape->orientation;
      shape->orientation = shape->orientation < -M_PI
                               ? shape->orientation + 2 * M_PI
                               : shape->orientation;
    }
    ongrid();
#pragma omp parallel for
    for (size_t i = 0; i < velInfo.size(); i++)
      memcpy(var.vold->infos[i]->block, velInfo[i]->block,
             2 * _BS_ * _BS_ * sizeof(Real));
    if (var.tmpV->UpdateFluxCorrection) {
      var.tmpV->UpdateFluxCorrection = false;
    }
    computeA(KernelAdvectDiffuse(), var.vel, 2);
#pragma omp parallel for
    for (size_t i = 0; i < velInfo.size(); i++) {
      Real *V = velInfo[i]->block;
      Real *Vold = var.vold->infos[i]->block;
      Real *tmpV = var.tmpV->infos[i]->block;
      Real ih2 = 0.5 / (velInfo[i]->h * velInfo[i]->h);
      for (int j = 0; j < 2 * _BS_ * _BS_; j++)
        V[j] = Vold[j] + tmpV[j] * ih2;
    }
    if (var.tmpV->UpdateFluxCorrection) {
      var.tmpV->UpdateFluxCorrection = false;
    }
    computeA(KernelAdvectDiffuse(), var.vel, 2);
#pragma omp parallel for
    for (size_t i = 0; i < velInfo.size(); i++) {
      Real *V = velInfo[i]->block;
      Real *Vold = var.vold->infos[i]->block;
      Real *tmpV = var.tmpV->infos[i]->block;
      Real ih2 = 1.0 / (velInfo[i]->h * velInfo[i]->h);
      for (int j = 0; j < 2 * _BS_ * _BS_; j++)
        V[j] = Vold[j] + tmpV[j] * ih2;
    }
    for (auto &shape : sim.shapes) {
      std::vector<Obstacle *> &oblock = shape->obstacleBlocks;
      Real PM = 0, PX = 0, PY = 0, UM = 0, VM = 0;
#pragma omp parallel for reduction(+ : PM, PX, PY, UM, VM)
      for (size_t i = 0; i < velInfo.size(); i++) {
        Real *VEL = velInfo[i]->block;
        Real hsq = velInfo[i]->h * velInfo[i]->h;
        if (oblock[velInfo[i]->id] == nullptr)
          continue;
        Real *chi = (Real *)oblock[velInfo[i]->id]->chi;
        Real *udef = (Real *)oblock[velInfo[i]->id]->udef;
        Real lambdt = sim.lambda * sim.dt;
        for (int iy = 0; iy < _BS_; ++iy)
          for (int ix = 0; ix < _BS_; ++ix) {
            int j = _BS_ * iy + ix;
            if (chi[j] <= 0)
              continue;
            Real udiff[2] = {VEL[2 * j + 0] - udef[2 * j + 0],
                             VEL[2 * j + 1] - udef[2 * j + 1]};
            Real Xlamdt = chi[j] >= 0.5 ? lambdt : 0.0;
            Real F = hsq * Xlamdt / (1 + Xlamdt);
            Real p[2];
            p[0] = velInfo[i]->origin[0] + velInfo[i]->h * (ix + 0.5);
            p[1] = velInfo[i]->origin[1] + velInfo[i]->h * (iy + 0.5);
            p[0] -= shape->x;
            p[1] -= shape->y;
            PM += F;
            PX += F * p[0];
            PY += F * p[1];
            UM += F * udiff[0];
            VM += F * udiff[1];
          }
      }
      Real quantities[] = {PM, PX, PY, UM, VM};
      PM = quantities[0];
      PX = quantities[1];
      PY = quantities[2];
      UM = quantities[3];
      VM = quantities[4];
      if (PM != 0) {
        shape->u = (PY * shape->omega + UM) / PM;
        shape->v = (VM - PX * shape->omega) / PM;
      }
    }
    auto &infos = var.chi->infos;
    size_t N = sim.shapes.size();
    std::vector<CollisionInfo> collisions(N);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j < N; ++j) {
        if (i == j)
          continue;
        auto &coll = collisions[i];
        auto &iBlocks = sim.shapes[i]->obstacleBlocks;
        auto &jBlocks = sim.shapes[j]->obstacleBlocks;
        for (size_t k = 0; k < iBlocks.size(); ++k) {
          if (iBlocks[k] == nullptr || jBlocks[k] == nullptr)
            continue;
          auto &iSDF = iBlocks[k]->dist;
          auto &jSDF = jBlocks[k]->dist;
          ScalarBlock &iChi = iBlocks[k]->chi;
          ScalarBlock &jChi = jBlocks[k]->chi;
          Real h = 1.0 / _BS_ / (1 << infos[k]->level);
          Real hsq = h * h;
          for (int iy = 0; iy < _BS_; ++iy)
            for (int ix = 0; ix < _BS_; ++ix) {
              if (iChi[iy][ix] <= 0.0 || jChi[iy][ix] <= 0.0)
                continue;
              Real pos[2];
              pos[0] = infos[k]->origin[0] + h * (ix + 0.5);
              pos[1] = infos[k]->origin[1] + h * (iy + 0.5);
              coll.iM += iChi[iy][ix] * hsq;
              coll.iPosX += iChi[iy][ix] * pos[0] * hsq;
              coll.iPosY += iChi[iy][ix] * pos[1] * hsq;
              coll.jM += jChi[iy][ix] * hsq;
              coll.jPosX += jChi[iy][ix] * pos[0] * hsq;
              coll.jPosY += jChi[iy][ix] * pos[1] * hsq;
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
      buffer[20 * i + 7] = coll.ivecX;
      buffer[20 * i + 8] = coll.ivecY;
      buffer[20 * i + 10] = coll.jM;
      buffer[20 * i + 11] = coll.jPosX;
      buffer[20 * i + 12] = coll.jPosY;
      buffer[20 * i + 17] = coll.jvecX;
      buffer[20 * i + 18] = coll.jvecY;
    }
    for (size_t i = 0; i < N; i++) {
      auto &coll = collisions[i];
      coll.iM = buffer[20 * i];
      coll.iPosX = buffer[20 * i + 1];
      coll.iPosY = buffer[20 * i + 2];
      coll.ivecX = buffer[20 * i + 7];
      coll.ivecY = buffer[20 * i + 8];
      coll.jM = buffer[20 * i + 10];
      coll.jPosX = buffer[20 * i + 11];
      coll.jPosY = buffer[20 * i + 12];
      coll.jvecX = buffer[20 * i + 17];
      coll.jvecY = buffer[20 * i + 18];
    }
    // #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = i + 1; j < N; ++j) {
        auto &coll = collisions[i];
        auto &coll_other = collisions[j];
        if (coll.iM > 0 && coll.jM > 0 && coll_other.iM > 0 &&
            coll_other.jM > 0) {
          Real norm_i = hypot(coll.ivecX, coll.ivecY);
          Real norm_j = hypot(coll.jvecX, coll.jvecY);
          Real mX = coll.ivecX / norm_i - coll.jvecX / norm_j;
          Real mY = coll.ivecY / norm_i - coll.jvecY / norm_j;
          Real inorm = 1.0 / hypot(mX, mY);
          Real NX = mX * inorm;
          Real NY = mY * inorm;
          Real mass = (coll.iM + coll.jM) / 2;
          Real du = 8 * NX * mass;
          Real dv = 8 * NY * mass;
          sim.shapes[i]->u += du;
          sim.shapes[i]->v += dv;
          sim.shapes[j]->u -= du;
          sim.shapes[j]->v -= dv;
          if (sim.rank == 0)
            printf("Collision between objects %ld and %ld\n"
                   " iM %g %g\n"
                   " jM %g %g\n"
                   " Normal vector = %g %g\n",
                   i, j, collisions[i].iM, collisions[j].jM, collisions[i].jM,
                   collisions[j].iM, NX, NY);
        }
      }
    }
    std::vector<Info *> &chiInfo = var.chi->infos;
#pragma omp parallel for
    for (size_t i = 0; i < velInfo.size(); i++)
      for (auto &shape : sim.shapes) {
        std::vector<Obstacle *> &oblock = shape->obstacleBlocks;
        Obstacle *o = oblock[velInfo[i]->id];
        if (o == nullptr)
          continue;
        Real *X = (Real *)o->chi;
        Real *UDEF = (Real *)o->udef;
        Real *CHI = chiInfo[i]->block;
        Real *V = velInfo[i]->block;
        for (int iy = 0; iy < _BS_; ++iy)
          for (int ix = 0; ix < _BS_; ++ix) {
            int j = _BS_ * iy + ix;
            if (CHI[j] > X[j])
              continue;
            if (X[j] <= 0)
              continue;
            Real p[2];
            p[0] = velInfo[i]->origin[0] + velInfo[i]->h * (ix + 0.5);
            p[1] = velInfo[i]->origin[1] + velInfo[i]->h * (iy + 0.5);
            p[0] -= shape->x;
            p[1] -= shape->y;
            Real alpha = X[j] > 0.5 ? 1 / (1 + sim.lambda * sim.dt) : 1;
            Real US = shape->u - shape->omega * p[1] + UDEF[2 * j + 0];
            Real VS = shape->v + shape->omega * p[0] + UDEF[2 * j + 1];
            V[2 * j + 0] = alpha * V[2 * j + 0] + (1 - alpha) * US;
            V[2 * j + 1] = alpha * V[2 * j + 1] + (1 - alpha) * VS;
          }
      }
    std::vector<Info *> &tmpVInfo = var.tmpV->infos;
#pragma omp parallel for
    for (size_t i = 0; i < velInfo.size(); i++)
      memset(tmpVInfo[i]->block, 0, 2 * _BS_ * _BS_ * sizeof(Real));
    for (auto &shape : sim.shapes) {
      std::vector<Obstacle *> &oblock = shape->obstacleBlocks;
#pragma omp parallel for
      for (size_t i = 0; i < velInfo.size(); i++) {
        if (oblock[tmpVInfo[i]->id] == nullptr)
          continue;
        Real *udef = (Real *)oblock[tmpVInfo[i]->id]->udef;
        Real *chi = (Real *)oblock[tmpVInfo[i]->id]->chi;
        Real *UDEF = tmpVInfo[i]->block;
        Real *CHI = chiInfo[i]->block;
        for (int iy = 0; iy < _BS_; iy++)
          for (int ix = 0; ix < _BS_; ix++) {
            int j = _BS_ * iy + ix;
            if (chi[j] < CHI[j])
              continue;
            UDEF[2 * j + 0] += udef[2 * j + 0];
            UDEF[2 * j + 1] += udef[2 * j + 1];
          }
      }
    }
    if (var.tmp->UpdateFluxCorrection) {
      var.tmp->UpdateFluxCorrection = false;
    }
    Stencil stencil{-1, -1, 2, 2, false};
    std::vector<Info *> &blk = var.vel->infos;
    std::vector<bool> ready(blk.size(), false);
    std::vector<Info *> &avail0 = var.vel->infos;
    std::vector<Info *> &avail02 = var.tmpV->infos;
    const int Ninner = avail0.size();
#pragma omp parallel
    {
      BlockLab lab(2);
      BlockLab lab2(2);
      lab.prepare(stencil);
      lab2.prepare(stencil);
#pragma omp for
      for (int i = 0; i < Ninner; i++) {
        Info *I = avail0[i];
        Info *I2 = avail02[i];
        lab.load(&var.vel->tree, &var.vel->all, stencil, I, true);
        lab2.load(&var.tmpV->tree, &var.tmpV->all, stencil, I2, true);
        pressure_rhs_fun(lab, lab2, I, I2);
        ready[I->id] = true;
      }
    }
    std::vector<Info *> &presInfo = var.pres->infos;
    std::vector<Info *> &poldInfo = var.pold->infos;
#pragma omp parallel for
    for (size_t i = 0; i < velInfo.size(); i++) {
      memcpy(poldInfo[i]->block, presInfo[i]->block,
             _BS_ * _BS_ * sizeof(Real));
      memset(presInfo[i]->block, 0, _BS_ * _BS_ * sizeof(Real));
    }
    computeA(pressure_rhs1(), var.pold, 1);
    const double max_error = sim.step < 10 ? 0.0 : sim.PoissonTol;
    const double max_rel_error = sim.step < 10 ? 0.0 : sim.PoissonTolRel;
    const int max_restarts = sim.step < 10 ? 100 : sim.maxPoissonRestarts;
    if (var.pres->UpdateFluxCorrection) {
      var.pres->UpdateFluxCorrection = false;

      update_blocks(&var.tmp->infos, &var.tmp->all, &var.tmp->tree);
      std::vector<Info *> &RhsInfo = var.tmp->infos;
      const int Nblocks = RhsInfo.size();
      const int N = _BS_ * _BS_ * Nblocks;
      sim.mat->reserve(N);
      sim.nblocks[0] = 0;
      sim.nrows[0] = 0;
      for (int i = 0; i < Nblocks; i++) {
        Info *&rhs_info = RhsInfo[i];
        const int aux = 1 << rhs_info->level;
        const int MAX_X_BLOCKS = aux - 1;
        const int MAX_Y_BLOCKS = aux - 1;
        std::array<bool, 4> isBoundary;
        isBoundary[0] = (rhs_info->index[0] == 0);
        isBoundary[1] = (rhs_info->index[0] == MAX_X_BLOCKS);
        isBoundary[2] = (rhs_info->index[1] == 0);
        isBoundary[3] = (rhs_info->index[1] == MAX_Y_BLOCKS);
        std::array<const Info *, 4> rhsNei;
        rhsNei[0] =
            getf(&var.tmp->all, rhs_info->level, rhs_info->Znei[1 - 1][1]);
        rhsNei[1] =
            getf(&var.tmp->all, rhs_info->level, rhs_info->Znei[1 + 1][1]);
        rhsNei[2] =
            getf(&var.tmp->all, rhs_info->level, rhs_info->Znei[1][1 - 1]);
        rhsNei[3] =
            getf(&var.tmp->all, rhs_info->level, rhs_info->Znei[1][1 + 1]);
        for (int iy = 0; iy < _BS_; iy++)
          for (int ix = 0; ix < _BS_; ix++) {
            const long long sfc_idx =
                sim.solver->GenericCell.This(rhs_info, ix, iy);
            if ((ix > 0 && ix < _BS_ - 1) && (iy > 0 && iy < _BS_ - 1)) {
              sim.mat->cooPushBackVal(
                  1, sfc_idx,
                  sim.solver->GenericCell.This(rhs_info, ix, iy - 1));
              sim.mat->cooPushBackVal(
                  1, sfc_idx,
                  sim.solver->GenericCell.This(rhs_info, ix - 1, iy));
              sim.mat->cooPushBackVal(-4, sfc_idx, sfc_idx);
              sim.mat->cooPushBackVal(
                  1, sfc_idx,
                  sim.solver->GenericCell.This(rhs_info, ix + 1, iy));
              sim.mat->cooPushBackVal(
                  1, sfc_idx,
                  sim.solver->GenericCell.This(rhs_info, ix, iy + 1));
            } else {
              std::array<bool, 4> validNei;
              validNei[0] = ix > 0;
              validNei[1] = ix < _BS_ - 1;
              validNei[2] = iy > 0;
              validNei[3] = iy < _BS_ - 1;
              std::array<long long, 4> idxNei;
              idxNei[0] = sim.solver->GenericCell.This(rhs_info, ix - 1, iy);
              idxNei[1] = sim.solver->GenericCell.This(rhs_info, ix + 1, iy);
              idxNei[2] = sim.solver->GenericCell.This(rhs_info, ix, iy - 1);
              idxNei[3] = sim.solver->GenericCell.This(rhs_info, ix, iy + 1);
              SpRowInfo row(Tree1(rhs_info, &var.tmp->tree), sfc_idx, 8);
              for (int j = 0; j < 4; j++) {
                if (validNei[j]) {
                  row.mapColVal(idxNei[j], 1);
                  row.mapColVal(sfc_idx, -1);
                } else if (!isBoundary[j]) {
                  sim.solver->makeFlux(rhs_info, ix, iy, rhsNei[j],
                                       sim.solver->edgeIndexers[j], row);
                }
              }
              sim.mat->cooPushBackRow(row);
            }
          }
      }
      sim.mat->make(sim.nrows);
      sim.solver->getVec();
      sim.mat->solveWithUpdate(max_error, max_rel_error, max_restarts);
    } else {
      sim.solver->getVec();
      sim.mat->solveNoUpdate(max_error, max_rel_error, max_restarts);
    }
    std::vector<Info *> &zInfo = var.pres->infos;
    const int NB = zInfo.size();
    const std::vector<double> &x = sim.mat->x_;
    Real avg, avg1;
    avg = 0;
    avg1 = 0;
#pragma omp parallel for reduction(+ : avg, avg1)
    for (int i = 0; i < NB; i++) {
      Real *P = zInfo[i]->block;
      const double vv = zInfo[i]->h * zInfo[i]->h;
      for (int j = 0; j < _BS_ * _BS_; j++) {
        P[j] = x[i * _BS_ * _BS_ + j];
        avg += P[j] * vv;
        avg1 += vv;
      }
    }
    avg = avg / avg1;
#pragma omp parallel for
    for (int i = 0; i < NB; i++) {
      Real *P = zInfo[i]->block;
      for (int j = 0; j < _BS_ * _BS_; j++)
        P[j] += -avg;
    }
    avg = 0;
    avg1 = 0;
#pragma omp parallel for reduction(+ : avg, avg1)
    for (size_t i = 0; i < velInfo.size(); i++) {
      Real *P = presInfo[i]->block;
      Real vv = presInfo[i]->h * presInfo[i]->h;
      for (int j = 0; j < _BS_ * _BS_; j++) {
        avg += P[j] * vv;
        avg1 += vv;
      }
    }
    avg = avg / avg1;
#pragma omp parallel for
    for (size_t i = 0; i < velInfo.size(); i++) {
      Real *pres = presInfo[i]->block;
      Real *pold = poldInfo[i]->block;
      for (int j = 0; j < _BS_ * _BS_; j++)
        pres[j] += pold[j] - avg;
    }
    computeA(pressureCorrectionKernel(), var.pres, 1);
#pragma omp parallel for
    for (size_t i = 0; i < velInfo.size(); i++) {
      Real ih2 = 1.0 / velInfo[i]->h / velInfo[i]->h;
      Real *V = velInfo[i]->block;
      Real *tmpV = tmpVInfo[i]->block;
      for (int j = 0; j < 2 * _BS_ * _BS_; j++)
        V[j] += tmpV[j] * ih2;
    }
    sim.time += sim.dt;
    sim.step++;
    if (sim.endTime > 0 && sim.time >= sim.endTime)
      break;
  }

  delete sim.mat;
  delete sim.solver;
  for (Shape *shape : sim.shapes) {
    for (Obstacle *oblock : shape->obstacleBlocks)
      delete oblock;
    free(shape->sdf);
    delete shape;
  }
}
