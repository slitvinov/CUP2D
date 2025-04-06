#define OMPI_SKIP_MPICXX 1
#include <algorithm>
#include <array>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mpi.h>
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

#define CHECK                                                                  \
  do {                                                                         \
    int req =                                                                  \
        unpack->ly == 0 ? 0 : unpack->LX * (unpack->ly - 1) + unpack->lx;      \
    if (1 || (dim * buf->recv_buffer_size[otherrank] - unpack->offset) <       \
                 dim * req) {                                                  \
      fprintf(stderr,                                                          \
              "ERROR: recv_buffer size mismatch on rank %d:\n"                 \
              "  otherrank = %d\n"                                             \
              "  recv_buffer_size[%d]   = %d\n"                                \
              "  unpack->offset         = %d\n"                                \
              "  req                    = %d\n"                                \
              "  dim                    = %d\n"                                \
              "  buf                    = %g\n",                               \
              sim.rank, otherrank, otherrank,                                  \
              buf->recv_buffer_size[otherrank], unpack->offset, req, dim,      \
              buf->recv_buffer[otherrank][unpack->offset + dim * req]);        \
      MPI_Abort(MPI_COMM_WORLD, 1);                                            \
    }                                                                          \
  } while (0)

typedef double Real;
#define MPI_Real MPI_DOUBLE
static constexpr unsigned int sizes[] = {_BS_, _BS_, 1};
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
struct Synchronizer;
static struct {
  int AdaptSteps;
  int levelMax;
  int levelStart;
  int maxPoissonRestarts;
  int rank;
  int size;
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
struct BlockCase;
struct Info {
  bool changed2;
  double h, origin[2];
  enum State state;
  int index[3], level;
  long long id, id2, halo_id, Z, Zchild[2][2], Znei[3][3], Zparent;
  Real *block = NULL;
  BlockCase *auxiliary;
};
struct BlockCase {
  Real *d[4];
  int level;
  long long Z;
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
struct Interface {
  Info *infos[2];
  int icode[2];
  bool CoarseStencil;
  bool ToBeKept;
  int dis;
  Interface(Info *i0, Info *i1, int a_icode0, int a_icode1) {
    infos[0] = i0;
    infos[1] = i1;
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
  std::vector<int> removed;
  int index;
  int sx;
  int sy;
  int ex;
  int ey;
  bool needed{true};
  bool avg_down{true};
};
static void remove(Range *q, Range *p) {
  q->removed.insert(q->removed.end(), p->removed.begin(), p->removed.end());
}
static bool contains(Range *q, Range *r) {
  if (q->avg_down != r->avg_down)
    return false;
  int V = (q->ey - q->sy) * (q->ex - q->sx);
  int Vr = (r->ey - r->sy) * (r->ex - r->sx);
  return q->sx <= r->sx && r->ex <= q->ex && q->sy <= r->sy && r->ey <= q->ey &&
         Vr < V;
}
struct UnPackInfo {
  int offset;
  int lx;
  int ly;
  int x;
  int y;
  int LX;
  int LY;
  int CoarseVersionOffset;
  int CoarseVersionLX;
  int CoarseVersionLY;
  int CoarseVersionx;
  int CoarseVersiony;
  int level;
  int icode;
  int rank;
  int index_0;
  int index_1;
};
struct HaloBlockGroup {
  std::vector<Info *> myblocks;
  std::set<int> myranks;
  bool ready = false;
};
struct PackInfo {
  Real *block;
  Real *pack;
  int sx;
  int sy;
  int ex;
  int ey;
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
            if (me[j2].needed && contains(&me[j2], &me[j1])) {
              me[j1].needed = false;
              me[j2].removed.push_back(me[j1].index);
              remove(&me[j2], &me[j1]);
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
                  if (m.needed && contains(&m, &o)) {
                    o.needed = false;
                    m.removed.push_back(o.index);
                    remove(&m, &o);
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
  void add(int r, int index) {
    if (sizes[r] == 0)
      positions[r] = index;
    sizes[r]++;
  }
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
  b->auxiliary = nullptr;
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
static void DetermineStencilLength(int *sLength, int level_sender,
                                   int level_receiver, int icode, int *L) {
  if (level_sender == level_receiver) {
    L[0] = sLength[3 * icode + 0];
    L[1] = sLength[3 * icode + 1];
    L[2] = 1;
  } else if (level_sender > level_receiver) {
    L[0] = sLength[3 * (icode + 27) + 0];
    L[1] = sLength[3 * (icode + 27) + 1];
    L[2] = 1;
  } else {
    L[0] = sLength[3 * (icode + 2 * 27) + 0];
    L[1] = sLength[3 * (icode + 2 * 27) + 1];
    L[2] = 1;
  }
}

static Range &DetermineStencil(std::array<Range, 3 * 27> &AllStencils,
                               Range &Coarse_Range, const Stencil &stencil,
                               const Interface *f, bool CoarseVersion) {
  if (CoarseVersion) {
    AllStencils[f->icode[1] + 2 * 27].needed = true;
    return AllStencils[f->icode[1] + 2 * 27];
  } else {
    if (f->infos[0]->level == f->infos[1]->level) {
      AllStencils[f->icode[1]].needed = true;
      return AllStencils[f->icode[1]];
    } else if (f->infos[0]->level > f->infos[1]->level) {
      AllStencils[f->icode[1] + 27].needed = true;
      return AllStencils[f->icode[1] + 27];
    } else {
      Coarse_Range.needed = true;
      const int code[3] = {f->icode[1] % 3 - 1, (f->icode[1] / 3) % 3 - 1,
                           (f->icode[1] / 9) % 3 - 1};
      const int s[3] = {
          code[0] < 1 ? (code[0] < 0 ? ((stencil.sx - 1) / 2 - 1) : 0)
                      : _BS_ / 2,
          code[1] < 1 ? (code[1] < 0 ? ((stencil.sy - 1) / 2 - 1) : 0)
                      : _BS_ / 2,
          code[2] < 1 ? (code[2] < 0 ? ((0 - 1) / 2) : 0) : 1 / 2};
      int e[3] = {code[0] < 1 ? (code[0] < 0 ? 0 : _BS_ / 2)
                              : _BS_ / 2 + stencil.ex / 2 + 1,
                  code[1] < 1 ? (code[1] < 0 ? 0 : _BS_ / 2)
                              : _BS_ / 2 + stencil.ey / 2 + 1,
                  code[2] < 1 ? (code[2] < 0 ? 0 : 1 / 2) : 1 / 2};
      int base[3] = {(f->infos[1]->index[0] + code[0]) % 2,
                     (f->infos[1]->index[1] + code[1]) % 2,
                     (f->infos[1]->index[2] + code[2]) % 2};
      int Cindex_true[3];
      for (int d = 0; d < 3; d++)
        Cindex_true[d] = f->infos[1]->index[d] + code[d];
      int CoarseEdge[2];
      CoarseEdge[0] = code[0] == 0 ? 0
                      : ((f->infos[1]->index[0] % 2 == 0) &&
                         (Cindex_true[0] > f->infos[1]->index[0])) ||
                              ((f->infos[1]->index[0] % 2 == 1) &&
                               (Cindex_true[0] < f->infos[1]->index[0]))
                          ? 1
                          : 0;
      CoarseEdge[1] = code[1] == 0 ? 0
                      : ((f->infos[1]->index[1] % 2 == 0) &&
                         (Cindex_true[1] > f->infos[1]->index[1])) ||
                              ((f->infos[1]->index[1] % 2 == 1) &&
                               (Cindex_true[1] < f->infos[1]->index[1]))
                          ? 1
                          : 0;
      Coarse_Range.sx = s[0] + std::max(code[0], 0) * _BS_ / 2 +
                        (1 - abs(code[0])) * base[0] * _BS_ / 2 -
                        code[0] * _BS_ + CoarseEdge[0] * code[0] * _BS_ / 2;
      Coarse_Range.sy = s[1] + std::max(code[1], 0) * _BS_ / 2 +
                        (1 - abs(code[1])) * base[1] * _BS_ / 2 -
                        code[1] * _BS_ + CoarseEdge[1] * code[1] * _BS_ / 2;
      Coarse_Range.ex = e[0] + std::max(code[0], 0) * _BS_ / 2 +
                        (1 - abs(code[0])) * base[0] * _BS_ / 2 -
                        code[0] * _BS_ + CoarseEdge[0] * code[0] * _BS_ / 2;
      Coarse_Range.ey = e[1] + std::max(code[1], 0) * _BS_ / 2 +
                        (1 - abs(code[1])) * base[1] * _BS_ / 2 -
                        code[1] * _BS_ + CoarseEdge[1] * code[1] * _BS_ / 2;
      return Coarse_Range;
    }
  }
}

static void FixDuplicates(std::array<Range, 3 * 27> &AllStencils,
                          Range &Coarse_Range, const Stencil &stencil,
                          const Interface *f, const Interface *f_dup, int lx,
                          int ly, int lz, int lx_dup, int ly_dup, int lz_dup,
                          int *sx, int *sy, int *sz) {
  Info *receiver = f->infos[1];
  Info *receiver_dup = f_dup->infos[1];
  if (receiver->level >= receiver_dup->level) {
    int icode_dup = f_dup->icode[1];
    const int code_dup[3] = {icode_dup % 3 - 1, (icode_dup / 3) % 3 - 1,
                             (icode_dup / 9) % 3 - 1};
    *sx = (lx == lx_dup || code_dup[0] != -1) ? 0 : lx - lx_dup;
    *sy = (ly == ly_dup || code_dup[1] != -1) ? 0 : ly - ly_dup;
    *sz = (lz == lz_dup || code_dup[2] != -1) ? 0 : lz - lz_dup;
  } else {
    Range &range =
        DetermineStencil(AllStencils, Coarse_Range, stencil, f, false);
    Range &range_dup =
        DetermineStencil(AllStencils, Coarse_Range, stencil, f_dup, false);
    *sx = range_dup.sx - range.sx;
    *sy = range_dup.sy - range.sy;
    *sz = 0;
  }
}

static void FixDuplicates2(std::array<Range, 3 * 27> &AllStencils,
                           Range &Coarse_Range, const Stencil &stencil,
                           const Interface *f, const Interface *f_dup, int *sx,
                           int *sy, int *sz) {
  if (f->infos[0]->level != f->infos[1]->level ||
      f_dup->infos[0]->level != f_dup->infos[1]->level)
    return;
  Range &range = DetermineStencil(AllStencils, Coarse_Range, stencil, f, true);
  Range &range_dup =
      DetermineStencil(AllStencils, Coarse_Range, stencil, f_dup, true);
  *sx = range_dup.sx - range.sx;
  *sy = range_dup.sy - range.sy;
  *sz = 0;
}

struct SyncBuf {
  std::set<int> Neighbors;
  std::vector<Info *> halo_blocks;
  std::vector<Info *> inner_blocks;
  std::vector<int> recv_buffer_size;
  std::vector<int> send_buffer_size;
  std::vector<MPI_Request> requests;
  std::vector<std::vector<Interface>> recv_interfaces;
  std::vector<std::vector<Interface>> send_interfaces;
  std::vector<std::vector<PackInfo>> send_packinfos;
  Real **recv_buffer;
  Real **send_buffer;
  std::vector<std::vector<UnPackInfo>> myunpacks;
};
static void
Setup(int dim, std::unordered_map<long long, int> *tree,
      std::unordered_map<long long, Info *> *all, std::vector<Info *> *infos,
      struct SyncBuf *buf, bool &use_averages,
      std::array<Range, 3 * 27> &AllStencils, Range &Coarse_Range,
      const Stencil &stencil, int *sLength,
      std::vector<std::vector<int>> &ToBeAveragedDown,
      std::unordered_map<std::string, HaloBlockGroup> &mapofHaloBlockGroups

) {
  DuplicatesManager DM;
  std::vector<int> offsets(sim.size, 0);
  std::vector<int> offsets_recv(sim.size, 0);
  DM.positions.resize(sim.size);
  DM.sizes.resize(sim.size);
  buf->Neighbors.clear();
  buf->inner_blocks.clear();
  buf->halo_blocks.clear();
  for (int r = 0; r < sim.size; r++) {
    buf->send_interfaces[r].clear();
    buf->recv_interfaces[r].clear();
    buf->send_buffer_size[r] = 0;
  }
  for (size_t i = 0; i < buf->myunpacks.size(); i++)
    buf->myunpacks[i].clear();
  buf->myunpacks.clear();
  std::vector<Range> compass[27];
  for (Info *info : *infos) {
    info->halo_id = -1;
    bool xskin =
        info->index[0] == 0 || info->index[0] == ((1 << info->level) - 1);
    bool yskin =
        info->index[1] == 0 || info->index[1] == ((1 << info->level) - 1);
    int xskip = info->index[0] == 0 ? -1 : 1;
    int yskip = info->index[1] == 0 ? -1 : 1;
    assert(xskip);
    assert(yskip);

    bool isInner = true;
    std::vector<int> ToBeChecked;
    bool Coarsened = false;
    for (int icode = 0; icode < 27; icode++) {
      if (icode == 1 * 1 + 3 * 1 + 9 * 1)
        continue;
      int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, (icode / 9) % 3 - 1};
      if (code[2] != 0)
        continue;
      if (code[0] == xskip && xskin)
        continue;
      if (code[1] == yskip && yskin)
        continue;
      int &infoNeiTree =
          treef(tree, info->level, info->Znei[1 + code[0]][1 + code[1]]);
      if (infoNeiTree >= 0 && infoNeiTree != sim.rank) {
        isInner = false;
        buf->Neighbors.insert(infoNeiTree);
        Info *infoNei =
            getf(all, info->level, info->Znei[1 + code[0]][1 + code[1]]);
        int icode2 = (-code[0] + 1) + (-code[1] + 1) * 3 + (-code[2] + 1) * 9;
        buf->send_interfaces[infoNeiTree].push_back(
            {info, infoNei, icode, icode2});
        buf->recv_interfaces[infoNeiTree].push_back(
            {infoNei, info, icode2, icode});
        ToBeChecked.push_back(infoNeiTree);
        ToBeChecked.push_back((int)buf->send_interfaces[infoNeiTree].size() -
                              1);
        ToBeChecked.push_back((int)buf->recv_interfaces[infoNeiTree].size() -
                              1);
        DM.add(infoNeiTree, (int)buf->send_interfaces[infoNeiTree].size() - 1);
      } else if (infoNeiTree == -2) {
        Coarsened = true;
        Info *infoNei =
            getf(all, info->level, info->Znei[1 + code[0]][1 + code[1]]);
        int infoNeiCoarserrank = treef(tree, info->level - 1, infoNei->Zparent);
        if (infoNeiCoarserrank != sim.rank) {
          isInner = false;
          buf->Neighbors.insert(infoNeiCoarserrank);
          Info *infoNeiCoarser =
              getf(all, infoNei->level - 1, infoNei->Zparent);
          int icode2 = (-code[0] + 1) + (-code[1] + 1) * 3 + (-code[2] + 1) * 9;
          int Bmax[3] = {1 << (info->level - 1), 1 << (info->level - 1),
                         1 << (info->level - 1)};
          int test_idx[3] = {
              (infoNeiCoarser->index[0] - code[0] + Bmax[0]) % Bmax[0],
              (infoNeiCoarser->index[1] - code[1] + Bmax[1]) % Bmax[1],
              (infoNeiCoarser->index[2] - code[2] + Bmax[2]) % Bmax[2]};
          if (info->index[0] / 2 == test_idx[0] &&
              info->index[1] / 2 == test_idx[1] &&
              info->index[2] / 2 == test_idx[2]) {
            buf->send_interfaces[infoNeiCoarserrank].push_back(
                {info, infoNeiCoarser, icode, icode2});
            buf->recv_interfaces[infoNeiCoarserrank].push_back(
                {infoNeiCoarser, info, icode2, icode});
            DM.add(infoNeiCoarserrank,
                   (int)buf->send_interfaces[infoNeiCoarserrank].size() - 1);
            if (abs(code[0]) + abs(code[1]) + abs(code[2]) == 1) {
              int d0 = abs(code[1] + 2 * code[2]);
              int d1 = (d0 + 1) % 3;
              int d2 = (d0 + 2) % 3;
              int code3[3];
              code3[d0] = code[d0];
              code3[d1] = -2 * (info->index[d1] % 2) + 1;
              code3[d2] = -2 * (info->index[d2] % 2) + 1;
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
                buf->recv_interfaces[infoNeiCoarserrank].push_back(
                    {infoNeiCoarser, info, icode2, icode3});
              if (code4[2] == 0)
                buf->recv_interfaces[infoNeiCoarserrank].push_back(
                    {infoNeiCoarser, info, icode2, icode4});
              if (code5[2] == 0)
                buf->recv_interfaces[infoNeiCoarserrank].push_back(
                    {infoNeiCoarser, info, icode2, icode5});
            }
          }
        }
      } else if (infoNeiTree == -1) {
        Info *infoNei =
            getf(all, info->level, info->Znei[1 + code[0]][1 + code[1]]);
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
              infoNei->Zchild[std::max(-code[0], 0) +
                              (B % 2) * std::max(0, 1 - abs(code[0]))]
                             [std::max(-code[1], 0) +
                              temp * std::max(0, 1 - abs(code[1]))];
          int infoNeiFinerrank = treef(tree, info->level + 1, nFine);
          if (infoNeiFinerrank != sim.rank) {
            isInner = false;
            buf->Neighbors.insert(infoNeiFinerrank);
            Info *infoNeiFiner = getf(all, info->level + 1, nFine);
            int icode2 =
                (-code[0] + 1) + (-code[1] + 1) * 3 + (-code[2] + 1) * 9;
            buf->send_interfaces[infoNeiFinerrank].push_back(
                {info, infoNeiFiner, icode, icode2});
            buf->recv_interfaces[infoNeiFinerrank].push_back(
                {infoNeiFiner, info, icode2, icode});
            DM.add(infoNeiFinerrank,
                   (int)buf->send_interfaces[infoNeiFinerrank].size() - 1);
            if (Bstep == 1) {
              int d0 = abs(code[1] + 2 * code[2]);
              int d1 = (d0 + 1) % 3;
              int d2 = (d0 + 2) % 3;
              int code3[3];
              code3[d0] = -code[d0];
              code3[d1] = -2 * (infoNeiFiner->index[d1] % 2) + 1;
              code3[d2] = -2 * (infoNeiFiner->index[d2] % 2) + 1;
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
                buf->send_interfaces[infoNeiFinerrank].push_back(
                    Interface(info, infoNeiFiner, icode, icode3));
                DM.add(infoNeiFinerrank,
                       (int)buf->send_interfaces[infoNeiFinerrank].size() - 1);
              }
              if (code4[2] == 0) {
                buf->send_interfaces[infoNeiFinerrank].push_back(
                    Interface(info, infoNeiFiner, icode, icode4));
                DM.add(infoNeiFinerrank,
                       (int)buf->send_interfaces[infoNeiFinerrank].size() - 1);
              }
              if (code5[2] == 0) {
                buf->send_interfaces[infoNeiFinerrank].push_back(
                    Interface(info, infoNeiFiner, icode, icode5));
                DM.add(infoNeiFinerrank,
                       (int)buf->send_interfaces[infoNeiFinerrank].size() - 1);
              }
            }
          }
        }
      }
    }
    if (isInner) {
      info->halo_id = -1;
      buf->inner_blocks.push_back(info);
    } else {
      info->halo_id = buf->halo_blocks.size();
      buf->halo_blocks.push_back(info);
      if (Coarsened) {
        for (size_t j = 0; j < ToBeChecked.size(); j += 3) {
          int r = ToBeChecked[j];
          int send = ToBeChecked[j + 1];
          int recv = ToBeChecked[j + 2];
          Info *a = buf->send_interfaces[r][send].infos[0];
          Info *b = buf->send_interfaces[r][send].infos[1];
          bool retval = false;
          if (!(a->level == 0 || !use_averages)) {
            int imin[2];
            int imax[2];
            int aux = 1 << a->level;
            int blocks[3] = {aux - 1, aux - 1};
            for (int d = 0; d < 2; d++) {
              imin[d] = (a->index[d] < b->index[d]) ? 0 : -1;
              imax[d] = (a->index[d] > b->index[d]) ? 0 : +1;
              if (a->index[d] == 0 && b->index[d] == 0)
                imin[d] = 0;
              if (a->index[d] == blocks[d] && b->index[d] == blocks[d])
                imax[d] = 0;
            }
            for (int i1 = imin[1]; i1 <= imax[1]; i1++)
              for (int i0 = imin[0]; i0 <= imax[0]; i0++) {
                if ((treef(tree, a->level, a->Znei[1 + i0][1 + i1])) == -2) {
                  retval = true;
                  break;
                }
              }
          }
          buf->send_interfaces[r][send].CoarseStencil = retval;
          buf->recv_interfaces[r][recv].CoarseStencil = retval;
        }
      }
      for (int r = 0; r < sim.size; r++)
        if (DM.sizes[r] > 0) {
          std::vector<Interface> &f = buf->send_interfaces[r];
          int &total_size = buf->send_buffer_size[r];
          bool skip_needed = false;
          std::sort(f.begin() + DM.positions[r],
                    f.begin() + DM.sizes[r] + DM.positions[r]);
          for (size_t i = 0; i < sizeof compass / sizeof *compass; i++)
            compass[i].clear();
          for (size_t i = 0; i < DM.sizes[r]; i++) {
            compass[f[i + DM.positions[r]].icode[0]].push_back(
                DetermineStencil(AllStencils, Coarse_Range, stencil,
                                 &f[i + DM.positions[r]], false));
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
          int Lc[2] = {0, 0};
          for (auto &i : keepEl(compass)) {
            const int k = i->index;
            DetermineStencilLength(sLength, f[k].infos[0]->level,
                                   f[k].infos[1]->level, f[k].icode[1], L);
            const int V = L[0] * L[1];
            total_size += V;
            f[k].dis = offsets[r];
            if (f[k].CoarseStencil) {
              Lc[0] = sLength[3 * (f[k].icode[1] + 2 * 27) + 0];
              Lc[1] = sLength[3 * (f[k].icode[1] + 2 * 27) + 1];
              int Vc = Lc[0] * Lc[1];
              total_size += Vc;
              offsets[r] += Vc * dim;
            }
            offsets[r] += V * dim;
            for (size_t kk = 0; kk < (*i).removed.size(); kk++)
              f[i->removed[kk]].dis = f[k].dis;
          }
          DM.sizes[r] = 0;
        }
    }
    getf(all, info->level, info->Z)->halo_id = info->halo_id;
  }
  buf->myunpacks.resize(buf->halo_blocks.size());
  for (int r = 0; r < sim.size; r++) {
    buf->recv_buffer_size[r] = 0;
    std::sort(buf->recv_interfaces[r].begin(), buf->recv_interfaces[r].end());
    size_t counter = 0;
    while (counter < buf->recv_interfaces[r].size()) {
      long long ID = buf->recv_interfaces[r][counter].infos[0]->id2;
      size_t start = counter;
      size_t finish = start + 1;
      counter++;
      size_t j;
      for (j = counter; j < buf->recv_interfaces[r].size(); j++) {
        if (buf->recv_interfaces[r][j].infos[0]->id2 == ID)
          finish++;
        else
          break;
      }
      counter = j;
      std::vector<Interface> &f = buf->recv_interfaces[r];
      int otherrank = r;
      bool skip_needed = false;
      for (size_t i = 0; i < sizeof compass / sizeof *compass; i++)
        compass[i].clear();
      for (size_t i = start; i < finish; i++) {
        compass[f[i].icode[0]].push_back(
            DetermineStencil(AllStencils, Coarse_Range, stencil, &f[i], false));
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
        int Lc[2] = {0, 0};
        DetermineStencilLength(sLength, f[k].infos[0]->level,
                               f[k].infos[1]->level, f[k].icode[1], L);
        const int V = L[0] * L[1];
        int Vc = 0;
        buf->recv_buffer_size[r] += V;
        f[k].dis = offsets_recv[otherrank];
        UnPackInfo info = {f[k].dis,
                           L[0],
                           L[1],
                           0,
                           0,
                           L[0],
                           L[1],
                           -1,
                           0,
                           0,
                           0,
                           0,
                           f[k].infos[0]->level,
                           f[k].icode[1],
                           otherrank,
                           f[k].infos[0]->index[0],
                           f[k].infos[0]->index[1]};
        if (f[k].CoarseStencil) {
          Lc[0] = sLength[3 * (f[k].icode[1] + 2 * 27) + 0];
          Lc[1] = sLength[3 * (f[k].icode[1] + 2 * 27) + 1];
          Vc = Lc[0] * Lc[1];
          buf->recv_buffer_size[r] += Vc;
          offsets_recv[otherrank] += Vc * dim;
          info.CoarseVersionOffset = V * dim;
          info.CoarseVersionLX = Lc[0];
          info.CoarseVersionLY = Lc[1];
        }
        offsets_recv[otherrank] += V * dim;
        buf->myunpacks[f[k].infos[1]->halo_id].push_back(info);
        for (size_t kk = 0; kk < (*i).removed.size(); kk++) {
          int remEl1 = i->removed[kk];
          DetermineStencilLength(sLength, f[remEl1].infos[0]->level,
                                 f[remEl1].infos[1]->level, f[remEl1].icode[1],
                                 &L[0]);
          int srcx, srcy, srcz;
          FixDuplicates(AllStencils, Coarse_Range, stencil, &f[k], &f[remEl1],
                        info.lx, info.ly, 1, L[0], L[1], L[2], &srcx, &srcy,
                        &srcz);
          int Csrcx = 0;
          int Csrcy = 0;
          int Csrcz = 0;
          if (f[k].CoarseStencil)
            FixDuplicates2(AllStencils, Coarse_Range, stencil, &f[k],
                           &f[remEl1], &Csrcx, &Csrcy, &Csrcz);
          buf->myunpacks[f[remEl1].infos[1]->halo_id].push_back(
              {info.offset, L[0], L[1], srcx, srcy, info.LX, info.LY,
               info.CoarseVersionOffset, info.CoarseVersionLX,
               info.CoarseVersionLY, Csrcx, Csrcy, f[remEl1].infos[0]->level,
               f[remEl1].icode[1], otherrank, f[remEl1].infos[0]->index[0],
               f[remEl1].infos[0]->index[1]});
          f[remEl1].dis = info.offset;
        }
      }
    }
    free(buf->send_buffer[r]);
    buf->send_buffer[r] =
        (Real *)malloc(dim * buf->send_buffer_size[r] * sizeof(Real));
    free(buf->recv_buffer[r]);
    buf->recv_buffer[r] =
        (Real *)malloc(dim * buf->recv_buffer_size[r] * sizeof(Real));
    buf->send_packinfos[r].clear();
    ToBeAveragedDown[r].clear();
    for (int i = 0; i < (int)buf->send_interfaces[r].size(); i++) {
      Interface *f = &buf->send_interfaces[r][i];
      if (!f->ToBeKept)
        continue;
      if (f->infos[0]->level <= f->infos[1]->level) {
        Range &range =
            DetermineStencil(AllStencils, Coarse_Range, stencil, f, false);
        buf->send_packinfos[r].push_back(
            {f->infos[0]->block, &buf->send_buffer[r][f->dis], range.sx,
             range.sy, range.ex, range.ey});
        if (f->CoarseStencil) {
          int V = (range.ex - range.sx) * (range.ey - range.sy);
          ToBeAveragedDown[r].push_back(i);
          ToBeAveragedDown[r].push_back(f->dis + V * dim);
        }
      } else {
        ToBeAveragedDown[r].push_back(i);
        ToBeAveragedDown[r].push_back(f->dis);
      }
    }
  }
  mapofHaloBlockGroups.clear();
  for (auto info : buf->halo_blocks) {
    int id = info->halo_id;
    UnPackInfo *unpacks = buf->myunpacks[id].data();
    std::set<int> ranks;
    for (size_t jj = 0; jj < buf->myunpacks[id].size(); jj++) {
      UnPackInfo *unpack = &unpacks[jj];
      ranks.insert(unpack->rank);
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
struct Synchronizer {
  bool use_averages;
  int sLength[3 * 27 * 3];
  std::array<Range, 3 * 27> AllStencils;
  std::unordered_map<int, MPI_Request *> mapofrequests;

  std::vector<MPI_Request *> reqs;
  std::vector<Real *> bufs;

  std::unordered_map<std::string, HaloBlockGroup> mapofHaloBlockGroups;
  std::vector<Info *> dummy_vector;
  std::vector<std::vector<int>> ToBeAveragedDown;
  struct Range Coarse_Range;
  struct SyncBuf *buf;
};
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
static void update_blocks(bool UpdateIDs, std::vector<Info *> *infos,
                          std::unordered_map<long long, Info *> *all,
                          std::unordered_map<long long, int> *tree) {
  std::vector<long long> myData;
  for (auto &info : *infos) {
    bool myflag = false;
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
          if (infoNeiTree >= 0 && infoNeiTree != sim.rank) {
            myflag = true;
            goto end;
          } else if (infoNeiTree == -2) {
            long long nCoarse = infoNei->Zparent;
            int infoNeiCoarserrank = treef(tree, infoNei->level - 1, nCoarse);
            if (infoNeiCoarserrank != sim.rank) {
              myflag = true;
              goto end;
            }
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
              int infoNeiFinerrank = treef(tree, infoNei->level + 1, nFine);
              if (infoNeiFinerrank != sim.rank) {
                myflag = true;
                goto end;
              }
            }
          } else if (infoNeiTree < 0) {
            myflag = true;
            goto end;
          }
        }
  end:
    if (myflag) {
      myData.push_back(info->level);
      myData.push_back(info->Z);
      if (UpdateIDs)
        myData.push_back(info->id);
    }
  }
  std::vector<int> neighbors;
  double *boxes;
  double box[4] = {DBL_MAX, DBL_MAX, -DBL_MAX, -DBL_MAX};
  for (auto &info : *infos) {
    double h = 1.0 / _BS_ / (1 << info->level);
    box[0] = std::min(box[0], info->origin[0] - 1.5 * h);
    box[1] = std::min(box[1], info->origin[1] - 1.5 * h);
    box[2] = std::max(box[2], info->origin[0] + h * _BS_ + 1.5 * h);
    box[3] = std::max(box[3], info->origin[1] + h * _BS_ + 1.5 * h);
  }
  boxes = (double *)malloc(sim.size * sizeof box);
  MPI_Allgather(box, 4, MPI_DOUBLE, boxes, 4, MPI_DOUBLE, MPI_COMM_WORLD);
  for (int i = 0; i < sim.size; i++) {
    if (i == sim.rank)
      continue;
    double *l2 = &boxes[i * 4];
    double *h2 = &boxes[i * 4 + 2];
    if (std::max(box[0], l2[0]) <= std::min(box[2], h2[0]) &&
        std::max(box[1], l2[1]) <= std::min(box[3], h2[1]))
      neighbors.push_back(i);
  }
  free(boxes);
  std::vector<std::vector<long long>> recv_buffer(neighbors.size());
  std::vector<std::vector<long long>> send_buffer(neighbors.size());
  std::vector<int> recv_size(neighbors.size());
  std::vector<MPI_Request> size_requests(2 * neighbors.size());
  int mysize = (int)myData.size();
  int kk = 0;
  for (auto r : neighbors) {
    MPI_Irecv(&recv_size[kk], 1, MPI_INT, r, 0, MPI_COMM_WORLD,
              &size_requests[2 * kk]);
    MPI_Isend(&mysize, 1, MPI_INT, r, 0, MPI_COMM_WORLD,
              &size_requests[2 * kk + 1]);
    kk++;
  }
  kk = 0;
  for (size_t j = 0; j < neighbors.size(); j++) {
    send_buffer[kk].resize(myData.size());
    for (size_t i = 0; i < myData.size(); i++)
      send_buffer[kk][i] = myData[i];
    kk++;
  }
  MPI_Waitall(size_requests.size(), size_requests.data(), MPI_STATUSES_IGNORE);
  std::vector<MPI_Request> requests(2 * neighbors.size());
  kk = 0;
  for (auto r : neighbors) {
    recv_buffer[kk].resize(recv_size[kk]);
    MPI_Irecv(recv_buffer[kk].data(), recv_buffer[kk].size(), MPI_LONG_LONG, r,
              0, MPI_COMM_WORLD, &requests[2 * kk]);
    MPI_Isend(send_buffer[kk].data(), send_buffer[kk].size(), MPI_LONG_LONG, r,
              0, MPI_COMM_WORLD, &requests[2 * kk + 1]);
    kk++;
  }
  MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
  kk = -1;
  int increment = UpdateIDs ? 3 : 2;
  for (auto r : neighbors) {
    kk++;
    for (size_t index = 0; index < recv_buffer[kk].size(); index += increment) {
      int level = (int)recv_buffer[kk][index];
      long long Z = recv_buffer[kk][index + 1];
      treef(tree, level, Z) = r;
      if (UpdateIDs)
        getf(all, level, Z)->id = recv_buffer[kk][index + 2];
      int p[2];
      sfc_inverse(Z, level, &p[0], &p[1]);
      if (level < sim.levelMax - 1)
        for (int j = 0; j < 2; j++)
          for (int i = 0; i < 2; i++) {
            long long nc = forward(level + 1, 2 * p[0] + i, 2 * p[1] + j);
            treef(tree, level + 1, nc) = -2;
          }
      if (level > 0) {
        long long nf = forward(level - 1, p[0] / 2, p[1] / 2);
        treef(tree, level - 1, nf) = -1;
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

struct Buffers {
  std::vector<std::vector<Face>> recv_faces;
  std::vector<std::vector<Face>> send_faces;
  std::vector<std::vector<Real>> recv_buffer;
  std::vector<std::vector<Real>> send_buffer;
  std::vector<BlockCase *> Cases;
  std::map<std::array<long long, 2>, BlockCase *> Map;
};
static void fillcase0(Face *F, Buffers *buf,
                      std::unordered_map<long long, int> *tree, int dim) {
  Info *info = F->infos[1];
  int icode = F->icode[1];
  int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, (icode / 9) % 3 - 1};
  int myFace = abs(code[0]) * std::max(0, code[0]) +
               abs(code[1]) * (std::max(0, code[1]) + 2) +
               abs(code[2]) * (std::max(0, code[2]) + 4);
  auto search = buf->Map.find({info->level, info->Z});
  assert(search != buf->Map.end());
  Real *CoarseFace = search->second->d[myFace];
  for (int B = 0; B <= 1; B++) {
    int aux = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
    long long Z = forward(info->level + 1,
                          2 * info->index[0] + std::max(code[0], 0) + code[0] +
                              (B % 2) * std::max(0, 1 - abs(code[0])),
                          2 * info->index[1] + std::max(code[1], 0) + code[1] +
                              aux * std::max(0, 1 - abs(code[1])));
    if (Z != F->infos[0]->Z)
      continue;
    int d = myFace / 2;
    int d1 = std::max((d + 1) % 3, (d + 2) % 3);
    int d2 = std::min((d + 1) % 3, (d + 2) % 3);
    int N1 = sizes[d1];
    int N2 = sizes[d2];
    int base = 0;
    if (B == 1)
      base = N2 / 2;
    else if (B == 2)
      base = (N1 / 2) * N2;
    else if (B == 3)
      base = (N2 / 2) + (N1 / 2) * N2;
    int r = treef(tree, F->infos[0]->level, F->infos[0]->Z);
    int dis = 0;
    for (int i2 = 0; i2 < N2; i2 += 2) {
      Real *s = &CoarseFace[dim * (base + (i2 / 2))];
      for (int j = 0; j < dim; j++)
        s[j] += buf->recv_buffer[r][F->offset + dis + j];
      dis += dim;
    }
  }
}
static void fillcase1(Face *F, int codex, int codey, Buffers *buf, int dim) {
  Info *info = F->infos[1];
  const int icode = F->icode[1];
  const int code[2] = {icode % 3 - 1, (icode / 3) % 3 - 1};
  if (abs(code[0]) != codex)
    return;
  if (abs(code[1]) != codey)
    return;
  const int myFace = abs(code[0]) * std::max(0, code[0]) +
                     abs(code[1]) * (std::max(0, code[1]) + 2);
  std::array<long long, 2> temp = {(long long)info->level, info->Z};
  auto search = buf->Map.find(temp);
  assert(search != buf->Map.end());
  BlockCase *CoarseCase = search->second;
  Real *CoarseFace = CoarseCase->d[myFace];
  Real *block = info->block;
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
static void prepare0(Buffers *buf, std::vector<Info *> *infos,
                     std::unordered_map<long long, Info *> *all,
                     std::unordered_map<long long, int> *tree, int dim) {
  buf->send_buffer.resize(sim.size);
  buf->recv_buffer.resize(sim.size);
  buf->send_faces.resize(sim.size);
  buf->recv_faces.resize(sim.size);
  for (int r = 0; r < sim.size; r++) {
    buf->send_faces[r].clear();
    buf->recv_faces[r].clear();
  }
  std::vector<int> send_buffer_size(sim.size, 0);
  std::vector<int> recv_buffer_size(sim.size, 0);
  for (size_t i = 0; i < buf->Cases.size(); i++) {
    for (int j = 0; j < 4; j++)
      free(buf->Cases[i]->d[j]);
    free(buf->Cases[i]);
  }
  buf->Cases.clear();
  buf->Map.clear();
  std::array<int, 6> icode = {1 * 2 + 3 * 1 + 9 * 1, 1 * 0 + 3 * 1 + 9 * 1,
                              1 * 1 + 3 * 2 + 9 * 1, 1 * 1 + 3 * 0 + 9 * 1,
                              1 * 1 + 3 * 1 + 9 * 2, 1 * 1 + 3 * 1 + 9 * 0};
  for (auto &info : *infos) {
    getf(all, info->level, info->Z)->auxiliary = nullptr;
    info->auxiliary = nullptr;
    int aux = 1 << info->level;
    bool xskin = info->index[0] == 0 || info->index[0] == aux - 1;
    bool yskin = info->index[1] == 0 || info->index[1] == aux - 1;
    int xskip = info->index[0] == 0 ? -1 : 1;
    int yskip = info->index[1] == 0 ? -1 : 1;

    bool storeFace[4] = {false, false, false, false};
    bool stored = false;
    for (int f = 0; f < 6; f++) {
      const int code[3] = {icode[f] % 3 - 1, (icode[f] / 3) % 3 - 1,
                           (icode[f] / 9) % 3 - 1};
      if (code[0] == xskip && xskin)
        continue;
      if (code[1] == yskip && yskin)
        continue;
      if (code[2] != 0)
        continue;
      if (!(treef(tree, info->level, info->Znei[1 + code[0]][1 + code[1]]) >=
            0)) {
        storeFace[abs(code[0]) * std::max(0, code[0]) +
                  abs(code[1]) * (std::max(0, code[1]) + 2)] = true;
        stored = true;
      }
      int L[3];
      L[0] = code[0] == 0 ? _BS_ / 2 : 1;
      L[1] = code[1] == 0 ? _BS_ / 2 : 1;
      int V = L[0] * L[1];
      if (treef(tree, info->level, info->Znei[1 + code[0]][1 + code[1]]) ==
          -2) {
        Info *infoNei =
            getf(all, info->level, info->Znei[1 + code[0]][1 + code[1]]);
        const long long nCoarse = infoNei->Zparent;
        Info *infoNeiCoarser = getf(all, info->level - 1, nCoarse);
        const int infoNeiCoarserrank = treef(tree, info->level - 1, nCoarse);
        int code2[3] = {-code[0], -code[1], -code[2]};
        int icode2 = (code2[0] + 1) + (code2[1] + 1) * 3 + (code2[2] + 1) * 9;
        buf->send_faces[infoNeiCoarserrank].push_back(
            Face(info, infoNeiCoarser, icode[f], icode2));
        send_buffer_size[infoNeiCoarserrank] += V;
      } else if (treef(tree, info->level,
                       info->Znei[1 + code[0]][1 + code[1]]) == -1) {
        Info *infoNei =
            getf(all, info->level, info->Znei[1 + code[0]][1 + code[1]]);
        int Bstep = 1;
        for (int B = 0; B <= 1; B += Bstep) {
          const int temp = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
          const long long nFine =
              infoNei->Zchild[std::max(-code[0], 0) +
                              (B % 2) * std::max(0, 1 - abs(code[0]))]
                             [std::max(-code[1], 0) +
                              temp * std::max(0, 1 - abs(code[1]))];
          const int infoNeiFinerrank = treef(tree, infoNei->level + 1, nFine);
          Info *infoNeiFiner = getf(all, infoNei->level + 1, nFine);
          int icode2 = (-code[0] + 1) + (-code[1] + 1) * 3 + (-code[2] + 1) * 9;
          buf->recv_faces[infoNeiFinerrank].push_back(
              Face(infoNeiFiner, info, icode2, icode[f]));
          assert(0 <= infoNeiFinerrank);
          assert(infoNeiFinerrank < sim.size);
          assert(recv_buffer_size.size() == sim.size);
          recv_buffer_size[infoNeiFinerrank] += V;
        }
      }
    }
    if (stored) {
      BlockCase *c = (BlockCase *)malloc(sizeof(BlockCase));
      c->level = info->level;
      c->Z = info->Z;
      for (int i = 0; i < 4; i++)
        c->d[i] =
            storeFace[i] ? (Real *)malloc(_BS_ * dim * sizeof(Real)) : nullptr;
      buf->Cases.push_back(c);
    }
  }
  size_t Cases_index = 0;
  if (buf->Cases.size() > 0)
    for (auto &info : *infos) {
      if (Cases_index == buf->Cases.size())
        break;
      if (buf->Cases[Cases_index]->level == info->level &&
          buf->Cases[Cases_index]->Z == info->Z) {
        buf->Map.insert(std::pair<std::array<long long, 2>, BlockCase *>(
            {buf->Cases[Cases_index]->level, buf->Cases[Cases_index]->Z},
            buf->Cases[Cases_index]));
        getf(all, buf->Cases[Cases_index]->level, buf->Cases[Cases_index]->Z)
            ->auxiliary = buf->Cases[Cases_index];
        info->auxiliary = buf->Cases[Cases_index];
        Cases_index++;
      }
    }
  for (int r = 0; r < sim.size; r++) {
    std::sort(buf->send_faces[r].begin(), buf->send_faces[r].end());
    std::sort(buf->recv_faces[r].begin(), buf->recv_faces[r].end());
  }
  for (int r = 0; r < sim.size; r++) {
    buf->send_buffer[r].resize(send_buffer_size[r] * dim);
    buf->recv_buffer[r].resize(recv_buffer_size[r] * dim);
    int offset = 0;
    for (int k = 0; k < (int)buf->recv_faces[r].size(); k++) {
      Face &f = buf->recv_faces[r][k];
      const int code[3] = {f.icode[1] % 3 - 1, (f.icode[1] / 3) % 3 - 1,
                           (f.icode[1] / 9) % 3 - 1};
      int V = ((code[0] == 0) ? _BS_ / 2 : 1) * ((code[1] == 0) ? _BS_ / 2 : 1);
      f.offset = offset;
      offset += V * dim;
    }
  }
}
static void fillcases(Buffers *buf, std::unordered_map<long long, int> *tree,
                      int dim) {
  for (int r = 0; r < sim.size; r++) {
    int displacement = 0;
    for (int k = 0; k < (int)buf->send_faces[r].size(); k++) {
      Face &f = buf->send_faces[r][k];
      Info *info = f.infos[0];
      auto search = buf->Map.find({(long long)info->level, info->Z});
      assert(search != buf->Map.end());
      BlockCase *FineCase = search->second;
      int icode = f.icode[0];
      assert((icode / 9) % 3 - 1 == 0);
      int code[2] = {icode % 3 - 1, (icode / 3) % 3 - 1};
      int myFace = abs(code[0]) * std::max(0, code[0]) +
                   abs(code[1]) * (std::max(0, code[1]) + 2);
      Real *FineFace = FineCase->d[myFace];
      int d = myFace / 2;
      assert(d == 0 || d == 1);
      int d2 = std::min((d + 1) % 3, (d + 2) % 3);
      int N2 = sizes[d2];
      for (int i2 = 0; i2 < N2; i2 += 2) {
        Real *a = &FineFace[dim * i2];
        Real *b = &FineFace[dim * (i2 + 1)];
        for (d = 0; d < dim; d++) {
          Real avg = a[d] + b[d];
          memcpy(&buf->send_buffer[r][displacement], &avg, sizeof(Real));
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
      if (buf->recv_buffer[r].size() != 0) {
        MPI_Request req{};
        recv_requests.push_back(req);
        MPI_Irecv(&buf->recv_buffer[r][0], buf->recv_buffer[r].size(), MPI_Real,
                  r, 123456, MPI_COMM_WORLD, &recv_requests.back());
      }
      if (buf->send_buffer[r].size() != 0) {
        MPI_Request req{};
        send_requests.push_back(req);
        MPI_Isend(&buf->send_buffer[r][0], buf->send_buffer[r].size(), MPI_Real,
                  r, 123456, MPI_COMM_WORLD, &send_requests.back());
      }
    }
  if (buf->recv_buffer[sim.rank].size() > 0 &&
      buf->send_buffer[sim.rank].size() > 0)
    memcpy(&buf->recv_buffer[sim.rank][0], &buf->send_buffer[sim.rank][0],
           buf->send_buffer[sim.rank].size() * sizeof(Real));
  for (int index = 0; index < (int)buf->recv_faces[sim.rank].size(); index++)
    fillcase0(&buf->recv_faces[sim.rank][index], buf, tree, dim);
  if (recv_requests.size() > 0)
    MPI_Waitall(recv_requests.size(), &recv_requests[0], MPI_STATUSES_IGNORE);
  for (int r = 0; r < sim.size; r++)
    if (r != sim.rank)
      for (int index = 0; index < (int)buf->recv_faces[r].size(); index++)
        fillcase0(&buf->recv_faces[r][index], buf, tree, dim);
  for (int r = 0; r < sim.size; r++)
    for (int index = 0; index < (int)buf->recv_faces[r].size(); index++)
      fillcase1(&buf->recv_faces[r][index], 1, 0, buf, dim);
  for (int r = 0; r < sim.size; r++)
    for (int index = 0; index < (int)buf->recv_faces[r].size(); index++)
      fillcase1(&buf->recv_faces[r][index], 0, 1, buf, dim);
  if (send_requests.size() > 0)
    MPI_Waitall(send_requests.size(), &send_requests[0], MPI_STATUSES_IGNORE);
}
static void update_boundary(bool clean, std::vector<Info *> *boundary,
                            std::unordered_map<long long, Info *> *all,
                            std::unordered_map<long long, int> *tree) {
  std::vector<std::vector<long long>> send_buffer(sim.size);
  std::vector<Info *> &bbb = *boundary;
  std::set<int> Neighbors;
  for (size_t jjj = 0; jjj < bbb.size(); jjj++) {
    Info *info = bbb[jjj];
    std::set<int> receivers;
    const int aux = 1 << info->level;
    const bool xskin = info->index[0] == 0 || info->index[0] == aux - 1;
    const bool yskin = info->index[1] == 0 || info->index[1] == aux - 1;
    const int xskip = info->index[0] == 0 ? -1 : 1;
    const int yskip = info->index[1] == 0 ? -1 : 1;

    for (int icode = 0; icode < 27; icode++) {
      if (icode == 1 * 1 + 3 * 1 + 9 * 1)
        continue;
      const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                           (icode / 9) % 3 - 1};
      if (code[0] == xskip && xskin)
        continue;
      if (code[1] == yskip && yskin)
        continue;
      if (code[2] != 0)
        continue;
      Info *infoNei =
          getf(all, info->level, info->Znei[1 + code[0]][1 + code[1]]);
      const int &infoNeiTree = treef(tree, infoNei->level, infoNei->Z);
      if (infoNeiTree >= 0 && infoNeiTree != sim.rank) {
        if (infoNei->state != Refine || clean)
          infoNei->state = Leave;
        receivers.insert(infoNeiTree);
        Neighbors.insert(infoNeiTree);
      } else if (infoNeiTree == -2) {
        const long long nCoarse = infoNei->Zparent;
        Info *infoNeiCoarser = getf(all, infoNei->level - 1, nCoarse);
        const int infoNeiCoarserrank = treef(tree, infoNei->level - 1, nCoarse);
        if (infoNeiCoarserrank != sim.rank) {
          assert(infoNeiCoarserrank >= 0);
          if (infoNeiCoarser->state != Refine || clean)
            infoNeiCoarser->state = Leave;
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
              infoNei->Zchild[std::max(-code[0], 0) +
                              (B % 2) * std::max(0, 1 - abs(code[0]))]
                             [std::max(-code[1], 0) +
                              temp * std::max(0, 1 - abs(code[1]))];
          Info *infoNeiFiner = getf(all, infoNei->level + 1, nFine);
          const int infoNeiFinerrank = treef(tree, infoNei->level + 1, nFine);
          if (infoNeiFinerrank != sim.rank) {
            if (infoNeiFiner->state != Refine || clean)
              infoNeiFiner->state = Leave;
            receivers.insert(infoNeiFinerrank);
            Neighbors.insert(infoNeiFinerrank);
          }
        }
      }
    }
    if (info->changed2 && info->state != Leave) {
      if (info->state == Refine)
        info->changed2 = false;
      std::set<int>::iterator it = receivers.begin();
      while (it != receivers.end()) {
        int temp = (info->state == Compress) ? 1 : 2;
        send_buffer[*it].push_back(info->level);
        send_buffer[*it].push_back(info->Z);
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
        getf(all, level, Z)->state =
            (recv_buffer[r][index + 2] == 1) ? Compress : Refine;
      }
};
static Synchronizer *sync1(const Stencil &stencil,
                           std::map<Stencil, Synchronizer *> *synchronizers,
                           std::unordered_map<long long, int> *tree,
                           std::unordered_map<long long, Info *> *all,
                           std::vector<Info *> *infos, size_t *timestamp,
                           int dim) {
  Synchronizer *s;
  auto itSynchronizerMPI = synchronizers->find(stencil);
  if (itSynchronizerMPI == synchronizers->end()) {
    s = new Synchronizer;
    s->buf = new SyncBuf;
    s->use_averages = stencil.tensorial || stencil.sx < -2 || stencil.sy < -2 ||
                      stencil.ex > 3 || stencil.ey > 3;
    s->buf->send_interfaces.resize(sim.size);
    s->buf->recv_interfaces.resize(sim.size);
    s->buf->send_packinfos.resize(sim.size);
    s->buf->send_buffer_size.resize(sim.size);
    s->buf->recv_buffer_size.resize(sim.size);
    s->buf->send_buffer = (Real **)malloc(sim.size * sizeof(Real *));
    s->buf->recv_buffer = (Real **)malloc(sim.size * sizeof(Real *));
    for (int i = 0; i < sim.size; i++) {
      s->buf->send_buffer[i] = NULL;
      s->buf->recv_buffer[i] = NULL;
    }

    s->ToBeAveragedDown.resize(sim.size);
    const int sC[3] = {(stencil.sx - 1) / 2 - 1, (stencil.sy - 1) / 2 - 1,
                       (0 - 1) / 2 + 0};
    const int eC[3] = {stencil.ex / 2 + 2, stencil.ey / 2 + 2, 1 / 2 + 1};
    for (int icode = 0; icode < 27; icode++) {
      const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                           (icode / 9) % 3 - 1};
      Range &range0 = s->AllStencils[icode];
      range0.sx = code[0] < 1 ? (code[0] < 0 ? _BS_ + stencil.sx : 0) : 0;
      range0.sy = code[1] < 1 ? (code[1] < 0 ? _BS_ + stencil.sy : 0) : 0;
      range0.ex = code[0] < 1 ? _BS_ : stencil.ex - 1;
      range0.ey = code[1] < 1 ? _BS_ : stencil.ey - 1;
      s->sLength[3 * icode + 0] = range0.ex - range0.sx;
      s->sLength[3 * icode + 1] = range0.ey - range0.sy;
      s->sLength[3 * icode + 2] = 1;
      Range &range1 = s->AllStencils[icode + 27];
      range1.sx = code[0] < 1 ? (code[0] < 0 ? _BS_ + 2 * stencil.sx : 0) : 0;
      range1.sy = code[1] < 1 ? (code[1] < 0 ? _BS_ + 2 * stencil.sy : 0) : 0;
      range1.ex = code[0] < 1 ? _BS_ : 2 * (stencil.ex - 1);
      range1.ey = code[1] < 1 ? _BS_ : 2 * (stencil.ey - 1);
      s->sLength[3 * (icode + 27) + 0] = (range1.ex - range1.sx) / 2;
      s->sLength[3 * (icode + 27) + 1] = (range1.ey - range1.sy) / 2;
      s->sLength[3 * (icode + 27) + 2] = 1;
      Range &range2 = s->AllStencils[icode + 2 * 27];
      range2.sx = code[0] < 1 ? (code[0] < 0 ? _BS_ / 2 + sC[0] : 0) : 0;
      range2.sy = code[1] < 1 ? (code[1] < 0 ? _BS_ / 2 + sC[1] : 0) : 0;
      range2.ex = code[0] < 1 ? _BS_ / 2 : eC[0] - 1;
      range2.ey = code[1] < 1 ? _BS_ / 2 : eC[1] - 1;
      s->sLength[3 * (icode + 2 * 27) + 0] = range2.ex - range2.sx;
      s->sLength[3 * (icode + 2 * 27) + 1] = range2.ey - range2.sy;
      s->sLength[3 * (icode + 2 * 27) + 2] = 1;
    }
    Setup(dim, tree, all, infos, s->buf, s->use_averages, s->AllStencils,
          s->Coarse_Range, stencil, s->sLength, s->ToBeAveragedDown,
          s->mapofHaloBlockGroups

    );
    (*synchronizers)[stencil] = s;
  } else {
    s = itSynchronizerMPI->second;
  }
  auto it = s->mapofHaloBlockGroups.begin();
  while (it != s->mapofHaloBlockGroups.end()) {
    (it->second).ready = false;
    it++;
  }
  s->reqs.clear();
  s->bufs.clear();
  s->mapofrequests.clear();
  s->buf->requests.clear();
  s->buf->requests.reserve(2 * sim.size);
  for (auto r : s->buf->Neighbors)
    if (s->buf->recv_buffer_size[r] > 0) {
      s->buf->requests.resize(s->buf->requests.size() + 1);
      Real *buf = &s->buf->recv_buffer[r][0];
      MPI_Request *req = &s->buf->requests.back();
      MPI_Irecv(buf, s->buf->recv_buffer_size[r] * dim, MPI_Real, r, *timestamp,
                MPI_COMM_WORLD, req);
      s->mapofrequests[r] = req;
      s->reqs.push_back(req);
      s->bufs.push_back(buf);
    }
  for (int r = 0; r < sim.size; r++)
    if (s->buf->send_buffer_size[r] != 0) {
#pragma omp parallel
      {
#pragma omp for
        for (size_t j = 0; j < s->ToBeAveragedDown[r].size(); j += 2) {
          int i = s->ToBeAveragedDown[r][j];
          int d = s->ToBeAveragedDown[r][j + 1];
          Interface &f = s->buf->send_interfaces[r][i];
          int code[3] = {-(f.icode[0] % 3 - 1), -((f.icode[0] / 3) % 3 - 1),
                         -((f.icode[0] / 9) % 3 - 1)};
          if (f.CoarseStencil) {
            Real *dst = s->buf->send_buffer[r] + d;
            const Info *const info = f.infos[0];
            int eC[2] = {(stencil.ex) / 2 + 2, (stencil.ey) / 2 + 2};
            int sC[2] = {(stencil.sx - 1) / 2 - 1, (stencil.sy - 1) / 2 - 1};
            int s[2] = {code[0] < 1 ? (code[0] < 0 ? sC[0] : 0) : _BS_ / 2,
                        code[1] < 1 ? (code[1] < 0 ? sC[1] : 0) : _BS_ / 2};
            int e[2] = {code[0] < 1 ? (code[0] < 0 ? 0 : _BS_ / 2)
                                    : _BS_ / 2 + eC[0] - 1,
                        code[1] < 1 ? (code[1] < 0 ? 0 : _BS_ / 2)
                                    : _BS_ / 2 + eC[1] - 1};
            Real *src = (*info).block;
            int pos = 0;
            for (int iy = s[1]; iy < e[1]; iy++) {
              int YY = 2 * (iy - s[1]) + s[1] +
                       std::max(code[1], 0) * _BS_ / 2 - code[1] * _BS_ +
                       std::min(0, code[1]) * (e[1] - s[1]);
              for (int ix = s[0]; ix < e[0]; ix++) {
                int XX = 2 * (ix - s[0]) + s[0] +
                         std::max(code[0], 0) * _BS_ / 2 - code[0] * _BS_ +
                         std::min(0, code[0]) * (e[0] - s[0]);
                for (int c = 0; c < dim; c++) {
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
            Real *dst = s->buf->send_buffer[r] + d;
            const Info *const info = f.infos[0];
            int s[2] = {code[0] < 1 ? (code[0] < 0 ? stencil.sx : 0) : _BS_,
                        code[1] < 1 ? (code[1] < 0 ? stencil.sy : 0) : _BS_};
            int e[2] = {
                code[0] < 1 ? (code[0] < 0 ? 0 : _BS_) : _BS_ + stencil.ex - 1,
                code[1] < 1 ? (code[1] < 0 ? 0 : _BS_) : _BS_ + stencil.ey - 1};
            Real *src = (*info).block;
            int xStep = (code[0] == 0) ? 2 : 1;
            int yStep = (code[1] == 0) ? 2 : 1;
            int pos = 0;
            for (int iy = s[1]; iy < e[1]; iy += yStep) {
              int YY = (abs(code[1]) == 1) ? 2 * (iy - code[1] * _BS_) +
                                                 std::min(0, code[1]) * _BS_
                                           : iy;
              for (int ix = s[0]; ix < e[0]; ix += xStep) {
                int XX = (abs(code[0]) == 1) ? 2 * (ix - code[0] * _BS_) +
                                                   std::min(0, code[0]) * _BS_
                                             : ix;
                for (int c = 0; c < dim; c++) {
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
        for (size_t i = 0; i < s->buf->send_packinfos[r].size(); i++) {
          const PackInfo &info = s->buf->send_packinfos[r][i];
          pack(info.block, info.pack, dim, info.sx, info.sy, info.ex, info.ey);
        }
      }
    }
  for (auto r : s->buf->Neighbors)
    if (s->buf->send_buffer_size[r] > 0) {
      s->buf->requests.resize(s->buf->requests.size() + 1);
      MPI_Isend(&s->buf->send_buffer[r][0], s->buf->send_buffer_size[r] * dim,
                MPI_Real, r, *timestamp, MPI_COMM_WORLD,
                &s->buf->requests.back());
    }
  *timestamp = (*timestamp + 1) % 32768;
  return s;
}
static void dealloc(int m, long long n, std::vector<Info *> *infos) {
  for (size_t j = 0; j < infos->size(); j++) {
    if ((*infos)[j]->level == m && (*infos)[j]->Z == n) {
      free((*infos)[j]->block);
      infos->erase(infos->begin() + j);
      return;
    }
  }
}
static Real *avail(int m, long long n, std::unordered_map<long long, int> *tree,
                   std::unordered_map<long long, Info *> *all) {
  return (treef(tree, m, n) == sim.rank) ? getf(all, m, n)->block : nullptr;
}
static Real *avail1(int ix, int iy, int m,
                    std::unordered_map<long long, int> *tree,
                    std::unordered_map<long long, Info *> *all) {
  const long long n = forward(m, ix, iy);
  return avail(m, n, tree, all);
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
  size_t timestamp;
  std::map<Stencil, Synchronizer *> *synchronizers;
  std::unordered_map<long long, Info *> all;
  std::unordered_map<long long, int> tree;
  std::vector<Info *> boundary;
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
            std::unordered_map<long long, Info *> *all, SyncBuf *buf,
            const Stencil &stencil, Info *info, bool applybc, int *sLength) {
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
                         info->level - 1, tree, all);
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
        myblocks[icode] =
            avail(info->level, info->Znei[1 + cx][1 + cy], tree, all);
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
                           info->level + 1, tree, all);
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
    if (sim.size == 1)
      post_load(info, applybc);
    int id = info->halo_id;
    if (id >= 0) {
      UnPackInfo *unpacks = buf->myunpacks[id].data();
      for (size_t jj = 0; jj < buf->myunpacks[id].size(); jj++) {
        UnPackInfo *unpack = &unpacks[jj];
        int cx = unpack->icode % 3 - 1;
        int cy = (unpack->icode / 3) % 3 - 1;
        int otherrank = unpack->rank;
        int s[3] = {cx < 1 ? (cx < 0 ? stencil.sx : 0) : _BS_,
                    cy < 1 ? (cy < 0 ? stencil.sy : 0) : _BS_,
                    0 < 1 ? (0 < 0 ? 0 : 0) : 1};
        int e[3] = {cx < 1 ? (cx < 0 ? 0 : _BS_) : _BS_ + stencil.ex - 1,
                    cy < 1 ? (cy < 0 ? 0 : _BS_) : _BS_ + stencil.ey - 1,
                    0 < 1 ? (0 < 0 ? 0 : 1) : 1};
        if (unpack->level == info->level) {
          Real *dstbase =
              m + ((s[2] - 0) * nm[0] * nm[1] + (s[1] - stencil.sy) * nm[0] +
                   s[0] - stencil.sx) *
                      dim;
          Real *srcbase = buf->recv_buffer[otherrank] + unpack->offset +
                          dim * (unpack->x + unpack->LX * unpack->y);
          CHECK;
          for (int yd = 0; yd < unpack->ly; ++yd) {
            Real *dst = dstbase + dim * nm[0] * yd;
            Real *src = srcbase + dim * unpack->LX * yd;
            std::memcpy(dst, src, sizeof(Real) * dim * unpack->lx);
          }
          if (unpack->CoarseVersionOffset >= 0) {
            int offset[3] = {(stencil.sx - 1) / 2 - 1, (stencil.sy - 1) / 2 - 1,
                             (0 - 1) / 2 + 0};
            int sC[3] = {cx < 1 ? (cx < 0 ? offset[0] : 0) : _BS_ / 2,
                         cy < 1 ? (cy < 0 ? offset[1] : 0) : _BS_ / 2,
                         0 < 1 ? (0 < 0 ? offset[2] : 0) : 1 / 2};
            Real *dst1 = c + ((sC[2] - offset[2]) * nc[0] * nc[1] +
                              (sC[1] - offset[1]) * nc[0] + sC[0] - offset[0]) *
                                 dim;
            int L[3];
            int icode = (-cx + 1) + 3 * (-cy + 1) + 9 * (-0 + 1);
            L[0] = sLength[3 * (icode + 2 * 27) + 0];
            L[1] = sLength[3 * (icode + 2 * 27) + 1];
            L[2] = sLength[3 * (icode + 2 * 27) + 2];
            unpack_subregion(buf->recv_buffer[otherrank] + unpack->offset +
                                 unpack->CoarseVersionOffset,
                             &dst1[0], dim, unpack->CoarseVersionx,
                             unpack->CoarseVersiony, unpack->CoarseVersionLX,
                             L[0], L[1], nc[0]);
          }
        } else if (unpack->level < info->level) {
          int offset[2] = {(stencil.sx - 1) / 2 - 1, (stencil.sy - 1) / 2 - 1};
          int C[2] = {cx < 1 ? (cx < 0 ? offset[0] : 0) : _BS_ / 2,
                      cy < 1 ? (cy < 0 ? offset[1] : 0) : _BS_ / 2};
          Real *dstbase =
              c + dim * (C[0] - offset[0] + (C[1] - offset[1]) * nc[0]);
          Real *srcbase = buf->recv_buffer[otherrank] + unpack->offset +
                          dim * (unpack->x + unpack->LX * unpack->y);
          int req = dim * (unpack->LX * unpack->ly + unpack->lx - unpack->LX);
          assert(unpack->lx == 0 || req + unpack->offset <=
                                        dim * buf->recv_buffer_size[otherrank]);
          CHECK;
          for (int yd = 0; yd < unpack->ly; ++yd) {
            Real *dst = dstbase + dim * nc[0] * yd;
            Real *src = srcbase + dim * unpack->LX * yd;
            std::memcpy(dst, src, sizeof(Real) * dim * unpack->lx);
          }
        } else {
          int B;
          if ((abs(cx) + abs(cy) + abs(0) == 3))
            B = 0;
          else if ((abs(cx) + abs(cy) + abs(0) == 2)) {
            int t;
            if (cx == 0)
              t = unpack->index_0 - 2 * xi;
            else if (cy == 0)
              t = unpack->index_1 - 2 * yi;
            else
              t = -2 * 0;
            assert(t == 0 || t == 1);
            B = (t == 1) ? 3 : 0;
          } else {
            int Bmod, Bdiv;
            if (abs(cx) == 1) {
              Bmod = unpack->index_1 - 2 * yi;
              Bdiv = -2 * 0;
            } else if (abs(cy) == 1) {
              Bmod = unpack->index_0 - 2 * xi;
              Bdiv = -2 * 0;
            } else {
              Bmod = unpack->index_0 - 2 * xi;
              Bdiv = unpack->index_1 - 2 * yi;
            }
            B = 2 * Bdiv + Bmod;
          }
          int aux1 = (abs(cx) == 1) ? (B % 2) : (B / 2);
          Real *dstbase =
              m +
              ((abs(0) * (s[2] - 0) +
                (1 - abs(0)) * (0 + (B / 2) * (e[2] - s[2]) / 2)) *
                   nm[0] * nm[1] +
               (abs(cy) * (s[1] - stencil.sy) +
                (1 - abs(cy)) * (-stencil.sy + aux1 * (e[1] - s[1]) / 2)) *
                   nm[0] +
               abs(cx) * (s[0] - stencil.sx) +
               (1 - abs(cx)) * (-stencil.sx + (B % 2) * (e[0] - s[0]) / 2)) *
                  dim;
          Real *srcbase = buf->recv_buffer[otherrank] + unpack->offset +
                          dim * (unpack->x + unpack->LX * unpack->y);
          CHECK;
          for (int yd = 0; yd < unpack->ly; ++yd) {
            Real *dst = dstbase + dim * nm[0] * yd;
            Real *src = srcbase + dim * unpack->LX * yd;
            std::memcpy(dst, src, sizeof(Real) * dim * unpack->lx);
          }
        }
      }
    }
    if (sim.size > 1)
      post_load(info, applybc);
  }

  void post_load(Info *info, bool applybc) {
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
    int aux = 1 << info->level;
    bool xskin = info->index[0] == 0 || info->index[0] == aux - 1;
    bool yskin = info->index[1] == 0 || info->index[1] == aux - 1;
    int xskip = info->index[0] == 0 ? -1 : 1;
    int yskip = info->index[1] == 0 ? -1 : 1;
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
static void _alloc(int level, long long Z,
                   std::unordered_map<long long, Info *> *all,
                   std::vector<Info *> *infos,
                   std::unordered_map<long long, int> *tree, int dim) {
  Info *new_info = getf(all, level, Z);
  new_info->block = (Real *)malloc(dim * _BS_ * _BS_ * sizeof(Real));
#pragma omp critical
  { infos->push_back(new_info); }
  treef(tree, level, Z) = sim.rank;
}
static void AddBlock(int dim, Grid *grid, int level, long long Z,
                     uint8_t *data) {
  _alloc(level, Z, &grid->all, &grid->infos, &grid->tree, dim);
  Info *info = getf(&grid->all, level, Z);
  memcpy(info->block, data, _BS_ * _BS_ * dim * sizeof(Real));
  int p[2];
  sfc_inverse(Z, level, &p[0], &p[1]);
  if (level < sim.levelMax - 1)
    for (int j1 = 0; j1 < 2; j1++)
      for (int i1 = 0; i1 < 2; i1++) {
        long long nc = forward(level + 1, 2 * p[0] + i1, 2 * p[1] + j1);
        treef(&grid->tree, level + 1, nc) = -2;
      }
  if (level > 0) {
    long long nf = forward(level - 1, p[0] / 2, p[1] / 2);
    treef(&grid->tree, level - 1, nf) = -1;
  }
}
struct MPI_Block {
  long long level;
  long long Z;
  uint8_t data[_BS_ * _BS_ * max_dim * sizeof(Real)];
};
template <typename Kernel>
static void computeA(Kernel &&kernel, Grid *g, int dim) {
  Synchronizer *Synch = sync1(kernel.stencil, g->synchronizers, &g->tree,
                              &g->all, &g->infos, &g->timestamp, dim);
  std::vector<Info *> *inner = &Synch->buf->inner_blocks;
  std::vector<Info *> *halo_next;
  bool done = false;
#pragma omp parallel
  {
    BlockLab lab(dim);
    lab.prepare(kernel.stencil);
#pragma omp for nowait
    for (std::size_t i = 0; i < inner->size(); ++i) {
      const auto &I = (*inner)[i];
      lab.load(&g->tree, &g->all, Synch->buf, kernel.stencil, I, true,
               Synch->sLength);
      kernel(lab.m, I);
    }
    while (done == false) {
#pragma omp master
      {
        for (;;) {
          bool all;
          all = true;
          for (auto &it : Synch->mapofHaloBlockGroups) {
            if (it.second.ready == false) {
              std::set<int> ranks = it.second.myranks;
              int flag = 0;
              for (auto r : ranks) {
                const auto retval = Synch->mapofrequests.find(r);
                MPI_Status status;
                int err;
                if ((err = MPI_Test(retval->second, &flag, &status)) !=
                    MPI_SUCCESS) {
                  int len;
                  char err_string[MPI_MAX_ERROR_STRING];
                  MPI_Error_string(err, err_string, &len);
                  fprintf(stderr, "%s:%d: error: %s\n", __FILE__, __LINE__,
                          err_string);
                  MPI_Abort(MPI_COMM_WORLD, 1);
                }
                if (flag == false)
                  break;
              }
              if (flag == 1) {
                it.second.ready = true;
                halo_next = &it.second.myblocks;
                goto done;
              }
            }
            all = all && it.second.ready;
          }
          if (all) {
            halo_next = &Synch->dummy_vector;
            goto done;
          }
        }
      done:;
      }
#pragma omp barrier
#pragma omp for nowait
      for (std::size_t i = 0; i < halo_next->size(); ++i) {
        const auto &I = (*halo_next)[i];
        lab.load(&g->tree, &g->all, Synch->buf, kernel.stencil, I, true,
                 Synch->sLength);
        kernel(lab.m, I);
      }
#pragma omp single
      {
        if (halo_next->size() == 0)
          done = true;
      }
    }
  }
  MPI_Waitall(Synch->buf->requests.size(), Synch->buf->requests.data(),
              MPI_STATUSES_IGNORE);
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
  struct Buffers *buf1, *buf2;
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
  BlockCase *tempCase = (BlockCase *)(tmpInfo[info->id]->auxiliary);
  Real *faceXm = nullptr;
  Real *faceXp = nullptr;
  Real *faceYm = nullptr;
  Real *faceYp = nullptr;
  if (tempCase != nullptr) {
    faceXm = tempCase->d[0];
    faceXp = tempCase->d[1];
    faceYm = tempCase->d[2];
    faceYp = tempCase->d[3];
  }
  if (faceXm != nullptr) {
    int ix = 0;
    for (int iy = 0; iy < _BS_; ++iy) {
      int ip0 = ix - stencil.sx;
      int jp0 = iy - stencil.sy;
      int im1 = ip0 - 1;
      Real *v0 = vm + 2 * (nm * jp0 + ip0) + 0;
      Real *u0 = um + 2 * (nm * jp0 + ip0) + 0;
      Real *v1 = vm + 2 * (nm * jp0 + im1) + 0;
      Real *u1 = um + 2 * (nm * jp0 + im1) + 0;
      faceXm[iy] =
          facDiv * (*v1 + *v0) - (facDiv * CHI[_BS_ * iy + ix]) * (*u1 + *u0);
    }
  }
  if (faceXp != nullptr) {
    int ix = _BS_ - 1;
    for (int iy = 0; iy < _BS_; ++iy) {
      int ip0 = ix - stencil.sx;
      int jp0 = iy - stencil.sy;
      int ip1 = ip0 + 1;
      Real *v0 = vm + 2 * (nm * jp0 + ip0) + 0;
      Real *u0 = um + 2 * (nm * jp0 + ip0) + 0;
      Real *v1 = vm + 2 * (nm * jp0 + ip1) + 0;
      Real *u1 = um + 2 * (nm * jp0 + ip1) + 0;
      faceXp[iy] =
          -facDiv * (*v1 + *v0) + (facDiv * CHI[_BS_ * iy + ix]) * (*u1 + *u0);
    }
  }
  if (faceYm != nullptr) {
    int iy = 0;
    for (int ix = 0; ix < _BS_; ++ix) {
      int ip0 = ix - stencil.sx;
      int jp0 = iy - stencil.sy;
      int jm1 = jp0 - 1;
      Real *v0 = vm + 2 * (nm * jp0 + ip0) + 1;
      Real *u0 = um + 2 * (nm * jp0 + ip0) + 1;
      Real *v1 = vm + 2 * (nm * jm1 + ip0) + 1;
      Real *u1 = um + 2 * (nm * jm1 + ip0) + 1;
      faceYm[ix] =
          facDiv * (*v1 + *v0) - (facDiv * CHI[_BS_ * iy + ix]) * (*u1 + *u0);
    }
  }
  if (faceYp != nullptr) {
    int iy = _BS_ - 1;
    for (int ix = 0; ix < _BS_; ++ix) {
      int ip0 = ix - stencil.sx;
      int jp0 = iy - stencil.sy;
      int jp1 = jp0 + 1;
      Real *v0 = vm + 2 * (nm * jp0 + ip0) + 1;
      Real *u0 = um + 2 * (nm * jp0 + ip0) + 1;
      Real *v1 = vm + 2 * (nm * jp1 + ip0) + 1;
      Real *u1 = um + 2 * (nm * jp1 + ip0) + 1;
      faceYp[ix] =
          -facDiv * (*v1 + *v0) + (facDiv * CHI[_BS_ * iy + ix]) * (*u1 + *u0);
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
  long i, j, k, x, y, offset, nblock;
  char xyz_path[FILENAME_MAX], attr_path[FILENAME_MAX];
  MPI_File mpi_file;
  float xyz[8 * _BS_ * _BS_];
  snprintf(xyz_path, sizeof xyz_path, "%s.xyz.raw", path);
  nblock = var.vel->infos.size();
  MPI_Exscan(&nblock, &offset, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  if (sim.rank == 0)
    offset = 0;
  if (sim.rank == sim.size - 1) {
    char *xyz_base, xdmf_path[FILENAME_MAX];
    long nblock_total = nblock + offset;
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
            time, _BS_ * _BS_ * nblock_total, 4 * _BS_ * _BS_ * nblock_total,
            xyz_base);
    for (size_t i = 0; i < sizeof var.F / sizeof *var.F; i++)
      if (var.F[i].prefix != NULL) {
        snprintf(attr_path, sizeof attr_path, "%s.%s.raw", path,
                 var.F[i].prefix);
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
                _BS_ * _BS_ * nblock_total, dim, sizeof(Real),
                attr_path + (xyz_path - xyz_base));
      }
    fprintf(xdmf, "    </Grid>\n"
                  "  </Domain>\n"
                  "</Xdmf>\n");
    fclose(xdmf);
  }
  MPI_File_open(MPI_COMM_WORLD, xyz_path, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &mpi_file);
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
    MPI_File_write_at(mpi_file, (offset + i) * sizeof xyz, xyz,
                      sizeof xyz / sizeof *xyz, MPI_FLOAT, MPI_STATUS_IGNORE);
  }
  MPI_File_close(&mpi_file);

  for (size_t i = 0; i < sizeof var.F / sizeof *var.F; i++)
    if (var.F[i].prefix != NULL) {
      Grid *g = *var.F[i].g;
      int dim = var.F[i].dim;
      snprintf(attr_path, sizeof attr_path, "%s.%s.raw", path, var.F[i].prefix);
      MPI_File_open(MPI_COMM_WORLD, attr_path,
                    MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL,
                    &mpi_file);
      for (j = 0; j < nblock; j++)
        MPI_File_write_at(
            mpi_file, (offset + j) * dim * _BS_ * _BS_ * sizeof(Real),
            g->infos[j]->block, dim * _BS_ * _BS_, MPI_Real, MPI_STATUS_IGNORE);
      MPI_File_close(&mpi_file);
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
    MPI_Allreduce(MPI_IN_PLACE, com, 3, MPI_Real, MPI_SUM, MPI_COMM_WORLD);
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
  bool movedBlocks = false;
  computeA(KernelVorticity(), var.vel, 2);
  computeA(GradChiOnTmp(), var.chi, 1);
  Stencil stencil{-1, -1, 2, 2, true};
  Synchronizer *Synch =
      sync1(stencil, var.tmp->synchronizers, &var.tmp->tree, &var.tmp->all,
            &var.tmp->infos, &var.tmp->timestamp, 1);
  bool CallValidStates = false;
  bool Reduction = false;
  MPI_Request Reduction_req;
  int tmp;
  std::vector<Info *> *halo = &Synch->buf->halo_blocks;
  std::vector<Info *> *infos[2] = {&Synch->buf->inner_blocks, halo};
  for (int iii = 0;; iii++) {
    std::vector<Info *> *I = infos[iii];
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
            CallValidStates = true;
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
    MPI_Waitall(Synch->buf->requests.size(), Synch->buf->requests.data(),
                MPI_STATUSES_IGNORE);
  }
  if (!Reduction) {
    tmp = CallValidStates ? 1 : 0;
    Reduction = true;
    MPI_Iallreduce(MPI_IN_PLACE, &tmp, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD,
                   &Reduction_req);
  }
  MPI_Wait(&Reduction_req, MPI_STATUS_IGNORE);
  var.tmp->boundary = *halo;
  if (tmp > 0) {
    int levelMin = 0;
    std::vector<Info *> &I = var.tmp->infos;
#pragma omp parallel for
    for (size_t j = 0; j < I.size(); j++) {
      Info *info = I[j];
      if ((info->state == Refine && info->level == sim.levelMax - 1) ||
          (info->state == Compress && info->level == levelMin)) {
        info->state = Leave;
        getf(&var.tmp->all, info->level, info->Z)->state = Leave;
      }
      if (info->state != Leave) {
        info->changed2 = true;
        (getf(&var.tmp->all, info->level, info->Z))->changed2 = info->changed2;
      }
    }
    bool clean_boundary = true;
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
      update_boundary(clean_boundary, &var.tmp->boundary, &var.tmp->all,
                      &var.tmp->tree);
      clean_boundary = false;
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
    bool boundary_needed = var.F[i].boundary_needed;
    int dim = var.F[i].dim;
    Synchronizer *Synch = nullptr;
    const Stencil stencil{-1, -1, 2, 2, true};
    if (basic == false) {
      Synch = sync1(stencil, g->synchronizers, &g->tree, &g->all, &g->infos,
                    &g->timestamp, dim);
      MPI_Waitall(Synch->buf->requests.size(), Synch->buf->requests.data(),
                  MPI_STATUSES_IGNORE);
      g->boundary = Synch->buf->halo_blocks;
      if (boundary_needed)
        update_boundary(false, &g->boundary, &g->all, &g->tree);
    }
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
    std::vector<long long> dealloc_IDs;
    BlockLab lab(dim);
    if (Synch != nullptr)
      lab.prepare(stencil);
    for (size_t i = 0; i < m_ref.size(); i++) {
      const int level = m_ref[i];
      const long long Z = n_ref[i];
      Info *parent = getf(&g->all, level, Z);
      parent->state = Leave;
      if (basic == false)
        lab.load(&g->tree, &g->all, Synch->buf, stencil, parent, true,
                 Synch->sLength);
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
    std::vector<std::vector<MPI_Block>> send_blocks(sim.size);
    std::vector<std::vector<MPI_Block>> recv_blocks(sim.size);
    for (auto &b : I) {
      const long long nBlock =
          forward(b->level, 2 * (b->index[0] / 2), 2 * (b->index[1] / 2));
      const Info *base = getf(&g->all, b->level, nBlock);
      if (!(Tree1(base, &g->tree) >= 0) || base->state != Compress)
        continue;
      const Info *bCopy = getf(&g->all, b->level, b->Z);
      const int baserank = treef(&g->tree, b->level, nBlock);
      const int brank = treef(&g->tree, b->level, b->Z);
      if (b->Z != nBlock) {
        if (baserank != sim.rank && brank == sim.rank) {
          MPI_Block x;
          x.level = bCopy->level;
          x.Z = bCopy->Z;
          std::memcpy(&x.data[0], bCopy->block,
                      _BS_ * _BS_ * dim * sizeof(Real));
          send_blocks[baserank].push_back(x);
          treef(&g->tree, b->level, b->Z) = baserank;
        }
      } else {
        for (int j = 0; j < 2; j++)
          for (int i = 0; i < 2; i++) {
            const long long n =
                forward(b->level, b->index[0] + i, b->index[1] + j);
            if (n == nBlock)
              continue;
            const int temprank = treef(&g->tree, b->level, n);
            if (temprank != sim.rank) {
              MPI_Block x;
              x.level = bCopy->level;
              x.Z = bCopy->Z;
              recv_blocks[temprank].push_back(x);
              treef(&g->tree, b->level, n) = baserank;
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
        dealloc(send_blocks[r][i].level, send_blocks[r][i].Z, &g->infos);
        treef(&g->tree, send_blocks[r][i].level, send_blocks[r][i].Z) = -2;
      }
    if (requests0.size() != 0) {
      movedBlocks = true;
      MPI_Waitall(requests0.size(), &requests0[0], MPI_STATUSES_IGNORE);
    }
    for (int r = 0; r < sim.size; r++)
      for (int i = 0; i < (int)recv_blocks[r].size(); i++) {
        const int level = (int)recv_blocks[r][i].level;
        const long long Z = recv_blocks[r][i].Z;
        Info *info = getf(&g->all, level, Z);
        info->block = (Real *)calloc(dim * _BS_ * _BS_, sizeof(Real));
#pragma omp critical
        { g->infos.push_back(info); }
        treef(&g->tree, level, Z) = sim.rank;
        std::memcpy(info->block, recv_blocks[r][i].data,
                    _BS_ * _BS_ * dim * sizeof(Real));
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
    MPI_Waitall(2, requests, MPI_STATUS_IGNORE);
    movedBlocks = false;
    long long max_b = block_distribution[0];
    long long min_b = block_distribution[0];
    for (auto &b : block_distribution) {
      max_b = std::max(max_b, b);
      min_b = std::min(min_b, b);
    }
    const double ratio = static_cast<double>(max_b) / min_b;
    if (ratio > 1.01 || min_b == 0) {
      std::sort(g->infos.begin(), g->infos.end(), info_cmp);
      long long total_load = 0;
      for (int r = 0; r < sim.size; r++)
        total_load += block_distribution[r];
      long long my_load = total_load / sim.size;
      if (sim.rank < (total_load % sim.size))
        my_load += 1;
      std::vector<long long> index_start(sim.size);
      index_start[0] = 0;
      for (int r = 1; r < sim.size; r++)
        index_start[r] = index_start[r - 1] + block_distribution[r - 1];
      long long ideal_index = (total_load / sim.size) * sim.rank;
      ideal_index +=
          sim.rank < (total_load % sim.size) ? sim.rank : total_load % sim.size;
      std::vector<std::vector<MPI_Block>> send_blocks(sim.size);
      std::vector<std::vector<MPI_Block>> recv_blocks(sim.size);
      for (int r = 0; r < sim.size; r++)
        if (sim.rank != r) {
          {
            long long a1 = ideal_index;
            long long a2 = ideal_index + my_load - 1;
            long long b1 = index_start[r];
            long long b2 = index_start[r] + block_distribution[r] - 1;
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
            long long b2 =
                index_start[sim.rank] + block_distribution[sim.rank] - 1;
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
            Info *info = g->infos[counter_S + i];
            MPI_Block *x = &send_blocks[r][i];
            x->level = info->level;
            x->Z = info->Z;
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
            Info *info = g->infos[g->infos.size() - 1 - (counter_E + i)];
            MPI_Block *x = &send_blocks[r][i];
            x->level = info->level;
            x->Z = info->Z;
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
              Info *info = g->infos[counter_S + i];
              deallocIDs.push_back(info->id2);
              treef(&g->tree, info->level, info->Z) = r;
            }
            counter_S += send_blocks[r].size();
          } else {
            for (size_t i = 0; i < send_blocks[r].size(); i++) {
              Info *info = g->infos[g->infos.size() - 1 - (counter_E + i)];
              deallocIDs.push_back(info->id2);
              treef(&g->tree, info->level, info->Z) = r;
            }
            counter_E += send_blocks[r].size();
          }
        }
      dealloc_many(deallocIDs, &g->infos);
      MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
#pragma omp parallel
      {
        for (int r = 0; r < sim.size; r++)
          if (recv_blocks[r].size() != 0) {
#pragma omp for
            for (size_t i = 0; i < recv_blocks[r].size(); i++)
              AddBlock(dim, g, recv_blocks[r][i].level, recv_blocks[r][i].Z,
                       recv_blocks[r][i].data);
          }
      }
      fill_pos(&g->infos, &g->all);
    } else {
      const int right =
          (sim.rank == sim.size - 1) ? MPI_PROC_NULL : sim.rank + 1;
      const int left = (sim.rank == 0) ? MPI_PROC_NULL : sim.rank - 1;
      const int my_blocks = g->infos.size();
      int right_blocks, left_blocks;
      MPI_Request reqs[4];
      MPI_Irecv(&left_blocks, 1, MPI_INT, left, 123, MPI_COMM_WORLD, &reqs[0]);
      MPI_Irecv(&right_blocks, 1, MPI_INT, right, 456, MPI_COMM_WORLD,
                &reqs[1]);
      MPI_Isend(&my_blocks, 1, MPI_INT, left, 456, MPI_COMM_WORLD, &reqs[2]);
      MPI_Isend(&my_blocks, 1, MPI_INT, right, 123, MPI_COMM_WORLD, &reqs[3]);
      MPI_Waitall(4, &reqs[0], MPI_STATUSES_IGNORE);
      const int nu = 4;
      const int flux_left =
          (sim.rank == 0) ? 0 : (my_blocks - left_blocks) / nu;
      const int flux_right =
          (sim.rank == sim.size - 1) ? 0 : (my_blocks - right_blocks) / nu;
      if (flux_right != 0 || flux_left != 0)
        std::sort(g->infos.begin(), g->infos.end(), info_cmp);
      std::vector<MPI_Block> send_left;
      std::vector<MPI_Block> recv_left;
      std::vector<MPI_Block> send_right;
      std::vector<MPI_Block> recv_right;
      std::vector<MPI_Request> request;
      if (flux_left > 0) {
        send_left.resize(flux_left);
#pragma omp parallel for schedule(runtime)
        for (int i = 0; i < flux_left; i++) {
          Info *info = g->infos[i];
          MPI_Block *x = &send_left[i];
          x->level = info->level;
          x->Z = info->Z;
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
          Info *info = g->infos[my_blocks - i - 1];
          MPI_Block *x = &send_right[i];
          x->level = info->level;
          x->Z = info->Z;
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
        Info *info = g->infos[my_blocks - i - 1];
        dealloc(info->level, info->Z, &g->infos);
        treef(&g->tree, info->level, info->Z) = right;
      }
      for (int i = 0; i < flux_left; i++) {
        Info *info = g->infos[i];
        dealloc(info->level, info->Z, &g->infos);
        treef(&g->tree, info->level, info->Z) = left;
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
        AddBlock(dim, g, recv_left[i].level, recv_left[i].Z, recv_left[i].data);
      for (int i = 0; i < -flux_right; i++)
        AddBlock(dim, g, recv_right[i].level, recv_right[i].Z,
                 recv_right[i].data);
      MPI_Wait(&request_reduction, MPI_STATUS_IGNORE);
      movedBlocks = (temp >= 1);
      fill_pos(&g->infos, &g->all);
    }
    if (result[0] > 0 || result[1] > 0 || movedBlocks) {
      g->UpdateFluxCorrection = true;
      update_blocks(false, &g->infos, &g->all, &g->tree);
      auto it = g->synchronizers->begin();
      while (it != g->synchronizers->end()) {
        Setup(dim, &g->tree, &g->all, &g->infos, it->second->buf,

              it->second->use_averages, it->second->AllStencils,
              it->second->Coarse_Range, stencil, it->second->sLength,
              it->second->ToBeAveragedDown, it->second->mapofHaloBlockGroups);
        it++;
      }
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
    BlockCase *tempCase = tmpVInfo[info->id]->auxiliary;
    Real *faceXm = nullptr;
    Real *faceXp = nullptr;
    Real *faceYm = nullptr;
    Real *faceYp = nullptr;
    if (tempCase != nullptr) {
      faceXm = tempCase->d[0];
      faceXp = tempCase->d[1];
      faceYm = tempCase->d[2];
      faceYp = tempCase->d[3];
    }
    if (faceXm != nullptr) {
      int ix = 0;
      for (int iy = 0; iy < _BS_; ++iy) {
        int ip0 = ix - stencil.sx;
        int jp0 = iy - stencil.sy;
        int im1 = ip0 - 1;
        Real *l0 = um + 2 * (nm * jp0 + ip0) + 0;
        Real *l1 = um + 2 * (nm * jp0 + im1) + 0;
        Real *l2 = um + 2 * (nm * jp0 + ip0) + 1;
        Real *l3 = um + 2 * (nm * jp0 + im1) + 1;
        faceXm[2 * iy] = dfac * (*l0 - *l1);
        faceXm[2 * iy + 1] = dfac * (*l2 - *l3);
      }
    }
    if (faceXp != nullptr) {
      int ix = _BS_ - 1;
      for (int iy = 0; iy < _BS_; ++iy) {
        int ip0 = ix - stencil.sx;
        int jp0 = iy - stencil.sy;
        int ip1 = ip0 + 1;
        Real *l0 = um + 2 * (nm * jp0 + ip0) + 0;
        Real *l1 = um + 2 * (nm * jp0 + ip1) + 0;
        Real *l2 = um + 2 * (nm * jp0 + ip0) + 1;
        Real *l3 = um + 2 * (nm * jp0 + ip1) + 1;
        faceXp[2 * iy] = dfac * (*l0 - *l1);
        faceXp[2 * iy + 1] = dfac * (*l2 - *l3);
      }
    }
    if (faceYm != nullptr) {
      int iy = 0;
      for (int ix = 0; ix < _BS_; ++ix) {
        int ip0 = ix - stencil.sx;
        int jp0 = iy - stencil.sy;
        int jm1 = jp0 - 1;
        Real *l0 = um + 2 * (nm * jp0 + ip0) + 0;
        Real *l1 = um + 2 * (nm * jm1 + ip0) + 0;
        Real *l2 = um + 2 * (nm * jp0 + ip0) + 1;
        Real *l3 = um + 2 * (nm * jm1 + ip0) + 1;
        faceYm[2 * ix] = dfac * (*l0 - *l1);
        faceYm[2 * ix + 1] = dfac * (*l2 - *l3);
      }
    }
    if (faceYp != nullptr) {
      int iy = _BS_ - 1;
      for (int ix = 0; ix < _BS_; ++ix) {
        int ip0 = ix - stencil.sx;
        int jp0 = iy - stencil.sy;
        int jp1 = jp0 + 1;
        Real *l0 = um + 2 * (nm * jp0 + ip0) + 0;
        Real *l1 = um + 2 * (nm * jp1 + ip0) + 0;
        Real *l2 = um + 2 * (nm * jp0 + ip0) + 1;
        Real *l3 = um + 2 * (nm * jp1 + ip0) + 1;
        faceYp[2 * ix] = dfac * (*l0 - *l1);
        faceYp[2 * ix + 1] = dfac * (*l2 - *l3);
      }
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
    BlockCase *tempCase = tmpVInfo[info->id]->auxiliary;
    Real *faceXm = nullptr;
    Real *faceXp = nullptr;
    Real *faceYm = nullptr;
    Real *faceYp = nullptr;
    if (tempCase != nullptr) {
      faceXm = tempCase->d[0];
      faceXp = tempCase->d[1];
      faceYm = tempCase->d[2];
      faceYp = tempCase->d[3];
    }
    if (faceXm != nullptr) {
      int ix = 0;
      for (int iy = 0; iy < _BS_; ++iy) {
        int ip0 = ix - stencil.sx;
        int jp0 = iy - stencil.sy;
        int im1 = ip0 - 1;
        Real *p0 = um + nm * jp0 + ip0;
        Real *p1 = um + nm * jp0 + im1;
        faceXm[2 * iy] = pFac * (*p1 + *p0);
        faceXm[2 * iy + 1] = 0;
      }
    }
    if (faceXp != nullptr) {
      int ix = _BS_ - 1;
      for (int iy = 0; iy < _BS_; ++iy) {
        int ip0 = ix - stencil.sx;
        int jp0 = iy - stencil.sy;
        int ip1 = ip0 + 1;
        Real *p0 = um + nm * jp0 + ip0;
        Real *p1 = um + nm * jp0 + ip1;
        faceXp[2 * iy] = -pFac * (*p1 + *p0);
        faceXp[2 * iy + 1] = 0;
      }
    }
    if (faceYm != nullptr) {
      int iy = 0;
      for (int ix = 0; ix < _BS_; ++ix) {
        int ip0 = ix - stencil.sx;
        int jp0 = iy - stencil.sy;
        int jm1 = jp0 - 1;
        Real *p0 = um + nm * jp0 + ip0;
        Real *p1 = um + nm * jm1 + ip0;
        faceYm[2 * ix] = 0;
        faceYm[2 * ix + 1] = pFac * (*p1 + *p0);
      }
    }
    if (faceYp != nullptr) {
      int iy = _BS_ - 1;
      for (int ix = 0; ix < _BS_; ++ix) {
        int ip0 = ix - stencil.sx;
        int jp0 = iy - stencil.sy;
        int jp1 = jp0 + 1;
        Real *p0 = um + nm * jp0 + ip0;
        Real *p1 = um + nm * jp1 + ip0;
        faceYp[2 * ix] = 0;
        faceYp[2 * ix + 1] = -pFac * (*p1 + *p0);
      }
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
    BlockCase *tempCase = (BlockCase *)(var.tmp->infos[info->id]->auxiliary);
    Real *faceXm = nullptr;
    Real *faceXp = nullptr;
    Real *faceYm = nullptr;
    Real *faceYp = nullptr;
    if (tempCase != nullptr) {
      faceXm = tempCase->d[0];
      faceXp = tempCase->d[1];
      faceYm = tempCase->d[2];
      faceYp = tempCase->d[3];
    }
    if (faceXm != nullptr) {
      int ix = 0;
      for (int iy = 0; iy < _BS_; ++iy) {
        int ip0 = ix - stencil.sx;
        int jp0 = iy - stencil.sy;
        int im1 = ip0 - 1;
        Real *l0 = um + nm * jp0 + ip0;
        Real *l1 = um + nm * jp0 + im1;
        faceXm[iy] = *l1 - *l0;
      }
    }
    if (faceXp != nullptr) {
      int ix = _BS_ - 1;
      for (int iy = 0; iy < _BS_; ++iy) {
        int ip0 = ix - stencil.sx;
        int jp0 = iy - stencil.sy;
        int ip1 = ip0 + 1;
        Real *l0 = um + nm * jp0 + ip0;
        Real *l1 = um + nm * jp0 + ip1;
        faceXp[iy] = *l1 - *l0;
      }
    }
    if (faceYm != nullptr) {
      int iy = 0;
      for (int ix = 0; ix < _BS_; ++ix) {
        int ip0 = ix - stencil.sx;
        int jp0 = iy - stencil.sy;
        int jm1 = jp0 - 1;
        Real *l0 = um + nm * jp0 + ip0;
        Real *l1 = um + nm * jm1 + ip0;
        faceYm[ix] = *l1 - *l0;
      }
    }
    if (faceYp != nullptr) {
      int iy = _BS_ - 1;
      for (int ix = 0; ix < _BS_; ++ix) {
        int ip0 = ix - stencil.sx;
        int jp0 = iy - stencil.sy;
        int jp1 = jp0 + 1;
        Real *l0 = um + nm * jp0 + ip0;
        Real *l1 = um + nm * jp1 + ip0;
        faceYp[ix] = *l1 - *l0;
      }
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

#include <csignal>
#include <execinfo.h>
#include <fenv.h>
#include <unistd.h>
static void handler(int) {
  void *array[10];
  size_t size, i;
  char **strings;
  size = backtrace(array, 10);
  fprintf(stderr, "%s:%d: error: floating point exception on rank %d\n",
          __FILE__, __LINE__, sim.rank);
  size = backtrace(array, 10);
  strings = backtrace_symbols(array, size);
  if (strings != NULL) {
    for (i = 0; i < size; i++)
      fprintf(stderr, "%s\n", strings[i]);
  }
  free(strings);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
  CommandlineParser parser(argc, argv);
  MPI_Comm_size(MPI_COMM_WORLD, &sim.size);
  MPI_Comm_rank(MPI_COMM_WORLD, &sim.rank);

  /* GNU extension */
  std::signal(SIGFPE, handler);
  feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);

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
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      char tag[3] = {0};
      float area, J, length, rmax;
      fread(tag, sizeof *tag, sizeof tag, file);
      if (tag[0] != 'S' || tag[1] != 'D' || tag[2] != 'F') {
        fprintf(stderr, "main.cpp: error: not and sdf file\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      if (fread(&length, sizeof(length), 1, file) != 1 ||
          fread(&area, sizeof(area), 1, file) != 1 ||
          fread(&J, sizeof(J), 1, file) != 1 ||
          fread(&rmax, sizeof(rmax), 1, file) != 1 ||
          fread(&shape->nr, sizeof(shape->nr), 1, file) != 1 ||
          fread(&shape->np, sizeof(shape->np), 1, file) != 1) {
        fprintf(stderr,
                "main.cpp: error: fail to read shape header from file.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      size_t ncount = shape->nr * shape->np;
      if ((shape->sdf = (float *)malloc(ncount * sizeof(float))) == NULL) {
        fprintf(stderr, "main.cpp: error: malloc() failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
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

  sim.nblocks.resize(sim.size + 1);
  sim.nrows.resize(sim.size + 1);
  sim.levels.resize(sim.levelMax);
  sim.levels[0] = 0;
  for (int m = 0; m < sim.levelMax - 1; m++)
    sim.levels[m + 1] = sim.levels[m] + (1 << (2 * m));
  long long total_blocks = 1LL << (2 * sim.levelStart);
  long long base = total_blocks / sim.size;
  long long rema = total_blocks % sim.size;
  long long my_blocks = base + (sim.rank < rema ? 1 : 0);
  long long n_start = sim.rank * base + (sim.rank < rema ? sim.rank : rema);
  var.buf1 = new Buffers;
  var.buf2 = new Buffers;

  for (size_t i = 0; i < sizeof var.F / sizeof *var.F; i++) {
    int dim = var.F[i].dim;
    Grid *g = *var.F[i].g = new Grid;
    g->synchronizers = new std::map<Stencil, Synchronizer *>;
    for (size_t i = 0; i < my_blocks; i++) {
      long long Z = n_start + i;
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
    g->timestamp = 0;
    g->UpdateFluxCorrection = true;
    update_blocks(false, &g->infos, &g->all, &g->tree);
    MPI_Barrier(MPI_COMM_WORLD);
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
  sim.mat = new LocalSpMatDnVec(MPI_COMM_WORLD, _BS_ * _BS_, 0, P_inv);
  sim.solver = new Solver;
  while (1) {
    if (sim.rank == 0 && sim.step % 5 == 0)
      fprintf(stderr, "main.cpp: %08d\n", sim.step);
    Real CFL = sim.CFL;
    Real h = std::numeric_limits<Real>::infinity();
    for (size_t i = 0; i < var.vel->infos.size(); i++)
      h = std::min(var.vel->infos[i]->h, h);
    MPI_Allreduce(MPI_IN_PLACE, &h, 1, MPI_Real, MPI_MIN, MPI_COMM_WORLD);
    Real umax = 0;
#pragma omp parallel for schedule(static) reduction(max : umax)
    for (size_t i = 0; i < velInfo.size(); i++) {
      Real *vel = velInfo[i]->block;
      for (int j = 0; j < 2 * _BS_ * _BS_; j++)
        umax = std::max(umax, std::fabs(vel[j]));
    }
    MPI_Allreduce(MPI_IN_PLACE, &umax, 1, MPI_Real, MPI_MAX, MPI_COMM_WORLD);
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
      prepare0(var.buf2, &var.tmpV->infos, &var.tmpV->all, &var.tmpV->tree, 2);
      var.tmpV->UpdateFluxCorrection = false;
    }
    computeA(KernelAdvectDiffuse(), var.vel, 2);
    fillcases(var.buf2, &var.tmpV->tree, 2);
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
      prepare0(var.buf2, &var.tmpV->infos, &var.tmpV->all, &var.tmpV->tree, 2);
      var.tmpV->UpdateFluxCorrection = false;
    }
    computeA(KernelAdvectDiffuse(), var.vel, 2);
    fillcases(var.buf2, &var.tmpV->tree, 2);
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
      MPI_Allreduce(MPI_IN_PLACE, quantities,
                    sizeof quantities / sizeof *quantities, MPI_Real, MPI_SUM,
                    MPI_COMM_WORLD);
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
    MPI_Allreduce(MPI_IN_PLACE, buffer.data(), buffer.size(), MPI_Real, MPI_SUM,
                  MPI_COMM_WORLD);
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
      prepare0(var.buf1, &var.tmp->infos, &var.tmp->all, &var.tmp->tree, 1);
      var.tmp->UpdateFluxCorrection = false;
    }
    Stencil stencil{-1, -1, 2, 2, false};
    Synchronizer *Synch =
        sync1(stencil, var.vel->synchronizers, &var.vel->tree, &var.vel->all,
              &var.vel->infos, &var.vel->timestamp, 2);
    Synchronizer *Synch2 =
        sync1(stencil, var.tmpV->synchronizers, &var.tmpV->tree, &var.tmpV->all,
              &var.tmpV->infos, &var.tmpV->timestamp, 2);
    std::vector<Info *> &blk = var.vel->infos;
    std::vector<bool> ready(blk.size(), false);
    std::vector<Info *> &avail0 = Synch->buf->inner_blocks;
    std::vector<Info *> &avail02 = Synch2->buf->inner_blocks;
    const int Ninner = avail0.size();
    std::vector<Info *> avail1;
    std::vector<Info *> avail12;
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
        lab.load(&var.vel->tree, &var.vel->all, Synch->buf, stencil, I, true,
                 Synch->sLength);
        lab2.load(&var.tmpV->tree, &var.tmpV->all, Synch2->buf, stencil, I2,
                  true, Synch2->sLength);
        pressure_rhs_fun(lab, lab2, I, I2);
        ready[I->id] = true;
      }
#pragma omp master
      {
        MPI_Waitall(Synch->buf->requests.size(), Synch->buf->requests.data(),
                    MPI_STATUSES_IGNORE);
        avail1 = Synch->buf->halo_blocks;

        MPI_Waitall(Synch2->buf->requests.size(), Synch2->buf->requests.data(),
                    MPI_STATUSES_IGNORE);
        avail12 = Synch2->buf->halo_blocks;
      }
#pragma omp barrier
      const int Nhalo = avail1.size();
#pragma omp for
      for (int i = 0; i < Nhalo; i++) {
        Info *I = avail1[i];
        Info *I2 = avail12[i];
        lab.load(&var.vel->tree, &var.vel->all, Synch->buf, stencil, I, true,
                 Synch->sLength);
        lab2.load(&var.tmpV->tree, &var.tmpV->all, Synch2->buf, stencil, I2,
                  true, Synch->sLength);
        pressure_rhs_fun(lab, lab2, I, I2);
      }
    }
    fillcases(var.buf1, &var.tmp->tree, 1);
    std::vector<Info *> &presInfo = var.pres->infos;
    std::vector<Info *> &poldInfo = var.pold->infos;
#pragma omp parallel for
    for (size_t i = 0; i < velInfo.size(); i++) {
      memcpy(poldInfo[i]->block, presInfo[i]->block,
             _BS_ * _BS_ * sizeof(Real));
      memset(presInfo[i]->block, 0, _BS_ * _BS_ * sizeof(Real));
    }
    if (var.tmp->UpdateFluxCorrection) {
      prepare0(var.buf1, &var.tmp->infos, &var.tmp->all, &var.tmp->tree, 1);
      var.tmp->UpdateFluxCorrection = false;
    }
    computeA(pressure_rhs1(), var.pold, 1);
    fillcases(var.buf1, &var.tmp->tree, 1);
    const double max_error = sim.step < 10 ? 0.0 : sim.PoissonTol;
    const double max_rel_error = sim.step < 10 ? 0.0 : sim.PoissonTolRel;
    const int max_restarts = sim.step < 10 ? 100 : sim.maxPoissonRestarts;
    if (var.pres->UpdateFluxCorrection) {
      var.pres->UpdateFluxCorrection = false;

      update_blocks(true, &var.tmp->infos, &var.tmp->all, &var.tmp->tree);
      std::vector<Info *> &RhsInfo = var.tmp->infos;
      const int Nblocks = RhsInfo.size();
      const int N = _BS_ * _BS_ * Nblocks;
      sim.mat->reserve(N);
      const long long Nblocks_long = Nblocks;
      MPI_Allgather(&Nblocks_long, 1, MPI_LONG_LONG, sim.nblocks.data(), 1,
                    MPI_LONG_LONG, MPI_COMM_WORLD);
      for (int i(sim.nblocks.size() - 1); i > 0; i--) {
        sim.nblocks[i] = sim.nblocks[i - 1];
      }
      sim.nblocks[0] = 0;
      sim.nrows[0] = 0;
      for (size_t i = 1; i < sim.nblocks.size(); i++) {
        sim.nblocks[i] += sim.nblocks[i - 1];
        sim.nrows[i] = (_BS_ * _BS_) * sim.nblocks[i];
      }
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
    Real avg, avg1, quantities[2];
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
    quantities[0] = avg;
    quantities[1] = avg1;
    MPI_Allreduce(MPI_IN_PLACE, &quantities, 2, MPI_Real, MPI_SUM,
                  MPI_COMM_WORLD);
    avg = quantities[0];
    avg1 = quantities[1];
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
    quantities[0] = avg;
    quantities[1] = avg1;
    MPI_Allreduce(MPI_IN_PLACE, &quantities, 2, MPI_Real, MPI_SUM,
                  MPI_COMM_WORLD);
    avg = quantities[0];
    avg1 = quantities[1];
    avg = avg / avg1;
#pragma omp parallel for
    for (size_t i = 0; i < velInfo.size(); i++) {
      Real *pres = presInfo[i]->block;
      Real *pold = poldInfo[i]->block;
      for (int j = 0; j < _BS_ * _BS_; j++)
        pres[j] += pold[j] - avg;
    }
    if (var.tmp->UpdateFluxCorrection) {
      prepare0(var.buf1, &var.tmp->infos, &var.tmp->all, &var.tmp->tree, 1);
      var.tmp->UpdateFluxCorrection = false;
    }
    computeA(pressureCorrectionKernel(), var.pres, 1);
    fillcases(var.buf1, &var.tmp->tree, 1);
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
  for (int i = 0; i < var.buf1->Cases.size(); i++) {
    for (int j = 0; j < 4; j++)
      free(var.buf1->Cases[i]->d[j]);
    free(var.buf1->Cases[i]);
  }
  delete var.buf1;
  delete var.buf2;
  MPI_Finalize();
}
