#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "cuda.h"

typedef double Real;
enum { BS = 8 };
enum {
  off_vel = 0,
  off_pres = 2,
  off_chi = 3,
  off_vold = 4,
  off_tmp = 6,
  off_pold = 7,
  off_tmpV = 8,
  off_n = 10,
};

static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
enum State : signed char { Leave = 0, Refine = 1, Compress = -1 };
enum TreeState : signed char {
  Active = 0,
  ChildrenAreActive = -1,
  ParentIsActive = -2,
};
struct Shape;
struct Info;
static struct Sim {
  int AdaptSteps;
  int levelMax;
  int levelStart;
  int maxPoissonRestarts;
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
  long long *levels;
  struct Shape **shapes;
  struct LocalSpMatDnVec *mat;
  long long n;
  int nshape;
  std::unordered_map<long long, TreeState> tree;
  std::unordered_map<long long, long long> map;
  Info **infos;
} sim;
#include "utils.h"
struct Info {
  double h, origin[2];
  int level;
  long long id, Z;
  Real *block = NULL;
};
struct Collision {
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
static TreeState Tree1(Info *info) {
  return sim.tree.at(sim.levels[info->level] + info->Z);
}
static void fill(Info *b, int level, long long Z) {
  int n = 1 << level;
  int ix, iy;
  sfc_inverse(Z, level, &ix, &iy);
  b->level = level;
  b->Z = Z;
  b->h = 1.0 / BS / n;
  b->origin[0] = (Real)ix / n;
  b->origin[1] = (Real)iy / n;
  int index[2] = {ix, iy};
  b->id = sfc_encode(level, index);
}
static int exist(int level, long long Z) {
  long long aux = sim.levels[level] + Z;
  return sim.map.find(aux) != sim.map.end();
}
static Info *getf0(int level, long long Z) {
  return sim.infos[sim.map.at(sim.levels[level] + Z)];
}
static Info getf1(int level, long long Z) {
  Info dummy;
  fill(&dummy, level, Z);
  auto r = sim.map.find(sim.levels[level] + Z);
  return (r == sim.map.end()) ? dummy : *sim.infos[r->second];
}
struct {
  int offset;
  int dim;
  const char *prefix;
} vars[] = {{off_vel, 2, "vel"}, {off_pres, 1, "pres"}, {off_chi, 1, "chi"},
            {off_vold, 2, NULL}, {off_tmp, 1, "tmp"},   {off_pold, 1, NULL},
            {off_tmpV, 2, NULL}};

static inline bool skin_skip(int c, int coord, int n) {
  bool skin = coord == 0 || coord == n - 1;
  int skip = coord == 0 ? -1 : 1;
  return c == skip && skin;
}
static void get_states(Info *info, TreeState nei[3][3]) {
  int xi, yi;
  int n = 1 << info->level;
  sfc_inverse(info->Z, info->level, &xi, &yi);
  for (int icode = 0; icode < 9; icode++) {
    int cx = icode % 3 - 1;
    int cy = icode / 3 - 1;
    if (cx == 0 && cy == 0)
      continue;
    if (skin_skip(cx, xi, n) || skin_skip(cy, yi, n))
      continue;
    long long Z = sfc_forward(info->level, (xi + cx + n) % n, (yi + cy + n) % n);
    long long id = sim.levels[info->level] + Z;
    assert(sim.tree.find(id) != sim.tree.end());
    nei[1 + cx][1 + cy] = sim.tree.at(id);
  }
}

enum OpType : int8_t {
  OP_COPY,
  OP_AVG,
  OP_INTERP9,
  OP_INTERP3,
  OP_LELI,
  OP_BC_SCALAR,
  OP_BC_VECTOR,
};
struct Op {
  OpType type;
  int8_t blk_idx;
  int8_t dst_idx;
  int8_t flags;
  int32_t src_off, dst_off, p1, p2;
};
struct BlkSrc {
  int8_t level_delta;
  int8_t xi_mul, yi_mul;
  int8_t xi_add, yi_add;
  int8_t xi_shift, yi_shift;
  bool is_self;
  int8_t self_idx;
};
enum { MAX_PRE = 32, MAX_POST = 48, MAX_OPS = MAX_PRE + MAX_POST };
struct TabEntry {
  int8_t n_blk;
  BlkSrc blk_src[2];
  int8_t _pad;
  int32_t n_pre;
  int32_t n_post;
  Op ops[MAX_OPS];
};
static void exec_program(Real *const blk[], Real *const dst[],
                         const Op *ops, int n, int dim, int nm, int nc) {
  Real *m = dst[0], *c = dst[1];
  for (int i = 0; i < n; i++) {
    const Op &o = ops[i];
    switch (o.type) {
    case OP_COPY:
      memcpy(dst[o.dst_idx] + o.dst_off, blk[o.blk_idx] + o.src_off,
             o.p1 * dim * sizeof(Real));
      break;
    case OP_AVG: {
      Real *src = blk[o.blk_idx] + o.src_off;
      Real *d = dst[o.dst_idx] + o.dst_off;
      Real *q1 = src + o.p2 * dim;
      for (int k = 0; k < o.p1; k++)
        for (int dd = 0; dd < dim; dd++)
          d[k * dim + dd] =
              (src[2 * k * dim + dd] + src[(2 * k + 1) * dim + dd] +
               q1[2 * k * dim + dd] + q1[(2 * k + 1) * dim + dd]) /
              4;
      break;
    }
    case OP_INTERP9: {
      static const int8_t W[4][9] = {
          {1, 10, -1, 10, 56, -6, -1, -6, 1},   // x=0,y=0
          {-1, 10, 1, -6, 56, 10, 1, -6, -1},    // x=1,y=0
          {-1, -6, 1, 10, 56, -6, 1, 10, -1},    // x=0,y=1
          {1, -6, -1, -6, 56, 10, -1, 10, 1},    // x=1,y=1
      };
      const int8_t *w = W[o.flags & 3];
      for (int d = 0; d < dim; d++) {
        Real sum = 0;
        for (int jj = 0; jj < 3; jj++)
          for (int ii = 0; ii < 3; ii++)
            sum += w[3 * jj + ii] * c[o.src_off + d + dim * ((ii - 1) + nc * (jj - 1))];
        m[o.dst_off + d] = sum / 64.0;
      }
      break;
    }
    case OP_INTERP3: {
      for (int d = 0; d < dim; d++)
        m[o.dst_off + d] =
            (o.blk_idx * c[o.src_off + d] + o.dst_idx * c[o.p1 + d] +
             o.flags * c[o.p2 + d]) /
            32.0;
      break;
    }
    case OP_LELI: {
      static const int8_t W[2][3] = {
          {8, 10, -3},  // LI: (8a + 10b - 3c) / 15
          {24, -15, 6}, // LE: (24a - 15b + 6c) / 15
      };
      const int8_t *w = W[o.flags & 1];
      for (int d = 0; d < dim; d++) {
        Real a = m[o.src_off + d], b = m[o.dst_off + d], cv = m[o.p1 + d];
        m[o.src_off + d] = (w[0] * a + w[1] * b + w[2] * cv) / 15.0;
      }
      break;
    }
    case OP_BC_SCALAR: {
      Real *buf = dst[o.dst_idx];
      for (int d = 0; d < dim; d++)
        buf[o.dst_off + d] = buf[o.src_off + d];
      break;
    }
    case OP_BC_VECTOR: {
      Real *buf = dst[o.dst_idx];
      int dir = o.flags & 1;
      buf[o.dst_off + dir] = -buf[o.src_off + dir];
      buf[o.dst_off + 1 - dir] = buf[o.src_off + 1 - dir];
      break;
    }
    }
  }
}

static const TabEntry (*load_cfg_tab(int ss, int dim))[3][2][2][6] {
  char fname[64];
  snprintf(fname, sizeof fname, "tab_ss%d_dim%d.bin", ss, dim);
  FILE *fp = fopen(fname, "rb");
  if (!fp) {
    fprintf(stderr, "main.cpp: cannot open %s\n", fname);
    exit(1);
  }
  size_t sz = 3 * 3 * 2 * 2 * 6 * sizeof(TabEntry);
  TabEntry *tab = (TabEntry *)malloc(sz);
  if (fread(tab, 1, sz, fp) != sz) {
    fprintf(stderr, "main.cpp: short read from %s\n", fname);
    exit(1);
  }
  fclose(fp);
  return (const TabEntry (*)[3][2][2][6])tab;
}

static const TabEntry (*g_tab[5][3])[3][2][2][6];
static void tab_load_all() {
  int configs[][2] = {{1,1}, {1,2}, {3,2}, {4,1}};
  for (auto &c : configs)
    g_tab[c[0]][c[1]] = load_cfg_tab(c[0], c[1]);
}
static Real *lab_alloc(int dim, int ss) {
  int nm = 2 * ss + BS;
  int nc = BS / 2 + ss + 3;
  return (Real *)malloc((nm * nm + nc * nc) * dim * sizeof(Real));
}
static void lab_load1(Real *m, int dim, int blk_offset, const TreeState *nei,
                      int ss, Info *info) {
  int nm = 2 * ss + BS;
  int nc = BS / 2 + ss + 3;
  int n = 1 << info->level;
  int level = info->level;
  int xi, yi;
  const TabEntry (*cfg_tab)[3][2][2][6] = g_tab[ss][dim];
  sfc_inverse(info->Z, level, &xi, &yi);

  Real *p0 = info->block + BS * BS * blk_offset;
  for (int i = 0; i < BS; i++)
    memcpy(m + dim * ((i + ss) * nm + ss), p0 + dim * BS * i,
           BS * dim * sizeof(Real));

  Real *c = m + nm * nm * dim;
  Real *dst[2] = {m, c};

  struct {
    const TabEntry *e;
    Real *blk[2];
  } dirs[8];
  int nd = 0;
  for (int icode = 0; icode < 9; icode++) {
    int cx = icode % 3 - 1, cy = icode / 3 - 1;
    if (!cx && !cy)
      continue;
    int s;
    bool xskin = skin_skip(cx, xi, n);
    bool yskin = skin_skip(cy, yi, n);
    if (xskin && yskin)
      s = 5;
    else if (xskin)
      s = 3;
    else if (yskin)
      s = 4;
    else
      s = -nei[3 * (1 + cx) + (1 + cy)];
    const TabEntry *te =
        &cfg_tab[cx + 1][cy + 1][xi % 2][yi % 2][s];
    Real *blk[2] = {nullptr, nullptr};
    for (int b = 0; b < te->n_blk; b++) {
      const BlkSrc &bs = te->blk_src[b];
      if (bs.is_self) {
        blk[b] = dst[bs.self_idx];
      } else {
        int L = level + bs.level_delta;
        int fx = (xi * bs.xi_mul + bs.xi_add) >> bs.xi_shift;
        int fy = (yi * bs.yi_mul + bs.yi_add) >> bs.yi_shift;
        blk[b] = getf0(L, forward(L, fx, fy))->block +
                 BS * BS * blk_offset;
      }
    }
    dirs[nd] = {te, {blk[0], blk[1]}};
    nd++;
  }

  for (int i = 0; i < nd; i++)
    exec_program(dirs[i].blk, dst, dirs[i].e->ops, dirs[i].e->n_pre,
                 dim, nm, nc);
  for (int i = 0; i < nd; i++)
    exec_program(dirs[i].blk, dst, dirs[i].e->ops + MAX_PRE,
                 dirs[i].e->n_post, dim, nm, nc);
}
static void lab_load(Real *m, int dim, int blk_offset, int ss, Info *info) {
  TreeState nei[3][3];
  get_states(info, nei);
  lab_load1(m, dim, blk_offset, &nei[0][0], ss, info);
}

typedef Real ScalarBlock[BS][BS];
static void pressure_rhs_fun(Real *vm, Real *um, size_t i) {
  int ss = 1, nm = 2 * ss + BS;
  Real h = sim.infos[i]->h;
  Real facDiv = 0.5 * h / sim.dt;
  Real *TMP = sim.infos[i]->block + BS * BS * off_tmp;
  Real *CHI = sim.infos[i]->block + BS * BS * off_chi;
  for (int iy = 0; iy < BS; ++iy)
    for (int ix = 0; ix < BS; ++ix) {
#define V(dx, dy, c) vm[2 * (nm * (iy + ss + (dy)) + ix + ss + (dx)) + (c)]
#define U(dx, dy, c) um[2 * (nm * (iy + ss + (dy)) + ix + ss + (dx)) + (c)]
      Real divV = V(1,0,0) - V(-1,0,0) + V(0,1,1) - V(0,-1,1);
      Real divU = U(1,0,0) - U(-1,0,0) + U(0,1,1) - U(0,-1,1);
      TMP[BS * iy + ix] = facDiv * divV - facDiv * CHI[BS * iy + ix] * divU;
#undef U
#undef V
    }
}
struct Obstacle {
  Real chi[BS][BS];
  Real dist[BS][BS];
  Real udef[BS][BS][2];
  Real COM_x = 0;
  Real COM_y = 0;
  Real Mass = 0;
};
static void compute_vorticity() {
#pragma omp parallel
  {
    Real *um = lab_alloc(2, 1);
#pragma omp for nowait
    for (long long id = 0; id < sim.n; ++id) {
      lab_load(um, 2, off_vel, 1, sim.infos[id]);
      Info *info = sim.infos[id];
      Real i2h = 0.5 * (1 << info->level) * BS;
      Real *TMP = info->block + BS * BS * off_tmp;
      int ss = 1, nm = 2 * ss + BS;
      for (int j = 0; j < BS; ++j)
        for (int i = 0; i < BS; ++i) {
#define V(dx, dy, c) um[2 * (nm * (j + ss + (dy)) + i + ss + (dx)) + (c)]
          TMP[j * BS + i] = i2h * (V(0,-1,0) - V(0,1,0) + V(1,0,1) - V(-1,0,1));
#undef V
        }
    }
    free(um);
  }
}
static void dump(Real time, Info **infos, char *path) {
  long i, j, k, x, y;
  char xyz_path[FILENAME_MAX], attr_path[FILENAME_MAX];
  FILE *file;
  float xyz[8 * BS * BS];
  char *xyz_base, xdmf_path[FILENAME_MAX];
  FILE *xdmf;
  if (snprintf(xyz_path, sizeof xyz_path, "%s.xyz.raw", path) >=
          (long)sizeof xyz_path ||
      snprintf(xdmf_path, sizeof xdmf_path, "%s.xdmf2", path) >=
          (long)sizeof xdmf_path) {
    fprintf(stderr, "main.cpp: output path '%s' is too long\n", path);
    exit(1);
  }
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
          "          Dimensions=\"%lld\"\n"
          "          TopologyType=\"Quadrilateral\"/>\n"
          "     <Geometry\n"
          "         GeometryType=\"XY\">\n"
          "       <DataItem\n"
          "           Dimensions=\"%lld 2\"\n"
          "           Format=\"Binary\">\n"
          "         %s\n"
          "       </DataItem>\n"
          "     </Geometry>\n",
          time, BS * BS * sim.n, 4 * BS * BS * sim.n, xyz_base);
  for (size_t i = 0; i < sizeof vars / sizeof *vars; i++)
    if (vars[i].prefix != NULL) {
      if (snprintf(attr_path, sizeof attr_path, "%s.%s.raw", path,
                   vars[i].prefix) > (long)sizeof attr_path) {
        fprintf(stderr, "main.cpp: output path '%s' is too long\n", path);
        exit(1);
      }
      int dim = vars[i].dim;
      fprintf(xdmf,
              "       <Attribute\n"
              "           AttributeType=\"%s\"\n"
              "           Name=\"%s\"\n"
              "           Center=\"Cell\">\n"
              "         <DataItem\n"
              "             Dimensions=\"%lld %d\"\n"
              "             Precision=\"%ld\"\n"
              "             Format=\"Binary\">\n"
              "           %s\n"
              "         </DataItem>\n"
              "       </Attribute>\n",
              dim == 2 ? "Vector" : "Scalar", vars[i].prefix, BS * BS * sim.n,
              dim, sizeof(Real), attr_path + (xyz_path - xyz_base));
    }
  fprintf(xdmf, "    </Grid>\n"
                "  </Domain>\n"
                "</Xdmf>\n");
  fclose(xdmf);
  file = fopen(xyz_path, "wb");
  for (i = 0; i < sim.n; i++) {
    Info *info = infos[i];
    k = 0;
    for (y = 0; y < BS; y++)
      for (x = 0; x < BS; x++) {
        double u0, v0, u1, v1, h;
        h = 1.0 / BS / (1 << info->level);
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

  for (size_t i = 0; i < sizeof vars / sizeof *vars; i++)
    if (vars[i].prefix != NULL) {
      int dim = vars[i].dim;
      int offset = vars[i].offset;
      if (snprintf(attr_path, sizeof attr_path, "%s.%s.raw", path,
                   vars[i].prefix) >= (long)sizeof attr_path) {
        fprintf(stderr, "main.cpp: output path '%s' is too long\n", path);
        exit(1);
      }
      file = fopen(attr_path, "wb");
      for (j = 0; j < sim.n; j++)
        fwrite(sim.infos[j]->block + offset * BS * BS, sizeof(Real),
               dim * BS * BS, file);
      fclose(file);
    }
}
struct Shape {
  float rmax;
  float *sdf;
  int nr;
  int np;
  Real x;
  Real y;
  Real length;
  Real mass;
  Real omega;
  Real orientation;
  Real u;
  Real v;
  std::vector<Obstacle *> blocks;
};
static void compute_chi_on_grid() {
#pragma omp parallel
  {
    Real *um = lab_alloc(1, 1);
#pragma omp for nowait
    for (long long id = 0; id < sim.n; ++id) {
      lab_load(um, 1, off_tmp, 1, sim.infos[id]);
      Info *info = sim.infos[id];
      int ss = 1, nm = 2 * ss + BS;
      for (int ishape = 0; ishape < sim.nshape; ishape++) {
        Shape *shape = sim.shapes[ishape];
        if (shape->blocks[id] == nullptr)
          continue;
        Real h = 1.0 / BS / (1 << info->level);
        Real h2 = h * h;
        Obstacle *o = shape->blocks[id];
        o->COM_x = 0;
        o->COM_y = 0;
        o->Mass = 0;
        Real *CHI = info->block + BS * BS * off_chi;
        Real *chi = (Real *)o->chi;
        Real *dist = (Real *)o->dist;
        for (int iy = 0; iy < BS; iy++)
          for (int ix = 0; ix < BS; ix++) {
#define D(dx, dy) um[nm * (iy + ss + (dy)) + ix + ss + (dx)]
            int j = BS * iy + ix;
            if (dist[j] > +h || dist[j] < -h) {
              chi[j] = dist[j] > 0 ? 1 : 0;
            } else {
              Real dpx = D(1,0), dmx = D(-1,0), dpy = D(0,1), dmy = D(0,-1);
              Real gradIX = std::max(0.0, dpx) - std::max(0.0, dmx);
              Real gradIY = std::max(0.0, dpy) - std::max(0.0, dmy);
              Real gradUX = dpx - dmx, gradUY = dpy - dmy;
              chi[j] = (gradIX * gradUX + gradIY * gradUY) /
                       (gradUX * gradUX + gradUY * gradUY + EPS);
            }
#undef D
            CHI[j] = std::max(CHI[j], chi[j]);
            if (chi[j] > 0) {
              Real px = info->origin[0] + info->h * (ix + 0.5);
              Real py = info->origin[1] + info->h * (iy + 0.5);
              o->COM_x += chi[j] * h2 * (px - shape->x);
              o->COM_y += chi[j] * h2 * (py - shape->y);
              o->Mass += chi[j] * h2;
            }
          }
      }
    }
    free(um);
  }
}
static void ongrid() {
#pragma omp parallel for
  for (long long i = 0; i < sim.n; i++) {
    memset(sim.infos[i]->block + BS * BS * off_chi, 0, BS * BS * sizeof(Real));
    std::fill(sim.infos[i]->block + BS * BS * off_tmp,
              sim.infos[i]->block + BS * BS * (off_tmp + 1), -1.0);
  }
  for (int ishape = 0; ishape < sim.nshape; ishape++) {
    Shape *shape = sim.shapes[ishape];
    for (auto &entry : shape->blocks)
      delete entry;
    shape->blocks.clear();
    shape->blocks = std::vector<Obstacle *>(sim.n, nullptr);
#pragma omp parallel for
    for (long long i = 0; i < sim.n; ++i) {
      shape->blocks[i] = new Obstacle();
    }
#pragma omp parallel for
    for (long long i = 0; i < sim.n; i++) {
      Obstacle *o = shape->blocks[sim.infos[i]->id];
      Info *info = sim.infos[i];
      Real *b = sim.infos[i]->block + BS * BS * off_tmp;
      Real h = info->h;
      Real co = std::cos(shape->orientation);
      Real si = std::sin(shape->orientation);
      memset(&o->chi[0][0], 0, sizeof(Real) * BS * BS);
      memset(&o->udef[0][0][0], 0, sizeof(Real) * BS * BS * 2);
      for (int iy = 0; iy < BS; ++iy)
        for (int ix = 0; ix < BS; ++ix) {
          Real x = info->origin[0] + h * (ix + 0.5) - shape->x;
          Real y = info->origin[1] + h * (iy + 0.5) - shape->y;
          Real x0 = co * x + si * y;
          Real y0 = -si * x + co * y;
          Real r = sqrt(x0 * x0 + y0 * y0);
          Real p = atan2(y0, x0);
          if (p < 0)
            p += 2 * M_PI;
          int ri = r * shape->nr / shape->rmax;
          if (ri >= shape->nr)
            ri = shape->nr - 1;
          int pi = p * (shape->np - 2) / (2 * M_PI);
          if (pi >= shape->np)
            pi = shape->np - 1;
          Real dist = shape->sdf[ri * shape->np + pi];
          o->dist[iy][ix] = dist;
          b[iy * BS + ix] = std::max(b[iy * BS + ix], dist);
        }
    }
  }

  compute_chi_on_grid();
  for (int ishape = 0; ishape < sim.nshape; ishape++) {
    Shape *shape = sim.shapes[ishape];
    Real com[3] = {0.0, 0.0, 0.0};
    std::vector<Obstacle *> &oblock = shape->blocks;
#pragma omp parallel for reduction(+ : com[ : 3])
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
  for (int ishape = 0; ishape < sim.nshape; ishape++) {
    Shape *shape = sim.shapes[ishape];
    Real x = 0, y = 0, m = 0, J = 0, u = 0, v = 0, a = 0;
#pragma omp parallel for reduction(+ : x, y, m, J, u, v, a)
    for (long long i = 0; i < sim.n; i++) {
      Real hsq = sim.infos[i]->h * sim.infos[i]->h;
      Obstacle *o = shape->blocks[i];
      Real *CHI = (Real *)o->chi;
      Real *UDEF = (Real *)o->udef;
      for (int iy = 0; iy < BS; ++iy)
        for (int ix = 0; ix < BS; ++ix) {
          int j = BS * iy + ix;
          if (CHI[j] <= 0)
            continue;
          Real p[2];
          p[0] = sim.infos[i]->origin[0] + sim.infos[i]->h * (ix + 0.5);
          p[1] = sim.infos[i]->origin[1] + sim.infos[i]->h * (iy + 0.5);
          Real chi = CHI[j] * hsq;
          p[0] -= shape->x;
          p[1] -= shape->y;
          x += chi * p[0];
          y += chi * p[1];
          m += chi;
          J += chi * (p[0] * p[0] + p[1] * p[1]);
          u += chi * UDEF[2 * j + 0];
          v += chi * UDEF[2 * j + 1];
          a += chi * (p[0] * UDEF[2 * j + 1] - p[1] * UDEF[2 * j + 0]);
        }
    }
    u /= m;
    v /= m;
    a /= J;
#pragma omp parallel for
    for (long long i = 0; i < sim.n; i++) {
      Obstacle *o = shape->blocks[i];
      for (int iy = 0; iy < BS; ++iy)
        for (int ix = 0; ix < BS; ++ix) {
          Real p[2];
          p[0] = sim.infos[i]->origin[0] + sim.infos[i]->h * (ix + 0.5);
          p[1] = sim.infos[i]->origin[1] + sim.infos[i]->h * (iy + 0.5);
          p[0] -= shape->x;
          p[1] -= shape->y;
          o->udef[iy][ix][0] -= u - a * p[1];
          o->udef[iy][ix][1] -= v + a * p[0];
        }
    }
  }
}
static void compute_grad_chi() {
#pragma omp parallel
  {
    Real *um = lab_alloc(1, 4);
#pragma omp for nowait
    for (long long id = 0; id < sim.n; ++id) {
      lab_load(um, 1, off_chi, 4, sim.infos[id]);
      Info *info = sim.infos[id];
      Real *TMP = info->block + BS * BS * off_tmp;
      int offset = (info->level == sim.levelMax - 1) ? 4 : 2;
      int ss = 4, nm = 2 * ss + BS;
      for (int y = -offset; y < BS + offset; ++y)
        for (int x = -offset; x < BS + offset; ++x) {
          int k = nm * (y + ss) + x + ss;
          um[k] = std::min(um[k], 1.0);
          um[k] = std::max(um[k], 0.0);
          if (0.0 < um[k] && um[k] < 0.1) {
            int i = BS / 2;
            int j = BS / 2 - 1;
            TMP[BS * i + j] = 2 * sim.Rtol;
            TMP[BS * j + j] = 2 * sim.Rtol;
            TMP[BS * i + i] = 2 * sim.Rtol;
            TMP[BS * j + i] = 2 * sim.Rtol;
            break;
          }
        }
    }
    free(um);
  }
}
static const Real refine_w[4][9] = {
  /*        (-1,-1)  (0,-1)  (1,-1)  (-1,0)  (0,0)  (1,0)  (-1,1)  (0,1)  (1,1) */
  /* (-1/4, -1/4) */ { 1./64, 10./64, -1./64, 10./64, 56./64, -6./64, -1./64, -6./64,  1./64},
  /* (+1/4, -1/4) */ {-1./64, 10./64,  1./64, -6./64, 56./64, 10./64,  1./64, -6./64, -1./64},
  /* (-1/4, +1/4) */ {-1./64, -6./64,  1./64, 10./64, 56./64, -6./64,  1./64, 10./64, -1./64},
  /* (+1/4, +1/4) */ { 1./64, -6./64, -1./64, -6./64, 56./64, 10./64, -1./64, 10./64,  1./64},
};
static int adapt() {
  long long cnt, nprev;
  std::vector<int> level_com;
  std::vector<int> level_ref;
  struct TreeStateMatrix {
    TreeState nei[3][3];
  };
  std::vector<TreeStateMatrix> m_tree;
  std::vector<long long> Z_com;
  std::vector<long long> Z_ref;
  std::unordered_set<long long> dealloc_IDs;
  compute_vorticity();
  compute_grad_chi();
  State *state = (State *)malloc(sim.n * sizeof *state);
  int Changed = 0;
  int More = 0;
  int ss = 1;

#pragma omp parallel for reduction(|| : Changed)
  for (long long i = 0; i < sim.n; i++) {
    Real *b = sim.infos[i]->block + BS * BS * off_tmp;
    double Linf = 0.0;
    for (int j = 0; j < BS * BS; j++)
      Linf = std::max(Linf, std::fabs(b[j]));
    state[i] = Linf > sim.Rtol ? Refine : Linf < sim.Ctol ? Compress : Leave;
    bool maxLevel =
        state[i] == Refine && sim.infos[i]->level == sim.levelMax - 1;
    bool minLevel = state[i] == Compress && sim.infos[i]->level == 0;
    if (maxLevel || minLevel)
      state[i] = Leave;
    if (state[i] != Leave) {
      Changed = 1 || Changed;
    }
  }
  if (!Changed)
    goto end;
  do {
    More = 0;
    for (long long j = 0; j < sim.n; j++) {
      if (state[j] == Refine) {
        int xi, yi;
        int n = 1 << sim.infos[j]->level;
        sfc_inverse(sim.infos[j]->Z, sim.infos[j]->level, &xi, &yi);
        for (int icode = 0; icode < 9; icode++) {
          int cx = icode % 3 - 1;
          int cy = icode / 3 - 1;
          if (cx == 0 && cy == 0)
            continue;
          if (skin_skip(cx, xi, n) || skin_skip(cy, yi, n))
            continue;
          long long Z = sim.infos[j]->Znei[1 + cx][1 + cy];
          long long id = sim.levels[sim.infos[j]->level] + Z;
          long long pid = sim.levels[sim.infos[j]->level - 1] + Z / 4;
          if (sim.map.find(pid) != sim.map.end()) {
            if (state[sim.map[pid]] != Refine) {
              state[sim.map[pid]] = Refine;
              More = 1 || More;
            }
          } else if (sim.map.find(id) != sim.map.end()) {
            if (state[sim.map[id]] == Compress)
              state[sim.map[id]] = Leave;
          }
        }
      }
    }
  } while (More);

  for (long long j = 0; j < sim.n; j++) {
    if (state[j] == Compress) {
      int xi, yi;
      sfc_inverse(sim.infos[j]->Z, sim.infos[j]->level, &xi, &yi);
      if (xi % 2 == 0 && yi % 2 == 0) {
        for (int dx = 0; dx < 2; dx++)
          for (int dy = 0; dy < 2; dy++) {
            long long Z = sfc_forward(sim.infos[j]->level, xi + dx, yi + dy);
            long long id = sim.levels[sim.infos[j]->level] + Z;
            if (sim.map.find(id) == sim.map.end() ||
                state[sim.map.at(id)] != Compress)
              state[j] = Leave;
          }
        int level = sim.infos[j]->level;
        int n = 1 << level;
        for (int dx = 0; dx < 2 && state[j] != Leave; dx++)
          for (int dy = 0; dy < 2 && state[j] != Leave; dy++) {
            int sx = xi + dx, sy = yi + dy;
            for (int icode = 0; icode < 9 && state[j] != Leave; icode++) {
              int cx = icode % 3 - 1, cy = icode / 3 - 1;
              if (cx == 0 && cy == 0) continue;
              if (skin_skip(cx, sx, n) || skin_skip(cy, sy, n)) continue;
              long long Z = forward(level, sx + cx, sy + cy);
              long long id = sim.levels[level] + Z;
              auto it = sim.tree.find(id);
              if (it != sim.tree.end() && it->second == ChildrenAreActive)
                state[j] = Leave;
            }
          }
      }
    }
  }

  for (long long j = 0; j < sim.n; j++) {
    int ix, iy;
    sfc_inverse(sim.infos[j]->Z, sim.infos[j]->level, &ix, &iy);
    if (state[j] == Refine) {
      level_ref.push_back(sim.infos[j]->level);
      Z_ref.push_back(sim.infos[j]->Z);
      TreeStateMatrix nei;
      get_states(sim.infos[j], nei.nei);
      m_tree.push_back(nei);
    } else if (state[j] == Compress && ix % 2 == 0 && iy % 2 == 0) {
      level_com.push_back(sim.infos[j]->level);
      Z_com.push_back(sim.infos[j]->Z);
    }
  }
  fprintf(stderr, "%s:%d: com/ref: %ld %ld\n", __FILE__, __LINE__,
          level_com.size(), level_ref.size());
  if (level_ref.size() == 0 && level_com.size() == 0)
    goto end;
  nprev = sim.n;
  sim.n += 4 * level_ref.size();
  sim.infos = (Info **)realloc(sim.infos, sim.n * sizeof *sim.infos);

#pragma omp parallel
  {
    Real *lm[2] = {lab_alloc(1, ss), lab_alloc(2, ss)};
#pragma omp for
    for (size_t k = 0; k < level_ref.size(); k++) {
      int px, py;
      assert(exist(level_ref[k], Z_ref[k]));
      sfc_inverse(Z_ref[k], level_ref[k], &px, &py);
      assert(level_ref[k] <= sim.levelMax - 1);
      Real *blocks[4];
      for (int J = 0; J < 2; J++)
        for (int I = 0; I < 2; I++) {
          long long Z = sfc_forward(level_ref[k] + 1, 2 * px + I, 2 * py + J);
#pragma omp critical
          assert(!exist(level_ref[k] + 1, Z));
          Info *child = new Info;
          fill(child, level_ref[k] + 1, Z);
          long long id = nprev + 4 * k + 2 * J + I;
#pragma omp critical
          sim.infos[id] = child;
          child->block = blocks[2 * J + I] =
              (Real *)malloc(off_n * BS * BS * sizeof(Real));
        }
      int nm = 2 * ss + BS;
      Info *parent = getf0(level_ref[k], Z_ref[k]);
      for (size_t m = 0; m < sizeof vars / sizeof *vars; m++) {
        int dim = vars[m].dim;
        int offset = vars[m].offset;
        lab_load1(lm[dim - 1], dim, offset, &m_tree[k].nei[0][0], ss, parent);
        Real *um = lm[dim - 1];
        for (int J = 0; J < 2; J++)
          for (int I = 0; I < 2; I++) {
            Real *b = blocks[J * 2 + I] + offset * BS * BS;
            for (int j = 0; j < BS; j += 2)
              for (int i = 0; i < BS; i += 2) {
                int i0 = i / 2 + I * (BS / 2) + ss;
                int j0 = j / 2 + J * (BS / 2) + ss;
                int sub[4] = {BS*j+i, BS*j+i+1, BS*(j+1)+i, BS*(j+1)+i+1};
                for (int s = 0; s < 4; s++)
                  for (int d = 0; d < dim; d++) {
                    Real val = 0;
                    for (int kk = 0; kk < 9; kk++)
                      val += refine_w[s][kk] * um[dim*(nm*(j0+kk/3-1)+i0+kk%3-1)+d];
                    b[dim * sub[s] + d] = val;
                  }
              }
          }
      }
    }
#pragma omp for
    for (size_t k = 0; k < level_ref.size(); k++) {
#pragma omp critical
      dealloc_IDs.insert(sim.levels[level_ref[k]] + Z_ref[k]);
    }
#pragma omp for
    for (size_t k = 0; k < level_com.size(); k++) {
      int x, y;
      sfc_inverse(Z_com[k], level_com[k], &x, &y);
      Real *Blocks[4];
      for (int J = 0; J < 2; J++)
        for (int I = 0; I < 2; I++) {
          int blk = J * 2 + I;
          long long Z = sfc_forward(level_com[k], x + I, y + J);
          long long id = sim.levels[level_com[k]] + Z;
          Blocks[blk] = sim.infos[sim.map.at(id)]->block;
          if (blk != 0)
#pragma omp critical
            dealloc_IDs.insert(sim.levels[level_com[k]] + Z);
        }
      for (size_t v = 0; v < sizeof vars / sizeof *vars; v++) {
        int dim = vars[v].dim;
        int offset = vars[v].offset;
        Real *dst = Blocks[0] + offset * BS * BS;
        for (int J = 0; J < 2; J++)
          for (int I = 0; I < 2; I++) {
            Real *src = Blocks[J * 2 + I] + offset * BS * BS;
            for (int j = 0; j < BS; j += 2)
              for (int i = 0; i < BS; i += 2) {
                int o = BS * (j / 2 + J * (BS / 2)) + i / 2 + I * (BS / 2);
                for (int d = 0; d < dim; d++)
                  dst[dim * o + d] =
                      (src[dim * (BS * j + i) + d] +
                       src[dim * (BS * j + i + 1) + d] +
                       src[dim * (BS * (j + 1) + i) + d] +
                       src[dim * (BS * (j + 1) + i + 1) + d]) / 4;
              }
          }
      }
      long long id = sim.levels[level_com[k]] + Z_com[k];
      fill(sim.infos[sim.map.at(id)], level_com[k] - 1, Z_com[k] / 4);
    }
    free(lm[0]);
    free(lm[1]);
  }
  cnt = 0;
  sim.map.clear();
  for (long long i = 0; i < sim.n; i++) {
    Info *info = sim.infos[i];
    long long id = sim.levels[info->level] + info->Z;
    if (dealloc_IDs.find(id) != dealloc_IDs.end()) {
      free(info->block);
      delete info;
    } else {
      sim.map[id] = cnt;
      sim.infos[cnt] = info;
      sim.infos[cnt]->id = cnt;
      cnt++;
    }
  }
  sim.n = cnt;
  sim.infos = (Info **)realloc(sim.infos, sim.n * sizeof *sim.infos);
  sim.tree.clear();
#pragma omp parallel for
  for (long long i = 0; i < sim.n; i++) {
    int ix, iy;
    Info *info = sim.infos[i];
    long long id = sim.levels[info->level] + info->Z;
#pragma omp critical
    sim.tree[id] = Active;
    assert(sim.map.at(id) == i);
    sfc_inverse(info->Z, info->level, &ix, &iy);
    if (info->level + 1 < sim.levelMax)
      for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) {
          long long Zchild =
              sfc_forward(info->level + 1, 2 * ix + i, 2 * iy + j);
#pragma omp critical
          sim.tree[sim.levels[info->level + 1] + Zchild] = ParentIsActive;
        }
    if (info->level > 0) {
#pragma omp critical
      sim.tree[sim.levels[info->level - 1] + info->Z / 4] = ChildrenAreActive;
    }
  }
end:
  free(state);
  return Changed;
}
static void compute_advect_diffuse() {
#pragma omp parallel
  {
    Real *um = lab_alloc(2, 3);
#pragma omp for nowait
    for (long long id = 0; id < sim.n; ++id) {
      lab_load(um, 2, off_vel, 3, sim.infos[id]);
      Info *info = sim.infos[id];
      Real h = info->h;
      Real dfac = sim.nu * sim.dt;
      Real afac = -sim.dt * h;
      Real *TMP = info->block + BS * BS * off_tmpV;
      int ss = 3, nm = 2 * ss + BS;
      for (int iy = 0; iy < BS; ++iy)
        for (int ix = 0; ix < BS; ++ix) {
#define V(dx, dy, c) um[2 * (nm * (iy + ss + (dy)) + ix + ss + (dx)) + (c)]
          Real u = V(0,0,0), v = V(0,0,1);
          Real dudx = derivative(u, V(-3,0,0), V(-2,0,0), V(-1,0,0), u, V(1,0,0), V(2,0,0), V(3,0,0));
          Real dudy = derivative(v, V(0,-3,0), V(0,-2,0), V(0,-1,0), u, V(0,1,0), V(0,2,0), V(0,3,0));
          Real dvdx = derivative(u, V(-3,0,1), V(-2,0,1), V(-1,0,1), v, V(1,0,1), V(2,0,1), V(3,0,1));
          Real dvdy = derivative(v, V(0,-3,1), V(0,-2,1), V(0,-1,1), v, V(0,1,1), V(0,2,1), V(0,3,1));
          TMP[2 * (BS * iy + ix)]     = afac * (u * dudx + v * dudy) + dfac * (V(1,0,0) + V(-1,0,0) + V(0,1,0) + V(0,-1,0) - 4*u);
          TMP[2 * (BS * iy + ix) + 1] = afac * (u * dvdx + v * dvdy) + dfac * (V(1,0,1) + V(-1,0,1) + V(0,1,1) + V(0,-1,1) - 4*v);
#undef V
        }
    }
    free(um);
  }
}
static long long This(Info *info, int ix, int iy) {
  return info->id * BS * BS + iy * BS + ix;
}
struct PoissonOp {
  int8_t blk_ref, cell_ix, cell_iy, _pad;
  float coeff;
};
enum { MAX_POISSON_OPS = 16 };
struct PoissonEntry {
  int32_t n_ops;
  PoissonOp ops[MAX_POISSON_OPS];
};
static const PoissonEntry *poisson_tab;
static void load_poisson() {
  FILE *fp = fopen("tab_poisson.bin", "rb");
  if (!fp) { fprintf(stderr, "main.cpp: cannot open tab_poisson.bin\n"); exit(1); }
  size_t sz = 4 * BS * 2 * 4 * sizeof(PoissonEntry);
  PoissonEntry *tab = (PoissonEntry *)malloc(sz);
  if (fread(tab, 1, sz, fp) != sz) {
    fprintf(stderr, "main.cpp: short read from tab_poisson.bin\n"); exit(1);
  }
  fclose(fp);
  poisson_tab = tab;
}
static void getVec() {
#pragma omp parallel for
  for (int i = 0; i < sim.n; i++) {
    Real h = sim.infos[i]->h;
    sim.mat->h2_[i] = h * h;
    long long offset = sim.infos[i]->id * BS * BS;
    memcpy(&sim.mat->b_[offset], sim.infos[i]->block + BS * BS * off_tmp,
           BS * BS * sizeof(Real));
    memcpy(&sim.mat->x_[offset], sim.infos[i]->block + BS * BS * off_pres,
           BS * BS * sizeof(Real));
  }
}
static void compute_pressure_correction() {
#pragma omp parallel
  {
    Real *um = lab_alloc(1, 1);
#pragma omp for nowait
    for (long long id = 0; id < sim.n; ++id) {
      lab_load(um, 1, off_pres, 1, sim.infos[id]);
      Info *info = sim.infos[id];
      int ss = 1, nm = 2 * ss + BS;
      Real pFac = -0.5 * sim.dt * info->h;
      Real *tmpV = info->block + BS * BS * off_tmpV;
      for (int iy = 0; iy < BS; ++iy)
        for (int ix = 0; ix < BS; ++ix) {
#define P(dx, dy) um[nm * (iy + ss + (dy)) + ix + ss + (dx)]
          tmpV[2 * (BS * iy + ix)]     = pFac * (P(1,0) - P(-1,0));
          tmpV[2 * (BS * iy + ix) + 1] = pFac * (P(0,1) - P(0,-1));
#undef P
        }
    }
    free(um);
  }
}
static void compute_pressure_laplacian() {
#pragma omp parallel
  {
    Real *um = lab_alloc(1, 1);
#pragma omp for nowait
    for (long long id = 0; id < sim.n; ++id) {
      lab_load(um, 1, off_pold, 1, sim.infos[id]);
      Real *TMP = sim.infos[id]->block + BS * BS * off_tmp;
      int ss = 1, nm = 2 * ss + BS;
      for (int iy = 0; iy < BS; ++iy)
        for (int ix = 0; ix < BS; ++ix) {
#define P(dx, dy) um[nm * (iy + ss + (dy)) + ix + ss + (dx)]
          TMP[BS * iy + ix] -= P(-1,0) + P(1,0) + P(0,-1) + P(0,1) - 4 * P(0,0);
#undef P
        }
    }
    free(um);
  }
}
static const struct {
  const char *name;
  int type;
  size_t off;
} tab[] = {
    {"levelMax", 0, offsetof(struct Sim, levelMax)},
    {"AdaptSteps", 0, offsetof(struct Sim, AdaptSteps)},
    {"levelStart", 0, offsetof(struct Sim, levelStart)},
    {"maxPoissonRestarts", 0, offsetof(struct Sim, maxPoissonRestarts)},
    {"Rtol", 1, offsetof(struct Sim, Rtol)},
    {"Ctol", 1, offsetof(struct Sim, Ctol)},
    {"CFL", 1, offsetof(struct Sim, CFL)},
    {"tend", 1, offsetof(struct Sim, endTime)},
    {"lambda", 1, offsetof(struct Sim, lambda)},
    {"nu", 1, offsetof(struct Sim, nu)},
    {"poissonTol", 1, offsetof(struct Sim, PoissonTol)},
    {"poissonTolRel", 1, offsetof(struct Sim, PoissonTolRel)},
    {"tdump", 1, offsetof(struct Sim, dumpTime)},
};
static const struct {
  const char *name;
  size_t off;
  Real scale;
} stab[] = {
    {"xcenter", offsetof(Shape, x), 1},
    {"ycenter", offsetof(Shape, y), 1},
    {"orientation", offsetof(Shape, orientation), M_PI / 180},
    {"omega", offsetof(Shape, omega), 1},
};
int main(int argc, char **argv) {
#ifdef _OPENMP
#pragma omp parallel
#pragma omp master
  fprintf(stderr, "main.cpp: %d threads\n", omp_get_num_threads());
#endif
  char *base = (char *)&sim;
  for (size_t i = 0; i < sizeof tab / sizeof *tab; i++)
    if (tab[i].type == 0)
      *(int *)(base + tab[i].off) = arg_int(argc, argv, tab[i].name);
    else
      *(Real *)(base + tab[i].off) = arg_real(argc, argv, tab[i].name);
  sim.nshape = 0;
  sim.shapes = NULL;
  const char *shapeArg = arg_find(argc, argv, "shapes");
  const char *sp = shapeArg;
  while (*sp) {
    while (*sp == '\n' || *sp == ',' || *sp == ' ')
      sp++;
    if (!*sp)
      break;
    const char *end = sp;
    while (*end && *end != '\n' && *end != ',')
      end++;
    size_t len = end - sp;
    char line[1024];
    if (len >= sizeof line) {
      fprintf(stderr, "main.cpp: shape line too long\n");
      exit(1);
    }
    memcpy(line, sp, len);
    line[len] = '\0';
    sp = end;
    Shape *shape = new Shape;
    char *base = (char *)shape;
    for (size_t i = 0; i < sizeof stab / sizeof *stab; i++)
      *(Real *)(base + stab[i].off) =
          kv_real(line, stab[i].name) * stab[i].scale;
    Real scale = kv_real(line, "scale");
    char pathbuf[FILENAME_MAX];
    const char *path = kv_str(line, "sdf", pathbuf, sizeof pathbuf);
    FILE *file = fopen(path, "r");
    char tag[3];
    float length, rmax;
    if (file == NULL) {
      fprintf(stderr, "main.cpp: error: fail to open '%s'\n", path);
      exit(1);
    }
    if (fread(tag, sizeof *tag, sizeof tag, file) != sizeof tag) {
      fprintf(stderr, "main.cpp: error: fail to read '%s'\n", path);
      exit(1);
    }
    if (tag[0] != 'S' || tag[1] != 'D' || tag[2] != 'F') {
      fprintf(stderr, "main.cpp: error: not and sdf file '%s'\n", path);
      exit(1);
    }
    if (fread(&length, sizeof(length), 1, file) != 1 ||
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
      fprintf(stderr, "main.cpp: error: fail to read arrays from '%s'\n", path);
    }
    shape->length = scale * length;
    shape->rmax = scale * rmax;
    for (size_t i = 0; i < ncount; i++)
      shape->sdf[i] *= scale;
    shape->u = 0;
    shape->v = 0;
    sim.nshape++;
    sim.shapes =
        (struct Shape **)realloc(sim.shapes, sim.nshape * sizeof sim.shapes);
    sim.shapes[sim.nshape - 1] = shape;
  }
  if (!sim.nshape && *shapeArg) {
    fprintf(stderr, "main.cpp: error: failed to parse shapes\n");
    exit(1);
  }
  sim.levels = (long long *)malloc(sim.levelMax * sizeof *sim.levels);
  sim.levels[0] = 0;
  for (int m = 0; m < sim.levelMax - 1; m++)
    sim.levels[m + 1] = sim.levels[m] + (1 << (2 * m));
  sim.n = 1LL << (2 * sim.levelStart);
  sim.infos = (Info **)malloc(sim.n * sizeof *sim.infos);
  for (long long i = 0; i < sim.n; i++) {
    long long Z = i;
    long long aux = sim.levels[sim.levelStart] + Z;
    Info *info = new Info;
    sim.map[aux] = i;
    fill(info, sim.levelStart, Z);
    info->block = (Real *)calloc(off_n * BS * BS, sizeof(Real));
    sim.infos[i] = info;
#pragma omp critical
    sim.tree[aux] = Active;
    int px, py;
    sfc_inverse(Z, sim.levelStart, &px, &py);
    if (sim.levelStart < sim.levelMax - 1)
      for (int j1 = 0; j1 < 2; j1++)
        for (int i1 = 0; i1 < 2; i1++) {
          long long n = forward(sim.levelStart + 1, 2 * px + i1, 2 * py + j1);
#pragma omp critical
          sim.tree[sim.levels[sim.levelStart + 1] + n] = ParentIsActive;
        }
    if (sim.levelStart > 0) {
      long long n = forward(sim.levelStart - 1, px / 2, py / 2);
#pragma omp critical
      sim.tree[sim.levels[sim.levelStart - 1] + n] = ChildrenAreActive;
    }
  }
  tab_load_all();
  int Changed = 0;
  for (long long j = 0; j < sim.n; j++)
    sim.infos[j]->id = j;
  for (int i = 0;; i++) {
    ongrid();
    if (i == sim.levelMax)
      break;
    Changed = adapt() || Changed;
  }
  for (int ishape = 0; ishape < sim.nshape; ishape++) {
    Shape *shape = sim.shapes[ishape];
    std::vector<Obstacle *> &oblock = shape->blocks;
#pragma omp parallel for
    for (long long i = 0; i < sim.n; i++) {
      if (oblock[sim.infos[i]->id] == nullptr)
        continue;
      Real *udef = (Real *)oblock[sim.infos[i]->id]->udef;
      Real *chi = (Real *)oblock[sim.infos[i]->id]->chi;
      Real *UDEF = sim.infos[i]->block + BS * BS * off_tmpV;
      Real *CHI = sim.infos[i]->block + BS * BS * off_chi;
      for (int j = 0; j < BS * BS; j++) {
        if (chi[j] < CHI[j])
          continue;
        UDEF[2 * j] += udef[2 * j];
        UDEF[2 * j + 1] += udef[2 * j + 1];
      }
    }
  }
#pragma omp parallel for schedule(static)
  for (long long i = 0; i < sim.n; i++) {
    Real *UF = sim.infos[i]->block + BS * BS * off_vel;
    Real *US = sim.infos[i]->block + BS * BS * off_tmpV;
    Real *X = sim.infos[i]->block + BS * BS * off_chi;
    for (int j = 0; j < BS * BS; j++) {
      UF[2 * j + 0] = UF[2 * j + 0] * (1 - X[j]) + US[2 * j + 0] * X[j];
      UF[2 * j + 1] = UF[2 * j + 1] * (1 - X[j]) + US[2 * j + 1] * X[j];
    }
  }
  std::vector<double> P_inv = precond();
  sim.mat = new LocalSpMatDnVec(BS * BS, 0, P_inv);
  load_poisson();
  while (1) {
    if (sim.step % 5 == 0)
      fprintf(stderr, "main.cpp: %08d %.16e\n", sim.step, sim.time);
    if (sim.dumpTime > 0 && sim.time >= sim.nextDumpTime) {
      sim.nextDumpTime += sim.dumpTime;
      compute_vorticity();
      char path[FILENAME_MAX];
      snprintf(path, sizeof path, "vel.%08d", sim.dump_count++);
      dump(sim.time, sim.infos, path);
    }
    if (sim.endTime > 0 && sim.time >= sim.endTime)
      break;
    Real CFL = sim.CFL;
    Real h = std::numeric_limits<Real>::infinity();
    for (long long i = 0; i < sim.n; i++)
      h = std::min(sim.infos[i]->h, h);
    Real umax = 0;
#pragma omp parallel for schedule(static) reduction(max : umax)
    for (long long i = 0; i < sim.n; i++) {
      Real *vel = sim.infos[i]->block + BS * BS * off_vel;
      for (int j = 0; j < 2 * BS * BS; j++)
        umax = std::max(umax, std::fabs(vel[j]));
    }
    sim.dt = real_min(CFL * h / (umax + 1e-8),
                      0.25 * h * h / (sim.nu + 0.25 * h * umax));
    if (sim.step <= 10 || sim.step % sim.AdaptSteps == 0)
      Changed = adapt() || Changed;
    for (int ishape = 0; ishape < sim.nshape; ishape++) {
      Shape *shape = sim.shapes[ishape];
      shape->x += sim.dt * shape->u;
      shape->y += sim.dt * shape->v;
      shape->orientation += sim.dt * shape->omega;
      if (shape->orientation < -M_PI)
        shape->orientation += 2 * M_PI;
      else if (shape->orientation > M_PI)
        shape->orientation -= 2 * M_PI;
    }
    ongrid();
#pragma omp parallel for
    for (long long i = 0; i < sim.n; i++)
      memcpy(sim.infos[i]->block + BS * BS * off_vold,
             sim.infos[i]->block + BS * BS * off_vel,
             2 * BS * BS * sizeof(Real));
    for (Real fac : {0.5, 1.0}) {
      compute_advect_diffuse();
#pragma omp parallel for
      for (long long i = 0; i < sim.n; i++) {
        Real *V = sim.infos[i]->block + BS * BS * off_vel;
        Real *Vold = sim.infos[i]->block + BS * BS * off_vold;
        Real *tmpV = sim.infos[i]->block + BS * BS * off_tmpV;
        Real ih2 = fac / (sim.infos[i]->h * sim.infos[i]->h);
        for (int j = 0; j < 2 * BS * BS; j++)
          V[j] = Vold[j] + tmpV[j] * ih2;
      }
    }
    for (int ishape = 0; ishape < sim.nshape; ishape++) {
      Shape *shape = sim.shapes[ishape];
      std::vector<Obstacle *> &oblock = shape->blocks;
      Real PM = 0, PX = 0, PY = 0, UM = 0, VM = 0;
#pragma omp parallel for reduction(+ : PM, PX, PY, UM, VM)
      for (long long i = 0; i < sim.n; i++) {
        Real *VEL = sim.infos[i]->block + BS * BS * off_vel;
        Real hsq = sim.infos[i]->h * sim.infos[i]->h;
        if (oblock[sim.infos[i]->id] == nullptr)
          continue;
        Real *chi = (Real *)oblock[sim.infos[i]->id]->chi;
        Real *udef = (Real *)oblock[sim.infos[i]->id]->udef;
        Real lambdt = sim.lambda * sim.dt;
        for (int iy = 0; iy < BS; ++iy)
          for (int ix = 0; ix < BS; ++ix) {
            int j = BS * iy + ix;
            if (chi[j] <= 0)
              continue;
            Real udiff[2] = {VEL[2 * j + 0] - udef[2 * j + 0],
                             VEL[2 * j + 1] - udef[2 * j + 1]};
            Real Xlamdt = chi[j] >= 0.5 ? lambdt : 0.0;
            Real F = hsq * Xlamdt / (1 + Xlamdt);
            Real p[2];
            p[0] = sim.infos[i]->origin[0] + sim.infos[i]->h * (ix + 0.5);
            p[1] = sim.infos[i]->origin[1] + sim.infos[i]->h * (iy + 0.5);
            p[0] -= shape->x;
            p[1] -= shape->y;
            PM += F;
            PX += F * p[0];
            PY += F * p[1];
            UM += F * udiff[0];
            VM += F * udiff[1];
          }
      }
      if (PM != 0) {
        shape->u = (PY * shape->omega + UM) / PM;
        shape->v = (VM - PX * shape->omega) / PM;
      }
    }
    std::vector<Collision> collisions(sim.nshape);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < sim.nshape; ++i)
      for (int j = 0; j < sim.nshape; ++j) {
        if (i == j)
          continue;
        Collision &coll = collisions[i];
        auto &iBlocks = sim.shapes[i]->blocks;
        auto &jBlocks = sim.shapes[j]->blocks;
        for (size_t k = 0; k < iBlocks.size(); ++k) {
          if (iBlocks[k] == nullptr || jBlocks[k] == nullptr)
            continue;
          auto &iSDF = iBlocks[k]->dist;
          auto &jSDF = jBlocks[k]->dist;
          ScalarBlock &iChi = iBlocks[k]->chi;
          ScalarBlock &jChi = jBlocks[k]->chi;
          Real h = 1.0 / BS / (1 << sim.infos[k]->level);
          Real hsq = h * h;
          for (int iy = 0; iy < BS; ++iy)
            for (int ix = 0; ix < BS; ++ix) {
              if (iChi[iy][ix] <= 0.0 || jChi[iy][ix] <= 0.0)
                continue;
              Real pos[2];
              pos[0] = sim.infos[k]->origin[0] + h * (ix + 0.5);
              pos[1] = sim.infos[k]->origin[1] + h * (iy + 0.5);
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
              } else if (ix == BS - 1) {
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
              } else if (iy == BS - 1) {
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
    // #pragma omp parallel for schedule(static)
    for (int i = 0; i < sim.nshape; ++i) {
      for (int j = i + 1; j < sim.nshape; ++j) {
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
          fprintf(stderr,
                  "Collision between objects %d and %d\n"
                  " iM %g %g\n"
                  " jM %g %g\n"
                  " Normal vector = %g %g\n",
                  i, j, collisions[i].iM, collisions[j].jM, collisions[i].jM,
                  collisions[j].iM, NX, NY);
        }
      }
    }
#pragma omp parallel for
    for (long long i = 0; i < sim.n; i++)
      for (int ishape = 0; ishape < sim.nshape; ishape++) {
        Shape *shape = sim.shapes[ishape];
        std::vector<Obstacle *> &oblock = shape->blocks;
        Obstacle *o = oblock[sim.infos[i]->id];
        if (o == nullptr)
          continue;
        Real *X = (Real *)o->chi;
        Real *UDEF = (Real *)o->udef;
        Real *CHI = sim.infos[i]->block + BS * BS * off_chi;
        Real *V = sim.infos[i]->block + BS * BS * off_vel;
        for (int iy = 0; iy < BS; ++iy)
          for (int ix = 0; ix < BS; ++ix) {
            int j = BS * iy + ix;
            if (CHI[j] > X[j])
              continue;
            if (X[j] <= 0)
              continue;
            Real p[2];
            p[0] = sim.infos[i]->origin[0] + sim.infos[i]->h * (ix + 0.5);
            p[1] = sim.infos[i]->origin[1] + sim.infos[i]->h * (iy + 0.5);
            p[0] -= shape->x;
            p[1] -= shape->y;
            Real alpha = X[j] > 0.5 ? 1 / (1 + sim.lambda * sim.dt) : 1;
            Real US = shape->u - shape->omega * p[1] + UDEF[2 * j + 0];
            Real VS = shape->v + shape->omega * p[0] + UDEF[2 * j + 1];
            V[2 * j + 0] = alpha * V[2 * j + 0] + (1 - alpha) * US;
            V[2 * j + 1] = alpha * V[2 * j + 1] + (1 - alpha) * VS;
          }
      }
#pragma omp parallel for
    for (long long i = 0; i < sim.n; i++)
      memset(sim.infos[i]->block + BS * BS * off_tmpV, 0,
             2 * BS * BS * sizeof(Real));
    for (int ishape = 0; ishape < sim.nshape; ishape++) {
      Shape *shape = sim.shapes[ishape];
      std::vector<Obstacle *> &oblock = shape->blocks;
#pragma omp parallel for
      for (long long i = 0; i < sim.n; i++) {
        if (oblock[sim.infos[i]->id] == nullptr)
          continue;
        Real *udef = (Real *)oblock[sim.infos[i]->id]->udef;
        Real *chi = (Real *)oblock[sim.infos[i]->id]->chi;
        Real *UDEF = sim.infos[i]->block + BS * BS * off_tmpV;
        Real *CHI = sim.infos[i]->block + BS * BS * off_chi;
        for (int iy = 0; iy < BS; iy++)
          for (int ix = 0; ix < BS; ix++) {
            int j = BS * iy + ix;
            if (chi[j] < CHI[j])
              continue;
            UDEF[2 * j + 0] += udef[2 * j + 0];
            UDEF[2 * j + 1] += udef[2 * j + 1];
          }
      }
    }
#pragma omp parallel
    {
      Real *vm = lab_alloc(2, 1);
      Real *um = lab_alloc(2, 1);
#pragma omp for
      for (int i = 0; i < sim.n; i++) {
        lab_load(vm, 2, off_vel, 1, sim.infos[i]);
        lab_load(um, 2, off_tmpV, 1, sim.infos[i]);
        pressure_rhs_fun(vm, um, i);
      }
      free(vm);
      free(um);
    }
#pragma omp parallel for
    for (long long i = 0; i < sim.n; i++) {
      memcpy(sim.infos[i]->block + BS * BS * off_pold,
             sim.infos[i]->block + BS * BS * off_pres, BS * BS * sizeof(Real));
      memset(sim.infos[i]->block + BS * BS * off_pres, 0,
             BS * BS * sizeof(Real));
    }
    compute_pressure_laplacian();
    double max_error = sim.step < 10 ? 0.0 : sim.PoissonTol;
    double max_rel_error = sim.step < 10 ? 0.0 : sim.PoissonTolRel;
    int max_restarts = sim.step < 10 ? 100 : sim.maxPoissonRestarts;
    int N = BS * BS * sim.n;
    sim.mat->reserve(N);
    for (int i = 0; i < sim.n; i++) {
      Info *info = sim.infos[i];
      int n = 1 << info->level;
      for (int iy = 0; iy < BS; iy++)
        for (int ix = 0; ix < BS; ix++) {
          long long sfc_idx = This(info, ix, iy);
          if ((ix > 0 && ix < BS - 1) && (iy > 0 && iy < BS - 1)) {
            sim.mat->cooPushBackVal(1, sfc_idx, This(info, ix, iy - 1));
            sim.mat->cooPushBackVal(1, sfc_idx, This(info, ix - 1, iy));
            sim.mat->cooPushBackVal(-4, sfc_idx, sfc_idx);
            sim.mat->cooPushBackVal(1, sfc_idx, This(info, ix + 1, iy));
            sim.mat->cooPushBackVal(1, sfc_idx, This(info, ix, iy + 1));
          } else {
            SpRowInfo row(Tree1(info), sfc_idx, 8);
            for (int j = 0; j < 4; j++) {
              int dir = j >> 1, side = j & 1;
              int sign = 2 * side - 1;
              int ec = dir == 0 ? ix : iy;
              int tc = dir == 0 ? iy : ix;
              Info *blk_infos[4] = {info, nullptr, nullptr, nullptr};
              int state;
              if (side == 0 ? ec > 0 : ec < BS - 1) {
                int dx = (1 - dir) * sign, dy = dir * sign;
                row.mapColVal(This(info, ix + dx, iy + dy), 1);
                row.mapColVal(sfc_idx, -1);
                continue;
              } else if (side == 0 ? info->index[dir] == 0
                                   : info->index[dir] == n - 1) {
                continue;
              } else {
                long long Z = dir == 0 ? info->Znei[1 + sign][1]
                                       : info->Znei[1][1 + sign];
                TreeState ts = sim.tree.at(sim.levels[info->level] + Z);
                if (ts == Active) {
                  state = 1;
                  blk_infos[1] = getf0(info->level, Z);
                } else if (ts == ParentIsActive) {
                  state = 2;
                  blk_infos[2] = getf0(info->level - 1, Z >> 2);
                } else if (ts == ChildrenAreActive) {
                  state = 3;
                  Info nei0 = getf1(info->level, Z);
                  int ct = tc >= BS / 2 ? 1 : 0, ce = 1 - side;
                  long long Zc = dir == 0 ? nei0.Zchild[ce][ct]
                                          : nei0.Zchild[ct][ce];
                  blk_infos[3] = getf0(info->level + 1, Zc);
                } else {
                  throw std::runtime_error(
                      "Neighbour doesn't exist, isn't coarser, nor finer...");
                }
              }
              int parity = dir == 0 ? info->index[1] % 2
                                    : info->index[0] % 2;
              const PoissonEntry &pe =
                  poisson_tab[((j * BS + tc) * 2 + parity) * 4 + state];
              for (int k = 0; k < pe.n_ops; k++) {
                const PoissonOp &op = pe.ops[k];
                row.mapColVal(
                    This(blk_infos[op.blk_ref], op.cell_ix, op.cell_iy),
                    (double)op.coeff);
              }
            }
            sim.mat->cooPushBackRow(row);
          }
        }
    }
    if (Changed) {
      sim.mat->make();
      getVec();
      sim.mat->solveWithUpdate(max_error, max_rel_error, max_restarts);
      Changed = 0;
    } else {
      getVec();
      sim.mat->solveNoUpdate(max_error, max_rel_error, max_restarts);
    }
    Real avg, avg1;
    avg = 0;
    avg1 = 0;
#pragma omp parallel for reduction(+ : avg, avg1)
    for (long long i = 0; i < sim.n; i++) {
      Real *P = sim.infos[i]->block + BS * BS * off_pres;
      Real vv = sim.infos[i]->h * sim.infos[i]->h;
      for (int j = 0; j < BS * BS; j++) {
        P[j] = sim.mat->x_[i * BS * BS + j];
        avg += P[j] * vv;
        avg1 += vv;
      }
    }
    avg = avg / avg1;
#pragma omp parallel for
    for (long long i = 0; i < sim.n; i++) {
      Real *P = sim.infos[i]->block + BS * BS * off_pres;
      for (int j = 0; j < BS * BS; j++)
        P[j] += -avg;
    }
    avg = 0;
    avg1 = 0;
#pragma omp parallel for reduction(+ : avg, avg1)
    for (long long i = 0; i < sim.n; i++) {
      Real *P = sim.infos[i]->block + BS * BS * off_pres;
      Real vv = sim.infos[i]->h * sim.infos[i]->h;
      for (int j = 0; j < BS * BS; j++) {
        avg += P[j] * vv;
        avg1 += vv;
      }
    }
    avg = avg / avg1;
#pragma omp parallel for
    for (long long i = 0; i < sim.n; i++) {
      Real *pres = sim.infos[i]->block + BS * BS * off_pres;
      Real *pold = sim.infos[i]->block + BS * BS * off_pold;
      for (int j = 0; j < BS * BS; j++)
        pres[j] += pold[j] - avg;
    }
    compute_pressure_correction();
#pragma omp parallel for
    for (long long i = 0; i < sim.n; i++) {
      Real ih2 = 1.0 / sim.infos[i]->h / sim.infos[i]->h;
      Real *V = sim.infos[i]->block + BS * BS * off_vel;
      Real *tmpV = sim.infos[i]->block + BS * BS * off_tmpV;
      for (int j = 0; j < 2 * BS * BS; j++)
        V[j] += tmpV[j] * ih2;
    }
    sim.time += sim.dt;
    sim.step++;
  }

  delete sim.mat;
  for (int ishape = 0; ishape < sim.nshape; ishape++) {
    Shape *shape = sim.shapes[ishape];
    for (Obstacle *oblock : shape->blocks)
      delete oblock;
    free(shape->sdf);
    delete shape;
  }
  free(sim.levels);
  fprintf(stderr, "main.cpp: end\n");
}
