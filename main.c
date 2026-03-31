#include <assert.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "solver.h"

typedef double Real;
enum { BS = 8 };
enum {
  F_VEL = 0,
  F_PRS = 2,
  F_VOL = 3,
  F_TMP = 5,
  F_POL = 6,
  F_TMV = 7,
  F_N = 9,
  BLK_S = F_N *BS *BS,
};

#define EPS DBL_EPSILON
enum AdSt { Leave = 0, Refine = 1, Compress = -1, Dealloc = 2 };
struct Blk;
struct HMap {
  long long *keys;
  int *vals;
  int cap;
};
static int hm_slot(const struct HMap *m, long long key) {
  unsigned long long h = (unsigned long long)key * 0x9E3779B97F4A7C15ULL;
  return (int)(h >> 32) & (m->cap - 1);
}
static int hm_get(const struct HMap *m, long long key) {
  int i = hm_slot(m, key);
  while (m->keys[i] >= 0) {
    if (m->keys[i] == key) return m->vals[i];
    i = (i + 1) & (m->cap - 1);
  }
  return -1;
}
static struct Sim {
  int AdaptSteps;
  int levelMax;
  int levelStart;
  int maxPoissonRestarts;
  int step;
  int dump_count;
  Real CFL;
  Real Ctol;
  Real dt;
  Real dumpTime;
  Real endTime;
  Real nextDumpTime;
  Real nu;
  Real PoissonTol;
  Real PoissonTolRel;
  Real Rtol;
  Real time;
  struct Solver *solver;
  int coo_nnz, coo_cap;
  double *coo_val;
  int *coo_row, *coo_col;
  double *sol_x, *sol_b, *sol_h2;
  long long n;
  struct HMap hm;
  struct Blk *blk;
  Real *fld;
} sim;
static long long hm_key(int level, int ix, int iy) {
  long long n = 1LL << level;
  return ((n * n) - 1) / 3 + iy * n + ix;
}
static double ps_Aloc(int I1, int I2) {
  int j1 = I1 / BS;
  int i1 = I1 % BS;
  int j2 = I2 / BS;
  int i2 = I2 % BS;
  if (i1 == i2 && j1 == j2)
    return 4.0;
  else if (abs(i1 - i2) + abs(j1 - j2) == 1)
    return -1.0;
  else
    return 0.0;
}
static const char *arg_find(int argc, char **argv, const char *key) {
  for (int i = 1; i < argc; i++)
    if (argv[i][0] == '-' && strcmp(argv[i] + 1, key) == 0) {
      if (i + 1 < argc)
        return argv[i + 1];
      fprintf(stderr, "main.c: error: option -%s has no value\n", key);
      exit(1);
    }
  fprintf(stderr, "main.c: error: option -%s is not set\n", key);
  exit(1);
}
static Real arg_r(int argc, char **argv, const char *key) {
  const char *s = arg_find(argc, argv, key);
  char *end;
  Real v = strtod(s, &end);
  if (end == s || *end != '\0') {
    fprintf(stderr, "main.c: error: -%s: bad real '%s'\n", key, s);
    exit(1);
  }
  return v;
}
static int arg_i(int argc, char **argv, const char *key) {
  const char *s = arg_find(argc, argv, key);
  char *end;
  long v = strtol(s, &end, 10);
  if (end == s || *end != '\0') {
    fprintf(stderr, "main.c: error: -%s: bad integer '%s'\n", key, s);
    exit(1);
  }
  return (int)v;
}
static void ps_prec(double *P_inv) {
  double L[64][64];
  double L_inv[64][64];
  memset(L, 0, sizeof L);
  memset(L_inv, 0, sizeof L_inv);
  for (int i = 0; i < BS * BS; i++)
    L_inv[i][i] = 1.0;
  for (int i = 0; i < BS * BS; i++) {
    double s1 = 0;
    for (int k = 0; k <= i - 1; k++)
      s1 += L[i][k] * L[i][k];
    L[i][i] = sqrt(ps_Aloc(i, i) - s1);
    for (int j = i + 1; j < BS * BS; j++) {
      double s2 = 0;
      for (int k = 0; k <= i - 1; k++)
        s2 += L[i][k] * L[j][k];
      L[j][i] = (ps_Aloc(j, i) - s2) / L[i][i];
    }
  }
  for (int br = 0; br < BS * BS; br++) {
    double bsf = 1. / L[br][br];
    for (int c = 0; c <= br; c++)
      L_inv[br][c] *= bsf;
    for (int wr = br + 1; wr < BS * BS; wr++) {
      double wsf = L[wr][br];
      for (int c = 0; c <= br; c++)
        L_inv[wr][c] -= wsf * L_inv[br][c];
    }
  }
  for (int i = 0; i < BS * BS; i++)
    for (int j = 0; j < BS * BS; j++) {
      double aux = 0.;
      for (int k = 0; k < BS * BS; k++)
        aux += i <= k && j <= k ? L_inv[k][i] * L_inv[k][j] : 0.;
      P_inv[i * BS * BS + j] = -aux;
    }
}
struct Blk {
  double h, origin[2];
  int level, n, ix, iy;
};
#define BLK(i) (sim.fld + (long long)(i) * BLK_S)
static void bl_fill(struct Blk *b, int level, int ix, int iy) {
  int n = 1 << level;
  b->level = level;
  b->n = n;
  b->ix = ix;
  b->iy = iy;
  b->h = 1.0 / BS / n;
  b->origin[0] = (Real)ix / n;
  b->origin[1] = (Real)iy / n;
}
struct {
  int offset;
  int dim;
  const char *prefix;
} fld_t[] = {{F_VEL, 2, "vel"}, {F_PRS, 1, "pres"},
            {F_VOL, 2, NULL}, {F_TMP, 1, "tmp"}, {F_POL, 1, NULL},
            {F_TMV, 2, NULL}};
enum { NVARS = sizeof fld_t / sizeof *fld_t };

static inline int nb_skin(int c, int coord, int n) {
  int skin = coord == 0 || coord == n - 1;
  int skip = coord == 0 ? -1 : 1;
  return c == skip && skin;
}
static const int nb_ch_off[9][2][2] = {
  [0] = {{-1, -1}, {0, 0}},
  [1] = {{ 0, -1}, {1, -1}},
  [2] = {{ 2, -1}, {0, 0}},
  [3] = {{-1,  0}, {-1, 1}},
  [4] = {{ 0,  0}, {0, 0}},
  [5] = {{ 2,  0}, {2, 1}},
  [6] = {{-1,  2}, {0, 0}},
  [7] = {{ 0,  2}, {1, 2}},
  [8] = {{ 2,  2}, {0, 0}},
};
static const int nb_ch_n[9] = {1, 2, 1, 2, 0, 2, 1, 2, 1};
static void hm_rebuild(void) {
  int cap = 1;
  while (cap < 4 * sim.n) cap <<= 1;
  if (sim.hm.cap != cap) {
    free(sim.hm.keys);
    free(sim.hm.vals);
    sim.hm.cap = cap;
    sim.hm.keys = malloc(cap * sizeof *sim.hm.keys);
    sim.hm.vals = malloc(cap * sizeof *sim.hm.vals);
  }
  memset(sim.hm.keys, 0xff, cap * sizeof *sim.hm.keys);
  for (long long i = 0; i < sim.n; i++) {
    long long key = hm_key(sim.blk[i].level, sim.blk[i].ix, sim.blk[i].iy);
    int s = hm_slot(&sim.hm, key);
    while (sim.hm.keys[s] >= 0 && sim.hm.keys[s] != key)
      s = (s + 1) & (cap - 1);
    sim.hm.keys[s] = key;
    sim.hm.vals[s] = i;
  }
}
struct Nb {
  int8_t s;
  int idx;
  int ch[2];
};
static struct Nb nb_find(int level, int ix, int iy, int icode) {
  struct Nb r = {0, -1, {-1, -1}};
  int cx = icode % 3 - 1, cy = icode / 3 - 1;
  int n = 1 << level;
  int xskin = nb_skin(cx, ix, n);
  int yskin = nb_skin(cy, iy, n);
  if (xskin && yskin) { r.s = 5; return r; }
  if (xskin) { r.s = 3; return r; }
  if (yskin) { r.s = 4; return r; }
  int nx = (ix + cx + n) % n, ny = (iy + cy + n) % n;
  int idx = hm_get(&sim.hm, hm_key(level, nx, ny));
  if (idx >= 0) {
    r.s = 0;
    r.idx = idx;
    return r;
  }
  if (level > 0) {
    idx = hm_get(&sim.hm, hm_key(level - 1, nx / 2, ny / 2));
    if (idx >= 0) {
      r.s = 2;
      r.idx = idx;
      return r;
    }
  }
  r.s = 1;
  int L1 = level + 1, nL1 = 1 << L1;
  for (int b = 0; b < nb_ch_n[icode]; b++) {
    int fx = (ix * 2 + nb_ch_off[icode][b][0] + nL1) % nL1;
    int fy = (iy * 2 + nb_ch_off[icode][b][1] + nL1) % nL1;
    r.ch[b] = hm_get(&sim.hm, hm_key(L1, fx, fy));
  }
  return r;
}
enum {
  OP_COPY,
  OP_AVG,
  OP_INTERP9,
  OP_INTERP3,
  OP_LELI,
  OP_BC_SCALAR,
  OP_BC_VECTOR,
};
struct LbOp {
  int8_t type;
  int8_t blk_idx;
  int8_t dst_idx;
  int8_t flags;
  int32_t src_off, dst_off, p1, p2;
};
struct LbSrc {
  int8_t level_delta;
  int8_t xi_mul, yi_mul;
  int8_t xi_add, yi_add;
  int8_t xi_shift, yi_shift;
  int8_t is_self;
  int8_t self_idx;
};
enum { MAX_PRE = 32, MAX_POST = 48, MAX_OPS = MAX_PRE + MAX_POST };
struct LbTab {
  int8_t n_blk;
  struct LbSrc blk_src[2];
  int8_t _pad;
  int32_t n_pre;
  int32_t n_post;
  struct LbOp ops[MAX_OPS];
};
static void lb_exec(Real *const blk[], Real *const dst[],
                         const struct LbOp*ops, int n, int dim, int nm, int nc) {
  Real *m = dst[0], *c = dst[1];
  for (int i = 0; i < n; i++) {
    const struct LbOp*o = &ops[i];
    switch (o->type) {
    case OP_COPY:
      memcpy(dst[o->dst_idx] + o->dst_off, blk[o->blk_idx] + o->src_off,
             o->p1 * dim * sizeof(Real));
      break;
    case OP_AVG: {
      Real *src = blk[o->blk_idx] + o->src_off;
      Real *d = dst[o->dst_idx] + o->dst_off;
      Real *q1 = src + o->p2 * dim;
      for (int k = 0; k < o->p1; k++)
        for (int dd = 0; dd < dim; dd++)
          d[k * dim + dd] =
              (src[2 * k * dim + dd] + src[(2 * k + 1) * dim + dd] +
               q1[2 * k * dim + dd] + q1[(2 * k + 1) * dim + dd]) /
              4;
      break;
    }
    case OP_INTERP9: {
      static const int8_t W[4][9] = {
          {1, 10, -1, 10, 56, -6, -1, -6, 1},
          {-1, 10, 1, -6, 56, 10, 1, -6, -1},
          {-1, -6, 1, 10, 56, -6, 1, 10, -1},
          {1, -6, -1, -6, 56, 10, -1, 10, 1},
      };
      const int8_t *w = W[o->flags & 3];
      for (int d = 0; d < dim; d++) {
        Real sum = 0;
        for (int jj = 0; jj < 3; jj++)
          for (int ii = 0; ii < 3; ii++)
            sum += w[3 * jj + ii] * c[o->src_off + d + dim * ((ii - 1) + nc * (jj - 1))];
        m[o->dst_off + d] = sum / 64.0;
      }
      break;
    }
    case OP_INTERP3: {
      for (int d = 0; d < dim; d++)
        m[o->dst_off + d] =
            (o->blk_idx * c[o->src_off + d] + o->dst_idx * c[o->p1 + d] +
             o->flags * c[o->p2 + d]) /
            32.0;
      break;
    }
    case OP_LELI: {
      static const int8_t W[2][3] = {
          {8, 10, -3},
          {24, -15, 6},
      };
      const int8_t *w = W[o->flags & 1];
      for (int d = 0; d < dim; d++) {
        Real a = m[o->src_off + d], b = m[o->dst_off + d], cv = m[o->p1 + d];
        m[o->src_off + d] = (w[0] * a + w[1] * b + w[2] * cv) / 15.0;
      }
      break;
    }
    case OP_BC_SCALAR: {
      Real *buf = dst[o->dst_idx];
      for (int d = 0; d < dim; d++)
        buf[o->dst_off + d] = buf[o->src_off + d];
      break;
    }
    case OP_BC_VECTOR: {
      Real *buf = dst[o->dst_idx];
      int dir = o->flags & 1;
      buf[o->dst_off + dir] = -buf[o->src_off + dir];
      buf[o->dst_off + 1 - dir] = buf[o->src_off + 1 - dir];
      break;
    }
    }
  }
}

static const struct LbTab (*lb_tab[5][3])[3][2][2][6];
static void lb_init(void) {
  int configs[][2] = {{1,1}, {1,2}, {4,1}};
  for (int ci = 0; ci < 3; ci++) {
    int ss = configs[ci][0], dim = configs[ci][1];
    char fname[64];
    snprintf(fname, sizeof fname, "tab_ss%d_dim%d.bin", ss, dim);
    FILE *fp = fopen(fname, "rb");
    if (!fp) {
      fprintf(stderr, "main.c: cannot open %s\n", fname);
      exit(1);
    }
    size_t sz = 3 * 3 * 2 * 2 * 6 * sizeof(struct LbTab);
    struct LbTab *tab = malloc(sz);
    if (fread(tab, 1, sz, fp) != sz) {
      fprintf(stderr, "main.c: short read from %s\n", fname);
      exit(1);
    }
    fclose(fp);
    lb_tab[ss][dim] = (const struct LbTab (*)[3][2][2][6])tab;
  }
}
enum { LB_BUF = ((2*4+BS)*(2*4+BS) + (BS/2+4+3)*(BS/2+4+3)) * 2 };
static void lb_load(Real *m, int dim, int blk_offset, int ss, long long info_idx) {
  struct Blk *info = &sim.blk[info_idx];
  int nm = 2 * ss + BS;
  int nc = BS / 2 + ss + 3;
  int level = info->level;
  int xi = info->ix, yi = info->iy;
  const struct LbTab (*cflb_tab)[3][2][2][6] = lb_tab[ss][dim];

  Real *p0 = BLK(info_idx) + BS * BS * blk_offset;
  for (int i = 0; i < BS; i++)
    memcpy(m + dim * ((i + ss) * nm + ss), p0 + dim * BS * i,
           BS * dim * sizeof(Real));

  Real *c = m + nm * nm * dim;
  Real *dst[2] = {m, c};

  struct {
    const struct LbTab *e;
    Real *blk[2];
  } dirs[8];
  int nd = 0;
  for (int icode = 0; icode < 9; icode++) {
    int cx = icode % 3 - 1, cy = icode / 3 - 1;
    if (!cx && !cy)
      continue;
    struct Nb nr = nb_find(level, xi, yi, icode);
    const struct LbTab *te =
        &cflb_tab[cx + 1][cy + 1][xi % 2][yi % 2][nr.s];
    Real *blk[2] = {NULL, NULL};
    for (int b = 0; b < te->n_blk; b++) {
      const struct LbSrc *bs = &te->blk_src[b];
      if (bs->is_self) {
        blk[b] = dst[bs->self_idx];
      } else if (bs->level_delta == 1) {
        blk[b] = BLK(nr.ch[b]) + BS * BS * blk_offset;
      } else {
        blk[b] = BLK(nr.idx) + BS * BS * blk_offset;
      }
    }
    dirs[nd].e = te;
    dirs[nd].blk[0] = blk[0];
    dirs[nd].blk[1] = blk[1];
    nd++;
  }

  for (int i = 0; i < nd; i++)
    lb_exec(dirs[i].blk, dst, dirs[i].e->ops, dirs[i].e->n_pre,
                 dim, nm, nc);
  for (int i = 0; i < nd; i++)
    lb_exec(dirs[i].blk, dst, dirs[i].e->ops + MAX_PRE,
                 dirs[i].e->n_post, dim, nm, nc);
}

static void cm_vort() {
#pragma omp parallel
  {
    Real um[LB_BUF];
#pragma omp for nowait
    for (long long id = 0; id < sim.n; ++id) {
      lb_load(um, 2, F_VEL, 1, id);
      struct Blk *info = &sim.blk[id];
      Real i2h = 0.5 * (1 << info->level) * BS;
      Real *TMP = BLK(id) + BS * BS * F_TMP;
      int ss = 1, nm = 2 * ss + BS;
      for (int j = 0; j < BS; ++j)
        for (int i = 0; i < BS; ++i) {
#define V(dx, dy, c) um[2 * (nm * (j + ss + (dy)) + i + ss + (dx)) + (c)]
          TMP[j * BS + i] = i2h * (V(0,-1,0) - V(0,1,0) + V(1,0,1) - V(-1,0,1));
#undef V
        }
    }
  }
}
static void dump(Real time, char *path) {
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
    fprintf(stderr, "main.c: output path '%s' is too long\n", path);
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
  for (size_t i = 0; i < NVARS; i++)
    if (fld_t[i].prefix != NULL) {
      if (snprintf(attr_path, sizeof attr_path, "%s.%s.raw", path,
                   fld_t[i].prefix) > (long)sizeof attr_path) {
        fprintf(stderr, "main.c: output path '%s' is too long\n", path);
        exit(1);
      }
      int dim = fld_t[i].dim;
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
              dim == 2 ? "Vector" : "Scalar", fld_t[i].prefix, BS * BS * sim.n,
              dim, sizeof(Real), attr_path + (xyz_path - xyz_base));
    }
  fprintf(xdmf, "    </Grid>\n"
                "  </Domain>\n"
                "</Xdmf>\n");
  fclose(xdmf);
  file = fopen(xyz_path, "wb");
  for (i = 0; i < sim.n; i++) {
    struct Blk *info = &sim.blk[i];
    k = 0;
    for (y = 0; y < BS; y++)
      for (x = 0; x < BS; x++) {
        double u0, v0, u1, v1, h;
        h = info->h;
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

  for (size_t i = 0; i < NVARS; i++)
    if (fld_t[i].prefix != NULL) {
      int dim = fld_t[i].dim;
      int offset = fld_t[i].offset;
      if (snprintf(attr_path, sizeof attr_path, "%s.%s.raw", path,
                   fld_t[i].prefix) >= (long)sizeof attr_path) {
        fprintf(stderr, "main.c: output path '%s' is too long\n", path);
        exit(1);
      }
      file = fopen(attr_path, "wb");
      for (j = 0; j < sim.n; j++)
        fwrite(BLK(j) + offset * BS * BS, sizeof(Real),
               dim * BS * BS, file);
      fclose(file);
    }
}
static const Real ad_ref_w[4][9] = {
  { 1./64, 10./64, -1./64, 10./64, 56./64, -6./64, -1./64, -6./64,  1./64},
  {-1./64, 10./64,  1./64, -6./64, 56./64, 10./64,  1./64, -6./64, -1./64},
  {-1./64, -6./64,  1./64, 10./64, 56./64, -6./64,  1./64, 10./64, -1./64},
  { 1./64, -6./64, -1./64, -6./64, 56./64, 10./64, -1./64, 10./64,  1./64},
};
static const int ad_sib_ic[4] = {-1, 5, 7, 8};
static int ad_run() {
  cm_vort();
  enum AdSt *state = calloc(sim.n, sizeof *state);
  long long *ref_idx = malloc(sim.n * sizeof *ref_idx);
  long long *com_idx = malloc(sim.n * sizeof *com_idx);
  long long n_ref = 0, n_com = 0;
  int Changed = 0;

#pragma omp parallel for reduction(|| : Changed)
  for (long long i = 0; i < sim.n; i++) {
    Real *b = BLK(i) + BS * BS * F_TMP;
    double Linf = 0.0;
    for (int j = 0; j < BS * BS; j++)
      Linf = fmax(Linf, fabs(b[j]));
    int lev = sim.blk[i].level;
    state[i] = Linf > sim.Rtol && lev < sim.levelMax - 1 ? Refine
             : Linf < sim.Ctol && lev > 0                ? Compress
             : Leave;
    Changed |= state[i] != Leave;
  }
  if (!Changed)
    goto done;

  for (int More = 1; More;) {
    More = 0;
    for (long long j = 0; j < sim.n; j++) {
      if (state[j] != Refine) continue;
      struct Blk *bj = &sim.blk[j];
      for (int ic = 0; ic < 9; ic++) {
        if (ic == 4) continue;
        struct Nb nr = nb_find(bj->level, bj->ix, bj->iy, ic);
        if (nr.s >= 3 || nr.idx < 0) continue;
        if (nr.s == 2 && state[nr.idx] != Refine)
          { state[nr.idx] = Refine; More = 1; }
        else if (nr.s == 0 && state[nr.idx] == Compress)
          state[nr.idx] = Leave;
      }
    }
  }

  for (long long j = 0; j < sim.n; j++) {
    if (state[j] != Compress) continue;
    struct Blk *bj = &sim.blk[j];
    if ((bj->ix | bj->iy) & 1) continue;
    long long sib[4] = {j};
    int ok = 1;
    for (int s = 1; s < 4 && ok; s++) {
      struct Nb nr = nb_find(bj->level, bj->ix, bj->iy, ad_sib_ic[s]);
      ok = nr.s == 0 && nr.idx >= 0 && state[nr.idx] == Compress;
      sib[s] = nr.idx;
    }
    for (int s = 0; s < 4 && ok; s++) {
      struct Blk *bs = &sim.blk[sib[s]];
      for (int ic = 0; ic < 9 && ok; ic++)
        if (ic != 4) ok = nb_find(bs->level, bs->ix, bs->iy, ic).s != 1;
    }
    if (!ok) state[j] = Leave;
  }
  for (long long j = 0; j < sim.n; j++)
    if (state[j] == Refine)
      ref_idx[n_ref++] = j;
    else if (state[j] == Compress && !((sim.blk[j].ix | sim.blk[j].iy) & 1))
      com_idx[n_com++] = j;
  fprintf(stderr, "%s:%d: com/ref: %lld %lld\n", __FILE__, __LINE__,
          n_com, n_ref);
  if (n_ref == 0 && n_com == 0)
    goto done;

  long long nprev = sim.n;
  sim.n += 4 * n_ref;
  sim.blk = realloc(sim.blk, sim.n * sizeof *sim.blk);
  sim.fld = realloc(sim.fld, sim.n * BLK_S * sizeof(Real));
  memset(BLK(nprev), 0, 4 * n_ref * BLK_S * sizeof(Real));
  state = realloc(state, sim.n * sizeof *state);
  for (long long i = nprev; i < sim.n; i++) state[i] = Leave;

#pragma omp parallel
  {
    Real lm0[LB_BUF], lm1[LB_BUF];
    Real *lm[2] = {lm0, lm1};
#pragma omp for
    for (long long k = 0; k < n_ref; k++) {
      struct Blk *par = &sim.blk[ref_idx[k]];
      int px = par->ix, py = par->iy;
      Real *blks[4];
      for (int J = 0; J < 2; J++)
        for (int I = 0; I < 2; I++) {
          long long ci = nprev + 4 * k + 2 * J + I;
          bl_fill(&sim.blk[ci], par->level + 1, 2 * px + I, 2 * py + J);
          blks[2 * J + I] = BLK(ci);
        }
      int nm = 2 + BS;
      for (size_t m = 0; m < NVARS; m++) {
        int dim = fld_t[m].dim;
        int offset = fld_t[m].offset;
        lb_load(lm[dim - 1], dim, offset, 1, ref_idx[k]);
        Real *um = lm[dim - 1];
        for (int J = 0; J < 2; J++)
          for (int I = 0; I < 2; I++) {
            Real *b = blks[J * 2 + I] + offset * BS * BS;
            for (int j = 0; j < BS; j += 2)
              for (int i = 0; i < BS; i += 2) {
                int i0 = i / 2 + I * (BS / 2) + 1;
                int j0 = j / 2 + J * (BS / 2) + 1;
                int sub[4] = {BS*j+i, BS*j+i+1, BS*(j+1)+i, BS*(j+1)+i+1};
                for (int s = 0; s < 4; s++)
                  for (int d = 0; d < dim; d++) {
                    Real val = 0;
                    for (int kk = 0; kk < 9; kk++)
                      val += ad_ref_w[s][kk] * um[dim*(nm*(j0+kk/3-1)+i0+kk%3-1)+d];
                    b[dim * sub[s] + d] = val;
                  }
              }
          }
      }
      state[ref_idx[k]] = Dealloc;
    }
#pragma omp for
    for (long long k = 0; k < n_com; k++) {
      long long ci = com_idx[k];
      struct Blk *p0 = &sim.blk[ci];
      int level = p0->level, x = p0->ix, y = p0->iy;
      Real *blk[4] = {BLK(ci)};
      for (int s = 1; s < 4; s++) {
        int si = nb_find(level, x, y, ad_sib_ic[s]).idx;
        blk[s] = BLK(si);
        state[si] = Dealloc;
      }
      for (size_t v = 0; v < NVARS; v++) {
        int dim = fld_t[v].dim, off = fld_t[v].offset;
        Real *dst = blk[0] + off * BS * BS;
        for (int J = 0; J < 2; J++)
          for (int I = 0; I < 2; I++) {
            Real *src = blk[J * 2 + I] + off * BS * BS;
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
      bl_fill(p0, level - 1, x / 2, y / 2);
    }
  }

  long long cnt = 0;
  for (long long i = 0; i < sim.n; i++) {
    if (state[i] == Dealloc) continue;
    if (cnt != i) {
      memmove(BLK(cnt), BLK(i), BLK_S * sizeof(Real));
      sim.blk[cnt] = sim.blk[i];
    }
    cnt++;
  }
  sim.n = cnt;
  sim.blk = realloc(sim.blk, sim.n * sizeof *sim.blk);
  sim.fld = realloc(sim.fld, sim.n * BLK_S * sizeof(Real));
  hm_rebuild();

done:
  free(state);
  free(ref_idx);
  free(com_idx);
  return Changed;
}
static void cm_advd() {
#pragma omp parallel
  {
    Real um[LB_BUF];
#pragma omp for nowait
    for (long long id = 0; id < sim.n; ++id) {
      lb_load(um, 2, F_VEL, 1, id);
      struct Blk *info = &sim.blk[id];
      Real h = info->h;
      Real dfac = sim.nu * sim.dt;
      Real afac = -0.5 * sim.dt * h;
      Real *TMP = BLK(id) + BS * BS * F_TMV;
      int ss = 1, nm = 2 * ss + BS;
      for (int iy = 0; iy < BS; ++iy)
        for (int ix = 0; ix < BS; ++ix) {
#define V(dx, dy, c) um[2 * (nm * (iy + ss + (dy)) + ix + ss + (dx)) + (c)]
          Real u = V(0,0,0), v = V(0,0,1);
          TMP[2 * (BS * iy + ix)]     = afac * (u * (V(1,0,0) - V(-1,0,0)) + v * (V(0,1,0) - V(0,-1,0))) + dfac * (V(1,0,0) + V(-1,0,0) + V(0,1,0) + V(0,-1,0) - 4*u);
          TMP[2 * (BS * iy + ix) + 1] = afac * (u * (V(1,0,1) - V(-1,0,1)) + v * (V(0,1,1) - V(0,-1,1))) + dfac * (V(1,0,1) + V(-1,0,1) + V(0,1,1) + V(0,-1,1) - 4*v);
#undef V
        }
    }
  }
}
struct PsOp {
  int8_t blk_ref, cell_ix, cell_iy, _pad;
  float coeff;
};
enum { PS_MAX_OPS = 16 };
struct PsEnt {
  int32_t n_ops;
  struct PsOp ops[PS_MAX_OPS];
};
static const struct PsEnt *ps_tab;
static void ps_load() {
  FILE *fp = fopen("tab_poisson.bin", "rb");
  if (!fp) { fprintf(stderr, "main.c: cannot open tab_poisson.bin\n"); exit(1); }
  size_t sz = 4 * BS * 2 * 4 * sizeof(struct PsEnt);
  struct PsEnt *tab = malloc(sz);
  if (fread(tab, 1, sz, fp) != sz) {
    fprintf(stderr, "main.c: short read from tab_poisson.bin\n"); exit(1);
  }
  fclose(fp);
  ps_tab = tab;
}
static void ps_lapl() {
#pragma omp parallel
  {
    Real um[LB_BUF];
#pragma omp for nowait
    for (long long id = 0; id < sim.n; ++id) {
      lb_load(um, 1, F_POL, 1, id);
      Real *TMP = BLK(id) + BS * BS * F_TMP;
      int ss = 1, nm = 2 * ss + BS;
      for (int iy = 0; iy < BS; ++iy)
        for (int ix = 0; ix < BS; ++ix) {
#define P(dx, dy) um[nm * (iy + ss + (dy)) + ix + ss + (dx)]
          TMP[BS * iy + ix] -= P(-1,0) + P(1,0) + P(0,-1) + P(0,1) - 4 * P(0,0);
#undef P
        }
    }
  }
}
static const struct {
  const char *name;
  int type;
  size_t off;
} param_tab[] = {
    {"levelMax", 0, offsetof(struct Sim, levelMax)},
    {"AdaptSteps", 0, offsetof(struct Sim, AdaptSteps)},
    {"levelStart", 0, offsetof(struct Sim, levelStart)},
    {"maxPoissonRestarts", 0, offsetof(struct Sim, maxPoissonRestarts)},
    {"Rtol", 1, offsetof(struct Sim, Rtol)},
    {"Ctol", 1, offsetof(struct Sim, Ctol)},
    {"CFL", 1, offsetof(struct Sim, CFL)},
    {"tend", 1, offsetof(struct Sim, endTime)},
    {"nu", 1, offsetof(struct Sim, nu)},
    {"poissonTol", 1, offsetof(struct Sim, PoissonTol)},
    {"poissonTolRel", 1, offsetof(struct Sim, PoissonTolRel)},
    {"tdump", 1, offsetof(struct Sim, dumpTime)},
};
int main(int argc, char **argv) {
#ifdef _OPENMP
#pragma omp parallel
#pragma omp master
  fprintf(stderr, "main.c: %d threads\n", omp_get_num_threads());
#endif
  char *base = (char *)&sim;
  for (size_t i = 0; i < sizeof param_tab / sizeof *param_tab; i++)
    if (param_tab[i].type == 0)
      *(int *)(base + param_tab[i].off) = arg_i(argc, argv, param_tab[i].name);
    else
      *(Real *)(base + param_tab[i].off) = arg_r(argc, argv, param_tab[i].name);
  {
    int ns = 1 << sim.levelStart;
    sim.n = (long long)ns * ns;
    sim.blk = calloc(sim.n, sizeof *sim.blk);
    sim.fld = calloc(sim.n * BLK_S, sizeof(Real));
    long long idx = 0;
    for (int iy = 0; iy < ns; iy++)
      for (int ix = 0; ix < ns; ix++)
        bl_fill(&sim.blk[idx++], sim.levelStart, ix, iy);
  }
  hm_rebuild();
  lb_init();
#pragma omp parallel for
  for (long long i = 0; i < sim.n; i++) {
    struct Blk *info = &sim.blk[i];
    Real *vel = BLK(i) + BS * BS * F_VEL;
    for (int iy = 0; iy < BS; iy++)
      for (int ix = 0; ix < BS; ix++) {
        Real px = info->origin[0] + info->h * (ix + 0.5);
        Real py = info->origin[1] + info->h * (iy + 0.5);
        int j = BS * iy + ix;
        Real rho = 30.0;
        vel[2 * j + 0] = py <= 0.5 ? tanh(rho * (py - 0.25))
                                    : tanh(rho * (0.75 - py));
        vel[2 * j + 1] = 0.05 * sin(2 * M_PI * px);
      }
  }
  int Changed = 0;
  for (int i = 0; i < sim.levelMax; i++)
    Changed = ad_run() || Changed;
  double P_inv[BS * BS * BS * BS];
  ps_prec(P_inv);
  sim.solver = solver_create(BS * BS, P_inv);
  sim.coo_val = NULL; sim.coo_row = NULL; sim.coo_col = NULL;
  sim.sol_x = NULL; sim.sol_b = NULL; sim.sol_h2 = NULL;
  sim.coo_cap = 0;
  ps_load();
  while (1) {
    if (sim.step % 5 == 0)
      fprintf(stderr, "main.c: %08d %.16e\n", sim.step, sim.time);
    if (sim.dumpTime > 0 && sim.time >= sim.nextDumpTime) {
      sim.nextDumpTime += sim.dumpTime;
      cm_vort();
      char path[FILENAME_MAX];
      snprintf(path, sizeof path, "vel.%08d", sim.dump_count++);
      dump(sim.time, path);
    }
    if (sim.endTime > 0 && sim.time >= sim.endTime)
      break;
    Real CFL = sim.CFL;
    Real h = INFINITY;
    for (long long i = 0; i < sim.n; i++)
      h = fmin(sim.blk[i].h, h);
    Real umax = 0;
#pragma omp parallel for schedule(static) reduction(max : umax)
    for (long long i = 0; i < sim.n; i++) {
      Real *vel = BLK(i) + BS * BS * F_VEL;
      for (int j = 0; j < 2 * BS * BS; j++)
        umax = fmax(umax, fabs(vel[j]));
    }
    sim.dt = fmin(CFL * h / (umax + 1e-8),
                      0.25 * h * h / (sim.nu + 0.25 * h * umax));
    if (sim.step <= 10 || sim.step % sim.AdaptSteps == 0)
      Changed = ad_run() || Changed;
#pragma omp parallel for
    for (long long i = 0; i < sim.n; i++)
      memcpy(BLK(i) + BS * BS * F_VOL,
             BLK(i) + BS * BS * F_VEL,
             2 * BS * BS * sizeof(Real));
    for (int rk = 0; rk < 2; rk++) {
      Real fac = rk ? 1.0 : 0.5;
      cm_advd();
#pragma omp parallel for
      for (long long i = 0; i < sim.n; i++) {
        Real *V = BLK(i) + BS * BS * F_VEL;
        Real *Vold = BLK(i) + BS * BS * F_VOL;
        Real *tmpV = BLK(i) + BS * BS * F_TMV;
        Real ih2 = fac / (sim.blk[i].h * sim.blk[i].h);
        for (int j = 0; j < 2 * BS * BS; j++)
          V[j] = Vold[j] + tmpV[j] * ih2;
      }
    }
#pragma omp parallel
    {
      Real um[LB_BUF];
#pragma omp for
      for (int i = 0; i < sim.n; i++) {
        lb_load(um, 2, F_VEL, 1, i);
        int ss = 1, nm = 2 * ss + BS;
        Real facDiv = 0.5 * sim.blk[i].h / sim.dt;
        Real *TMP = BLK(i) + BS * BS * F_TMP;
        for (int iy = 0; iy < BS; ++iy)
          for (int ix = 0; ix < BS; ++ix) {
#define V(dx, dy, c) um[2 * (nm * (iy + ss + (dy)) + ix + ss + (dx)) + (c)]
            TMP[BS * iy + ix] = facDiv * (V(1,0,0) - V(-1,0,0) + V(0,1,1) - V(0,-1,1));
#undef V
          }
      }
    }
#pragma omp parallel for
    for (long long i = 0; i < sim.n; i++) {
      memcpy(BLK(i) + BS * BS * F_POL,
             BLK(i) + BS * BS * F_PRS, BS * BS * sizeof(Real));
      memset(BLK(i) + BS * BS * F_PRS, 0,
             BS * BS * sizeof(Real));
    }
    ps_lapl();
    double max_error = sim.step < 10 ? 0.0 : sim.PoissonTol;
    double max_rel_error = sim.step < 10 ? 0.0 : sim.PoissonTolRel;
    int max_restarts = sim.step < 10 ? 100 : sim.maxPoissonRestarts;
    int N = BS * BS * sim.n;
    sim.sol_x = realloc(sim.sol_x, N * sizeof(double));
    sim.sol_b = realloc(sim.sol_b, N * sizeof(double));
    sim.sol_h2 = realloc(sim.sol_h2, sim.n * sizeof(double));
    sim.coo_nnz = 0;
    if (sim.coo_cap < 16 * N) {
      sim.coo_cap = 16 * N;
      sim.coo_val = realloc(sim.coo_val, sim.coo_cap * sizeof(double));
      sim.coo_row = realloc(sim.coo_row, sim.coo_cap * sizeof(int));
      sim.coo_col = realloc(sim.coo_col, sim.coo_cap * sizeof(int));
    }
    static const int ps_ic[4] = {3, 5, 1, 7};
    static const int ps_doff[4] = {-1, 1, -BS, BS};
#define COO(v, r, c) do { \
    assert(sim.coo_nnz < sim.coo_cap); \
    sim.coo_val[sim.coo_nnz] = (v); \
    sim.coo_row[sim.coo_nnz] = (r); \
    sim.coo_col[sim.coo_nnz] = (c); \
    sim.coo_nnz++; \
  } while(0)
    for (int i = 0; i < sim.n; i++) {
      struct Blk *info = &sim.blk[i];
      int n = info->n, bix = info->ix, biy = info->iy;
      for (int iy = 0; iy < BS; iy++)
        for (int ix = 0; ix < BS; ix++) {
          int sfc = i * BS * BS + iy * BS + ix;
          if (ix > 0 && ix < BS-1 && iy > 0 && iy < BS-1) {
            COO(-4, sfc, sfc);
            for (int j = 0; j < 4; j++)
              COO(1, sfc, sfc + ps_doff[j]);
            continue;
          }
          int xy[2] = {ix, iy}, bxy[2] = {bix, biy};
          for (int j = 0; j < 4; j++) {
            int d = j/2, s = j&1, ec = xy[d], tc = xy[1-d];
            if (s ? ec < BS-1 : ec > 0) {
              COO(1, sfc, sfc + ps_doff[j]);
              COO(-1, sfc, sfc);
              continue;
            }
            if (s ? bxy[d] == n-1 : bxy[d] == 0)
              continue;
            struct Nb pnr = nb_find(info->level, bix, biy, ps_ic[j]);
            long long bi[4] = {i};
            int st;
            if (pnr.s == 0)      { st = 1; bi[1] = pnr.idx; }
            else if (pnr.s == 2) { st = 2; bi[2] = pnr.idx; }
            else                  { st = 3; bi[3] = pnr.ch[tc >= BS/2]; }
            const struct PsEnt *pe =
                &ps_tab[((j*BS + tc)*2 + bxy[1-d]%2)*4 + st];
            for (int k = 0; k < pe->n_ops; k++) {
              const struct PsOp *op = &pe->ops[k];
              COO((double)op->coeff, sfc,
                  (int)(bi[op->blk_ref]*BS*BS + op->cell_iy*BS + op->cell_ix));
            }
          }
        }
    }
#undef COO
#pragma omp parallel for
    for (int i = 0; i < sim.n; i++) {
      Real h = sim.blk[i].h;
      sim.sol_h2[i] = h * h;
      long long offset = (long long)i * BS * BS;
      memcpy(&sim.sol_b[offset], BLK(i) + BS * BS * F_TMP,
             BS * BS * sizeof(Real));
      memcpy(&sim.sol_x[offset], BLK(i) + BS * BS * F_PRS,
             BS * BS * sizeof(Real));
    }
    solver_solve(sim.solver, Changed, N, sim.coo_nnz,
        sim.coo_val, sim.coo_row, sim.coo_col,
        sim.sol_x, sim.sol_b, sim.sol_h2, -1,
        max_error, max_rel_error, max_restarts);
    Changed = 0;
    Real avg = 0, avg1 = 0;
#pragma omp parallel for reduction(+ : avg, avg1)
    for (long long i = 0; i < sim.n; i++) {
      Real *P = BLK(i) + BS * BS * F_PRS;
      Real vv = sim.blk[i].h * sim.blk[i].h;
      for (int j = 0; j < BS * BS; j++) {
        P[j] = sim.sol_x[i * BS * BS + j];
        avg += P[j] * vv;
        avg1 += vv;
      }
    }
    avg /= avg1;
#pragma omp parallel for
    for (long long i = 0; i < sim.n; i++) {
      Real *pres = BLK(i) + BS * BS * F_PRS;
      Real *pold = BLK(i) + BS * BS * F_POL;
      for (int j = 0; j < BS * BS; j++)
        pres[j] += pold[j] - avg;
    }
#pragma omp parallel
    {
      Real um[LB_BUF];
#pragma omp for nowait
      for (long long id = 0; id < sim.n; ++id) {
        lb_load(um, 1, F_PRS, 1, id);
        struct Blk *info = &sim.blk[id];
        int ss = 1, nm = 2 * ss + BS;
        Real pFac = -0.5 * sim.dt * info->h;
        Real *tmpV = BLK(id) + BS * BS * F_TMV;
        for (int iy = 0; iy < BS; ++iy)
          for (int ix = 0; ix < BS; ++ix) {
#define P(dx, dy) um[nm * (iy + ss + (dy)) + ix + ss + (dx)]
            tmpV[2 * (BS * iy + ix)]     = pFac * (P(1,0) - P(-1,0));
            tmpV[2 * (BS * iy + ix) + 1] = pFac * (P(0,1) - P(0,-1));
#undef P
          }
      }
    }
#pragma omp parallel for
    for (long long i = 0; i < sim.n; i++) {
      Real ih2 = 1.0 / sim.blk[i].h / sim.blk[i].h;
      Real *V = BLK(i) + BS * BS * F_VEL;
      Real *tmpV = BLK(i) + BS * BS * F_TMV;
      for (int j = 0; j < 2 * BS * BS; j++)
        V[j] += tmpV[j] * ih2;
    }
    sim.time += sim.dt;
    sim.step++;
  }

  solver_destroy(sim.solver);
  free(sim.coo_val); free(sim.coo_row); free(sim.coo_col);
  free(sim.sol_x); free(sim.sol_b); free(sim.sol_h2);
  fprintf(stderr, "main.c: end\n");
}
