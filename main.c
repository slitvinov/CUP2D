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
  off_vel = 0,
  off_pres = 2,
  off_chi = 3,
  off_vold = 4,
  off_tmp = 6,
  off_pold = 7,
  off_tmpV = 8,
  off_n = 10,
  BSTRIDE = off_n *BS *BS,
};

#define EPS DBL_EPSILON
enum State { Leave = 0, Refine = 1, Compress = -1, Dealloc = 2 };
struct Shape;
struct Info;
struct LkEntry {
  long long key;
  int val;
};
static int lk_cmp(const void *a, const void *b) {
  long long ka = ((const struct LkEntry *)a)->key;
  long long kb = ((const struct LkEntry *)b)->key;
  return (ka > kb) - (ka < kb);
}
static int lk_find(long long key, const struct LkEntry *lk, int n) {
  int lo = 0, hi = n - 1;
  while (lo <= hi) {
    int mid = lo + (hi - lo) / 2;
    if (lk[mid].key < key) lo = mid + 1;
    else if (lk[mid].key > key) hi = mid - 1;
    else return lk[mid].val;
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
  Real lambda;
  Real nextDumpTime;
  Real nu;
  Real PoissonTol;
  Real PoissonTolRel;
  Real Rtol;
  Real time;
  struct Shape **shapes;
  struct Solver *solver;
  int coo_nnz, coo_cap;
  double *coo_val;
  int *coo_row, *coo_col;
  double *sol_x, *sol_b, *sol_h2;
  long long n;
  int nshape;
  struct LkEntry *lk;
  int lk_n;
  struct Info *infos;
  Real *blocks;
} sim;
static long long level_id(int level, long long Z) {
  return ((1LL << (2 * level)) - 1) / 3 + Z;
}
static Real real_min(Real a, Real b) { return a < b ? a : b; }
static double getA_local(int I1, int I2) {
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
  return U > 0 ? weno5_plus(um2, um1, u, up1, up2) -
                     weno5_plus(um3, um2, um1, u, up1)
               : weno5_minus(um1, u, up1, up2, up3) -
                     weno5_minus(um2, um1, u, up1, up2);
}
static void sfc_rot(long long n, int *x, int *y, long long rx, long long ry) {
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
static long long sfc_forward(const int l, int i, int j) {
  if (l >= sim.levelMax)
    return 0;
  int n = 1 << l;
  int rx, ry, s, d = 0;
  for (s = n / 2; s > 0; s /= 2) {
    rx = (i & s) > 0;
    ry = (j & s) > 0;
    d += s * s * ((3 * rx) ^ ry);
    sfc_rot(n, &i, &j, rx, ry);
  }
  return d;
}
static void sfc_inverse(long long Z, int l, int *i, int *j) {
  int n = 1 << l;
  long long rx, ry, s;
  *i = 0;
  *j = 0;
  for (s = 1; s < n; s *= 2) {
    rx = 1 & (Z / 2);
    ry = 1 & (Z ^ rx);
    sfc_rot(s, i, j, rx, ry);
    *i += s * rx;
    *j += s * ry;
    Z /= 4;
  }
}
static long long forward(int level, int i, int j) {
  return sfc_forward(level, i % (1 << level), j % (1 << level));
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
static Real arg_real(int argc, char **argv, const char *key) {
  const char *s = arg_find(argc, argv, key);
  char *end;
  Real v = strtod(s, &end);
  if (end == s || *end != '\0') {
    fprintf(stderr, "main.c: error: -%s: bad real '%s'\n", key, s);
    exit(1);
  }
  return v;
}
static int arg_int(int argc, char **argv, const char *key) {
  const char *s = arg_find(argc, argv, key);
  char *end;
  long v = strtol(s, &end, 10);
  if (end == s || *end != '\0') {
    fprintf(stderr, "main.c: error: -%s: bad integer '%s'\n", key, s);
    exit(1);
  }
  return (int)v;
}
static int kv_sep(int c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }
static const char *kv_find(const char *line, const char *key) {
  const char *p = line;
  size_t klen = strlen(key);
  while (*p) {
    while (kv_sep(*p)) p++;
    if (strncmp(p, key, klen) == 0 && p[klen] == '=')
      return p + klen + 1;
    while (*p && !kv_sep(*p)) p++;
  }
  fprintf(stderr, "main.c: error: key '%s' not found in '%s'\n", key, line);
  exit(1);
}
static Real kv_real(const char *line, const char *key) {
  const char *s = kv_find(line, key);
  char *end;
  Real v = strtod(s, &end);
  if (end == s) {
    fprintf(stderr, "main.c: error: %s: bad real in '%s'\n", key, line);
    exit(1);
  }
  return v;
}
static const char *kv_str(const char *line, const char *key, char *buf, size_t n) {
  const char *s = kv_find(line, key);
  size_t i = 0;
  while (s[i] && !kv_sep(s[i]) && i < n - 1) { buf[i] = s[i]; i++; }
  buf[i] = '\0';
  return buf;
}
static void precond(double *P_inv) {
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
    L[i][i] = sqrt(getA_local(i, i) - s1);
    for (int j = i + 1; j < BS * BS; j++) {
      double s2 = 0;
      for (int k = 0; k <= i - 1; k++)
        s2 += L[i][k] * L[j][k];
      L[j][i] = (getA_local(j, i) - s2) / L[i][i];
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
struct Info {
  double h, origin[2];
  int level, ix, iy;
  long long Z;
  int parent;
  int children[4];
  int nb[9];
  int8_t nb_s[9];
};
#define BLK(i) (sim.blocks + (long long)(i) * BSTRIDE)
struct Collision {
  Real iM, ivecX, ivecY, jM, jvecX, jvecY;
};
static void fill(struct Info *b, int level, long long Z) {
  int n = 1 << level;
  sfc_inverse(Z, level, &b->ix, &b->iy);
  b->level = level;
  b->Z = Z;
  b->h = 1.0 / BS / n;
  b->origin[0] = (Real)b->ix / n;
  b->origin[1] = (Real)b->iy / n;
}
static long long getf0(int level, long long Z) {
  int idx = lk_find(level_id(level, Z), sim.lk, sim.lk_n);
  if (idx < 0) {
    fprintf(stderr, "main.c: getf0: level=%d Z=%lld not found\n", level, Z);
    abort();
  }
  return idx;
}
struct {
  int offset;
  int dim;
  const char *prefix;
} vars[] = {{off_vel, 2, "vel"}, {off_pres, 1, "pres"}, {off_chi, 1, "chi"},
            {off_vold, 2, NULL}, {off_tmp, 1, "tmp"},   {off_pold, 1, NULL},
            {off_tmpV, 2, NULL}};
enum { NVARS = sizeof vars / sizeof *vars };

static inline int skin_skip(int c, int coord, int n) {
  int skin = coord == 0 || coord == n - 1;
  int skip = coord == 0 ? -1 : 1;
  return c == skip && skin;
}
static void build_tree() {
  sim.lk = realloc(sim.lk, sim.n * sizeof *sim.lk);
  sim.lk_n = sim.n;
  for (long long i = 0; i < sim.n; i++) {
    sim.lk[i].key = level_id(sim.infos[i].level, sim.infos[i].Z);
    sim.lk[i].val = i;
  }
  qsort(sim.lk, sim.lk_n, sizeof *sim.lk, lk_cmp);
  for (long long i = 0; i < sim.n; i++) {
    struct Info *info = &sim.infos[i];
    int level = info->level;
    int n = 1 << level;
    info->parent = -1;
    if (level > 0)
      info->parent = lk_find(level_id(level - 1, info->Z / 4),
                              sim.lk, sim.lk_n);
    for (int s = 0; s < 4; s++) info->children[s] = -1;
    if (level + 1 < sim.levelMax)
      for (int J = 0; J < 2; J++)
        for (int I = 0; I < 2; I++) {
          long long Zc = sfc_forward(level + 1,
                                      2 * info->ix + I, 2 * info->iy + J);
          info->children[2 * J + I] =
              lk_find(level_id(level + 1, Zc), sim.lk, sim.lk_n);
        }
    for (int icode = 0; icode < 9; icode++) {
      info->nb[icode] = -1;
      info->nb_s[icode] = 0;
    }
    for (int icode = 0; icode < 9; icode++) {
      int cx = icode % 3 - 1, cy = icode / 3 - 1;
      if (!cx && !cy) continue;
      int xskin = skin_skip(cx, info->ix, n);
      int yskin = skin_skip(cy, info->iy, n);
      if (xskin && yskin) {
        info->nb_s[icode] = 5;
      } else if (xskin) {
        info->nb_s[icode] = 3;
      } else if (yskin) {
        info->nb_s[icode] = 4;
      } else {
        long long Z = sfc_forward(level,
            (info->ix + cx + n) % n, (info->iy + cy + n) % n);
        int idx = lk_find(level_id(level, Z), sim.lk, sim.lk_n);
        if (idx >= 0) {
          info->nb_s[icode] = 0;
          info->nb[icode] = idx;
        } else if (level > 0 &&
                   (idx = lk_find(level_id(level - 1, Z / 4),
                                   sim.lk, sim.lk_n)) >= 0) {
          info->nb_s[icode] = 2;
          info->nb[icode] = idx;
        } else {
          info->nb_s[icode] = 1;
          info->nb[icode] = -1;
        }
      }
    }
  }
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
struct Op {
  int8_t type;
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
  int8_t is_self;
  int8_t self_idx;
};
enum { MAX_PRE = 32, MAX_POST = 48, MAX_OPS = MAX_PRE + MAX_POST };
struct TabEntry {
  int8_t n_blk;
  struct BlkSrc blk_src[2];
  int8_t _pad;
  int32_t n_pre;
  int32_t n_post;
  struct Op ops[MAX_OPS];
};
static void exec_program(Real *const blk[], Real *const dst[],
                         const struct Op *ops, int n, int dim, int nm, int nc) {
  Real *m = dst[0], *c = dst[1];
  for (int i = 0; i < n; i++) {
    const struct Op *o = &ops[i];
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

static const struct TabEntry (*load_cfg_tab(int ss, int dim))[3][2][2][6] {
  char fname[64];
  snprintf(fname, sizeof fname, "tab_ss%d_dim%d.bin", ss, dim);
  FILE *fp = fopen(fname, "rb");
  if (!fp) {
    fprintf(stderr, "main.c: cannot open %s\n", fname);
    exit(1);
  }
  size_t sz = 3 * 3 * 2 * 2 * 6 * sizeof(struct TabEntry);
  struct TabEntry *tab = malloc(sz);
  if (fread(tab, 1, sz, fp) != sz) {
    fprintf(stderr, "main.c: short read from %s\n", fname);
    exit(1);
  }
  fclose(fp);
  return (const struct TabEntry (*)[3][2][2][6])tab;
}

static const struct TabEntry (*g_tab[5][3])[3][2][2][6];
static void tab_load_all() {
  int configs[][2] = {{1,1}, {1,2}, {3,2}, {4,1}};
  for (int ci = 0; ci < 4; ci++)
    g_tab[configs[ci][0]][configs[ci][1]] = load_cfg_tab(configs[ci][0], configs[ci][1]);
}
enum { LAB_BUF = ((2*4+BS)*(2*4+BS) + (BS/2+4+3)*(BS/2+4+3)) * 2 };
static void lab_load(Real *m, int dim, int blk_offset, int ss, long long info_idx) {
  struct Info *info = &sim.infos[info_idx];
  int nm = 2 * ss + BS;
  int nc = BS / 2 + ss + 3;
  int n = 1 << info->level;
  int level = info->level;
  int xi = info->ix, yi = info->iy;
  const struct TabEntry (*cfg_tab)[3][2][2][6] = g_tab[ss][dim];

  Real *p0 = BLK(info_idx) + BS * BS * blk_offset;
  for (int i = 0; i < BS; i++)
    memcpy(m + dim * ((i + ss) * nm + ss), p0 + dim * BS * i,
           BS * dim * sizeof(Real));

  Real *c = m + nm * nm * dim;
  Real *dst[2] = {m, c};

  struct {
    const struct TabEntry *e;
    Real *blk[2];
  } dirs[8];
  int nd = 0;
  for (int icode = 0; icode < 9; icode++) {
    int cx = icode % 3 - 1, cy = icode / 3 - 1;
    if (!cx && !cy)
      continue;
    int s = info->nb_s[icode];
    const struct TabEntry *te =
        &cfg_tab[cx + 1][cy + 1][xi % 2][yi % 2][s];
    Real *blk[2] = {NULL, NULL};
    for (int b = 0; b < te->n_blk; b++) {
      const struct BlkSrc *bs = &te->blk_src[b];
      if (bs->is_self) {
        blk[b] = dst[bs->self_idx];
      } else {
        int L = level + bs->level_delta;
        int fx = (xi * bs->xi_mul + bs->xi_add) >> bs->xi_shift;
        int fy = (yi * bs->yi_mul + bs->yi_add) >> bs->yi_shift;
        blk[b] = BLK(getf0(L, forward(L, fx, fy))) +
                 BS * BS * blk_offset;
      }
    }
    dirs[nd].e = te;
    dirs[nd].blk[0] = blk[0];
    dirs[nd].blk[1] = blk[1];
    nd++;
  }

  for (int i = 0; i < nd; i++)
    exec_program(dirs[i].blk, dst, dirs[i].e->ops, dirs[i].e->n_pre,
                 dim, nm, nc);
  for (int i = 0; i < nd; i++)
    exec_program(dirs[i].blk, dst, dirs[i].e->ops + MAX_PRE,
                 dirs[i].e->n_post, dim, nm, nc);
}

static void pressure_rhs_fun(Real *vm, Real *um, size_t i) {
  int ss = 1, nm = 2 * ss + BS;
  Real h = sim.infos[i].h;
  Real facDiv = 0.5 * h / sim.dt;
  Real *TMP = BLK(i) + BS * BS * off_tmp;
  Real *CHI = BLK(i) + BS * BS * off_chi;
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
static void compute_vorticity() {
#pragma omp parallel
  {
    Real um[LAB_BUF];
#pragma omp for nowait
    for (long long id = 0; id < sim.n; ++id) {
      lab_load(um, 2, off_vel, 1, id);
      struct Info *info = &sim.infos[id];
      Real i2h = 0.5 * (1 << info->level) * BS;
      Real *TMP = BLK(id) + BS * BS * off_tmp;
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
    if (vars[i].prefix != NULL) {
      if (snprintf(attr_path, sizeof attr_path, "%s.%s.raw", path,
                   vars[i].prefix) > (long)sizeof attr_path) {
        fprintf(stderr, "main.c: output path '%s' is too long\n", path);
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
    struct Info *info = &sim.infos[i];
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

  for (size_t i = 0; i < NVARS; i++)
    if (vars[i].prefix != NULL) {
      int dim = vars[i].dim;
      int offset = vars[i].offset;
      if (snprintf(attr_path, sizeof attr_path, "%s.%s.raw", path,
                   vars[i].prefix) >= (long)sizeof attr_path) {
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
  Real *o_chi;
  Real *o_dist;
  Real *o_udef;
  Real *o_com;
};
static void compute_chi_on_grid() {
#pragma omp parallel
  {
    Real um[LAB_BUF];
#pragma omp for nowait
    for (long long id = 0; id < sim.n; ++id) {
      lab_load(um, 1, off_tmp, 1, id);
      struct Info *info = &sim.infos[id];
      int ss = 1, nm = 2 * ss + BS;
      for (int ishape = 0; ishape < sim.nshape; ishape++) {
        struct Shape *shape = sim.shapes[ishape];
        Real h = 1.0 / BS / (1 << info->level);
        Real h2 = h * h;
        Real *chi = shape->o_chi + id * BS * BS;
        Real *dist = shape->o_dist + id * BS * BS;
        Real *oc = shape->o_com + id * 3;
        oc[0] = oc[1] = oc[2] = 0;
        Real *CHI = BLK(id) + BS * BS * off_chi;
        for (int iy = 0; iy < BS; iy++)
          for (int ix = 0; ix < BS; ix++) {
#define D(dx, dy) um[nm * (iy + ss + (dy)) + ix + ss + (dx)]
            int j = BS * iy + ix;
            if (dist[j] > +h || dist[j] < -h) {
              chi[j] = dist[j] > 0 ? 1 : 0;
            } else {
              Real dpx = D(1,0), dmx = D(-1,0), dpy = D(0,1), dmy = D(0,-1);
              Real gradIX = fmax(0.0, dpx) - fmax(0.0, dmx);
              Real gradIY = fmax(0.0, dpy) - fmax(0.0, dmy);
              Real gradUX = dpx - dmx, gradUY = dpy - dmy;
              chi[j] = (gradIX * gradUX + gradIY * gradUY) /
                       (gradUX * gradUX + gradUY * gradUY + EPS);
            }
#undef D
            CHI[j] = fmax(CHI[j], chi[j]);
            if (chi[j] > 0) {
              Real px = info->origin[0] + info->h * (ix + 0.5);
              Real py = info->origin[1] + info->h * (iy + 0.5);
              oc[0] += chi[j] * h2;
              oc[1] += chi[j] * h2 * (px - shape->x);
              oc[2] += chi[j] * h2 * (py - shape->y);
            }
          }
      }
    }
  }
}
static void ongrid() {
#pragma omp parallel for
  for (long long i = 0; i < sim.n; i++) {
    memset(BLK(i) + BS * BS * off_chi, 0, BS * BS * sizeof(Real));
    Real *p = BLK(i) + BS * BS * off_tmp;
    for (int j = 0; j < BS * BS; j++) p[j] = -1.0;
  }
  for (int ishape = 0; ishape < sim.nshape; ishape++) {
    struct Shape *shape = sim.shapes[ishape];
    shape->o_chi = realloc(shape->o_chi, sim.n * BS * BS * sizeof(Real));
    shape->o_dist = realloc(shape->o_dist, sim.n * BS * BS * sizeof(Real));
    shape->o_udef = realloc(shape->o_udef, sim.n * BS * BS * 2 * sizeof(Real));
    shape->o_com = realloc(shape->o_com, sim.n * 3 * sizeof(Real));
#pragma omp parallel for
    for (long long i = 0; i < sim.n; i++) {
      struct Info *info = &sim.infos[i];
      Real *b = BLK(i) + BS * BS * off_tmp;
      Real h = info->h;
      Real co = cos(shape->orientation);
      Real si = sin(shape->orientation);
      Real *o_chi = shape->o_chi + i * BS * BS;
      Real *o_dist = shape->o_dist + i * BS * BS;
      Real *o_udef = shape->o_udef + i * BS * BS * 2;
      memset(o_chi, 0, BS * BS * sizeof(Real));
      memset(o_udef, 0, BS * BS * 2 * sizeof(Real));
      for (int iy = 0; iy < BS; ++iy)
        for (int ix = 0; ix < BS; ++ix) {
          Real x = info->origin[0] + h * (ix + 0.5) - shape->x;
          Real y = info->origin[1] + h * (iy + 0.5) - shape->y;
          Real x0 = co * x + si * y;
          Real y0 = -si * x + co * y;
          Real r = sqrt(x0 * x0 + y0 * y0);
          Real pa = atan2(y0, x0);
          if (pa < 0)
            pa += 2 * M_PI;
          int ri = r * shape->nr / shape->rmax;
          if (ri >= shape->nr)
            ri = shape->nr - 1;
          int pi = pa * (shape->np - 2) / (2 * M_PI);
          if (pi >= shape->np)
            pi = shape->np - 1;
          Real dist = shape->sdf[ri * shape->np + pi];
          o_dist[iy * BS + ix] = dist;
          b[iy * BS + ix] = fmax(b[iy * BS + ix], dist);
        }
    }
  }

  compute_chi_on_grid();
  for (int ishape = 0; ishape < sim.nshape; ishape++) {
    struct Shape *shape = sim.shapes[ishape];
    Real com[3] = {0.0, 0.0, 0.0};
#pragma omp parallel for reduction(+ : com[ : 3])
    for (long long i = 0; i < sim.n; i++) {
      Real *oc = shape->o_com + i * 3;
      com[0] += oc[0];
      com[1] += oc[1];
      com[2] += oc[2];
    }
    shape->x += com[1] / com[0];
    shape->y += com[2] / com[0];
  }
  for (int ishape = 0; ishape < sim.nshape; ishape++) {
    struct Shape *shape = sim.shapes[ishape];
    Real x = 0, y = 0, m = 0, J = 0, u = 0, v = 0, a = 0;
#pragma omp parallel for reduction(+ : x, y, m, J, u, v, a)
    for (long long i = 0; i < sim.n; i++) {
      Real hsq = sim.infos[i].h * sim.infos[i].h;
      Real *CHI = shape->o_chi + i * BS * BS;
      Real *UDEF = shape->o_udef + i * BS * BS * 2;
      for (int iy = 0; iy < BS; ++iy)
        for (int ix = 0; ix < BS; ++ix) {
          int j = BS * iy + ix;
          if (CHI[j] <= 0)
            continue;
          Real p[2];
          p[0] = sim.infos[i].origin[0] + sim.infos[i].h * (ix + 0.5);
          p[1] = sim.infos[i].origin[1] + sim.infos[i].h * (iy + 0.5);
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
      Real *o_udef = shape->o_udef + i * BS * BS * 2;
      for (int iy = 0; iy < BS; ++iy)
        for (int ix = 0; ix < BS; ++ix) {
          int j = BS * iy + ix;
          Real p[2];
          p[0] = sim.infos[i].origin[0] + sim.infos[i].h * (ix + 0.5);
          p[1] = sim.infos[i].origin[1] + sim.infos[i].h * (iy + 0.5);
          p[0] -= shape->x;
          p[1] -= shape->y;
          o_udef[2 * j + 0] -= u - a * p[1];
          o_udef[2 * j + 1] -= v + a * p[0];
        }
    }
  }
}
static void compute_grad_chi() {
#pragma omp parallel
  {
    Real um[LAB_BUF];
#pragma omp for nowait
    for (long long id = 0; id < sim.n; ++id) {
      lab_load(um, 1, off_chi, 4, id);
      struct Info *info = &sim.infos[id];
      Real *TMP = BLK(id) + BS * BS * off_tmp;
      int offset = (info->level == sim.levelMax - 1) ? 4 : 2;
      int ss = 4, nm = 2 * ss + BS;
      for (int y = -offset; y < BS + offset; ++y)
        for (int x = -offset; x < BS + offset; ++x) {
          int k = nm * (y + ss) + x + ss;
          um[k] = fmin(um[k], 1.0);
          um[k] = fmax(um[k], 0.0);
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
  }
}
static const Real refine_w[4][9] = {
  { 1./64, 10./64, -1./64, 10./64, 56./64, -6./64, -1./64, -6./64,  1./64},
  {-1./64, 10./64,  1./64, -6./64, 56./64, 10./64,  1./64, -6./64, -1./64},
  {-1./64, -6./64,  1./64, 10./64, 56./64, -6./64,  1./64, 10./64, -1./64},
  { 1./64, -6./64, -1./64, -6./64, 56./64, 10./64, -1./64, 10./64,  1./64},
};
static int adapt() {
  compute_vorticity();
  compute_grad_chi();
  enum State *state = malloc(sim.n * sizeof *state);
  int Changed = 0;

#pragma omp parallel for reduction(|| : Changed)
  for (long long i = 0; i < sim.n; i++) {
    Real *b = BLK(i) + BS * BS * off_tmp;
    double Linf = 0.0;
    for (int j = 0; j < BS * BS; j++)
      Linf = fmax(Linf, fabs(b[j]));
    state[i] = Linf > sim.Rtol ? Refine : Linf < sim.Ctol ? Compress : Leave;
    int maxLevel =
        state[i] == Refine && sim.infos[i].level == sim.levelMax - 1;
    int minLevel = state[i] == Compress && sim.infos[i].level == 0;
    if (maxLevel || minLevel)
      state[i] = Leave;
    if (state[i] != Leave)
      Changed = 1;
  }
  if (!Changed)
    goto end;
  for (;;) {
    int More = 0;
    for (long long j = 0; j < sim.n; j++) {
      if (state[j] == Refine) {
        for (int icode = 0; icode < 9; icode++) {
          if (icode == 4) continue;
          int8_t ns = sim.infos[j].nb_s[icode];
          int nb_idx = sim.infos[j].nb[icode];
          if (ns >= 3) continue;
          if (ns == 2) {
            if (nb_idx >= 0 && state[nb_idx] != Refine) {
              state[nb_idx] = Refine;
              More = 1;
            }
          } else if (ns == 0) {
            if (nb_idx >= 0 && state[nb_idx] == Compress)
              state[nb_idx] = Leave;
          }
        }
      }
    }
    if (!More) break;
  }

  for (long long j = 0; j < sim.n; j++) {
    if (state[j] == Compress) {
      int xi = sim.infos[j].ix, yi = sim.infos[j].iy;
      if (xi % 2 == 0 && yi % 2 == 0) {
        static const int sib_ic[4] = {-1, 5, 7, 8};
        long long sib_idx[4];
        sib_idx[0] = j;
        for (int s = 1; s < 4 && state[j] != Leave; s++) {
          if (sim.infos[j].nb_s[sib_ic[s]] != 0) { state[j] = Leave; break; }
          sib_idx[s] = sim.infos[j].nb[sib_ic[s]];
          if (sib_idx[s] < 0 || state[sib_idx[s]] != Compress) { state[j] = Leave; break; }
        }
        for (int s = 0; s < 4 && state[j] != Leave; s++)
          for (int icode = 0; icode < 9 && state[j] != Leave; icode++) {
            if (icode == 4) continue;
            if (sim.infos[sib_idx[s]].nb_s[icode] == 1)
              state[j] = Leave;
          }
      }
    }
  }

  {
    long long n_ref = 0, n_com = 0;
    long long *ref_idx = malloc(sim.n * sizeof(long long));
    long long *com_idx = malloc(sim.n * sizeof(long long));
    for (long long j = 0; j < sim.n; j++) {
      if (state[j] == Refine) ref_idx[n_ref++] = j;
      else if (state[j] == Compress && sim.infos[j].ix % 2 == 0 && sim.infos[j].iy % 2 == 0) com_idx[n_com++] = j;
    }
    fprintf(stderr, "%s:%d: com/ref: %lld %lld\n", __FILE__, __LINE__,
            n_com, n_ref);
    if (n_ref == 0 && n_com == 0) {
      free(ref_idx);
      free(com_idx);
      goto end;
    }
    long long nprev = sim.n;
    sim.n += 4 * n_ref;
    sim.infos = realloc(sim.infos, sim.n * sizeof *sim.infos);
    sim.blocks = realloc(sim.blocks, sim.n * BSTRIDE * sizeof(Real));
    memset(BLK(nprev), 0, 4 * n_ref * BSTRIDE * sizeof(Real));
    state = realloc(state, sim.n * sizeof *state);
    for (long long i = nprev; i < sim.n; i++) state[i] = Leave;

    int ss = 1;
#pragma omp parallel
    {
      Real lm0[LAB_BUF], lm1[LAB_BUF];
      Real *lm[2] = {lm0, lm1};
#pragma omp for
      for (long long k = 0; k < n_ref; k++) {
        struct Info *par = &sim.infos[ref_idx[k]];
        int px = par->ix, py = par->iy;
        Real *blocks[4];
        for (int J = 0; J < 2; J++)
          for (int I = 0; I < 2; I++) {
            long long Z = sfc_forward(par->level + 1, 2 * px + I, 2 * py + J);
            long long ci = nprev + 4 * k + 2 * J + I;
            fill(&sim.infos[ci], par->level + 1, Z);
            blocks[2 * J + I] = BLK(ci);
          }
        int nm = 2 * ss + BS;
        for (size_t m = 0; m < NVARS; m++) {
          int dim = vars[m].dim;
          int offset = vars[m].offset;
          lab_load(lm[dim - 1], dim, offset, ss, ref_idx[k]);
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
        state[ref_idx[k]] = Dealloc;
      }
#pragma omp for
      for (long long k = 0; k < n_com; k++) {
        struct Info *p0 = &sim.infos[com_idx[k]];
        int level = p0->level;
        int x = p0->ix, y = p0->iy;
        long long Z0 = p0->Z;
        static const int sib_ic[4] = {-1, 5, 7, 8};
        Real *Blocks[4];
        Blocks[0] = BLK(com_idx[k]);
        for (int s = 1; s < 4; s++) {
          long long si = sim.infos[com_idx[k]].nb[sib_ic[s]];
          Blocks[s] = BLK(si);
          state[si] = Dealloc;
        }
        for (size_t v = 0; v < NVARS; v++) {
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
        fill(p0, level - 1, Z0 / 4);
      }
    }
    long long cnt = 0;
    for (long long i = 0; i < sim.n; i++) {
      if (state[i] != Dealloc) {
        if (cnt != i) {
          memmove(BLK(cnt), BLK(i), BSTRIDE * sizeof(Real));
          sim.infos[cnt] = sim.infos[i];
        }
        cnt++;
      }
    }
    sim.n = cnt;
    sim.infos = realloc(sim.infos, sim.n * sizeof *sim.infos);
    sim.blocks = realloc(sim.blocks, sim.n * BSTRIDE * sizeof(Real));
    build_tree();
    free(ref_idx);
    free(com_idx);
  }
end:
  free(state);
  return Changed;
}
static void compute_advect_diffuse() {
#pragma omp parallel
  {
    Real um[LAB_BUF];
#pragma omp for nowait
    for (long long id = 0; id < sim.n; ++id) {
      lab_load(um, 2, off_vel, 3, id);
      struct Info *info = &sim.infos[id];
      Real h = info->h;
      Real dfac = sim.nu * sim.dt;
      Real afac = -sim.dt * h;
      Real *TMP = BLK(id) + BS * BS * off_tmpV;
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
  }
}
struct PoissonOp {
  int8_t blk_ref, cell_ix, cell_iy, _pad;
  float coeff;
};
enum { MAX_POISSON_OPS = 16 };
struct PoissonEntry {
  int32_t n_ops;
  struct PoissonOp ops[MAX_POISSON_OPS];
};
static const struct PoissonEntry *poisson_tab;
static void load_poisson() {
  FILE *fp = fopen("tab_poisson.bin", "rb");
  if (!fp) { fprintf(stderr, "main.c: cannot open tab_poisson.bin\n"); exit(1); }
  size_t sz = 4 * BS * 2 * 4 * sizeof(struct PoissonEntry);
  struct PoissonEntry *tab = malloc(sz);
  if (fread(tab, 1, sz, fp) != sz) {
    fprintf(stderr, "main.c: short read from tab_poisson.bin\n"); exit(1);
  }
  fclose(fp);
  poisson_tab = tab;
}
static void getVec() {
#pragma omp parallel for
  for (int i = 0; i < sim.n; i++) {
    Real h = sim.infos[i].h;
    sim.sol_h2[i] = h * h;
    long long offset = (long long)i * BS * BS;
    memcpy(&sim.sol_b[offset], BLK(i) + BS * BS * off_tmp,
           BS * BS * sizeof(Real));
    memcpy(&sim.sol_x[offset], BLK(i) + BS * BS * off_pres,
           BS * BS * sizeof(Real));
  }
}
static void compute_pressure_correction() {
#pragma omp parallel
  {
    Real um[LAB_BUF];
#pragma omp for nowait
    for (long long id = 0; id < sim.n; ++id) {
      lab_load(um, 1, off_pres, 1, id);
      struct Info *info = &sim.infos[id];
      int ss = 1, nm = 2 * ss + BS;
      Real pFac = -0.5 * sim.dt * info->h;
      Real *tmpV = BLK(id) + BS * BS * off_tmpV;
      for (int iy = 0; iy < BS; ++iy)
        for (int ix = 0; ix < BS; ++ix) {
#define P(dx, dy) um[nm * (iy + ss + (dy)) + ix + ss + (dx)]
          tmpV[2 * (BS * iy + ix)]     = pFac * (P(1,0) - P(-1,0));
          tmpV[2 * (BS * iy + ix) + 1] = pFac * (P(0,1) - P(0,-1));
#undef P
        }
    }
  }
}
static void compute_pressure_laplacian() {
#pragma omp parallel
  {
    Real um[LAB_BUF];
#pragma omp for nowait
    for (long long id = 0; id < sim.n; ++id) {
      lab_load(um, 1, off_pold, 1, id);
      Real *TMP = BLK(id) + BS * BS * off_tmp;
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
    {"xcenter", offsetof(struct Shape, x), 1},
    {"ycenter", offsetof(struct Shape, y), 1},
    {"orientation", offsetof(struct Shape, orientation), M_PI / 180},
    {"omega", offsetof(struct Shape, omega), 1},
};
int main(int argc, char **argv) {
#ifdef _OPENMP
#pragma omp parallel
#pragma omp master
  fprintf(stderr, "main.c: %d threads\n", omp_get_num_threads());
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
      fprintf(stderr, "main.c: shape line too long\n");
      exit(1);
    }
    memcpy(line, sp, len);
    line[len] = '\0';
    sp = end;
    struct Shape *shape = calloc(1, sizeof(struct Shape));
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
      fprintf(stderr, "main.c: error: fail to open '%s'\n", path);
      exit(1);
    }
    if (fread(tag, sizeof *tag, sizeof tag, file) != sizeof tag) {
      fprintf(stderr, "main.c: error: fail to read '%s'\n", path);
      exit(1);
    }
    if (tag[0] != 'S' || tag[1] != 'D' || tag[2] != 'F') {
      fprintf(stderr, "main.c: error: not and sdf file '%s'\n", path);
      exit(1);
    }
    if (fread(&length, sizeof(length), 1, file) != 1 ||
        fread(&rmax, sizeof(rmax), 1, file) != 1 ||
        fread(&shape->nr, sizeof(shape->nr), 1, file) != 1 ||
        fread(&shape->np, sizeof(shape->np), 1, file) != 1) {
      fprintf(stderr,
              "main.c: error: fail to read shape header from file.\n");
      exit(1);
    }
    size_t ncount = shape->nr * shape->np;
    if ((shape->sdf = malloc(ncount * sizeof(float))) == NULL) {
      fprintf(stderr, "main.c: error: malloc() failed\n");
      exit(1);
    }
    if (fread(shape->sdf, sizeof *shape->sdf, ncount, file) != ncount) {
      fprintf(stderr, "main.c: error: fail to read arrays from '%s'\n", path);
    }
    shape->length = scale * length;
    shape->rmax = scale * rmax;
    for (size_t i = 0; i < ncount; i++)
      shape->sdf[i] *= scale;
    shape->u = 0;
    shape->v = 0;
    sim.nshape++;
    sim.shapes =
        realloc(sim.shapes, sim.nshape * sizeof sim.shapes);
    sim.shapes[sim.nshape - 1] = shape;
  }
  if (!sim.nshape && *shapeArg) {
    fprintf(stderr, "main.c: error: failed to parse shapes\n");
    exit(1);
  }
  sim.lk = NULL;
  sim.lk_n = 0;
  sim.n = 1LL << (2 * sim.levelStart);
  sim.infos = calloc(sim.n, sizeof *sim.infos);
  sim.blocks = calloc(sim.n * BSTRIDE, sizeof(Real));
  for (long long i = 0; i < sim.n; i++)
    fill(&sim.infos[i], sim.levelStart, i);
  build_tree();
  tab_load_all();
  int Changed = 0;
  for (int i = 0;; i++) {
    ongrid();
    if (i == sim.levelMax)
      break;
    Changed = adapt() || Changed;
  }
  for (int ishape = 0; ishape < sim.nshape; ishape++) {
    struct Shape *shape = sim.shapes[ishape];
#pragma omp parallel for
    for (long long i = 0; i < sim.n; i++) {
      Real *udef = shape->o_udef + i * BS * BS * 2;
      Real *chi = shape->o_chi + i * BS * BS;
      Real *UDEF = BLK(i) + BS * BS * off_tmpV;
      Real *CHI = BLK(i) + BS * BS * off_chi;
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
    Real *UF = BLK(i) + BS * BS * off_vel;
    Real *US = BLK(i) + BS * BS * off_tmpV;
    Real *X = BLK(i) + BS * BS * off_chi;
    for (int j = 0; j < BS * BS; j++) {
      UF[2 * j + 0] = UF[2 * j + 0] * (1 - X[j]) + US[2 * j + 0] * X[j];
      UF[2 * j + 1] = UF[2 * j + 1] * (1 - X[j]) + US[2 * j + 1] * X[j];
    }
  }
  double P_inv[BS * BS * BS * BS];
  precond(P_inv);
  sim.solver = solver_create(BS * BS, P_inv);
  sim.coo_val = NULL; sim.coo_row = NULL; sim.coo_col = NULL;
  sim.sol_x = NULL; sim.sol_b = NULL; sim.sol_h2 = NULL;
  sim.coo_cap = 0;
  load_poisson();
  while (1) {
    if (sim.step % 5 == 0)
      fprintf(stderr, "main.c: %08d %.16e\n", sim.step, sim.time);
    if (sim.dumpTime > 0 && sim.time >= sim.nextDumpTime) {
      sim.nextDumpTime += sim.dumpTime;
      compute_vorticity();
      char path[FILENAME_MAX];
      snprintf(path, sizeof path, "vel.%08d", sim.dump_count++);
      dump(sim.time, path);
    }
    if (sim.endTime > 0 && sim.time >= sim.endTime)
      break;
    Real CFL = sim.CFL;
    Real h = INFINITY;
    for (long long i = 0; i < sim.n; i++)
      h = fmin(sim.infos[i].h, h);
    Real umax = 0;
#pragma omp parallel for schedule(static) reduction(max : umax)
    for (long long i = 0; i < sim.n; i++) {
      Real *vel = BLK(i) + BS * BS * off_vel;
      for (int j = 0; j < 2 * BS * BS; j++)
        umax = fmax(umax, fabs(vel[j]));
    }
    sim.dt = real_min(CFL * h / (umax + 1e-8),
                      0.25 * h * h / (sim.nu + 0.25 * h * umax));
    if (sim.step <= 10 || sim.step % sim.AdaptSteps == 0)
      Changed = adapt() || Changed;
    for (int ishape = 0; ishape < sim.nshape; ishape++) {
      struct Shape *shape = sim.shapes[ishape];
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
      memcpy(BLK(i) + BS * BS * off_vold,
             BLK(i) + BS * BS * off_vel,
             2 * BS * BS * sizeof(Real));
    for (int rk = 0; rk < 2; rk++) {
      Real fac = rk ? 1.0 : 0.5;
      compute_advect_diffuse();
#pragma omp parallel for
      for (long long i = 0; i < sim.n; i++) {
        Real *V = BLK(i) + BS * BS * off_vel;
        Real *Vold = BLK(i) + BS * BS * off_vold;
        Real *tmpV = BLK(i) + BS * BS * off_tmpV;
        Real ih2 = fac / (sim.infos[i].h * sim.infos[i].h);
        for (int j = 0; j < 2 * BS * BS; j++)
          V[j] = Vold[j] + tmpV[j] * ih2;
      }
    }
    for (int ishape = 0; ishape < sim.nshape; ishape++) {
      struct Shape *shape = sim.shapes[ishape];
      Real PM = 0, PX = 0, PY = 0, UM = 0, VM = 0;
#pragma omp parallel for reduction(+ : PM, PX, PY, UM, VM)
      for (long long i = 0; i < sim.n; i++) {
        Real *VEL = BLK(i) + BS * BS * off_vel;
        Real hsq = sim.infos[i].h * sim.infos[i].h;
        Real *chi = shape->o_chi + i * BS * BS;
        Real *udef = shape->o_udef + i * BS * BS * 2;
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
            p[0] = sim.infos[i].origin[0] + sim.infos[i].h * (ix + 0.5);
            p[1] = sim.infos[i].origin[1] + sim.infos[i].h * (iy + 0.5);
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
    struct Collision collisions[16];
    assert(sim.nshape <= 16);
    memset(collisions, 0, sizeof(struct Collision) * sim.nshape);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < sim.nshape; ++i)
      for (int j = 0; j < sim.nshape; ++j) {
        if (i == j)
          continue;
        struct Collision *coll = &collisions[i];
        for (long long k = 0; k < sim.n; ++k) {
          Real *iSDF = sim.shapes[i]->o_dist + k * BS * BS;
          Real *jSDF = sim.shapes[j]->o_dist + k * BS * BS;
          Real *iChi = sim.shapes[i]->o_chi + k * BS * BS;
          Real *jChi = sim.shapes[j]->o_chi + k * BS * BS;
          Real h = 1.0 / BS / (1 << sim.infos[k].level);
          Real hsq = h * h;
          for (int iy = 0; iy < BS; ++iy)
            for (int ix = 0; ix < BS; ++ix) {
              int idx = iy * BS + ix;
              if (iChi[idx] <= 0.0 || jChi[idx] <= 0.0)
                continue;
              coll->iM += iChi[idx] * hsq;
              coll->jM += jChi[idx] * hsq;
              Real dSDFdx_i, dSDFdx_j;
              if (ix == 0) {
                dSDFdx_i = iSDF[idx + 1] - iSDF[idx];
                dSDFdx_j = jSDF[idx + 1] - jSDF[idx];
              } else if (ix == BS - 1) {
                dSDFdx_i = iSDF[idx] - iSDF[idx - 1];
                dSDFdx_j = jSDF[idx] - jSDF[idx - 1];
              } else {
                dSDFdx_i = 0.5 * (iSDF[idx + 1] - iSDF[idx - 1]);
                dSDFdx_j = 0.5 * (jSDF[idx + 1] - jSDF[idx - 1]);
              }
              Real dSDFdy_i, dSDFdy_j;
              if (iy == 0) {
                dSDFdy_i = iSDF[idx + BS] - iSDF[idx];
                dSDFdy_j = jSDF[idx + BS] - jSDF[idx];
              } else if (iy == BS - 1) {
                dSDFdy_i = iSDF[idx] - iSDF[idx - BS];
                dSDFdy_j = jSDF[idx] - jSDF[idx - BS];
              } else {
                dSDFdy_i = 0.5 * (iSDF[idx + BS] - iSDF[idx - BS]);
                dSDFdy_j = 0.5 * (jSDF[idx + BS] - jSDF[idx - BS]);
              }
              coll->ivecX += iChi[idx] * dSDFdx_i;
              coll->ivecY += iChi[idx] * dSDFdy_i;
              coll->jvecX += jChi[idx] * dSDFdx_j;
              coll->jvecY += jChi[idx] * dSDFdy_j;
            }
        }
      }
    for (int i = 0; i < sim.nshape; ++i) {
      for (int j = i + 1; j < sim.nshape; ++j) {
        struct Collision *coll = &collisions[i];
        struct Collision *coll_other = &collisions[j];
        if (coll->iM > 0 && coll->jM > 0 && coll_other->iM > 0 &&
            coll_other->jM > 0) {
          Real norm_i = hypot(coll->ivecX, coll->ivecY);
          Real norm_j = hypot(coll->jvecX, coll->jvecY);
          Real mX = coll->ivecX / norm_i - coll->jvecX / norm_j;
          Real mY = coll->ivecY / norm_i - coll->jvecY / norm_j;
          Real inorm = 1.0 / hypot(mX, mY);
          Real NX = mX * inorm;
          Real NY = mY * inorm;
          Real mass = (coll->iM + coll->jM) / 2;
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
        struct Shape *shape = sim.shapes[ishape];
        Real *X = shape->o_chi + i * BS * BS;
        Real *UDEF = shape->o_udef + i * BS * BS * 2;
        Real *CHI = BLK(i) + BS * BS * off_chi;
        Real *V = BLK(i) + BS * BS * off_vel;
        for (int iy = 0; iy < BS; ++iy)
          for (int ix = 0; ix < BS; ++ix) {
            int j = BS * iy + ix;
            if (CHI[j] > X[j])
              continue;
            if (X[j] <= 0)
              continue;
            Real p[2];
            p[0] = sim.infos[i].origin[0] + sim.infos[i].h * (ix + 0.5);
            p[1] = sim.infos[i].origin[1] + sim.infos[i].h * (iy + 0.5);
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
      memset(BLK(i) + BS * BS * off_tmpV, 0,
             2 * BS * BS * sizeof(Real));
    for (int ishape = 0; ishape < sim.nshape; ishape++) {
      struct Shape *shape = sim.shapes[ishape];
#pragma omp parallel for
      for (long long i = 0; i < sim.n; i++) {
        Real *udef = shape->o_udef + i * BS * BS * 2;
        Real *chi = shape->o_chi + i * BS * BS;
        Real *UDEF = BLK(i) + BS * BS * off_tmpV;
        Real *CHI = BLK(i) + BS * BS * off_chi;
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
      Real vm[LAB_BUF], um[LAB_BUF];
#pragma omp for
      for (int i = 0; i < sim.n; i++) {
        lab_load(vm, 2, off_vel, 1, i);
        lab_load(um, 2, off_tmpV, 1, i);
        pressure_rhs_fun(vm, um, i);
      }
    }
#pragma omp parallel for
    for (long long i = 0; i < sim.n; i++) {
      memcpy(BLK(i) + BS * BS * off_pold,
             BLK(i) + BS * BS * off_pres, BS * BS * sizeof(Real));
      memset(BLK(i) + BS * BS * off_pres, 0,
             BS * BS * sizeof(Real));
    }
    compute_pressure_laplacian();
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
#define COO_PUSH(v, r, c) do { \
    assert(sim.coo_nnz < sim.coo_cap); \
    sim.coo_val[sim.coo_nnz] = (v); \
    sim.coo_row[sim.coo_nnz] = (r); \
    sim.coo_col[sim.coo_nnz] = (c); \
    sim.coo_nnz++; \
  } while(0)
    for (int i = 0; i < sim.n; i++) {
      struct Info *info = &sim.infos[i];
      int n = 1 << info->level;
      int bix = info->ix, biy = info->iy;
      for (int iy = 0; iy < BS; iy++)
        for (int ix = 0; ix < BS; ix++) {
          int sfc_idx = i * BS * BS + iy * BS + ix;
          if ((ix > 0 && ix < BS - 1) && (iy > 0 && iy < BS - 1)) {
            COO_PUSH(1, sfc_idx, sfc_idx - BS);
            COO_PUSH(1, sfc_idx, sfc_idx - 1);
            COO_PUSH(-4, sfc_idx, sfc_idx);
            COO_PUSH(1, sfc_idx, sfc_idx + 1);
            COO_PUSH(1, sfc_idx, sfc_idx + BS);
          } else {
            for (int j = 0; j < 4; j++) {
              int dir = j >> 1, side = j & 1;
              int sign = 2 * side - 1;
              int ec = dir == 0 ? ix : iy;
              int tc = dir == 0 ? iy : ix;
              long long blk_idx[4] = {i, -1, -1, -1};
              int state;
              if (side == 0 ? ec > 0 : ec < BS - 1) {
                int dx = (1 - dir) * sign, dy = dir * sign;
                COO_PUSH(1, sfc_idx, sfc_idx + dy * BS + dx);
                COO_PUSH(-1, sfc_idx, sfc_idx);
                continue;
              } else if (side == 0 ? (dir == 0 ? bix : biy) == 0
                                   : (dir == 0 ? bix : biy) == n - 1) {
                continue;
              } else {
                static const int poisson_ic[4] = {3, 5, 1, 7};
                int ic = poisson_ic[j];
                int8_t ns = info->nb_s[ic];
                int nb_idx = info->nb[ic];
                if (ns == 0) {
                  state = 1;
                  blk_idx[1] = nb_idx;
                } else if (ns == 2) {
                  state = 2;
                  blk_idx[2] = nb_idx;
                } else if (ns == 1) {
                  state = 3;
                  int nix = dir == 0 ? (bix + sign + n) % n : bix;
                  int niy = dir == 0 ? biy : (biy + sign + n) % n;
                  int ct = tc >= BS / 2 ? 1 : 0, ce = 1 - side;
                  long long Zc = dir == 0
                      ? sfc_forward(info->level + 1, 2 * nix + ce, 2 * niy + ct)
                      : sfc_forward(info->level + 1, 2 * nix + ct, 2 * niy + ce);
                  blk_idx[3] = getf0(info->level + 1, Zc);
                } else {
                  fprintf(stderr, "main.c: bad neighbour state\n");
                  exit(1);
                }
              }
              int parity = dir == 0 ? biy % 2 : bix % 2;
              const struct PoissonEntry *pe =
                  &poisson_tab[((j * BS + tc) * 2 + parity) * 4 + state];
              for (int k = 0; k < pe->n_ops; k++) {
                const struct PoissonOp *op = &pe->ops[k];
                COO_PUSH((double)op->coeff, sfc_idx,
                    (int)(blk_idx[op->blk_ref] * BS * BS + op->cell_iy * BS + op->cell_ix));
              }
            }
          }
        }
    }
#undef COO_PUSH
    getVec();
    solver_solve(sim.solver, Changed, N, sim.coo_nnz,
        sim.coo_val, sim.coo_row, sim.coo_col,
        sim.sol_x, sim.sol_b, sim.sol_h2, -1,
        max_error, max_rel_error, max_restarts);
    Changed = 0;
    Real avg = 0, avg1 = 0;
#pragma omp parallel for reduction(+ : avg, avg1)
    for (long long i = 0; i < sim.n; i++) {
      Real *P = BLK(i) + BS * BS * off_pres;
      Real vv = sim.infos[i].h * sim.infos[i].h;
      for (int j = 0; j < BS * BS; j++) {
        P[j] = sim.sol_x[i * BS * BS + j];
        avg += P[j] * vv;
        avg1 += vv;
      }
    }
    avg /= avg1;
#pragma omp parallel for
    for (long long i = 0; i < sim.n; i++) {
      Real *pres = BLK(i) + BS * BS * off_pres;
      Real *pold = BLK(i) + BS * BS * off_pold;
      for (int j = 0; j < BS * BS; j++)
        pres[j] += pold[j] - avg;
    }
    compute_pressure_correction();
#pragma omp parallel for
    for (long long i = 0; i < sim.n; i++) {
      Real ih2 = 1.0 / sim.infos[i].h / sim.infos[i].h;
      Real *V = BLK(i) + BS * BS * off_vel;
      Real *tmpV = BLK(i) + BS * BS * off_tmpV;
      for (int j = 0; j < 2 * BS * BS; j++)
        V[j] += tmpV[j] * ih2;
    }
    sim.time += sim.dt;
    sim.step++;
  }

  solver_destroy(sim.solver);
  free(sim.coo_val); free(sim.coo_row); free(sim.coo_col);
  free(sim.sol_x); free(sim.sol_b); free(sim.sol_h2);
  for (int ishape = 0; ishape < sim.nshape; ishape++) {
    struct Shape *shape = sim.shapes[ishape];
    free(shape->o_chi);
    free(shape->o_dist);
    free(shape->o_udef);
    free(shape->o_com);
    free(shape->sdf);
    free(shape);
  }
  free(sim.lk);
  fprintf(stderr, "main.c: end\n");
}
