#include <assert.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "solver.h"
#ifdef _OPENMP
#include <omp.h>
#endif

typedef double Real;
enum { BS = 8 };
enum {
  F_U = 0,   /* u velocity */
  F_V = 1,   /* v velocity */
  F_P = 2,   /* pressure */
  F_PHI = 3, /* pressure correction (Poisson solve) */
  F_W = 4,   /* vorticity (output/indicator) */
  F_TMP = 5, /* scratch */
  F_N = 6,
  BLK_S = F_N *BS *BS,
};

enum { BC_WALL = 0, BC_SYMMETRY, BC_INFLOW, BC_OUTFLOW, BC_PERIODIC };
struct FaceBC { int type; Real val[F_N]; };
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
  int step;
  int dump_count;
  Real CFL;
  Real Ctol;
  Real dt;
  Real dumpTime;
  Real endTime;
  Real nextDumpTime;
  Real Rtol;
  Real time;
  Real L[2];
  int nb[2];
  long long n;
  struct HMap hm;
  struct FaceBC bc[4]; /* x-, x+, y-, y+ */
  struct Blk *blk;
  Real *fld;
  /* Solvers */
  struct Solver *solver;      /* Poisson (Cholesky precond) */
  struct Solver *helm_solver; /* Helmholtz (identity precond) */
  int coo_nnz, coo_cap;
  double *coo_val, *sol_x, *sol_b, *sol_h2;
  int *coo_row, *coo_col;
} sim;
static long long hm_key(int level, int ix, int iy) {
  long long n = 1LL << level;
  return ((n * n) - 1) / 3 + iy * n + ix;
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
static int arg_i_opt(int argc, char **argv, const char *key, int def) {
  for (int i = 1; i < argc; i++)
    if (argv[i][0] == '-' && strcmp(argv[i] + 1, key) == 0 && i + 1 < argc) {
      char *end;
      long v = strtol(argv[i + 1], &end, 10);
      if (end != argv[i + 1]) return (int)v;
    }
  return def;
}
struct Blk {
  double h, origin[2];
  int level, n, ix, iy;
};
#define BLK(i) (sim.fld + (long long)(i) * BLK_S)
static void bl_fill(struct Blk *b, int level, int ix, int iy) {
  int scale = 1 << (level - sim.levelStart);
  b->level = level;
  b->n = 1 << level;
  b->ix = ix;
  b->iy = iy;
  b->h = sim.L[0] / (BS * sim.nb[0] * scale);
  b->origin[0] = b->h * BS * ix;
  b->origin[1] = b->h * BS * iy;
}
struct {
  int offset;
  int dim;
  const char *prefix;
} fld_t[] = {{F_U, 1, "u"}, {F_V, 1, "v"}, {F_P, 1, "p"},
            {F_PHI, 1, NULL}, {F_W, 1, "vort"}, {F_TMP, 1, NULL}};
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
  int scale = 1 << (level - sim.levelStart);
  int nd[2] = {sim.nb[0]*scale, sim.nb[1]*scale};
  int pos[2] = {ix, iy};
  int c[2] = {cx, cy};
  int skin[2], nbc = 0;
  for (int d = 0; d < 2; d++) { skin[d] = nb_skin(c[d], pos[d], nd[d]); nbc += skin[d]; }
  /* Periodic BCs: skip boundary handling, use wrapped coordinates */
  if (nbc > 0) {
    int all_periodic = 1;
    for (int d = 0; d < 2; d++) {
      if (!skin[d]) continue;
      int face = 2*d + (pos[d] == nd[d]-1 ? 1 : 0);
      if (sim.bc[face].type != BC_PERIODIC) all_periodic = 0;
    }
    if (all_periodic) { nbc = 0; for (int d = 0; d < 2; d++) skin[d] = 0; }
  }
  if (nbc > 0) {
    if (nbc == 1) {
      for (int d = 0; d < 2; d++) {
        if (!skin[d]) continue;
        int face = 2*d + (pos[d] == nd[d]-1 ? 1 : (pos[d] == 0 && c[d] == -1 ? 0 : 1));
        int bt = sim.bc[face].type;
        if (bt == BC_OUTFLOW) { r.s = 6+d; return r; }
        if (bt == BC_INFLOW) { r.s = 8+d; return r; }
        r.s = 3+d; return r; /* wall/symmetry */
      }
    }
    /* Multi-axis: outflow/inflow dominates */
    for (int d = 0; d < 2; d++) {
      if (!skin[d]) continue;
      int face = 2*d + (pos[d] == nd[d]-1 ? 1 : (pos[d] == 0 && c[d] == -1 ? 0 : 1));
      int bt = sim.bc[face].type;
      if (bt == BC_OUTFLOW) { r.s = 6+d; return r; }
      if (bt == BC_INFLOW) { r.s = 8+d; return r; }
    }
    r.s = 5; return r; /* wall corner */
  }
  int nx = (ix + cx + nd[0]) % nd[0], ny = (iy + cy + nd[1]) % nd[1];
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
  if (level > 1) {
    idx = hm_get(&sim.hm, hm_key(level - 2, nx / 4, ny / 4));
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
  OP_BC_CORNER,
  OP_BC_FIXED,
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
    case OP_BC_CORNER: {
      Real *buf = dst[o->dst_idx];
      buf[o->dst_off] = -buf[o->src_off];
      buf[o->dst_off + 1] = -buf[o->src_off + 1];
      break;
    }
    case OP_BC_FIXED: {
      Real *src = dst[o->blk_idx]; /* bc_const buffer */
      Real *buf = dst[o->dst_idx];
      for (int d = 0; d < dim; d++)
        buf[o->dst_off + d] = src[d];
      break;
    }
    }
  }
}

enum { N_STATUS = 10 };
static const struct LbTab (*lb_tab[5][3])[3][2][2][N_STATUS];
static void lb_init(void) {
  int configs[][2] = {{1,1}, {1,2}, {2,1}, {4,1}};
  for (int ci = 0; ci < 3; ci++) {
    int ss = configs[ci][0], dim = configs[ci][1];
    char fname[64];
    snprintf(fname, sizeof fname, "tab_ss%d_dim%d.bin", ss, dim);
    FILE *fp = fopen(fname, "rb");
    if (!fp) {
      fprintf(stderr, "main.c: cannot open %s\n", fname);
      exit(1);
    }
    size_t sz = 3 * 3 * 2 * 2 * N_STATUS * sizeof(struct LbTab);
    struct LbTab *tab = malloc(sz);
    if (fread(tab, 1, sz, fp) != sz) {
      fprintf(stderr, "main.c: short read from %s\n", fname);
      exit(1);
    }
    fclose(fp);
    lb_tab[ss][dim] = (const struct LbTab (*)[3][2][2][N_STATUS])tab;
  }
}
enum { LB_BUF = ((2*4+BS)*(2*4+BS) + (BS/2+4+3)*(BS/2+4+3)) * 2 };
static void lb_load(Real *m, int dim, int blk_offset, int ss, long long info_idx) {
  struct Blk *info = &sim.blk[info_idx];
  int nm = 2 * ss + BS;
  int nc = BS / 2 + ss + 3;
  int level = info->level;
  int xi = info->ix, yi = info->iy;
  const struct LbTab (*cflb_tab)[3][2][2][N_STATUS] = lb_tab[ss][dim];

  Real *p0 = BLK(info_idx) + BS * BS * blk_offset;
  for (int i = 0; i < BS; i++)
    memcpy(m + dim * ((i + ss) * nm + ss), p0 + dim * BS * i,
           BS * dim * sizeof(Real));

  Real *c = m + nm * nm * dim;
  Real bc_const[2] = {0};
  Real *dst[3] = {m, c, bc_const};
  int scale = 1 << (level - sim.levelStart);
  int nd2[2] = {sim.nb[0]*scale, sim.nb[1]*scale};
  int pos[2] = {xi, yi};

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

    /* Fill bc_const for inflow faces */
    if (nr.s == 8 || nr.s == 9) {
      int axis = nr.s - 8;
      int face = 2*axis + (pos[axis] == nd2[axis]-1 ? 1 : 0);
      memcpy(bc_const, &sim.bc[face].val[blk_offset], dim * sizeof(Real));
    }
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

/* ---- Poisson solver infrastructure ---- */
/* Wide Laplacian: (-4φ + φ_{i+2} + φ_{i-2} + φ_{j+2} + φ_{j-2}) / (4h²)
   Local block stencil: stride-2 neighbors */
static double ps_Aloc(int I1, int I2) {
  int j1=I1/BS, i1=I1%BS, j2=I2/BS, i2=I2%BS;
  if (i1==i2 && j1==j2) return 4.0;
  if ((abs(i1-i2)==2 && j1==j2) || (i1==i2 && abs(j1-j2)==2)) return -1.0;
  return 0.0;
}
static void ps_prec(double *P_inv) {
  double L[64][64], L_inv[64][64];
  memset(L, 0, sizeof L); memset(L_inv, 0, sizeof L_inv);
  for (int i=0; i<BS*BS; i++) L_inv[i][i]=1.0;
  for (int i=0; i<BS*BS; i++) {
    double s1=0; for (int k=0; k<i; k++) s1+=L[i][k]*L[i][k];
    L[i][i]=sqrt(ps_Aloc(i,i)-s1);
    for (int j=i+1; j<BS*BS; j++) {
      double s2=0; for (int k=0; k<i; k++) s2+=L[i][k]*L[j][k];
      L[j][i]=(ps_Aloc(j,i)-s2)/L[i][i];
    }
  }
  for (int br=0; br<BS*BS; br++) {
    double bsf=1./L[br][br];
    for (int c=0; c<=br; c++) L_inv[br][c]*=bsf;
    for (int wr=br+1; wr<BS*BS; wr++) {
      double wsf=L[wr][br];
      for (int c=0; c<=br; c++) L_inv[wr][c]-=wsf*L_inv[br][c];
    }
  }
  for (int i=0; i<BS*BS; i++) for (int j=0; j<BS*BS; j++) {
    double aux=0;
    for (int k=0; k<BS*BS; k++) aux += i<=k && j<=k ? L_inv[k][i]*L_inv[k][j] : 0;
    P_inv[i*BS*BS+j] = -aux;
  }
}
struct PsOp { int8_t blk_ref, cell_ix, cell_iy, _pad; float coeff; };
enum { PS_MAX_OPS = 16 };
struct PsEnt { int32_t n_ops; struct PsOp ops[PS_MAX_OPS]; };
static const struct PsEnt *ps_tab;
static void ps_load(void) {
  FILE *fp = fopen("tab_poisson.bin", "rb");
  if (!fp) { fprintf(stderr, "cannot open tab_poisson.bin\n"); exit(1); }
  size_t sz = 4*BS*2*4*sizeof(struct PsEnt);
  struct PsEnt *tab = malloc(sz);
  if (fread(tab,1,sz,fp) != sz) { fprintf(stderr, "short read tab_poisson.bin\n"); exit(1); }
  fclose(fp); ps_tab = tab;
}

/* ---- Incompressible NS: Godunov-projection (Brown & Minion 1995) ---- */

static Real NU = 1e-4; /* kinematic viscosity */

static inline Real minmod(Real a, Real b) {
  return a * b <= 0 ? 0 : fabs(a) < fabs(b) ? a : b;
}

/* Compute vorticity: w = dv/dx - du/dy */
static void compute_vorticity(void) {
#pragma omp parallel
  {
    Real bu[LB_BUF], bv[LB_BUF];
#pragma omp for
    for (long long id = 0; id < sim.n; id++) {
      lb_load(bu, 1, F_U, 1, id);
      lb_load(bv, 1, F_V, 1, id);
      Real *w = BLK(id) + BS*BS*F_W;
      int ss=1, nm=2*ss+BS;
      Real ih = 0.5 / sim.blk[id].h;
      for (int j = 0; j < BS; j++)
        for (int i = 0; i < BS; i++) {
#define U(di,dj) bu[nm*((j)+(dj)+ss)+(i)+(di)+ss]
#define V(di,dj) bv[nm*((j)+(dj)+ss)+(i)+(di)+ss]
          w[j*BS+i] = (V(1,0)-V(-1,0))*ih - (U(0,1)-U(0,-1))*ih;
#undef U
#undef V
        }
    }
  }
}

/* Refinement indicator based on vorticity magnitude */
static void compute_indicator(void) {
  compute_vorticity();
#pragma omp parallel for
  for (long long id = 0; id < sim.n; id++) {
    Real *w = BLK(id) + BS*BS*F_W;
    Real *t = BLK(id) + BS*BS*F_TMP;
    Real h = sim.blk[id].h;
    for (int j = 0; j < BS*BS; j++)
      t[j] = fabs(w[j]) * h; /* scale by h for resolution-independent threshold */
  }
}

/* Output dump */
static void dump(Real time, int step, char *path) {
  long i, j, k, x, y;
  char xyz_path[FILENAME_MAX], attr_path[FILENAME_MAX];
  FILE *file;
  float xyz[4 * BS * BS][2];
  char *xyz_base, xdmf_path[FILENAME_MAX];
  FILE *xdmf;
  long long ncell = 0;
  for (i = 0; i < sim.n; i++) ncell += BS * BS;
  snprintf(xyz_path, sizeof xyz_path, "%s.xyz.raw", path);
  file = fopen(xyz_path, "wb");
  for (i = 0; i < sim.n; i++) {
    Real h = sim.blk[i].h, ox = sim.blk[i].origin[0], oy = sim.blk[i].origin[1];
    for (j = 0; j < BS; j++)
      for (k = 0; k < BS; k++) {
        int c = j * BS + k;
        float x0=ox+k*h, y0=oy+j*h, x1=x0+h, y1=y0+h;
        xyz[4*c+0][0]=x0; xyz[4*c+0][1]=y0;
        xyz[4*c+1][0]=x0; xyz[4*c+1][1]=y1;
        xyz[4*c+2][0]=x1; xyz[4*c+2][1]=y1;
        xyz[4*c+3][0]=x1; xyz[4*c+3][1]=y0;
      }
    fwrite(xyz, sizeof(float), 4*2*BS*BS, file);
  }
  fclose(file);
  snprintf(xdmf_path, sizeof xdmf_path, "%s.xdmf2", path);
  xdmf = fopen(xdmf_path, "w");
  fprintf(xdmf,
    "<Xdmf Version=\"2.0\">\n<Domain><Grid>\n"
    "  <Time Value=\"%.16e\"/>\n"
    "  <Information Name=\"Step\" Value=\"%d\"/>\n"
    "  <Topology Dimensions=\"%lld\" TopologyType=\"Quadrilateral\"/>\n"
    "  <Geometry Type=\"XY\">\n"
    "    <DataItem Dimensions=\"%lld 4 2\" Format=\"Binary\""
    " DataType=\"Float\" Precision=\"4\" Endian=\"Little\">%s</DataItem>\n"
    "  </Geometry>\n", time, step, ncell, ncell, xyz_path);
  for (size_t fi = 0; fi < NVARS; fi++)
    if (fld_t[fi].prefix) {
      int dim = fld_t[fi].dim, offset = fld_t[fi].offset;
      snprintf(attr_path, sizeof attr_path, "%s.%s.raw", path, fld_t[fi].prefix);
      fprintf(xdmf,
        "  <Attribute Name=\"%s\" Center=\"Cell\" AttributeType=\"%s\">\n"
        "    <DataItem Dimensions=\"%lld %s\" Format=\"Binary\""
        " DataType=\"Float\" Precision=\"8\" Endian=\"Little\">%s</DataItem>\n"
        "  </Attribute>\n",
        fld_t[fi].prefix,
        dim > 1 ? "Vector" : "Scalar", ncell,
        dim > 1 ? "2" : "1", attr_path);
      file = fopen(attr_path, "wb");
      for (j = 0; j < sim.n; j++)
        fwrite(BLK(j) + offset*BS*BS, sizeof(Real), dim*BS*BS, file);
      fclose(file);
    }
  fprintf(xdmf, "</Grid></Domain></Xdmf>\n");
  fclose(xdmf);
}

/* AMR adaptation (reuse from main.c but simpler indicator) */
static const Real ad_ref_w[4][9] = {
  { 1./64, 10./64, -1./64, 10./64, 56./64, -6./64, -1./64, -6./64,  1./64},
  {-1./64, 10./64,  1./64, -6./64, 56./64, 10./64,  1./64, -6./64, -1./64},
  {-1./64, -6./64,  1./64, 10./64, 56./64, -6./64,  1./64, 10./64, -1./64},
  { 1./64, -6./64, -1./64, -6./64, 56./64, 10./64, -1./64, 10./64,  1./64},
};
static const int ad_sib_ic[4] = {-1, 5, 7, 8};
static int ad_run(void) {
  compute_indicator();
  enum AdSt *state = calloc(sim.n, sizeof *state);
  long long *ref_idx = malloc(sim.n * sizeof *ref_idx);
  long long *com_idx = malloc(sim.n * sizeof *com_idx);
  long long n_ref = 0, n_com = 0;
  int Changed = 0;
#pragma omp parallel for reduction(|| : Changed)
  for (long long i = 0; i < sim.n; i++) {
    Real *b = BLK(i) + BS*BS*F_TMP;
    double Linf = 0;
    for (int j = 0; j < BS*BS; j++) Linf = fmax(Linf, fabs(b[j]));
    int lev = sim.blk[i].level;
    state[i] = Linf > sim.Rtol && lev < sim.levelMax-1 ? Refine
             : Linf < sim.Ctol && lev > sim.levelStart ? Compress
             : Leave;
    Changed |= state[i] != Leave;
  }
  if (!Changed) goto done;
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
    if (state[j] == Refine) ref_idx[n_ref++] = j;
    else if (state[j] == Compress && !((sim.blk[j].ix|sim.blk[j].iy)&1))
      com_idx[n_com++] = j;
  fprintf(stderr, "  ad: com/ref %lld/%lld\n", n_com, n_ref);
  if (n_ref == 0 && n_com == 0) goto done;
  long long nprev = sim.n;
  sim.n += 4 * n_ref;
  sim.blk = realloc(sim.blk, sim.n * sizeof *sim.blk);
  sim.fld = realloc(sim.fld, sim.n * BLK_S * sizeof(Real));
  memset(BLK(nprev), 0, 4*n_ref*BLK_S*sizeof(Real));
  state = realloc(state, sim.n * sizeof *state);
  for (long long i = nprev; i < sim.n; i++) state[i] = Leave;
#pragma omp parallel
  {
    Real lm[LB_BUF];
#pragma omp for
    for (long long k = 0; k < n_ref; k++) {
      struct Blk *par = &sim.blk[ref_idx[k]];
      int px=par->ix, py=par->iy;
      Real *blks[4];
      for (int J=0;J<2;J++) for (int I=0;I<2;I++) {
        long long ci = nprev + 4*k + 2*J+I;
        bl_fill(&sim.blk[ci], par->level+1, 2*px+I, 2*py+J);
        blks[2*J+I] = BLK(ci);
      }
      int nm = 2 + BS;
      for (size_t m = 0; m < NVARS; m++) {
        int dim=fld_t[m].dim, offset=fld_t[m].offset;
        lb_load(lm, dim, offset, 1, ref_idx[k]);
        for (int J=0;J<2;J++) for (int I=0;I<2;I++) {
          Real *b = blks[J*2+I] + offset*BS*BS;
          for (int j=0;j<BS;j+=2) for (int i=0;i<BS;i+=2) {
            int i0=i/2+I*(BS/2)+1, j0=j/2+J*(BS/2)+1;
            int sub[4]={BS*j+i, BS*j+i+1, BS*(j+1)+i, BS*(j+1)+i+1};
            for (int s=0;s<4;s++) for (int d=0;d<dim;d++) {
              Real val=0;
              for (int kk=0;kk<9;kk++)
                val += ad_ref_w[s][kk]*lm[dim*(nm*(j0+kk/3-1)+i0+kk%3-1)+d];
              b[dim*sub[s]+d] = val;
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
      int level=p0->level, x=p0->ix, y=p0->iy;
      Real *blk[4] = {BLK(ci)};
      for (int s=1;s<4;s++) { blk[s]=BLK(nb_find(level,x,y,ad_sib_ic[s]).idx); state[nb_find(level,x,y,ad_sib_ic[s]).idx]=Dealloc; }
      for (size_t v=0;v<NVARS;v++) {
        int dim=fld_t[v].dim, off=fld_t[v].offset;
        Real *dst=blk[0]+off*BS*BS;
        for (int J=0;J<2;J++) for (int I=0;I<2;I++) {
          Real *src=blk[J*2+I]+off*BS*BS;
          for (int j=0;j<BS;j+=2) for (int i=0;i<BS;i+=2) {
            int o=BS*(j/2+J*(BS/2))+i/2+I*(BS/2);
            for (int d=0;d<dim;d++)
              dst[dim*o+d]=(src[dim*(BS*j+i)+d]+src[dim*(BS*j+i+1)+d]+src[dim*(BS*(j+1)+i)+d]+src[dim*(BS*(j+1)+i+1)+d])/4;
          }
        }
      }
      bl_fill(p0, level-1, x/2, y/2);
    }
  }
  long long cnt=0;
  for (long long i=0;i<sim.n;i++) {
    if (state[i]==Dealloc) continue;
    if (cnt!=i) { memmove(BLK(cnt),BLK(i),BLK_S*sizeof(Real)); sim.blk[cnt]=sim.blk[i]; }
    cnt++;
  }
  sim.n=cnt;
  sim.blk=realloc(sim.blk,sim.n*sizeof*sim.blk);
  sim.fld=realloc(sim.fld,sim.n*BLK_S*sizeof(Real));
  hm_rebuild();
done:
  free(state); free(ref_idx); free(com_idx);
  return Changed;
}

/* ---- Incompressible NS time step ---- */

/*
 * 4th-order monotone limited slope (Brown & Minion Eq. 20 right column).
 * D^C = (φ_{i+1} - φ_{i-1}) / 2
 * D^L = φ_i - φ_{i-1}
 * D^R = φ_{i+1} - φ_i
 * δ^lim = { min(2|D^L|, 2|D^R|) if D^L*D^R > 0, else 0 } * sign(D^C)
 * δ' = min(|D^C|, δ^lim * sign(D^C))
 * δ = min(|4*D^C/3 - (δ'_{i+1} + δ'_{i-1})/6|, δ^lim) * sign(D^C)
 */
static inline Real slope4(Real phim2, Real phim1, Real phi0, Real phip1, Real phip2) {
  Real DC = 0.5*(phip1 - phim1);
  Real DL = phi0 - phim1;
  Real DR = phip1 - phi0;
  Real dlim = DL*DR > 0 ? fmin(2*fabs(DL), 2*fabs(DR)) : 0;
  Real dprime = fmin(fabs(DC), dlim) * (DC > 0 ? 1 : (DC < 0 ? -1 : 0));

  /* For the full 4th-order slope we need δ'_{i+1} and δ'_{i-1}.
     Compute them inline. */
  Real DC_p = 0.5*(phip2 - phi0);
  Real DL_p = phip1 - phi0;
  Real DR_p = phip2 - phip1;
  Real dlim_p = DL_p*DR_p > 0 ? fmin(2*fabs(DL_p), 2*fabs(DR_p)) : 0;
  Real dp_p = fmin(fabs(DC_p), dlim_p) * (DC_p > 0 ? 1 : (DC_p < 0 ? -1 : 0));

  Real DC_m = 0.5*(phi0 - phim2);
  Real DL_m = phim1 - phim2;
  Real DR_m = phi0 - phim1;
  Real dlim_m = DL_m*DR_m > 0 ? fmin(2*fabs(DL_m), 2*fabs(DR_m)) : 0;
  Real dp_m = fmin(fabs(DC_m), dlim_m) * (DC_m > 0 ? 1 : (DC_m < 0 ? -1 : 0));

  Real d4 = 4.0/3.0 * DC - (dp_p + dp_m) / 6.0;
  Real sgn = DC > 0 ? 1 : (DC < 0 ? -1 : 0);
  return fmin(fabs(d4), dlim) * sgn;
}

/* Multigrid Poisson solver for periodic cell-centered grid.
   Solves (-4u + u_{i+1} + u_{i-1} + u_{j+1} + u_{j-1}) = f on M×M periodic grid.
   Uses W-cycle with cell-centered bilinear prolongation. */
static void mg_smooth(double *u, const double *f, int m, int niter) {
  for (int sw = 0; sw < niter; sw++) {
    for (int color = 0; color < 2; color++)
      for (int j = 0; j < m; j++)
        for (int i = 0; i < m; i++) {
          if ((i + j) % 2 != color) continue;
          int ip = (i+1)%m, im = (i-1+m)%m, jp = (j+1)%m, jm = (j-1+m)%m;
          u[j*m+i] = 0.25 * (u[j*m+ip]+u[j*m+im]+u[jp*m+i]+u[jm*m+i] - f[j*m+i]);
        }
    double mn = 0;
    for (int k = 0; k < m*m; k++) mn += u[k];
    mn /= m*m;
    for (int k = 0; k < m*m; k++) u[k] -= mn;
  }
}

static void mg_residual(const double *u, const double *f, double *r, int m) {
  for (int j = 0; j < m; j++)
    for (int i = 0; i < m; i++) {
      int ip = (i+1)%m, im = (i-1+m)%m, jp = (j+1)%m, jm = (j-1+m)%m;
      r[j*m+i] = f[j*m+i] - (-4*u[j*m+i]+u[j*m+ip]+u[j*m+im]+u[jp*m+i]+u[jm*m+i]);
    }
}

static void mg_restrict(const double *rf, double *rc, int mf) {
  /* Full-weighting restriction (9-point stencil, variational pair of bilinear prolongation) */
  int mc = mf / 2;
  for (int j = 0; j < mc; j++)
    for (int i = 0; i < mc; i++) {
      int i2=2*i, j2=2*j;
      int i2p=(i2+1)%mf, i2m=(i2-1+mf)%mf, j2p=(j2+1)%mf, j2m=(j2-1+mf)%mf;
      rc[j*mc+i] = (4*rf[j2*mf+i2]
        + 2*(rf[j2*mf+i2p]+rf[j2*mf+i2m]+rf[j2p*mf+i2]+rf[j2m*mf+i2])
        + rf[j2p*mf+i2p]+rf[j2p*mf+i2m]+rf[j2m*mf+i2p]+rf[j2m*mf+i2m]) / 16.0;
    }
}

static void mg_prolong_add(const double *ec, double *uf, int mc) {
  /* Cell-centered bilinear prolongation.
     Fine cell (2j, 2i) center at (j+1/4)*hc — closest to coarse(j,i), then (j-1,i-1).
     Weights: 9/16 from nearest, 3/16 from face-adjacent, 1/16 from diagonal. */
  int mf = mc * 2;
  for (int j = 0; j < mc; j++)
    for (int i = 0; i < mc; i++) {
      int im = (i-1+mc)%mc, jm = (j-1+mc)%mc;
      int ip = (i+1)%mc,    jp = (j+1)%mc;
      double cij = ec[j*mc+i];
      int fi = 2*i, fj = 2*j, fi1 = (2*i+1)%mf, fj1 = (2*j+1)%mf;
      /* fine(2j,  2i)   — lower-left quarter: uses (j,i), (j,i-1), (j-1,i), (j-1,i-1) */
      uf[fj*mf+fi]   += (9*cij + 3*ec[j*mc+im] + 3*ec[jm*mc+i] + ec[jm*mc+im]) / 16.0;
      /* fine(2j,  2i+1) — lower-right quarter: uses (j,i), (j,i+1), (j-1,i), (j-1,i+1) */
      uf[fj*mf+fi1]  += (9*cij + 3*ec[j*mc+ip] + 3*ec[jm*mc+i] + ec[jm*mc+ip]) / 16.0;
      /* fine(2j+1,2i)   — upper-left quarter: uses (j,i), (j,i-1), (j+1,i), (j+1,i-1) */
      uf[fj1*mf+fi]  += (9*cij + 3*ec[j*mc+im] + 3*ec[jp*mc+i] + ec[jp*mc+im]) / 16.0;
      /* fine(2j+1,2i+1) — upper-right quarter: uses (j,i), (j,i+1), (j+1,i), (j+1,i+1) */
      uf[fj1*mf+fi1] += (9*cij + 3*ec[j*mc+ip] + 3*ec[jp*mc+i] + ec[jp*mc+ip]) / 16.0;
    }
}

static void mg_vcycle(double *u, double *f, double *r, int m) {
  if (m <= 4) { mg_smooth(u, f, m, 50); return; }
  int mc = m / 2;
  double *uc = calloc(mc*mc, sizeof(double));
  double *fc = calloc(mc*mc, sizeof(double));
  double *rc = calloc(mc*mc, sizeof(double));

  mg_smooth(u, f, m, 4);
  mg_residual(u, f, r, m);
  mg_restrict(r, fc, m);
  mg_vcycle(uc, fc, rc, mc);
  mg_prolong_add(uc, u, mc);
  mg_smooth(u, f, m, 4);

  free(uc); free(fc); free(rc);
}

/* PCG solver for (-4φ + Σφ_nb) = f on M×M periodic grid.
   Uses multigrid V-cycle as preconditioner. */
static void mg_solve_periodic(double *x, const double *f, int M, double tol) {
  int N = M*M;
  double *rr = malloc(N*sizeof(double));
  double *z = calloc(N, sizeof(double));
  double *p = malloc(N*sizeof(double));
  double *Ap = malloc(N*sizeof(double));
  double *r_tmp = malloc(N*sizeof(double));

  /* r = f - A*x */
  mg_residual(x, f, rr, M);
  { double mn=0; for(int k=0;k<N;k++) mn+=rr[k]; mn/=N; for(int k=0;k<N;k++) rr[k]-=mn; }

  /* z = M^{-1} r (one V-cycle) */
  memset(z, 0, N*sizeof(double));
  mg_vcycle(z, rr, r_tmp, M);
  { double mn=0; for(int k=0;k<N;k++) mn+=z[k]; mn/=N; for(int k=0;k<N;k++) z[k]-=mn; }
  memcpy(p, z, N*sizeof(double));
  double rz = 0; for(int k=0;k<N;k++) rz += rr[k]*z[k];

  for (int it = 0; it < 100; it++) {
    /* Ap = A*p */
    for (int j=0;j<M;j++) for (int i=0;i<M;i++) {
      int ip=(i+1)%M,im=(i-1+M)%M,jp=(j+1)%M,jm=(j-1+M)%M;
      Ap[j*M+i] = -4*p[j*M+i]+p[j*M+ip]+p[j*M+im]+p[jp*M+i]+p[jm*M+i];
    }
    double pAp = 0; for(int k=0;k<N;k++) pAp += p[k]*Ap[k];
    if (fabs(pAp) < 1e-30) break;
    double alpha = rz / pAp;
    for(int k=0;k<N;k++) { x[k] += alpha*p[k]; rr[k] -= alpha*Ap[k]; }
    { double mn=0; for(int k=0;k<N;k++) mn+=rr[k]; mn/=N; for(int k=0;k<N;k++) rr[k]-=mn; }
    { double mn=0; for(int k=0;k<N;k++) mn+=x[k]; mn/=N; for(int k=0;k<N;k++) x[k]-=mn; }

    double rmax = 0; for(int k=0;k<N;k++) if(fabs(rr[k])>rmax) rmax=fabs(rr[k]);
    if (rmax < tol) break;

    /* z = M^{-1} r */
    memset(z, 0, N*sizeof(double));
    mg_vcycle(z, rr, r_tmp, M);
    { double mn=0; for(int k=0;k<N;k++) mn+=z[k]; mn/=N; for(int k=0;k<N;k++) z[k]-=mn; }
    double rz2 = 0; for(int k=0;k<N;k++) rz2 += rr[k]*z[k];
    double beta = rz2 / (rz + 1e-30);
    for(int k=0;k<N;k++) p[k] = z[k] + beta*p[k];
    rz = rz2;
  }
  { double mn=0; for(int k=0;k<N;k++) mn+=x[k]; mn/=N; for(int k=0;k<N;k++) x[k]-=mn; }
  free(rr); free(z); free(p); free(Ap); free(r_tmp);
}

/* Advect + diffuse with MAC projection (Brown & Minion Eq. 18-23).
   Steps: (A) Predict edge velocities (B) Transverse+viscous+pressure corrections
   (C) MAC projection (D) Advection from projected edges (E) CN update */
static void advect_diffuse(Real dt) {
  Real h0 = sim.blk[0].h;
  int ns = sim.nb[0] * (1 << (sim.blk[0].level - sim.levelStart));
  int Ng = ns * BS;
  Real ih = 1.0/h0;

  /* Flat MAC face arrays: umac at x-faces (Ng+1)*Ng, vmac at y-faces Ng*(Ng+1)
     On periodic grid, face (Ng) wraps to face (0), so store Ng faces each. */
  double *umac = calloc(Ng*Ng, sizeof(double)); /* u at face (i+1/2, j) for i=0..Ng-1 */
  double *vmac = calloc(Ng*Ng, sizeof(double)); /* v at face (i, j+1/2) for j=0..Ng-1 */
  /* Also store transported quantities at faces for advection flux */
  double *umac_v = calloc(Ng*Ng, sizeof(double)); /* v at x-faces */
  double *vmac_u = calloc(Ng*Ng, sizeof(double)); /* u at y-faces */

  /* Pass 1: Godunov edge prediction → fill umac, vmac, umac_v, vmac_u */
#pragma omp parallel
  {
    Real bu[LB_BUF], bv[LB_BUF], bp[LB_BUF];
#pragma omp for
    for (long long id = 0; id < sim.n; id++) {
      lb_load(bu, 1, F_U, 2, id);
      lb_load(bv, 1, F_V, 2, id);
      lb_load(bp, 1, F_P, 1, id);
      int ss=2, nm=2*ss+BS;
      int ss1=1, nm1=2*ss1+BS;
      int bx = (int)(sim.blk[id].origin[0]/h0 + 0.5);
      int by = (int)(sim.blk[id].origin[1]/h0 + 0.5);
      Real c_dt = 0.5*dt*ih;
#define U(di,dj) bu[nm*((j)+(dj)+ss)+(i)+(di)+ss]
#define V(di,dj) bv[nm*((j)+(dj)+ss)+(i)+(di)+ss]
#define P(di,dj) bp[nm1*((j)+(dj)+ss1)+(i)+(di)+ss1]
      for (int j = 0; j < BS; j++)
        for (int i = 0; i < BS; i++) {
          Real uc = U(0,0), vc = V(0,0);
          /* Slopes */
          Real su = slope4(U(-2,0),U(-1,0),U(0,0),U(1,0),U(2,0));
          Real sv = slope4(V(-2,0),V(-1,0),V(0,0),V(1,0),V(2,0));
          Real su_p = minmod(U(1,0)-U(0,0), U(2,0)-U(1,0));
          Real sv_p = minmod(V(1,0)-V(0,0), V(2,0)-V(1,0));
          Real su_y = slope4(U(0,-2),U(0,-1),U(0,0),U(0,1),U(0,2));
          Real sv_y = slope4(V(0,-2),V(0,-1),V(0,0),V(0,1),V(0,2));
          Real su_yp = minmod(U(0,1)-U(0,0), U(0,2)-U(0,1));
          Real sv_yp = minmod(V(0,1)-V(0,0), V(0,2)-V(0,1));

          /* Right face (i+1/2): Riemann solve */
          Real sL = uc > 0 ? 1 : 0;
          Real sR = U(1,0) < 0 ? 1 : 0;
          Real uR_L = U(0,0) + (0.5 - sL*c_dt*uc)*su;
          Real vR_L = V(0,0) + (0.5 - sL*c_dt*uc)*sv;
          Real uR_R = U(1,0) + (-0.5 - sR*c_dt*U(1,0))*su_p;
          Real vR_R = V(1,0) + (-0.5 - sR*c_dt*U(1,0))*sv_p;
          Real u_xR, v_xR;
          if (uc > 0 && U(1,0) > 0)      { u_xR=uR_L; v_xR=vR_L; }
          else if (uc < 0 && U(1,0) < 0) { u_xR=uR_R; v_xR=vR_R; }
          else                            { u_xR=0.5*(uR_L+uR_R); v_xR=0.5*(vR_L+vR_R); }

          /* Top face (j+1/2): Riemann solve */
          sL = vc > 0 ? 1 : 0;
          sR = V(0,1) < 0 ? 1 : 0;
          Real uT_L = U(0,0) + (0.5 - sL*c_dt*vc)*su_y;
          Real vT_L = V(0,0) + (0.5 - sL*c_dt*vc)*sv_y;
          Real uT_R = U(0,1) + (-0.5 - sR*c_dt*V(0,1))*su_yp;
          Real vT_R = V(0,1) + (-0.5 - sR*c_dt*V(0,1))*sv_yp;
          Real u_yT, v_yT;
          if (vc > 0 && V(0,1) > 0)      { u_yT=uT_L; v_yT=vT_L; }
          else if (vc < 0 && V(0,1) < 0) { u_yT=uT_R; v_yT=vT_R; }
          else                            { u_yT=0.5*(uT_L+uT_R); v_yT=0.5*(vT_L+vT_R); }

          /* Add transverse + viscous + pressure corrections (Eq. 22) */
          Real lap_u = (U(1,0)+U(-1,0)+U(0,1)+U(0,-1)-4*uc)*ih*ih;
          Real lap_v = (V(1,0)+V(-1,0)+V(0,1)+V(0,-1)-4*vc)*ih*ih;
          Real dpx = (P(1,0)-P(-1,0))*0.5*ih;
          Real dpy = (P(0,1)-P(0,-1))*0.5*ih;
          Real cu = 0.5*dt*(NU*lap_u - dpx);
          Real cv = 0.5*dt*(NU*lap_v - dpy);

          /* Store at right face (i+1/2, j) — each face written once by the left cell */
          int fi = (bx+i+1)%Ng, fj = by+j;
          umac[fj*Ng+fi] = u_xR + cu;
          umac_v[fj*Ng+fi] = v_xR + cv;

          /* Store at top face (i, j+1/2) */
          fi = bx+i; fj = (by+j+1)%Ng;
          vmac[fj*Ng+fi] = v_yT + cv;
          vmac_u[fj*Ng+fi] = u_yT + cu;
        }
#undef U
#undef V
#undef P
    }
  }

  /* Pass 2: MAC projection — make umac,vmac divergence-free.
     D_MAC = (umac_{i+1/2} - umac_{i-1/2})/h + (vmac_{j+1/2} - vmac_{j-1/2})/h
     Solve: ∆⁵ φ_mac = D_MAC  (standard 5-point compact Laplacian)
     Correct: umac -= (φ_{i+1}-φ_i)/h, vmac -= (φ_{j+1}-φ_j)/h */
  {
    double *div_mac = calloc(Ng*Ng, sizeof(double));
    double *phi_mac = calloc(Ng*Ng, sizeof(double));
    /* Compute MAC divergence */
    for (int j=0;j<Ng;j++) for (int i=0;i<Ng;i++) {
      int ip1=(i+1)%Ng, jp1=(j+1)%Ng;
      div_mac[j*Ng+i] = (umac[j*Ng+ip1]-umac[j*Ng+i]+vmac[jp1*Ng+i]-vmac[j*Ng+i])*ih;
    }
    /* Solve ∆⁵ φ = div*h² using multigrid-preconditioned CG */
    for (int k=0;k<Ng*Ng;k++) div_mac[k] *= h0*h0;
    mg_solve_periodic(phi_mac, div_mac, Ng, 1e-10);

    /* Correct MAC velocities: umac -= (φ_{i}-φ_{i-1})/h, vmac -= (φ_{j}-φ_{j-1})/h
       Note: face (i+1/2) stored at index i+1, so φ gradient uses φ[i+1]-φ[i] */
    for (int j=0;j<Ng;j++) for (int i=0;i<Ng;i++) {
      int im=(i-1+Ng)%Ng, jm=(j-1+Ng)%Ng;
      umac[j*Ng+i]   -= (phi_mac[j*Ng+i]-phi_mac[j*Ng+im])*ih;
      umac_v[j*Ng+i] -= 0; /* only correct advecting velocity, not transported */
      vmac[j*Ng+i]   -= (phi_mac[j*Ng+i]-phi_mac[jm*Ng+i])*ih;
      vmac_u[j*Ng+i] -= 0;
    }
    free(div_mac); free(phi_mac);
  }

  /* Pass 3: Compute advection from MAC-projected faces and form CN RHS */
#pragma omp parallel
  {
    Real bu[LB_BUF], bv[LB_BUF], bp[LB_BUF];
#pragma omp for
    for (long long id = 0; id < sim.n; id++) {
      lb_load(bu, 1, F_U, 1, id);
      lb_load(bv, 1, F_V, 1, id);
      lb_load(bp, 1, F_P, 1, id);
      Real *u = BLK(id)+BS*BS*F_U;
      Real *v = BLK(id)+BS*BS*F_V;
      int ss1=1, nm1=2*ss1+BS;
      int bx = (int)(sim.blk[id].origin[0]/h0 + 0.5);
      int by = (int)(sim.blk[id].origin[1]/h0 + 0.5);
#define UU(di,dj) bu[nm1*((j)+(dj)+ss1)+(i)+(di)+ss1]
#define VV(di,dj) bv[nm1*((j)+(dj)+ss1)+(i)+(di)+ss1]
#define PP(di,dj) bp[nm1*((j)+(dj)+ss1)+(i)+(di)+ss1]
      for (int j=0;j<BS;j++) for (int i=0;i<BS;i++) {
        int k = j*BS+i;
        Real uc = UU(0,0), vc = VV(0,0);
        int gi=bx+i, gj=by+j;
        int gip=(gi+1)%Ng, gjp=(gj+1)%Ng;
        /* MAC face values */
        Real uR = umac[gj*Ng+gip];    /* u at right face (i+1/2) */
        Real uL = umac[gj*Ng+gi];     /* u at left face (i-1/2) stored at i */
        Real vT = vmac[gjp*Ng+gi];    /* v at top face (j+1/2) */
        Real vB = vmac[gj*Ng+gi];     /* v at bottom face (j-1/2) stored at j */
        /* Transported quantities at faces */
        Real u_at_yT = vmac_u[gjp*Ng+gi];
        Real u_at_yB = vmac_u[gj*Ng+gi];
        Real v_at_xR = umac_v[gj*Ng+gip];
        Real v_at_xL = umac_v[gj*Ng+gi];
        /* Advection: (u·∇)U using MAC-projected advecting velocity */
        Real adv_u = uR*(uR>0?uc:UU(1,0))*ih - uL*(uL>0?UU(-1,0):uc)*ih
                   + vT*u_at_yT*ih - vB*u_at_yB*ih;
        Real adv_v = uR*v_at_xR*ih - uL*v_at_xL*ih
                   + vT*(vT>0?vc:VV(0,1))*ih - vB*(vB>0?VV(0,-1):vc)*ih;
        /* Viscous and pressure (centered at cell) */
        Real lap_u = (UU(1,0)+UU(-1,0)+UU(0,1)+UU(0,-1)-4*uc)*ih*ih;
        Real lap_v = (VV(1,0)+VV(-1,0)+VV(0,1)+VV(0,-1)-4*vc)*ih*ih;
        Real dpx = (PP(1,0)-PP(-1,0))*0.5*ih;
        Real dpy = (PP(0,1)-PP(0,-1))*0.5*ih;
        /* CN RHS */
        Real alpha = NU*dt*0.5;
        u[k] = uc + alpha*lap_u + dt*(-adv_u - dpx);
        v[k] = vc + alpha*lap_v + dt*(-adv_v - dpy);
      }
#undef UU
#undef VV
#undef PP
    }
  }
  free(umac); free(vmac); free(umac_v); free(vmac_u);
}

/* Helmholtz solve: (I - α∆)U* = RHS for one scalar field.
   After advect_diffuse, F_U and F_V contain the RHS.
   Solve in-place: on entry field has RHS, on exit has solution U*. */
static void helmholtz_solve(Real dt, int field) {
  Real alpha = NU * dt * 0.5;
  int N = BS * BS * sim.n;
  sim.sol_x = realloc(sim.sol_x, N * sizeof(double));
  sim.sol_b = realloc(sim.sol_b, N * sizeof(double));
  sim.sol_h2 = realloc(sim.sol_h2, sim.n * sizeof(double));
  sim.coo_nnz = 0;
  if (sim.coo_cap < 8 * N) {
    sim.coo_cap = 8 * N;
    sim.coo_val = realloc(sim.coo_val, sim.coo_cap * sizeof(double));
    sim.coo_row = realloc(sim.coo_row, sim.coo_cap * sizeof(int));
    sim.coo_col = realloc(sim.coo_col, sim.coo_cap * sizeof(int));
  }
#define COO(v, r, c) do { \
    sim.coo_val[sim.coo_nnz]=(v); sim.coo_row[sim.coo_nnz]=(r); \
    sim.coo_col[sim.coo_nnz]=(c); sim.coo_nnz++; } while(0)
  /* Assemble (I - α∆_h) where ∆_h = (-4 + Σnb)/h².
     Matrix entry: (1 + 4α/h²) on diagonal, (-α/h²) on neighbors. */
  static const int nb_dx[4] = {-1, 1, 0, 0};
  static const int nb_dy[4] = {0, 0, -1, 1};
  static const int nb_ic[4] = {3, 5, 1, 7};
  for (long long i = 0; i < sim.n; i++) {
    struct Blk *info = &sim.blk[i];
    Real h = info->h;
    Real ah2 = alpha / (h * h);
    for (int iy = 0; iy < BS; iy++)
      for (int ix = 0; ix < BS; ix++) {
        int sfc = i * BS * BS + iy * BS + ix;
        COO(1.0 + 4.0 * ah2, sfc, sfc);
        for (int d = 0; d < 4; d++) {
          int nx = ix + nb_dx[d], ny = iy + nb_dy[d];
          if (nx >= 0 && nx < BS && ny >= 0 && ny < BS) {
            COO(-ah2, sfc, i * BS * BS + ny * BS + nx);
          } else {
            struct Nb pnr = nb_find(info->level, info->ix, info->iy, nb_ic[d]);
            if (pnr.s != 0 || pnr.idx < 0) continue;
            int nnx = ((nx % BS) + BS) % BS;
            int nny = ((ny % BS) + BS) % BS;
            COO(-ah2, sfc, pnr.idx * BS * BS + nny * BS + nnx);
          }
        }
      }
  }
#undef COO
  /* Gather RHS and initial guess */
#pragma omp parallel for
  for (long long i = 0; i < sim.n; i++) {
    sim.sol_h2[i] = sim.blk[i].h * sim.blk[i].h;
    memcpy(&sim.sol_b[i*BS*BS], BLK(i) + BS*BS*field, BS*BS*sizeof(Real));
    memcpy(&sim.sol_x[i*BS*BS], BLK(i) + BS*BS*field, BS*BS*sizeof(Real));
  }
  solver_solve(sim.helm_solver, 1, N, sim.coo_nnz,
      sim.coo_val, sim.coo_row, sim.coo_col,
      sim.sol_x, sim.sol_b, sim.sol_h2, -1,
      1e-10, 1e-6, 50);
  /* Scatter solution */
#pragma omp parallel for
  for (long long i = 0; i < sim.n; i++)
    memcpy(BLK(i) + BS*BS*field, &sim.sol_x[i*BS*BS], BS*BS*sizeof(Real));
}

/* Poisson solve: Laplacian(phi) = div(u*)/dt */
static void poisson_solve(Real dt) {
  /* RHS = div(u*)/dt stored in F_TMP */
#pragma omp parallel
  {
    Real bu[LB_BUF], bv[LB_BUF];
#pragma omp for
    for (long long id = 0; id < sim.n; id++) {
      lb_load(bu, 1, F_U, 1, id);
      lb_load(bv, 1, F_V, 1, id);
      Real *rhs = BLK(id)+BS*BS*F_TMP;
      int ss=1, nm=2*ss+BS;
      Real h = sim.blk[id].h;
      /* RHS = (4h²) * div(u*) / dt for wide Laplacian L = (-4φ + Σφ_{±2})/(4h²)
         div = (u_{i+1}-u_{i-1})/(2h) + (v_{j+1}-v_{j-1})/(2h)
         So rhs = 4h² * div / dt = 2h/dt * (u_{i+1}-u_{i-1}+v_{j+1}-v_{j-1}) */
      Real fac = 2.0 * h / dt;
      for (int j=0;j<BS;j++) for (int i=0;i<BS;i++)
        rhs[j*BS+i] = fac * (bu[nm*(j+ss)+i+1+ss]-bu[nm*(j+ss)+i-1+ss]
                             +bv[nm*(j+1+ss)+i+ss]-bv[nm*(j-1+ss)+i+ss]);
    }
  }
  /* Multigrid Poisson solve on flat periodic grid.
     The wide Laplacian L = (-4φ + φ_{i±2} + φ_{j±2}) decouples into 4 sub-grids
     based on (i%2, j%2). Each sub-grid is a standard 5-point Laplacian on (N/2)².
     Solve each sub-grid independently with multigrid V-cycles. */
  Real h = sim.blk[0].h;
  int ns = sim.nb[0] * (1 << (sim.blk[0].level - sim.levelStart));
  int Ng = ns * BS; /* full grid size */
  int M = Ng / 2;  /* sub-grid size */

  /* Gather RHS and phi from blocks into flat N×N array */
  double *rhs_full = calloc(Ng * Ng, sizeof(double));
  double *phi_full = calloc(Ng * Ng, sizeof(double));
  for (long long id = 0; id < sim.n; id++) {
    Real *rhs = BLK(id)+BS*BS*F_TMP;
    Real *phi = BLK(id)+BS*BS*F_PHI;
    int bx = (int)(sim.blk[id].origin[0] / h + 0.5);
    int by = (int)(sim.blk[id].origin[1] / h + 0.5);
    for (int j=0;j<BS;j++) for (int i=0;i<BS;i++) {
      rhs_full[(by+j)*Ng + bx+i] = rhs[j*BS+i];
      phi_full[(by+j)*Ng + bx+i] = phi[j*BS+i];
    }
  }

  /* For each sub-grid (sx, sy) in {0,1}², extract, solve, scatter back */
  for (int sy = 0; sy < 2; sy++)
    for (int sx = 0; sx < 2; sx++) {
      /* Extract sub-grid: cell (i,j) in full grid → (i/2, j/2) in sub-grid
         if i%2==sx && j%2==sy */
      double *f = calloc(M * M, sizeof(double));
      double *x = calloc(M * M, sizeof(double));
      for (int j=0;j<M;j++) for (int i=0;i<M;i++) {
        f[j*M+i] = rhs_full[(2*j+sy)*Ng + 2*i+sx];
        x[j*M+i] = phi_full[(2*j+sy)*Ng + 2*i+sx];
      }

      /* Solve using multigrid-preconditioned CG */
      mg_solve_periodic(x, f, M, 1e-10);

      /* Scatter back to full grid */
      for (int j=0;j<M;j++) for (int i=0;i<M;i++)
        phi_full[(2*j+sy)*Ng + 2*i+sx] = x[j*M+i];

      free(f); free(x);
    }

  /* Check wide Laplacian residual on full grid */
  { double rmax = 0;
    for (int j=0;j<Ng;j++) for (int i=0;i<Ng;i++) {
      int ip=(i+2)%Ng, im=(i-2+Ng)%Ng, jp=(j+2)%Ng, jm=(j-2+Ng)%Ng;
      double r = rhs_full[j*Ng+i] - (-4*phi_full[j*Ng+i]+phi_full[j*Ng+ip]+phi_full[j*Ng+im]+phi_full[jp*Ng+i]+phi_full[jm*Ng+i]);
      if (fabs(r) > rmax) rmax = fabs(r);
    }
    if (rmax > 1e-4) fprintf(stderr, "  poisson res=%.2e\n", rmax);
  }

  /* Scatter back to blocks */
  for (long long id = 0; id < sim.n; id++) {
    Real *phi = BLK(id)+BS*BS*F_PHI;
    int bx = (int)(sim.blk[id].origin[0] / h + 0.5);
    int by = (int)(sim.blk[id].origin[1] / h + 0.5);
    for (int j=0;j<BS;j++) for (int i=0;i<BS;i++)
      phi[j*BS+i] = phi_full[(by+j)*Ng + bx+i];
  }
  free(rhs_full); free(phi_full);
}

/* Project: u^{n+1} = u* - dt*grad(phi), update pressure */
static void project(Real dt) {
#pragma omp parallel
  {
    Real bp[LB_BUF];
#pragma omp for
    for (long long id = 0; id < sim.n; id++) {
      lb_load(bp, 1, F_PHI, 1, id);
      Real *u = BLK(id)+BS*BS*F_U;
      Real *v = BLK(id)+BS*BS*F_V;
      Real *p = BLK(id)+BS*BS*F_P;
      Real *phi = BLK(id)+BS*BS*F_PHI;
      int ss=1, nm=2*ss+BS;
      Real ih = 0.5/sim.blk[id].h;
      for (int j=0;j<BS;j++) for (int i=0;i<BS;i++) {
        int k = j*BS+i;
#define PH(di,dj) bp[nm*((j)+(dj)+ss)+(i)+(di)+ss]
        u[k] -= dt * (PH(1,0)-PH(-1,0))*ih;
        v[k] -= dt * (PH(0,1)-PH(0,-1))*ih;
        p[k] += phi[k]; /* accumulate pressure */
#undef PH
      }
    }
  }
}

static const struct {
  const char *name; int type; size_t off;
} param_tab[] = {
  {"levelMax", 0, offsetof(struct Sim, levelMax)},
  {"AdaptSteps", 0, offsetof(struct Sim, AdaptSteps)},
  {"levelStart", 0, offsetof(struct Sim, levelStart)},
  {"Rtol", 1, offsetof(struct Sim, Rtol)},
  {"Ctol", 1, offsetof(struct Sim, Ctol)},
  {"CFL", 1, offsetof(struct Sim, CFL)},
  {"tend", 1, offsetof(struct Sim, endTime)},
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
  int dumpSteps = arg_i_opt(argc, argv, "sdump", 0);
  NU = arg_r(argc, argv, "nu");

  /* Domain: [0,1]^2, doubly periodic (Brown & Minion 1995) */
  sim.L[0] = 1.0; sim.L[1] = 1.0;
  sim.bc[0].type = BC_PERIODIC; sim.bc[1].type = BC_PERIODIC;
  sim.bc[2].type = BC_PERIODIC; sim.bc[3].type = BC_PERIODIC;
  {
    int ns = 1 << sim.levelStart;
    sim.nb[0] = ns; sim.nb[1] = ns;
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
  ps_load();
  { double P_inv[BS*BS*BS*BS];
    ps_prec(P_inv); sim.solver = solver_create(BS*BS, P_inv);
    /* Identity preconditioner for Helmholtz (well-conditioned) */
    memset(P_inv, 0, sizeof P_inv);
    for (int i = 0; i < BS*BS; i++) P_inv[i*BS*BS+i] = -1.0;
    sim.helm_solver = solver_create(BS*BS, P_inv);
  }

  /* IC: double shear layer (Eq. 27-28) */
  Real rho_layer = 30.0, delta = 0.05;
  fprintf(stderr, "main.c: IC rho=%g delta=%g nu=%g\n", rho_layer, delta, NU);
#pragma omp parallel for
  for (long long i = 0; i < sim.n; i++) {
    struct Blk *info = &sim.blk[i];
    Real *u = BLK(i)+BS*BS*F_U;
    Real *v = BLK(i)+BS*BS*F_V;
    Real h = info->h;
    for (int iy = 0; iy < BS; iy++)
      for (int ix = 0; ix < BS; ix++) {
        int j = BS*iy+ix;
        Real x = info->origin[0] + (ix+0.5)*h;
        Real y = info->origin[1] + (iy+0.5)*h;
        u[j] = y <= 0.5 ? tanh(rho_layer*(y-0.25)) : tanh(rho_layer*(0.75-y));
        v[j] = delta * sin(2*M_PI*x);
      }
  }

  /* Compute initial pressure p^{-1/2} by iteration (paper Sec. 3).
     Run a few projection steps with dt→0 to establish pressure field. */
  {
    Real dt0 = sim.blk[0].h / 10.0; /* small dt */
    for (int iter = 0; iter < 5; iter++) {
      advect_diffuse(dt0);
      helmholtz_solve(dt0, F_U);
      helmholtz_solve(dt0, F_V);
      poisson_solve(dt0);
      /* Don't project — just accumulate pressure, then restore velocity */
#pragma omp parallel for
      for (long long i = 0; i < sim.n; i++) {
        Real *p = BLK(i)+BS*BS*F_P;
        Real *phi = BLK(i)+BS*BS*F_PHI;
        for (int j = 0; j < BS*BS; j++) p[j] += phi[j];
      }
      /* Restore IC velocity */
#pragma omp parallel for
      for (long long i = 0; i < sim.n; i++) {
        struct Blk *info = &sim.blk[i];
        Real *uu = BLK(i)+BS*BS*F_U;
        Real *vv = BLK(i)+BS*BS*F_V;
        Real hh = info->h;
        for (int iy = 0; iy < BS; iy++)
          for (int ix = 0; ix < BS; ix++) {
            int j = BS*iy+ix;
            Real x = info->origin[0]+(ix+0.5)*hh;
            Real y = info->origin[1]+(iy+0.5)*hh;
            uu[j] = y <= 0.5 ? tanh(rho_layer*(y-0.25)) : tanh(rho_layer*(0.75-y));
            vv[j] = delta * sin(2*M_PI*x);
          }
      }
    }
    fprintf(stderr, "main.c: initial pressure computed\n");
  }

  /* Main loop */
  while (1) {
    if (sim.step % 10 == 0) {
      compute_vorticity();
      fprintf(stderr, "main.c: %08d %.6e dt=%.3e blk=%lld\n",
              sim.step, sim.time, sim.dt, sim.n);
    }
    {
      int do_dump = 0;
      if (sim.dumpTime > 0 && sim.time >= sim.nextDumpTime) {
        sim.nextDumpTime += sim.dumpTime; do_dump = 1;
      }
      if (dumpSteps > 0 && sim.step % dumpSteps == 0) do_dump = 1;
      if (do_dump) {
        compute_vorticity();
        char path[FILENAME_MAX];
        snprintf(path, sizeof path, "vel.%08d", sim.dump_count++);
        dump(sim.time, sim.step, path);
      }
    }
    if (sim.endTime > 0 && sim.time >= sim.endTime) break;

    /* CFL */
    Real smax = 0;
#pragma omp parallel for reduction(max:smax)
    for (long long i = 0; i < sim.n; i++) {
      Real *u = BLK(i)+BS*BS*F_U;
      Real *v = BLK(i)+BS*BS*F_V;
      Real ih = 1.0/sim.blk[i].h;
      for (int j = 0; j < BS*BS; j++)
        smax = fmax(smax, fmax(fabs(u[j]), fabs(v[j]))*ih);
    }
    sim.dt = sim.CFL / (smax + 1e-30);

    if (sim.step > 0 && sim.step % sim.AdaptSteps == 0) ad_run();

    /* Projection method time step */
    advect_diffuse(sim.dt);       /* Godunov advection → RHS in F_U, F_V */
    helmholtz_solve(sim.dt, F_U); /* Crank-Nicolson viscosity for u */
    helmholtz_solve(sim.dt, F_V); /* Crank-Nicolson viscosity for v */
    poisson_solve(sim.dt);        /* ∆φ = ∇·U* */
    project(sim.dt);              /* U^{n+1} = U* - dt∇φ */

    sim.time += sim.dt;
    sim.step++;
  }
  solver_destroy(sim.solver);
  solver_destroy(sim.helm_solver);
  free(sim.coo_val); free(sim.coo_row); free(sim.coo_col);
  free(sim.sol_x); free(sim.sol_b); free(sim.sol_h2);
  fprintf(stderr, "main.c: end\n");
}
