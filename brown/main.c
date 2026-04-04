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
#else
#define omp_get_max_threads() 1
#endif

typedef double Real;
enum { BS = 8 };
enum {
  F_U = 0,
  F_V = 1,
  F_P = 2,
  F_PHI = 3,
  F_W = 4,
  F_TMP = 5,
  F_N = 6,
  BLK_S = F_N * BS * BS,
};

enum AdSt { Leave = 0, Refine = 1, Compress = -1, Dealloc = 2 };
struct Blk;
struct HMEntry {
  long long key;
  int val;
};
struct HMap {
  struct HMEntry *e;
  int cap;
};
static int hm_slot(const struct HMap *m, long long key) {
  unsigned long long h = (unsigned long long)key * 0x9E3779B97F4A7C15ULL;
  return (int)(h >> 32) & (m->cap - 1);
}
static int hm_get(const struct HMap *m, long long key) {
  int i = hm_slot(m, key);
  while (m->e[i].key >= 0) {
    if (m->e[i].key == key)
      return m->e[i].val;
    i = (i + 1) & (m->cap - 1);
  }
  return -1;
}
static struct Sim {
  int AdaptSteps;
  int levelMax;
  int levelStart;
  int step;
  int sdump;
  int dump_count;
  Real CFL;
  Real dt;
  Real dumpTime;
  Real endTime;
  Real nextDumpTime;
  Real Rtol;
  Real nu;
  Real time;
  Real L[2];
  int nb[2];
  long long n;
  struct HMap hm;
  struct Blk *blk;
  Real *fld;
} sim;
static long long hm_key(int level, int ix, int iy) {
  long long n = 1LL << level;
  return ((n * n) - 1) / 3 + iy * n + ix;
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
} fld_t[] = {{F_U, 1, "u"},    {F_V, 1, "v"},    {F_P, 1, "p"},
             {F_PHI, 1, NULL}, {F_W, 1, "vort"}, {F_TMP, 1, NULL}};
enum { NVARS = sizeof fld_t / sizeof *fld_t };

static const int nb_ch_off[9][2][2] = {
    [0] = {{-1, -1}, {0, 0}}, [1] = {{0, -1}, {1, -1}}, [2] = {{2, -1}, {0, 0}},
    [3] = {{-1, 0}, {-1, 1}}, [4] = {{0, 0}, {0, 0}},   [5] = {{2, 0}, {2, 1}},
    [6] = {{-1, 2}, {0, 0}},  [7] = {{0, 2}, {1, 2}},   [8] = {{2, 2}, {0, 0}},
};
static const int nb_ch_n[9] = {1, 2, 1, 2, 0, 2, 1, 2, 1};
static void hm_rebuild(void) {
  int cap = 1;
  while (cap < 4 * sim.n)
    cap <<= 1;
  if (sim.hm.cap != cap) {
    free(sim.hm.e);
    sim.hm.cap = cap;
    sim.hm.e = malloc(cap * sizeof *sim.hm.e);
  }
  for (int j = 0; j < cap; j++)
    sim.hm.e[j].key = -1;
  for (long long i = 0; i < sim.n; i++) {
    long long key = hm_key(sim.blk[i].level, sim.blk[i].ix, sim.blk[i].iy);
    int s = hm_slot(&sim.hm, key);
    while (sim.hm.e[s].key >= 0 && sim.hm.e[s].key != key)
      s = (s + 1) & (cap - 1);
    sim.hm.e[s].key = key;
    sim.hm.e[s].val = i;
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
  int nd[2] = {sim.nb[0] * scale, sim.nb[1] * scale};
  int pos[2] = {ix, iy};
  int c[2] = {cx, cy};

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
                    const struct LbOp *ops, int n, int dim, int nm, int nc) {
  Real *m = dst[0], *c = dst[1];
  for (int i = 0; i < n; i++) {
    const struct LbOp *o = &ops[i];
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
            sum += w[3 * jj + ii] *
                   c[o->src_off + d + dim * ((ii - 1) + nc * (jj - 1))];
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
    }
  }
}

enum { N_STATUS = 10 };
static const struct LbTab (*lb_tab[5][3])[3][2][2][N_STATUS];
static void lb_init(void) {
  int configs[][2] = {{1, 1}, {1, 2}, {2, 1}, {4, 1}};
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
    lb_tab[ss][dim] = (const struct LbTab(*)[3][2][2][N_STATUS])tab;
  }
}
enum {
  LB_BUF =
      ((2 * 4 + BS) * (2 * 4 + BS) + (BS / 2 + 4 + 3) * (BS / 2 + 4 + 3)) * 2
};
static void lb_load(Real *m, int dim, int blk_offset, int ss,
                    long long info_idx) {
  struct Blk *info = &sim.blk[info_idx];
  int nm = 2 * ss + BS;
  int nc = BS / 2 + ss + 3;
  int level = info->level;
  int xi = info->ix, yi = info->iy;
  const struct LbTab(*cflb_tab)[3][2][2][N_STATUS] = lb_tab[ss][dim];

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
    const struct LbTab *te = &cflb_tab[cx + 1][cy + 1][xi % 2][yi % 2][nr.s];
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
    lb_exec(dirs[i].blk, dst, dirs[i].e->ops, dirs[i].e->n_pre, dim, nm, nc);
  for (int i = 0; i < nd; i++)
    lb_exec(dirs[i].blk, dst, dirs[i].e->ops + MAX_PRE, dirs[i].e->n_post, dim,
            nm, nc);
}


static inline Real minmod(Real a, Real b) {
  return a * b <= 0 ? 0 : fabs(a) < fabs(b) ? a : b;
}

static void compute_vorticity(void) {
#pragma omp parallel
  {
    Real bu[LB_BUF], bv[LB_BUF];
#pragma omp for
    for (long long id = 0; id < sim.n; id++) {
      lb_load(bu, 1, F_U, 1, id);
      lb_load(bv, 1, F_V, 1, id);
      Real *w = BLK(id) + BS * BS * F_W;
      int ss = 1, nm = 2 * ss + BS;
      Real ih = 0.5 / sim.blk[id].h;
      for (int j = 0; j < BS; j++)
        for (int i = 0; i < BS; i++) {
#define U(di, dj) bu[nm * ((j) + (dj) + ss) + (i) + (di) + ss]
#define V(di, dj) bv[nm * ((j) + (dj) + ss) + (i) + (di) + ss]
          w[j * BS + i] = (V(1, 0) - V(-1, 0)) * ih - (U(0, 1) - U(0, -1)) * ih;
#undef U
#undef V
        }
    }
  }
}

static const Real ad_ref_w[4][9] = {
    {1. / 64, 10. / 64, -1. / 64, 10. / 64, 56. / 64, -6. / 64, -1. / 64,
     -6. / 64, 1. / 64},
    {-1. / 64, 10. / 64, 1. / 64, -6. / 64, 56. / 64, 10. / 64, 1. / 64,
     -6. / 64, -1. / 64},
    {-1. / 64, -6. / 64, 1. / 64, 10. / 64, 56. / 64, -6. / 64, 1. / 64,
     10. / 64, -1. / 64},
    {1. / 64, -6. / 64, -1. / 64, -6. / 64, 56. / 64, 10. / 64, -1. / 64,
     10. / 64, 1. / 64},
};

static void compute_indicator(void) {
#pragma omp parallel
  {
    Real bu[LB_BUF], bv[LB_BUF];
    int ss = 1, nm = 2 * ss + BS;

    int nc = BS / 2 + 2;
    Real cu[nc * nc], cv[nc * nc];
#pragma omp for
    for (long long id = 0; id < sim.n; id++) {
      lb_load(bu, 1, F_U, ss, id);
      lb_load(bv, 1, F_V, ss, id);

      for (int jc = 0; jc < nc; jc++)
        for (int ic = 0; ic < nc; ic++) {
          int fi = 2 * ic - 1, fj = 2 * jc - 1;
          Real su = 0, sv = 0;
          for (int dj = 0; dj < 2; dj++)
            for (int di = 0; di < 2; di++) {
              su += bu[nm * (fj + dj + ss) + fi + di + ss];
              sv += bv[nm * (fj + dj + ss) + fi + di + ss];
            }
          cu[jc * nc + ic] = su * 0.25;
          cv[jc * nc + ic] = sv * 0.25;
        }

      Real *t = BLK(id) + BS * BS * F_TMP;
      for (int j = 0; j < BS; j += 2)
        for (int i = 0; i < BS; i += 2) {
          int ic = i / 2 + 1, jc = j / 2 + 1;
          for (int s = 0; s < 4; s++) {
            int di = s & 1, dj = s >> 1;
            Real pu = 0, pv = 0;
            for (int kk = 0; kk < 9; kk++) {
              int ci = ic + kk % 3 - 1, cj = jc + kk / 3 - 1;
              pu += ad_ref_w[s][kk] * cu[cj * nc + ci];
              pv += ad_ref_w[s][kk] * cv[cj * nc + ci];
            }
            Real au = bu[nm * (j + dj + ss) + i + di + ss];
            Real av = bv[nm * (j + dj + ss) + i + di + ss];
            t[BS * (j + dj) + i + di] = fmax(fabs(au - pu), fabs(av - pv));
          }
        }
    }
  }
}

static void dump(Real time, int step, char *path) {
  long i, j, k;
  char attr_path[FILENAME_MAX], xyz_path[FILENAME_MAX];
  FILE *file;
  float xyz[4 * BS * BS][2];
  snprintf(xyz_path, sizeof xyz_path, "%s.xyz.raw", path);
  file = fopen(xyz_path, "wb");
  for (i = 0; i < sim.n; i++) {
    Real h = sim.blk[i].h, ox = sim.blk[i].origin[0], oy = sim.blk[i].origin[1];
    for (j = 0; j < BS; j++)
      for (k = 0; k < BS; k++) {
        int c = j * BS + k;
        float x0 = ox + k * h, y0 = oy + j * h, x1 = x0 + h, y1 = y0 + h;
        xyz[4 * c + 0][0] = x0;
        xyz[4 * c + 0][1] = y0;
        xyz[4 * c + 1][0] = x0;
        xyz[4 * c + 1][1] = y1;
        xyz[4 * c + 2][0] = x1;
        xyz[4 * c + 2][1] = y1;
        xyz[4 * c + 3][0] = x1;
        xyz[4 * c + 3][1] = y0;
      }
    fwrite(xyz, sizeof(float), 4 * 2 * BS * BS, file);
  }
  fclose(file);
  for (size_t fi = 0; fi < NVARS; fi++)
    if (fld_t[fi].prefix) {
      int dim = fld_t[fi].dim, offset = fld_t[fi].offset;
      snprintf(attr_path, sizeof attr_path, "%s.%s.raw", path,
               fld_t[fi].prefix);
      file = fopen(attr_path, "wb");
      for (j = 0; j < sim.n; j++)
        fwrite(BLK(j) + offset * BS * BS, sizeof(Real), dim * BS * BS, file);
      fclose(file);
    }
}

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
    Real *b = BLK(i) + BS * BS * F_TMP;
    double Linf = 0;
    for (int j = 0; j < BS * BS; j++)
      Linf = fmax(Linf, fabs(b[j]));
    int lev = sim.blk[i].level;
    state[i] = Linf > sim.Rtol && lev < sim.levelMax           ? Refine
               : Linf < sim.Rtol / 1.5 && lev > sim.levelStart ? Compress
                                                               : Leave;
    Changed |= state[i] != Leave;
  }
  if (!Changed)
    goto done;
  for (int More = 1; More;) {
    More = 0;
    for (long long j = 0; j < sim.n; j++) {
      if (state[j] != Refine)
        continue;
      struct Blk *bj = &sim.blk[j];
      for (int ic = 0; ic < 9; ic++) {
        if (ic == 4)
          continue;
        struct Nb nr = nb_find(bj->level, bj->ix, bj->iy, ic);
        if (nr.s >= 3 || nr.idx < 0)
          continue;
        if (nr.s == 2 && state[nr.idx] != Refine) {
          state[nr.idx] = Refine;
          More = 1;
        } else if (nr.s == 0 && state[nr.idx] == Compress)
          state[nr.idx] = Leave;
      }
    }
  }
  for (long long j = 0; j < sim.n; j++) {
    if (state[j] != Compress)
      continue;
    struct Blk *bj = &sim.blk[j];
    if ((bj->ix | bj->iy) & 1)
      continue;
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
        if (ic != 4)
          ok = nb_find(bs->level, bs->ix, bs->iy, ic).s != 1;
    }
    if (!ok)
      state[j] = Leave;
  }
  for (long long j = 0; j < sim.n; j++)
    if (state[j] == Refine)
      ref_idx[n_ref++] = j;
    else if (state[j] == Compress && !((sim.blk[j].ix | sim.blk[j].iy) & 1))
      com_idx[n_com++] = j;
  fprintf(stderr, "  ad: com/ref %lld/%lld\n", n_com, n_ref);
  if (n_ref == 0 && n_com == 0)
    goto done;
  long long nprev = sim.n;
  sim.n += 4 * n_ref;
  sim.blk = realloc(sim.blk, sim.n * sizeof *sim.blk);
  sim.fld = realloc(sim.fld, sim.n * BLK_S * sizeof(Real));
  memset(BLK(nprev), 0, 4 * n_ref * BLK_S * sizeof(Real));
  state = realloc(state, sim.n * sizeof *state);
  for (long long i = nprev; i < sim.n; i++)
    state[i] = Leave;
#pragma omp parallel
  {
    Real lm[LB_BUF];
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
        int dim = fld_t[m].dim, offset = fld_t[m].offset;
        lb_load(lm, dim, offset, 1, ref_idx[k]);
        for (int J = 0; J < 2; J++)
          for (int I = 0; I < 2; I++) {
            Real *b = blks[J * 2 + I] + offset * BS * BS;
            for (int j = 0; j < BS; j += 2)
              for (int i = 0; i < BS; i += 2) {
                int i0 = i / 2 + I * (BS / 2) + 1,
                    j0 = j / 2 + J * (BS / 2) + 1;
                int sub[4] = {BS * j + i, BS * j + i + 1, BS * (j + 1) + i,
                              BS * (j + 1) + i + 1};
                for (int s = 0; s < 4; s++)
                  for (int d = 0; d < dim; d++) {
                    Real val = 0;
                    for (int kk = 0; kk < 9; kk++)
                      val +=
                          ad_ref_w[s][kk] *
                          lm[dim * (nm * (j0 + kk / 3 - 1) + i0 + kk % 3 - 1) +
                             d];
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
        blk[s] = BLK(nb_find(level, x, y, ad_sib_ic[s]).idx);
        state[nb_find(level, x, y, ad_sib_ic[s]).idx] = Dealloc;
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
                  dst[dim * o + d] = (src[dim * (BS * j + i) + d] +
                                      src[dim * (BS * j + i + 1) + d] +
                                      src[dim * (BS * (j + 1) + i) + d] +
                                      src[dim * (BS * (j + 1) + i + 1) + d]) /
                                     4;
              }
          }
      }
      bl_fill(p0, level - 1, x / 2, y / 2);
    }
  }
  long long cnt = 0;
  for (long long i = 0; i < sim.n; i++) {
    if (state[i] == Dealloc)
      continue;
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

static inline Real slope4(Real phim2, Real phim1, Real phi0, Real phip1,
                          Real phip2) {
  Real DC = 0.5 * (phip1 - phim1);
  Real DL = phi0 - phim1;
  Real DR = phip1 - phi0;
  Real dlim = DL * DR > 0 ? fmin(2 * fabs(DL), 2 * fabs(DR)) : 0;
  Real dprime = fmin(fabs(DC), dlim) * (DC > 0 ? 1 : (DC < 0 ? -1 : 0));

  Real DC_p = 0.5 * (phip2 - phi0);
  Real DL_p = phip1 - phi0;
  Real DR_p = phip2 - phip1;
  Real dlim_p = DL_p * DR_p > 0 ? fmin(2 * fabs(DL_p), 2 * fabs(DR_p)) : 0;
  Real dp_p = fmin(fabs(DC_p), dlim_p) * (DC_p > 0 ? 1 : (DC_p < 0 ? -1 : 0));

  Real DC_m = 0.5 * (phi0 - phim2);
  Real DL_m = phim1 - phim2;
  Real DR_m = phi0 - phim1;
  Real dlim_m = DL_m * DR_m > 0 ? fmin(2 * fabs(DL_m), 2 * fabs(DR_m)) : 0;
  Real dp_m = fmin(fabs(DC_m), dlim_m) * (DC_m > 0 ? 1 : (DC_m < 0 ? -1 : 0));

  Real d4 = 4.0 / 3.0 * DC - (dp_p + dp_m) / 6.0;
  Real sgn = DC > 0 ? 1 : (DC < 0 ? -1 : 0);
  return fmin(fabs(d4), dlim) * sgn;
}

static void subtract_mean(double *v, int n) {
  double mn = 0;
  for (int k = 0; k < n; k++)
    mn += v[k];
  mn /= n;
  for (int k = 0; k < n; k++)
    v[k] -= mn;
}

static void mg_smooth(double *u, const double *f, int m, int niter) {
  for (int sw = 0; sw < niter; sw++) {
    for (int color = 0; color < 2; color++)
      for (int j = 0; j < m; j++)
        for (int i = 0; i < m; i++) {
          if ((i + j) % 2 != color)
            continue;
          int ip = (i + 1) % m, im = (i - 1 + m) % m, jp = (j + 1) % m,
              jm = (j - 1 + m) % m;
          u[j * m + i] = 0.25 * (u[j * m + ip] + u[j * m + im] + u[jp * m + i] +
                                 u[jm * m + i] - f[j * m + i]);
        }
    subtract_mean(u, m * m);
  }
}

static void mg_residual(const double *u, const double *f, double *r, int m,
                        void *ctx) {
  (void)ctx;
  for (int j = 0; j < m; j++)
    for (int i = 0; i < m; i++) {
      int ip = (i + 1) % m, im = (i - 1 + m) % m, jp = (j + 1) % m,
          jm = (j - 1 + m) % m;
      r[j * m + i] =
          f[j * m + i] - (-4 * u[j * m + i] + u[j * m + ip] + u[j * m + im] +
                          u[jp * m + i] + u[jm * m + i]);
    }
}

static void mg_restrict(const double *rf, double *rc, int mf) {

  int mc = mf / 2;
  for (int j = 0; j < mc; j++)
    for (int i = 0; i < mc; i++) {
      int i2 = 2 * i, j2 = 2 * j;
      int i2p = (i2 + 1) % mf, i2m = (i2 - 1 + mf) % mf, j2p = (j2 + 1) % mf,
          j2m = (j2 - 1 + mf) % mf;
      rc[j * mc + i] = (4 * rf[j2 * mf + i2] +
                        2 * (rf[j2 * mf + i2p] + rf[j2 * mf + i2m] +
                             rf[j2p * mf + i2] + rf[j2m * mf + i2]) +
                        rf[j2p * mf + i2p] + rf[j2p * mf + i2m] +
                        rf[j2m * mf + i2p] + rf[j2m * mf + i2m]) /
                       16.0;
    }
}

static void mg_prolong_add(const double *ec, double *uf, int mc) {

  int mf = mc * 2;
  for (int j = 0; j < mc; j++)
    for (int i = 0; i < mc; i++) {
      int im = (i - 1 + mc) % mc, jm = (j - 1 + mc) % mc;
      int ip = (i + 1) % mc, jp = (j + 1) % mc;
      double cij = ec[j * mc + i];
      int fi = 2 * i, fj = 2 * j, fi1 = (2 * i + 1) % mf,
          fj1 = (2 * j + 1) % mf;

      uf[fj * mf + fi] += (9 * cij + 3 * ec[j * mc + im] + 3 * ec[jm * mc + i] +
                           ec[jm * mc + im]) /
                          16.0;

      uf[fj * mf + fi1] += (9 * cij + 3 * ec[j * mc + ip] +
                            3 * ec[jm * mc + i] + ec[jm * mc + ip]) /
                           16.0;

      uf[fj1 * mf + fi] += (9 * cij + 3 * ec[j * mc + im] +
                            3 * ec[jp * mc + i] + ec[jp * mc + im]) /
                           16.0;

      uf[fj1 * mf + fi1] += (9 * cij + 3 * ec[j * mc + ip] +
                             3 * ec[jp * mc + i] + ec[jp * mc + ip]) /
                            16.0;
    }
}

static void mg_vcycle(double *u, double *f, double *r, int m, double *w) {
  if (m <= 4) {
    mg_smooth(u, f, m, 50);
    return;
  }
  int mc = m / 2, nc = mc * mc;
  double *uc = w, *fc = w + nc, *rc = w + 2 * nc;
  memset(uc, 0, nc * sizeof(double));

  mg_smooth(u, f, m, 4);
  mg_residual(u, f, r, m, NULL);
  mg_restrict(r, fc, m);
  mg_vcycle(uc, fc, rc, mc, w + 3 * nc);
  mg_prolong_add(uc, u, mc);
  mg_smooth(u, f, m, 4);
}

typedef void (*mg_op)(const double *u, const double *f, double *r, int m, void *ctx);
typedef void (*mg_vc)(double *u, double *f, double *r, int m, double *w, void *ctx);

static void pcg_solve(double *x, const double *f, int M, double tol,
                      mg_op residual, mg_op matvec, mg_vc vcycle,
                      int do_mean, void *ctx) {
  int N = M * M;
  static double *buf;
  static int bufn;
  if (N > bufn) {
    buf = realloc(buf, 6 * N * sizeof(double));
    bufn = N;
  }
  double *rr = buf, *z = buf + N, *p = buf + 2 * N, *Ap = buf + 3 * N,
         *r_tmp = buf + 4 * N, *mgw = buf + 5 * N;
  memset(z, 0, N * sizeof(double));

  residual(x, f, rr, M, ctx);
  if (do_mean) subtract_mean(rr, N);
  vcycle(z, rr, r_tmp, M, mgw, ctx);
  if (do_mean) subtract_mean(z, N);
  memcpy(p, z, N * sizeof(double));
  double rz = 0;
  for (int k = 0; k < N; k++)
    rz += rr[k] * z[k];

  for (int it = 0; it < 100; it++) {
    matvec(p, NULL, Ap, M, ctx);
    double pAp = 0;
    for (int k = 0; k < N; k++)
      pAp += p[k] * Ap[k];
    if (fabs(pAp) < 1e-30)
      break;
    double al = rz / pAp;
    for (int k = 0; k < N; k++) {
      x[k] += al * p[k];
      rr[k] -= al * Ap[k];
    }
    if (do_mean) { subtract_mean(rr, N); subtract_mean(x, N); }

    double rmax = 0;
    for (int k = 0; k < N; k++)
      if (fabs(rr[k]) > rmax)
        rmax = fabs(rr[k]);
    if (rmax < tol)
      break;

    memset(z, 0, N * sizeof(double));
    vcycle(z, rr, r_tmp, M, mgw, ctx);
    if (do_mean) subtract_mean(z, N);
    double rz2 = 0;
    for (int k = 0; k < N; k++)
      rz2 += rr[k] * z[k];
    double beta = rz2 / (rz + 1e-30);
    for (int k = 0; k < N; k++)
      p[k] = z[k] + beta * p[k];
    rz = rz2;
  }
  if (do_mean) subtract_mean(x, N);
}

static void mg_matvec_lap(const double *u, const double *f, double *r, int m,
                          void *ctx) {
  (void)f; (void)ctx;
  for (int j = 0; j < m; j++)
    for (int i = 0; i < m; i++) {
      int ip = (i + 1) % m, im = (i - 1 + m) % m, jp = (j + 1) % m,
          jm = (j - 1 + m) % m;
      r[j * m + i] = -4 * u[j * m + i] + u[j * m + ip] + u[j * m + im] +
                     u[jp * m + i] + u[jm * m + i];
    }
}

static void mg_vcycle_wrap(double *u, double *f, double *r, int m, double *w,
                           void *ctx) {
  (void)ctx;
  mg_vcycle(u, f, r, m, w);
}

static void mg_solve_periodic(double *x, const double *f, int M, double tol) {
  pcg_solve(x, f, M, tol, mg_residual, mg_matvec_lap, mg_vcycle_wrap, 1, NULL);
}

static void mg_smooth_helm(double *u, const double *f, int m, int niter,
                           double alpha_h2) {
  double a = 1.0 + 4.0 * alpha_h2;
  double ia = 1.0 / a;
  for (int it = 0; it < niter; it++)
    for (int color = 0; color < 2; color++)
      for (int j = 0; j < m; j++)
        for (int i = 0; i < m; i++) {
          if ((i + j) % 2 != color)
            continue;
          int ip = (i + 1) % m, im = (i - 1 + m) % m, jp = (j + 1) % m,
              jm = (j - 1 + m) % m;
          u[j * m + i] =
              (f[j * m + i] + alpha_h2 * (u[j * m + ip] + u[j * m + im] +
                                          u[jp * m + i] + u[jm * m + i])) *
              ia;
        }
}

static void mg_residual_helm(const double *u, const double *f, double *r, int m,
                             double alpha_h2) {
  double a = 1.0 + 4.0 * alpha_h2;
  for (int j = 0; j < m; j++)
    for (int i = 0; i < m; i++) {
      int ip = (i + 1) % m, im = (i - 1 + m) % m, jp = (j + 1) % m,
          jm = (j - 1 + m) % m;
      r[j * m + i] =
          f[j * m + i] -
          (a * u[j * m + i] - alpha_h2 * (u[j * m + ip] + u[j * m + im] +
                                          u[jp * m + i] + u[jm * m + i]));
    }
}

static void mg_vcycle_helm(double *u, double *f, double *r, int m, double alpha,
                           double h, double *w) {
  double alpha_h2 = alpha / (h * h);
  if (m <= 4) {
    mg_smooth_helm(u, f, m, 50, alpha_h2);
    return;
  }
  int mc = m / 2, nc = mc * mc;
  double *uc = w, *fc = w + nc, *rc = w + 2 * nc;
  memset(uc, 0, nc * sizeof(double));

  mg_smooth_helm(u, f, m, 4, alpha_h2);
  mg_residual_helm(u, f, r, m, alpha_h2);
  mg_restrict(r, fc, m);
  mg_vcycle_helm(uc, fc, rc, mc, alpha, 2 * h, w + 3 * nc);
  mg_prolong_add(uc, u, mc);
  mg_smooth_helm(u, f, m, 4, alpha_h2);
}

struct HelmCtx { double alpha, h; };

static void mg_residual_helm_w(const double *u, const double *f, double *r,
                               int m, void *ctx) {
  struct HelmCtx *c = ctx;
  mg_residual_helm(u, f, r, m, c->alpha / (c->h * c->h));
}

static void mg_matvec_helm(const double *u, const double *f, double *r, int m,
                           void *ctx) {
  (void)f;
  struct HelmCtx *c = ctx;
  double ah2 = c->alpha / (c->h * c->h), a = 1.0 + 4.0 * ah2;
  for (int j = 0; j < m; j++)
    for (int i = 0; i < m; i++) {
      int ip = (i + 1) % m, im = (i - 1 + m) % m, jp = (j + 1) % m,
          jm = (j - 1 + m) % m;
      r[j * m + i] =
          a * u[j * m + i] -
          ah2 * (u[j * m + ip] + u[j * m + im] + u[jp * m + i] + u[jm * m + i]);
    }
}

static void mg_vcycle_helm_w(double *u, double *f, double *r, int m, double *w,
                             void *ctx) {
  struct HelmCtx *c = ctx;
  mg_vcycle_helm(u, f, r, m, c->alpha, c->h, w);
}

static void mg_solve_helmholtz(double *x, const double *f, int M, double alpha,
                               double h, double tol) {
  struct HelmCtx ctx = {alpha, h};
  pcg_solve(x, f, M, tol, mg_residual_helm_w, mg_matvec_helm, mg_vcycle_helm_w,
            0, &ctx);
}

static Real amr_finest_h(void) {
  Real hmin = sim.blk[0].h;
  for (long long i = 1; i < sim.n; i++)
    if (sim.blk[i].h < hmin)
      hmin = sim.blk[i].h;
  return hmin;
}
static int amr_finest_N(Real hf) { return (int)(sim.L[0] / hf + 0.5); }

static void amr_gather(double *dst, int field, int Ng, Real hf) {
  memset(dst, 0, (size_t)Ng * Ng * sizeof(double));
  Real lb[LB_BUF];
  for (long long id = 0; id < sim.n; id++) {
    Real h = sim.blk[id].h;
    int ratio = (int)(h / hf + 0.5);
    int bx = (int)(sim.blk[id].origin[0] / hf + 0.5);
    int by = (int)(sim.blk[id].origin[1] / hf + 0.5);
    if (ratio == 1) {

      Real *src = BLK(id) + BS * BS * field;
      for (int j = 0; j < BS; j++)
        for (int i = 0; i < BS; i++)
          dst[(by + j) * Ng + bx + i] = src[j * BS + i];
    } else {

      lb_load(lb, 1, field, 1, id);
      int ss = 1, nm = 2 * ss + BS;

      for (int j = 0; j < BS; j++)
        for (int i = 0; i < BS; i++)
          for (int dj = 0; dj < ratio; dj++)
            for (int di = 0; di < ratio; di++) {
              double fx = ((double)di + 0.5) / ratio - 0.5;
              double fy = ((double)dj + 0.5) / ratio - 0.5;
              int i0 = (fx < 0) ? -1 : 0, j0 = (fy < 0) ? -1 : 0;
              double wx = fx - i0, wy = fy - j0;

#define LB(ci, cj) lb[nm * ((j) + (cj) + ss) + (i) + (ci) + ss]
              double v00 = LB(i0, j0);
              double v10 = LB(i0 + 1, j0);
              double v01 = LB(i0, j0 + 1);
              double v11 = LB(i0 + 1, j0 + 1);
#undef LB
              double val = (1 - wx) * (1 - wy) * v00 + wx * (1 - wy) * v10 +
                           (1 - wx) * wy * v01 + wx * wy * v11;
              int gx = bx + i * ratio + di;
              int gy = by + j * ratio + dj;
              if (gx >= 0 && gx < Ng && gy >= 0 && gy < Ng)
                dst[gy * Ng + gx] = val;
            }
    }
  }
}

static void amr_scatter(double *src, int field, int Ng, Real hf) {
  for (long long id = 0; id < sim.n; id++) {
    Real *dst = BLK(id) + BS * BS * field;
    Real h = sim.blk[id].h;
    int ratio = (int)(h / hf + 0.5);
    int bx = (int)(sim.blk[id].origin[0] / hf + 0.5);
    int by = (int)(sim.blk[id].origin[1] / hf + 0.5);
    Real inv = 1.0 / (ratio * ratio);
    for (int j = 0; j < BS; j++)
      for (int i = 0; i < BS; i++) {
        double sum = 0;
        for (int dj = 0; dj < ratio; dj++)
          for (int di = 0; di < ratio; di++)
            sum += src[(by + j * ratio + dj) * Ng + bx + i * ratio + di];
        dst[j * BS + i] = sum * inv;
      }
  }
}

static void advect_diffuse(Real dt) {
  Real h0 = amr_finest_h();
  int N = amr_finest_N(h0);
  Real ih = 1.0 / h0;
  Real dtdx = dt / h0, dtdy = dt / h0;
  int NN = N * N;
#define IDX(i, j) (((j) + N) % N * N + ((i) + N) % N)

  enum { NSLOT = 14 };
  static double *abuf;
  static int abufn;
  if (NN > abufn) {
    abuf = realloc(abuf, NSLOT * NN * sizeof(double));
    abufn = NN;
  }
  memset(abuf, 0, NSLOT * NN * sizeof(double));
  double *qu = abuf, *qv = abuf + NN, *umac = abuf + 2 * NN,
         *vmac = abuf + 3 * NN;
  double *xedge_u = abuf + 4 * NN, *xedge_v = abuf + 5 * NN,
         *yedge_u = abuf + 6 * NN, *yedge_v = abuf + 7 * NN;

  amr_gather(qu, F_U, N, h0);
  amr_gather(qv, F_V, N, h0);

  for (int j = 0; j < N; j++)
    for (int i = 0; i < N; i++) {

      Real su_i = slope4(qu[IDX(i - 2, j)], qu[IDX(i - 1, j)], qu[IDX(i, j)],
                         qu[IDX(i + 1, j)], qu[IDX(i + 2, j)]);
      Real su_ip = slope4(qu[IDX(i - 1, j)], qu[IDX(i, j)], qu[IDX(i + 1, j)],
                          qu[IDX(i + 2, j)], qu[IDX(i + 3, j)]);
      Real uc = qu[IDX(i, j)], un = qu[IDX(i + 1, j)];
      Real sL = uc > 0 ? 1 : 0, sR = un < 0 ? 1 : 0;
      Real lo = uc + (0.5 - sL * 0.5 * dtdx * uc) * su_i;
      Real hi = un + (-0.5 - sR * 0.5 * dtdx * un) * su_ip;
      Real uface = (lo + hi) * 0.5;
      umac[IDX(i + 1, j)] = (uface >= 0) ? lo : hi;
      if (fabs(uface) < 1e-10)
        umac[IDX(i + 1, j)] = 0.5 * (lo + hi);

      Real sv_j = slope4(qv[IDX(i, j - 2)], qv[IDX(i, j - 1)], qv[IDX(i, j)],
                         qv[IDX(i, j + 1)], qv[IDX(i, j + 2)]);
      Real sv_jp = slope4(qv[IDX(i, j - 1)], qv[IDX(i, j)], qv[IDX(i, j + 1)],
                          qv[IDX(i, j + 2)], qv[IDX(i, j + 3)]);
      Real vc = qv[IDX(i, j)], vn = qv[IDX(i, j + 1)];
      sL = vc > 0 ? 1 : 0;
      sR = vn < 0 ? 1 : 0;
      lo = vc + (0.5 - sL * 0.5 * dtdy * vc) * sv_j;
      hi = vn + (-0.5 - sR * 0.5 * dtdy * vn) * sv_jp;
      Real vface = (lo + hi) * 0.5;
      vmac[IDX(i, j + 1)] = (vface >= 0) ? lo : hi;
      if (fabs(vface) < 1e-10)
        vmac[IDX(i, j + 1)] = 0.5 * (lo + hi);
    }

  {
    double *div = abuf + 8 * NN, *phi = abuf + 9 * NN;
    memset(div, 0, NN * sizeof(double));
    memset(phi, 0, NN * sizeof(double));
    for (int j = 0; j < N; j++)
      for (int i = 0; i < N; i++)
        div[IDX(i, j)] = (umac[IDX(i + 1, j)] - umac[IDX(i, j)] +
                          vmac[IDX(i, j + 1)] - vmac[IDX(i, j)]) *
                         ih;
    for (int k = 0; k < NN; k++)
      div[k] *= h0 * h0;
    mg_solve_periodic(phi, div, N, 1e-10);
    for (int j = 0; j < N; j++)
      for (int i = 0; i < N; i++) {
        umac[IDX(i, j)] -= (phi[IDX(i, j)] - phi[IDX(i - 1, j)]) * ih;
        vmac[IDX(i, j)] -= (phi[IDX(i, j)] - phi[IDX(i, j - 1)]) * ih;
      }
  }

  for (int n = 0; n < 2; n++) {
    double *q = (n == 0) ? qu : qv;
    double *xedge = (n == 0) ? xedge_u : xedge_v;
    double *yedge = (n == 0) ? yedge_u : yedge_v;

    double *xlo = abuf + 8 * NN, *xhi = abuf + 9 * NN, *ylo = abuf + 10 * NN,
           *yhi = abuf + 11 * NN;
    memset(xlo, 0, 4 * NN * sizeof(double));

    for (int j = 0; j < N; j++)
      for (int i = 0; i < N; i++) {
        Real s = slope4(q[IDX(i - 2, j)], q[IDX(i - 1, j)], q[IDX(i, j)],
                        q[IDX(i + 1, j)], q[IDX(i + 2, j)]);
        Real uc = qu[IDX(i, j)];

        xlo[IDX(i + 1, j)] =
            q[IDX(i, j)] + 0.5 * (1.0 - umac[IDX(i + 1, j)] * dtdx) * s;

        xhi[IDX(i, j)] =
            q[IDX(i, j)] + 0.5 * (-1.0 - umac[IDX(i, j)] * dtdx) * s;

        Real sy = slope4(q[IDX(i, j - 2)], q[IDX(i, j - 1)], q[IDX(i, j)],
                         q[IDX(i, j + 1)], q[IDX(i, j + 2)]);
        Real vc = qv[IDX(i, j)];
        ylo[IDX(i, j + 1)] =
            q[IDX(i, j)] + 0.5 * (1.0 - vmac[IDX(i, j + 1)] * dtdy) * sy;
        yhi[IDX(i, j)] =
            q[IDX(i, j)] + 0.5 * (-1.0 - vmac[IDX(i, j)] * dtdy) * sy;
      }

    double *yzlo = abuf + 12 * NN;
    memset(yzlo, 0, NN * sizeof(double));
    for (int j = 0; j < N; j++)
      for (int i = 0; i < N; i++) {
        Real vad = vmac[IDX(i, j)];
        Real lo_v = ylo[IDX(i, j)], hi_v = yhi[IDX(i, j)];
        yzlo[IDX(i, j)] = (fabs(vad) < 1e-10) ? 0.5 * (lo_v + hi_v)
                                              : ((vad >= 0) ? lo_v : hi_v);
      }

    for (int j = 0; j < N; j++)
      for (int i = 0; i < N; i++) {

        Real quxl = (umac[IDX(i, j)] - umac[IDX(i - 1, j)]) * q[IDX(i - 1, j)];
        Real stl = xlo[IDX(i, j)] - 0.5 * dtdx * quxl -
                   0.5 * dtdy *
                       (yzlo[IDX(i - 1, j + 1)] * vmac[IDX(i - 1, j + 1)] -
                        yzlo[IDX(i - 1, j)] * vmac[IDX(i - 1, j)]);

        Real quxh = (umac[IDX(i + 1, j)] - umac[IDX(i, j)]) * q[IDX(i, j)];
        Real sth = xhi[IDX(i, j)] - 0.5 * dtdx * quxh -
                   0.5 * dtdy *
                       (yzlo[IDX(i, j + 1)] * vmac[IDX(i, j + 1)] -
                        yzlo[IDX(i, j)] * vmac[IDX(i, j)]);

        Real uad = umac[IDX(i, j)];
        xedge[IDX(i, j)] =
            (fabs(uad) < 1e-10) ? 0.5 * (stl + sth) : ((uad >= 0) ? stl : sth);
      }

    double *xzlo = abuf + 13 * NN;
    memset(xzlo, 0, NN * sizeof(double));
    for (int j = 0; j < N; j++)
      for (int i = 0; i < N; i++) {
        Real uad = umac[IDX(i, j)];
        Real lo_u = xlo[IDX(i, j)], hi_u = xhi[IDX(i, j)];
        xzlo[IDX(i, j)] = (fabs(uad) < 1e-10) ? 0.5 * (lo_u + hi_u)
                                              : ((uad >= 0) ? lo_u : hi_u);
      }

    for (int j = 0; j < N; j++)
      for (int i = 0; i < N; i++) {
        Real qvyl = (vmac[IDX(i, j)] - vmac[IDX(i, j - 1)]) * q[IDX(i, j - 1)];
        Real stl = ylo[IDX(i, j)] - 0.5 * dtdy * qvyl -
                   0.5 * dtdx *
                       (xzlo[IDX(i + 1, j - 1)] * umac[IDX(i + 1, j - 1)] -
                        xzlo[IDX(i, j - 1)] * umac[IDX(i, j - 1)]);

        Real qvyh = (vmac[IDX(i, j + 1)] - vmac[IDX(i, j)]) * q[IDX(i, j)];
        Real sth = yhi[IDX(i, j)] - 0.5 * dtdy * qvyh -
                   0.5 * dtdx *
                       (xzlo[IDX(i + 1, j)] * umac[IDX(i + 1, j)] -
                        xzlo[IDX(i, j)] * umac[IDX(i, j)]);

        Real vad = vmac[IDX(i, j)];
        yedge[IDX(i, j)] =
            (fabs(vad) < 1e-10) ? 0.5 * (stl + sth) : ((vad >= 0) ? stl : sth);
      }
  }

  {
    double *div = abuf + 8 * NN, *phi = abuf + 9 * NN;
    memset(div, 0, NN * sizeof(double));
    memset(phi, 0, NN * sizeof(double));
    for (int j = 0; j < N; j++)
      for (int i = 0; i < N; i++)
        div[IDX(i, j)] = (xedge_u[IDX(i + 1, j)] - xedge_u[IDX(i, j)] +
                          yedge_v[IDX(i, j + 1)] - yedge_v[IDX(i, j)]) *
                         ih;
    for (int k = 0; k < NN; k++)
      div[k] *= h0 * h0;
    mg_solve_periodic(phi, div, N, 1e-10);
    for (int j = 0; j < N; j++)
      for (int i = 0; i < N; i++) {
        xedge_u[IDX(i, j)] -= (phi[IDX(i, j)] - phi[IDX(i - 1, j)]) * ih;
        yedge_v[IDX(i, j)] -= (phi[IDX(i, j)] - phi[IDX(i, j - 1)]) * ih;
      }
  }

  {
    double *qp = abuf + 8 * NN, *u_new = abuf + 9 * NN, *v_new = abuf + 10 * NN;
    memset(qp, 0, 3 * NN * sizeof(double));
    amr_gather(qp, F_P, N, h0);
    for (int gj = 0; gj < N; gj++)
      for (int gi = 0; gi < N; gi++) {
        Real uc = qu[IDX(gi, gj)], vc = qv[IDX(gi, gj)];
        Real uR = xedge_u[IDX(gi + 1, gj)], uL = xedge_u[IDX(gi, gj)];
        Real vT = yedge_v[IDX(gi, gj + 1)], vB = yedge_v[IDX(gi, gj)];
        Real uu_R = xedge_u[IDX(gi + 1, gj)], uu_L = xedge_u[IDX(gi, gj)];
        Real vv_T = yedge_v[IDX(gi, gj + 1)], vv_B = yedge_v[IDX(gi, gj)];
        Real u_yT = yedge_u[IDX(gi, gj + 1)], u_yB = yedge_u[IDX(gi, gj)];
        Real v_xR = xedge_v[IDX(gi + 1, gj)], v_xL = xedge_v[IDX(gi, gj)];
        Real adv_u = 0.5 * (uR + uL) * (uu_R - uu_L) * ih +
                     0.5 * (vT + vB) * (u_yT - u_yB) * ih;
        Real adv_v = 0.5 * (uR + uL) * (v_xR - v_xL) * ih +
                     0.5 * (vT + vB) * (vv_T - vv_B) * ih;
        Real lap_u = (qu[IDX(gi + 1, gj)] + qu[IDX(gi - 1, gj)] +
                      qu[IDX(gi, gj + 1)] + qu[IDX(gi, gj - 1)] - 4 * uc) *
                     ih * ih;
        Real lap_v = (qv[IDX(gi + 1, gj)] + qv[IDX(gi - 1, gj)] +
                      qv[IDX(gi, gj + 1)] + qv[IDX(gi, gj - 1)] - 4 * vc) *
                     ih * ih;
        Real dpx = (qp[IDX(gi + 1, gj)] - qp[IDX(gi - 1, gj)]) * 0.5 * ih;
        Real dpy = (qp[IDX(gi, gj + 1)] - qp[IDX(gi, gj - 1)]) * 0.5 * ih;
        Real alpha = sim.nu * dt * 0.5;
        u_new[IDX(gi, gj)] = uc + alpha * lap_u + dt * (-adv_u - dpx);
        v_new[IDX(gi, gj)] = vc + alpha * lap_v + dt * (-adv_v - dpy);
      }

    amr_scatter(u_new, F_U, N, h0);
    amr_scatter(v_new, F_V, N, h0);
  }
#undef IDX
}
static void helmholtz_solve(Real dt, int field) {
  Real alpha = sim.nu * dt * 0.5;
  Real hf = amr_finest_h();
  int Ng = amr_finest_N(hf);
  double *flat_f = calloc(Ng * Ng, sizeof(double));
  double *flat_x = calloc(Ng * Ng, sizeof(double));
  amr_gather(flat_f, field, Ng, hf);
  memcpy(flat_x, flat_f, Ng * Ng * sizeof(double));
  mg_solve_helmholtz(flat_x, flat_f, Ng, alpha, hf, 1e-10);
  amr_scatter(flat_x, field, Ng, hf);
  free(flat_f);
  free(flat_x);
}

static void poisson_solve(Real dt) {

#pragma omp parallel
  {
    Real bu[LB_BUF], bv[LB_BUF];
#pragma omp for
    for (long long id = 0; id < sim.n; id++) {
      lb_load(bu, 1, F_U, 1, id);
      lb_load(bv, 1, F_V, 1, id);
      Real *rhs = BLK(id) + BS * BS * F_TMP;
      int ss = 1, nm = 2 * ss + BS;
      Real h = sim.blk[id].h;

      Real fac = 2.0 * h / dt;
      for (int j = 0; j < BS; j++)
        for (int i = 0; i < BS; i++)
          rhs[j * BS + i] =
              fac *
              (bu[nm * (j + ss) + i + 1 + ss] - bu[nm * (j + ss) + i - 1 + ss] +
               bv[nm * (j + 1 + ss) + i + ss] - bv[nm * (j - 1 + ss) + i + ss]);
    }
  }

  Real hf = amr_finest_h();
  int Ng = amr_finest_N(hf);
  int M = Ng / 2;

  double *rhs_full = calloc(Ng * Ng, sizeof(double));
  double *phi_full = calloc(Ng * Ng, sizeof(double));
  {
    double *gu = calloc(Ng * Ng, 8), *gv = calloc(Ng * Ng, 8);
    amr_gather(gu, F_U, Ng, hf);
    amr_gather(gv, F_V, Ng, hf);
    Real fac = 2.0 * hf / dt;
#define GI(i, j) (((j) + Ng) % Ng * Ng + ((i) + Ng) % Ng)
    for (int j = 0; j < Ng; j++)
      for (int i = 0; i < Ng; i++)
        rhs_full[j * Ng + i] = fac * (gu[GI(i + 1, j)] - gu[GI(i - 1, j)] +
                                      gv[GI(i, j + 1)] - gv[GI(i, j - 1)]);
#undef GI
    free(gu);
    free(gv);
  }
  amr_gather(phi_full, F_PHI, Ng, hf);

  for (int sy = 0; sy < 2; sy++)
    for (int sx = 0; sx < 2; sx++) {

      double *f = calloc(M * M, sizeof(double));
      double *x = calloc(M * M, sizeof(double));
      for (int j = 0; j < M; j++)
        for (int i = 0; i < M; i++) {
          f[j * M + i] = rhs_full[(2 * j + sy) * Ng + 2 * i + sx];
          x[j * M + i] = phi_full[(2 * j + sy) * Ng + 2 * i + sx];
        }

      mg_solve_periodic(x, f, M, 1e-10);

      for (int j = 0; j < M; j++)
        for (int i = 0; i < M; i++)
          phi_full[(2 * j + sy) * Ng + 2 * i + sx] = x[j * M + i];

      free(f);
      free(x);
    }

  {
    double rmax = 0;
    for (int j = 0; j < Ng; j++)
      for (int i = 0; i < Ng; i++) {
        int ip = (i + 2) % Ng, im = (i - 2 + Ng) % Ng, jp = (j + 2) % Ng,
            jm = (j - 2 + Ng) % Ng;
        double r = rhs_full[j * Ng + i] -
                   (-4 * phi_full[j * Ng + i] + phi_full[j * Ng + ip] +
                    phi_full[j * Ng + im] + phi_full[jp * Ng + i] +
                    phi_full[jm * Ng + i]);
        if (fabs(r) > rmax)
          rmax = fabs(r);
      }
    if (rmax > 1e-4)
      fprintf(stderr, "  poisson res=%.2e\n", rmax);
  }

  amr_scatter(phi_full, F_PHI, Ng, hf);
  free(rhs_full);
  free(phi_full);
}

static void project(Real dt) {
#pragma omp parallel
  {
    Real bp[LB_BUF];
#pragma omp for
    for (long long id = 0; id < sim.n; id++) {
      lb_load(bp, 1, F_PHI, 1, id);
      Real *u = BLK(id) + BS * BS * F_U;
      Real *v = BLK(id) + BS * BS * F_V;
      Real *p = BLK(id) + BS * BS * F_P;
      Real *phi = BLK(id) + BS * BS * F_PHI;
      int ss = 1, nm = 2 * ss + BS;
      Real ih = 0.5 / sim.blk[id].h;
      for (int j = 0; j < BS; j++)
        for (int i = 0; i < BS; i++) {
          int k = j * BS + i;
#define PH(di, dj) bp[nm * ((j) + (dj) + ss) + (i) + (di) + ss]
          u[k] -= dt * (PH(1, 0) - PH(-1, 0)) * ih;
          v[k] -= dt * (PH(0, 1) - PH(0, -1)) * ih;
          p[k] += phi[k];
#undef PH
        }
    }
  }
}

static const struct {
  const char *name;
  int type;
  size_t off;
} param_tab[] = {
    {"levelStart", 0, offsetof(struct Sim, levelStart)},
    {"levelMax", 0, offsetof(struct Sim, levelMax)},
    {"AdaptSteps", 0, offsetof(struct Sim, AdaptSteps)},
    {"sdump", 0, offsetof(struct Sim, sdump)},
    {"Rtol", 1, offsetof(struct Sim, Rtol)},
    {"CFL", 1, offsetof(struct Sim, CFL)},
    {"nu", 1, offsetof(struct Sim, nu)},
    {"tend", 1, offsetof(struct Sim, endTime)},
    {"tdump", 1, offsetof(struct Sim, dumpTime)},
};

int main(int argc, char **argv) {
  fprintf(stderr, "main.c: %d threads\n", omp_get_max_threads());
  char *base = (char *)&sim;
  for (size_t i = 0; i < sizeof param_tab / sizeof *param_tab; i++) {
    const char *key = param_tab[i].name;
    const char *val = NULL;
    for (int a = 1; a < argc; a++)
      if (argv[a][0] == '-' && strcmp(argv[a] + 1, key) == 0) {
        if (a + 1 >= argc) { fprintf(stderr, "-%s: no value\n", key); exit(1); }
        val = argv[a + 1];
        break;
      }
    if (!val) { fprintf(stderr, "-%s: not set\n", key); exit(1); }
    char *end;
    if (param_tab[i].type == 0) {
      *(int *)(base + param_tab[i].off) = (int)strtol(val, &end, 10);
    } else {
      *(Real *)(base + param_tab[i].off) = strtod(val, &end);
    }
    if (end == val || *end) { fprintf(stderr, "-%s: bad value '%s'\n", key, val); exit(1); }
  }

  sim.L[0] = 1.0;
  sim.L[1] = 1.0;
  {
    int ns = 1 << sim.levelStart;
    sim.nb[0] = ns;
    sim.nb[1] = ns;
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

  Real rho_layer = 30.0, delta = 0.05;
  fprintf(stderr, "main.c: IC rho=%g delta=%g nu=%g\n", rho_layer, delta, sim.nu);
#pragma omp parallel for
  for (long long i = 0; i < sim.n; i++) {
    struct Blk *info = &sim.blk[i];
    Real *u = BLK(i) + BS * BS * F_U;
    Real *v = BLK(i) + BS * BS * F_V;
    Real h = info->h;
    for (int iy = 0; iy < BS; iy++)
      for (int ix = 0; ix < BS; ix++) {
        int j = BS * iy + ix;
        Real x = info->origin[0] + (ix + 0.5) * h;
        Real y = info->origin[1] + (iy + 0.5) * h;
        u[j] = y <= 0.5 ? tanh(rho_layer * (y - 0.25))
                        : tanh(rho_layer * (0.75 - y));
        v[j] = delta * sin(2 * M_PI * x);
      }
  }

  while (1) {
    if (sim.step % 10 == 0) {
      compute_vorticity();
      fprintf(stderr, "main.c: %08d %.6e dt=%.3e blk=%lld\n", sim.step,
              sim.time, sim.dt, sim.n);
    }
    {
      int do_dump = 0;
      if (sim.dumpTime > 0 && sim.time >= sim.nextDumpTime) {
        sim.nextDumpTime += sim.dumpTime;
        do_dump = 1;
      }
      if (sim.sdump > 0 && sim.step % sim.sdump == 0)
        do_dump = 1;
      if (do_dump) {
        compute_vorticity();
        char path[FILENAME_MAX];
        snprintf(path, sizeof path, "%08d", sim.dump_count++);
        dump(sim.time, sim.step, path);
      }
    }
    if (sim.endTime > 0 && sim.time >= sim.endTime)
      break;

    Real smax = 0;
#pragma omp parallel for reduction(max : smax)
    for (long long i = 0; i < sim.n; i++) {
      Real *u = BLK(i) + BS * BS * F_U;
      Real *v = BLK(i) + BS * BS * F_V;
      Real ih = 1.0 / sim.blk[i].h;
      for (int j = 0; j < BS * BS; j++)
        smax = fmax(smax, fmax(fabs(u[j]), fabs(v[j])) * ih);
    }
    sim.dt = sim.CFL / (smax + 1e-30);

    if (sim.step > 0 && sim.step % sim.AdaptSteps == 0)
      ad_run();

    advect_diffuse(sim.dt);
    helmholtz_solve(sim.dt, F_U);
    helmholtz_solve(sim.dt, F_V);
    poisson_solve(sim.dt);
    project(sim.dt);

    sim.time += sim.dt;
    sim.step++;
  }
  fprintf(stderr, "main.c: end\n");
}
