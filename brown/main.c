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
  int nb;
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
  b->h = 1.0 / (BS * sim.nb * scale);
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
  long long key;
  int s;
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
    key = hm_key(sim.blk[i].level, sim.blk[i].ix, sim.blk[i].iy);
    s = hm_slot(&sim.hm, key);
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
  int nd = sim.nb * scale;
  int nx = (ix + cx + nd) % nd, ny = (iy + cy + nd) % nd;
  int idx = hm_get(&sim.hm, hm_key(level, nx, ny));
  int L1, nL1;
  int fx, fy;
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
  L1 = level + 1; nL1 = 1 << L1;
  for (int b = 0; b < nb_ch_n[icode]; b++) {
    fx = (ix * 2 + nb_ch_off[icode][b][0] + nL1) % nL1;
    fy = (iy * 2 + nb_ch_off[icode][b][1] + nL1) % nL1;
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
  const struct LbOp *o;
  Real *avg_src, *avg_d, *avg_q1;
  const int8_t *w;
  Real sum;
  Real a, b, cv;
  for (int i = 0; i < n; i++) {
    o = &ops[i];
    switch (o->type) {
    case OP_COPY:
      memcpy(dst[o->dst_idx] + o->dst_off, blk[o->blk_idx] + o->src_off,
             o->p1 * dim * sizeof(Real));
      break;
    case OP_AVG:
      avg_src = blk[o->blk_idx] + o->src_off;
      avg_d = dst[o->dst_idx] + o->dst_off;
      avg_q1 = avg_src + o->p2 * dim;
      for (int k = 0; k < o->p1; k++)
        for (int dd = 0; dd < dim; dd++)
          avg_d[k * dim + dd] =
              (avg_src[2 * k * dim + dd] + avg_src[(2 * k + 1) * dim + dd] +
               avg_q1[2 * k * dim + dd] + avg_q1[(2 * k + 1) * dim + dd]) /
              4;
      break;
    case OP_INTERP9: {
      static const int8_t W[4][9] = {
          {1, 10, -1, 10, 56, -6, -1, -6, 1},
          {-1, 10, 1, -6, 56, 10, 1, -6, -1},
          {-1, -6, 1, 10, 56, -6, 1, 10, -1},
          {1, -6, -1, -6, 56, 10, -1, 10, 1},
      };
      w = W[o->flags & 3];
      for (int d = 0; d < dim; d++) {
        sum = 0;
        for (int jj = 0; jj < 3; jj++)
          for (int ii = 0; ii < 3; ii++)
            sum += w[3 * jj + ii] *
                   c[o->src_off + d + dim * ((ii - 1) + nc * (jj - 1))];
        m[o->dst_off + d] = sum / 64.0;
      }
      break;
    }
    case OP_INTERP3:
      for (int d = 0; d < dim; d++)
        m[o->dst_off + d] =
            (o->blk_idx * c[o->src_off + d] + o->dst_idx * c[o->p1 + d] +
             o->flags * c[o->p2 + d]) /
            32.0;
      break;
    case OP_LELI: {
      static const int8_t W[2][3] = {
          {8, 10, -3},
          {24, -15, 6},
      };
      w = W[o->flags & 1];
      for (int d = 0; d < dim; d++) {
        a = m[o->src_off + d]; b = m[o->dst_off + d]; cv = m[o->p1 + d];
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
  int configs[][2] = {{1, 1}, {1, 2}, {2, 1}};
  int ss, dim;
  char fname[64];
  FILE *fp;
  size_t sz;
  struct LbTab *tab;
  for (int ci = 0; ci < 3; ci++) {
    ss = configs[ci][0]; dim = configs[ci][1];
    snprintf(fname, sizeof fname, "tab_ss%d_dim%d.bin", ss, dim);
    fp = fopen(fname, "rb");
    if (!fp) {
      fprintf(stderr, "main.c: cannot open %s\n", fname);
      exit(1);
    }
    sz = 3 * 3 * 2 * 2 * N_STATUS * sizeof(struct LbTab);
    tab = malloc(sz);
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
  Real *c;
  Real *dst[2];
  struct {
    const struct LbTab *e;
    Real *blk[2];
  } dirs[8];
  int nd = 0;
  int cx, cy;
  struct Nb nr;
  const struct LbTab *te;
  Real *lblk[2];
  const struct LbSrc *bs;

  for (int i = 0; i < BS; i++)
    memcpy(m + dim * ((i + ss) * nm + ss), p0 + dim * BS * i,
           BS * dim * sizeof(Real));

  c = m + nm * nm * dim;
  dst[0] = m;
  dst[1] = c;

  for (int icode = 0; icode < 9; icode++) {
    cx = icode % 3 - 1; cy = icode / 3 - 1;
    lblk[0] = NULL; lblk[1] = NULL;
    if (!cx && !cy)
      continue;
    nr = nb_find(level, xi, yi, icode);
    te = &cflb_tab[cx + 1][cy + 1][xi % 2][yi % 2][nr.s];
    for (int b = 0; b < te->n_blk; b++) {
      bs = &te->blk_src[b];
      if (bs->is_self) {
        lblk[b] = dst[bs->self_idx];
      } else if (bs->level_delta == 1) {
        lblk[b] = BLK(nr.ch[b]) + BS * BS * blk_offset;
      } else {
        lblk[b] = BLK(nr.idx) + BS * BS * blk_offset;
      }
    }
    dirs[nd].e = te;
    dirs[nd].blk[0] = lblk[0];
    dirs[nd].blk[1] = lblk[1];
    nd++;
  }

  for (int i = 0; i < nd; i++)
    lb_exec(dirs[i].blk, dst, dirs[i].e->ops, dirs[i].e->n_pre, dim, nm, nc);
  for (int i = 0; i < nd; i++)
    lb_exec(dirs[i].blk, dst, dirs[i].e->ops + MAX_PRE, dirs[i].e->n_post, dim,
            nm, nc);
}



static void compute_vorticity(void) {
#pragma omp parallel
  {
    Real bu[LB_BUF], bv[LB_BUF];
    Real *w;
    int nm;
    Real ih;
#pragma omp for
    for (long long id = 0; id < sim.n; id++) {
      lb_load(bu, 1, F_U, 1, id);
      lb_load(bv, 1, F_V, 1, id);
      w = BLK(id) + BS * BS * F_W;
      nm = BS + 2;
      ih = 0.5 / sim.blk[id].h;
      for (int j = 0; j < BS; j++)
        for (int i = 0; i < BS; i++) {
#define U(di, dj) bu[nm * ((j) + (dj) + 1) + (i) + (di) + 1]
#define V(di, dj) bv[nm * ((j) + (dj) + 1) + (i) + (di) + 1]
          w[j * BS + i] = (V(1, 0) - V(-1, 0)) * ih - (U(0, 1) - U(0, -1)) * ih;
#undef U
#undef V
        }
    }
  }
}

static void restrict_2to1(const Real *fine, Real *coarse, int dim,
                          int fs, int cs, int ni, int nj) {
  for (int j = 0; j < nj; j++)
    for (int i = 0; i < ni; i++)
      for (int d = 0; d < dim; d++)
        coarse[dim * (j * cs + i) + d] =
            0.25 * (fine[dim * ((2*j) * fs + 2*i) + d] +
                    fine[dim * ((2*j) * fs + 2*i+1) + d] +
                    fine[dim * ((2*j+1) * fs + 2*i) + d] +
                    fine[dim * ((2*j+1) * fs + 2*i+1) + d]);
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

static void prolong_2to1(const Real *coarse, Real *fine, int dim,
                         int cs, int fs, int ni, int nj,
                         int ci0, int cj0) {
  int ic, jc, di, dj;
  Real val;
  for (int j = 0; j < nj; j++)
    for (int i = 0; i < ni; i++) {
      ic = i + ci0; jc = j + cj0;
      for (int s = 0; s < 4; s++) {
        di = s & 1; dj = s >> 1;
        for (int d = 0; d < dim; d++) {
          val = 0;
          for (int kk = 0; kk < 9; kk++)
            val += ad_ref_w[s][kk] *
                   coarse[dim * ((jc + kk / 3 - 1) * cs + ic + kk % 3 - 1) + d];
          fine[dim * ((2 * j + dj) * fs + 2 * i + di) + d] = val;
        }
      }
    }
}

static void compute_indicator(void) {
#pragma omp parallel
  {
    Real bu[LB_BUF], bv[LB_BUF];
    int nm = BS + 2;
    int nc = BS / 2 + 2;
    Real cu[nc * nc], cv[nc * nc];
    Real *t;
    int fi, fj, ic, jc, di, dj, ci, cj;
    Real su, sv, pu, pv, au, av;
#pragma omp for
    for (long long id = 0; id < sim.n; id++) {
      lb_load(bu, 1, F_U, 1, id);
      lb_load(bv, 1, F_V, 1, id);

      for (int jc0 = 0; jc0 < nc; jc0++)
        for (int ic0 = 0; ic0 < nc; ic0++) {
          fi = 2 * ic0 - 1; fj = 2 * jc0 - 1;
          su = 0; sv = 0;
          for (int dj0 = 0; dj0 < 2; dj0++)
            for (int di0 = 0; di0 < 2; di0++) {
              su += bu[nm * (fj + dj0 + 1) + fi + di0 + 1];
              sv += bv[nm * (fj + dj0 + 1) + fi + di0 + 1];
            }
          cu[jc0 * nc + ic0] = su * 0.25;
          cv[jc0 * nc + ic0] = sv * 0.25;
        }

      t = BLK(id) + BS * BS * F_TMP;
      for (int j = 0; j < BS; j += 2)
        for (int i = 0; i < BS; i += 2) {
          ic = i / 2 + 1; jc = j / 2 + 1;
          for (int s = 0; s < 4; s++) {
            di = s & 1; dj = s >> 1;
            pu = 0; pv = 0;
            for (int kk = 0; kk < 9; kk++) {
              ci = ic + kk % 3 - 1; cj = jc + kk / 3 - 1;
              pu += ad_ref_w[s][kk] * cu[cj * nc + ci];
              pv += ad_ref_w[s][kk] * cv[cj * nc + ci];
            }
            au = bu[nm * (j + dj + 1) + i + di + 1];
            av = bv[nm * (j + dj + 1) + i + di + 1];
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
  int c;
  float x0, y0, x1, y1;
  Real h, ox, oy;
  int dim, offset;
  snprintf(xyz_path, sizeof xyz_path, "%s.xyz.raw", path);
  file = fopen(xyz_path, "wb");
  for (i = 0; i < sim.n; i++) {
    h = sim.blk[i].h; ox = sim.blk[i].origin[0]; oy = sim.blk[i].origin[1];
    for (j = 0; j < BS; j++)
      for (k = 0; k < BS; k++) {
        c = j * BS + k;
        x0 = ox + k * h; y0 = oy + j * h; x1 = x0 + h; y1 = y0 + h;
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
  enum AdSt *state;
  long long *ref_idx;
  long long *com_idx;
  long long n_ref = 0, n_com = 0;
  int Changed = 0;
  long long nprev;
  long long cnt;
  struct Blk *bj;
  struct Nb nr;
  long long sib_ad[4];
  int ok;
  struct Blk *bs_blk;
  long long ci_ad;
  struct Blk *p0_ad;
  compute_indicator();
  state = calloc(sim.n, sizeof *state);
  ref_idx = malloc(sim.n * sizeof *ref_idx);
  com_idx = malloc(sim.n * sizeof *com_idx);
#pragma omp parallel for reduction(|| : Changed)
  for (long long i = 0; i < sim.n; i++) {
    Real *b = BLK(i) + BS * BS * F_TMP;
    double Linf = 0;
    int lev;
    for (int j = 0; j < BS * BS; j++)
      Linf = fmax(Linf, fabs(b[j]));
    lev = sim.blk[i].level;
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
      bj = &sim.blk[j];
      for (int ic = 0; ic < 9; ic++) {
        if (ic == 4)
          continue;
        nr = nb_find(bj->level, bj->ix, bj->iy, ic);
        if (nr.s == 1 || nr.idx < 0)
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
    bj = &sim.blk[j];
    if ((bj->ix | bj->iy) & 1)
      continue;
    sib_ad[0] = j; sib_ad[1] = 0; sib_ad[2] = 0; sib_ad[3] = 0;
    ok = 1;
    for (int s = 1; s < 4 && ok; s++) {
      nr = nb_find(bj->level, bj->ix, bj->iy, ad_sib_ic[s]);
      ok = nr.s == 0 && nr.idx >= 0 && state[nr.idx] == Compress;
      sib_ad[s] = nr.idx;
    }
    for (int s = 0; s < 4 && ok; s++) {
      bs_blk = &sim.blk[sib_ad[s]];
      for (int ic = 0; ic < 9 && ok; ic++)
        if (ic != 4)
          ok = nb_find(bs_blk->level, bs_blk->ix, bs_blk->iy, ic).s != 1;
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
  nprev = sim.n;
  sim.n += 4 * n_ref;
  sim.blk = realloc(sim.blk, sim.n * sizeof *sim.blk);
  sim.fld = realloc(sim.fld, sim.n * BLK_S * sizeof(Real));
  memset(BLK(nprev), 0, 4 * n_ref * BLK_S * sizeof(Real));
  state = realloc(state, sim.n * sizeof *state);
  for (long long i = nprev; i < sim.n; i++)
    state[i] = Leave;
  for (long long k = 0; k < n_com; k++) {
    ci_ad = com_idx[k];
    p0_ad = &sim.blk[ci_ad];
    for (int s = 1; s < 4; s++)
      state[nb_find(p0_ad->level, p0_ad->ix, p0_ad->iy, ad_sib_ic[s]).idx] = Dealloc;
  }
#pragma omp parallel
  {
    Real lm[LB_BUF];
    struct Blk *par;
    int px, py;
    Real *blks[4];
    int nm_ad;
    long long ci_omp;
    struct Blk *p0_omp;
    int level_omp, x_omp, y_omp;
    long long sib_omp[4];
    Real *blk_omp[4];
    int dim_omp, off_omp;
    Real *dst_omp;
#pragma omp for
    for (long long k = 0; k < n_ref; k++) {
      par = &sim.blk[ref_idx[k]];
      px = par->ix; py = par->iy;
      nm_ad = 2 + BS;
      for (int J = 0; J < 2; J++)
        for (int I = 0; I < 2; I++) {
          ci_omp = nprev + 4 * k + 2 * J + I;
          bl_fill(&sim.blk[ci_omp], par->level + 1, 2 * px + I, 2 * py + J);
          blks[2 * J + I] = BLK(ci_omp);
        }
      for (size_t m = 0; m < NVARS; m++) {
        dim_omp = fld_t[m].dim; off_omp = fld_t[m].offset;
        lb_load(lm, dim_omp, off_omp, 1, ref_idx[k]);
        for (int J = 0; J < 2; J++)
          for (int I = 0; I < 2; I++)
            prolong_2to1(lm, blks[J * 2 + I] + off_omp * BS * BS,
                         dim_omp, nm_ad, BS, BS / 2, BS / 2,
                         I * (BS / 2) + 1, J * (BS / 2) + 1);
      }
      state[ref_idx[k]] = Dealloc;
    }
#pragma omp for
    for (long long k = 0; k < n_com; k++) {
      ci_omp = com_idx[k];
      p0_omp = &sim.blk[ci_omp];
      level_omp = p0_omp->level; x_omp = p0_omp->ix; y_omp = p0_omp->iy;
      sib_omp[0] = ci_omp; sib_omp[1] = 0; sib_omp[2] = 0; sib_omp[3] = 0;
      for (int s = 1; s < 4; s++)
        sib_omp[s] = nb_find(level_omp, x_omp, y_omp, ad_sib_ic[s]).idx;
      for (int s = 0; s < 4; s++)
        blk_omp[s] = BLK(sib_omp[s]);
      for (size_t v = 0; v < NVARS; v++) {
        dim_omp = fld_t[v].dim; off_omp = fld_t[v].offset;
        dst_omp = blk_omp[0] + off_omp * BS * BS;
        for (int J = 0; J < 2; J++)
          for (int I = 0; I < 2; I++)
            restrict_2to1(blk_omp[J * 2 + I] + off_omp * BS * BS,
                          dst_omp + dim_omp * (J * (BS / 2) * BS + I * (BS / 2)),
                          dim_omp, BS, BS, BS / 2, BS / 2);
      }
      bl_fill(p0_omp, level_omp - 1, x_omp / 2, y_omp / 2);
    }
  }
  cnt = 0;
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
  Real DC, DL, DR, dlim;
  Real DC_p, DL_p, DR_p, dlim_p, dp_p;
  Real DC_m, DL_m, DR_m, dlim_m, dp_m;
  Real d4, sgn;

  DC = 0.5 * (phip1 - phim1);
  DL = phi0 - phim1;
  DR = phip1 - phi0;
  dlim = DL * DR > 0 ? fmin(2 * fabs(DL), 2 * fabs(DR)) : 0;

  DC_p = 0.5 * (phip2 - phi0);
  DL_p = phip1 - phi0;
  DR_p = phip2 - phip1;
  dlim_p = DL_p * DR_p > 0 ? fmin(2 * fabs(DL_p), 2 * fabs(DR_p)) : 0;
  dp_p = fmin(fabs(DC_p), dlim_p) * (DC_p > 0 ? 1 : (DC_p < 0 ? -1 : 0));

  DC_m = 0.5 * (phi0 - phim2);
  DL_m = phim1 - phim2;
  DR_m = phi0 - phim1;
  dlim_m = DL_m * DR_m > 0 ? fmin(2 * fabs(DL_m), 2 * fabs(DR_m)) : 0;
  dp_m = fmin(fabs(DC_m), dlim_m) * (DC_m > 0 ? 1 : (DC_m < 0 ? -1 : 0));

  d4 = 4.0 / 3.0 * DC - (dp_p + dp_m) / 6.0;
  sgn = DC > 0 ? 1 : (DC < 0 ? -1 : 0);
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
          int ip, im, jp, jm;
          if ((i + j) % 2 != color)
            continue;
          ip = (i + 1) % m; im = (i - 1 + m) % m; jp = (j + 1) % m;
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
  int mc, nc;
  double *uc, *fc, *rc;
  if (m <= 4) {
    mg_smooth(u, f, m, 50);
    return;
  }
  mc = m / 2; nc = mc * mc;
  uc = w; fc = w + nc; rc = w + 2 * nc;
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
  static double *buf; /* not reentrant: single-threaded solve only */
  static int bufn;
  double *rr, *z, *p, *Ap, *r_tmp, *mgw;
  double rz;
  double pAp, al, rmax, rz2, beta;
  if (N > bufn) {
    buf = realloc(buf, 6 * N * sizeof(double));
    bufn = N;
  }
  rr = buf; z = buf + N; p = buf + 2 * N; Ap = buf + 3 * N;
  r_tmp = buf + 4 * N; mgw = buf + 5 * N;
  memset(z, 0, N * sizeof(double));

  residual(x, f, rr, M, ctx);
  if (do_mean) subtract_mean(rr, N);
  vcycle(z, rr, r_tmp, M, mgw, ctx);
  if (do_mean) subtract_mean(z, N);
  memcpy(p, z, N * sizeof(double));
  rz = 0;
  for (int k = 0; k < N; k++)
    rz += rr[k] * z[k];

  for (int it = 0; it < 100; it++) {
    pAp = 0;
    rmax = 0;
    rz2 = 0;
    matvec(p, NULL, Ap, M, ctx);
    for (int k = 0; k < N; k++)
      pAp += p[k] * Ap[k];
    if (fabs(pAp) < 1e-30)
      break;
    al = rz / pAp;
    for (int k = 0; k < N; k++) {
      x[k] += al * p[k];
      rr[k] -= al * Ap[k];
    }
    if (do_mean) { subtract_mean(rr, N); subtract_mean(x, N); }

    for (int k = 0; k < N; k++)
      if (fabs(rr[k]) > rmax)
        rmax = fabs(rr[k]);
    if (rmax < tol)
      break;

    memset(z, 0, N * sizeof(double));
    vcycle(z, rr, r_tmp, M, mgw, ctx);
    if (do_mean) subtract_mean(z, N);
    for (int k = 0; k < N; k++)
      rz2 += rr[k] * z[k];
    beta = rz2 / (rz + 1e-30);
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
          int ip, im, jp, jm;
          if ((i + j) % 2 != color)
            continue;
          ip = (i + 1) % m; im = (i - 1 + m) % m; jp = (j + 1) % m;
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
  int mc, nc;
  double *uc, *fc, *rc;
  if (m <= 4) {
    mg_smooth_helm(u, f, m, 50, alpha_h2);
    return;
  }
  mc = m / 2; nc = mc * mc;
  uc = w; fc = w + nc; rc = w + 2 * nc;
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
  struct HelmCtx *c = ctx;
  double ah2 = c->alpha / (c->h * c->h), a = 1.0 + 4.0 * ah2;
  (void)f;
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
static int amr_finest_N(Real hf) { return (int)(1.0 / hf + 0.5); }

static void amr_gather(double *dst, int field, int Ng, Real hf) {
  Real lb[LB_BUF];
  Real h;
  int ratio, bx, by;
  Real *gsrc;
  int nm_g;
  double gfx, gfy, wx, wy, v00, v10, v01, v11, val;
  int i0, j0, gx, gy;
  memset(dst, 0, (size_t)Ng * Ng * sizeof(double));
  for (long long id = 0; id < sim.n; id++) {
    h = sim.blk[id].h;
    ratio = (int)(h / hf + 0.5);
    bx = (int)(sim.blk[id].origin[0] / hf + 0.5);
    by = (int)(sim.blk[id].origin[1] / hf + 0.5);
    if (ratio == 1) {
      gsrc = BLK(id) + BS * BS * field;
      for (int j = 0; j < BS; j++)
        for (int i = 0; i < BS; i++)
          dst[(by + j) * Ng + bx + i] = gsrc[j * BS + i];
    } else {
      nm_g = BS + 2;
      lb_load(lb, 1, field, 1, id);

      for (int j = 0; j < BS; j++)
        for (int i = 0; i < BS; i++)
          for (int dj = 0; dj < ratio; dj++)
            for (int di = 0; di < ratio; di++) {
              gfx = ((double)di + 0.5) / ratio - 0.5;
              gfy = ((double)dj + 0.5) / ratio - 0.5;
              i0 = (gfx < 0) ? -1 : 0; j0 = (gfy < 0) ? -1 : 0;
              wx = gfx - i0; wy = gfy - j0;

#define LB(ci, cj) lb[nm_g * ((j) + (cj) + 1) + (i) + (ci) + 1]
              v00 = LB(i0, j0);
              v10 = LB(i0 + 1, j0);
              v01 = LB(i0, j0 + 1);
              v11 = LB(i0 + 1, j0 + 1);
#undef LB
              val = (1 - wx) * (1 - wy) * v00 + wx * (1 - wy) * v10 +
                           (1 - wx) * wy * v01 + wx * wy * v11;
              gx = bx + i * ratio + di;
              gy = by + j * ratio + dj;
              if (gx >= 0 && gx < Ng && gy >= 0 && gy < Ng)
                dst[gy * Ng + gx] = val;
            }
    }
  }
}

static void amr_scatter(double *src, int field, int Ng, Real hf) {
  Real *sdst;
  Real sh;
  int sratio, sbx, sby;
  Real sinv;
  double ssum;
  for (long long id = 0; id < sim.n; id++) {
    sdst = BLK(id) + BS * BS * field;
    sh = sim.blk[id].h;
    sratio = (int)(sh / hf + 0.5);
    sbx = (int)(sim.blk[id].origin[0] / hf + 0.5);
    sby = (int)(sim.blk[id].origin[1] / hf + 0.5);
    sinv = 1.0 / (sratio * sratio);
    for (int j = 0; j < BS; j++)
      for (int i = 0; i < BS; i++) {
        ssum = 0;
        for (int dj = 0; dj < sratio; dj++)
          for (int di = 0; di < sratio; di++)
            ssum += src[(sby + j * sratio + dj) * Ng + sbx + i * sratio + di];
        sdst[j * BS + i] = ssum * sinv;
      }
  }
}

static void advect_diffuse(Real dt) {
  Real h0 = amr_finest_h();
  int N = amr_finest_N(h0);
  Real ih = 1.0 / h0;
  Real dtdx = dt / h0, dtdy = dt / h0;
  int NN = N * N;
  double *qu, *qv, *umac, *vmac;
  double *xedge_u, *xedge_v, *yedge_u, *yedge_v;
  Real su_i, su_ip, ad_uc, ad_un, sL, sR, lo, hi, uface;
  Real sv_j, sv_jp, ad_vc, ad_vn, vface;
  double *div_ad, *phi_ad;
  double *q, *xedge, *yedge, *xlo, *xhi, *ylo, *yhi, *yzlo, *xzlo;
  Real s_ad, sy, vad, lo_v, hi_v;
  Real quxl, stl, quxh, sth, uad;
  Real lo_u, hi_u;
  Real qvyl, qvyh;
  double *qp, *u_new, *v_new;
  Real uc_g, vc_g, uR, uL, vT, vB, uu_R, uu_L, vv_T, vv_B;
  Real u_yT, u_yB, v_xR, v_xL, adv_u, adv_v, lap_u, lap_v, dpx, dpy, alpha_g;
#define IDX(i, j) (((j) + N) % N * N + ((i) + N) % N)

  enum { NSLOT = 14 };
  static double *abuf;
  static int abufn;
  if (NN > abufn) {
    abuf = realloc(abuf, NSLOT * NN * sizeof(double));
    abufn = NN;
  }
  memset(abuf, 0, NSLOT * NN * sizeof(double));
  qu = abuf; qv = abuf + NN; umac = abuf + 2 * NN;
  vmac = abuf + 3 * NN;
  xedge_u = abuf + 4 * NN; xedge_v = abuf + 5 * NN;
  yedge_u = abuf + 6 * NN; yedge_v = abuf + 7 * NN;

  amr_gather(qu, F_U, N, h0);
  amr_gather(qv, F_V, N, h0);

  for (int j = 0; j < N; j++)
    for (int i = 0; i < N; i++) {
      su_i = slope4(qu[IDX(i - 2, j)], qu[IDX(i - 1, j)], qu[IDX(i, j)],
                         qu[IDX(i + 1, j)], qu[IDX(i + 2, j)]);
      su_ip = slope4(qu[IDX(i - 1, j)], qu[IDX(i, j)], qu[IDX(i + 1, j)],
                          qu[IDX(i + 2, j)], qu[IDX(i + 3, j)]);
      ad_uc = qu[IDX(i, j)]; ad_un = qu[IDX(i + 1, j)];
      sL = ad_uc > 0 ? 1 : 0; sR = ad_un < 0 ? 1 : 0;
      lo = ad_uc + (0.5 - sL * 0.5 * dtdx * ad_uc) * su_i;
      hi = ad_un + (-0.5 - sR * 0.5 * dtdx * ad_un) * su_ip;
      uface = (lo + hi) * 0.5;
      umac[IDX(i + 1, j)] = (uface >= 0) ? lo : hi;
      if (fabs(uface) < 1e-10)
        umac[IDX(i + 1, j)] = 0.5 * (lo + hi);

      sv_j = slope4(qv[IDX(i, j - 2)], qv[IDX(i, j - 1)], qv[IDX(i, j)],
                         qv[IDX(i, j + 1)], qv[IDX(i, j + 2)]);
      sv_jp = slope4(qv[IDX(i, j - 1)], qv[IDX(i, j)], qv[IDX(i, j + 1)],
                          qv[IDX(i, j + 2)], qv[IDX(i, j + 3)]);
      ad_vc = qv[IDX(i, j)]; ad_vn = qv[IDX(i, j + 1)];
      sL = ad_vc > 0 ? 1 : 0;
      sR = ad_vn < 0 ? 1 : 0;
      lo = ad_vc + (0.5 - sL * 0.5 * dtdy * ad_vc) * sv_j;
      hi = ad_vn + (-0.5 - sR * 0.5 * dtdy * ad_vn) * sv_jp;
      vface = (lo + hi) * 0.5;
      vmac[IDX(i, j + 1)] = (vface >= 0) ? lo : hi;
      if (fabs(vface) < 1e-10)
        vmac[IDX(i, j + 1)] = 0.5 * (lo + hi);
    }

  div_ad = abuf + 8 * NN; phi_ad = abuf + 9 * NN;
  memset(div_ad, 0, NN * sizeof(double));
  memset(phi_ad, 0, NN * sizeof(double));
  for (int j = 0; j < N; j++)
    for (int i = 0; i < N; i++)
      div_ad[IDX(i, j)] = (umac[IDX(i + 1, j)] - umac[IDX(i, j)] +
                        vmac[IDX(i, j + 1)] - vmac[IDX(i, j)]) *
                       ih;
  for (int k = 0; k < NN; k++)
    div_ad[k] *= h0 * h0;
  mg_solve_periodic(phi_ad, div_ad, N, 1e-10);
  for (int j = 0; j < N; j++)
    for (int i = 0; i < N; i++) {
      umac[IDX(i, j)] -= (phi_ad[IDX(i, j)] - phi_ad[IDX(i - 1, j)]) * ih;
      vmac[IDX(i, j)] -= (phi_ad[IDX(i, j)] - phi_ad[IDX(i, j - 1)]) * ih;
    }

  for (int n = 0; n < 2; n++) {
    q = (n == 0) ? qu : qv;
    xedge = (n == 0) ? xedge_u : xedge_v;
    yedge = (n == 0) ? yedge_u : yedge_v;
    xlo = abuf + 8 * NN; xhi = abuf + 9 * NN; ylo = abuf + 10 * NN;
    yhi = abuf + 11 * NN;
    yzlo = abuf + 12 * NN;
    xzlo = abuf + 13 * NN;
    memset(xlo, 0, 4 * NN * sizeof(double));

    for (int j = 0; j < N; j++)
      for (int i = 0; i < N; i++) {
        s_ad = slope4(q[IDX(i - 2, j)], q[IDX(i - 1, j)], q[IDX(i, j)],
                        q[IDX(i + 1, j)], q[IDX(i + 2, j)]);

        xlo[IDX(i + 1, j)] =
            q[IDX(i, j)] + 0.5 * (1.0 - umac[IDX(i + 1, j)] * dtdx) * s_ad;

        xhi[IDX(i, j)] =
            q[IDX(i, j)] + 0.5 * (-1.0 - umac[IDX(i, j)] * dtdx) * s_ad;

        sy = slope4(q[IDX(i, j - 2)], q[IDX(i, j - 1)], q[IDX(i, j)],
                         q[IDX(i, j + 1)], q[IDX(i, j + 2)]);
        ylo[IDX(i, j + 1)] =
            q[IDX(i, j)] + 0.5 * (1.0 - vmac[IDX(i, j + 1)] * dtdy) * sy;
        yhi[IDX(i, j)] =
            q[IDX(i, j)] + 0.5 * (-1.0 - vmac[IDX(i, j)] * dtdy) * sy;
      }

    memset(yzlo, 0, NN * sizeof(double));
    for (int j = 0; j < N; j++)
      for (int i = 0; i < N; i++) {
        vad = vmac[IDX(i, j)];
        lo_v = ylo[IDX(i, j)]; hi_v = yhi[IDX(i, j)];
        yzlo[IDX(i, j)] = (fabs(vad) < 1e-10) ? 0.5 * (lo_v + hi_v)
                                              : ((vad >= 0) ? lo_v : hi_v);
      }

    for (int j = 0; j < N; j++)
      for (int i = 0; i < N; i++) {
        quxl = (umac[IDX(i, j)] - umac[IDX(i - 1, j)]) * q[IDX(i - 1, j)];
        stl = xlo[IDX(i, j)] - 0.5 * dtdx * quxl -
                   0.5 * dtdy *
                       (yzlo[IDX(i - 1, j + 1)] * vmac[IDX(i - 1, j + 1)] -
                        yzlo[IDX(i - 1, j)] * vmac[IDX(i - 1, j)]);

        quxh = (umac[IDX(i + 1, j)] - umac[IDX(i, j)]) * q[IDX(i, j)];
        sth = xhi[IDX(i, j)] - 0.5 * dtdx * quxh -
                   0.5 * dtdy *
                       (yzlo[IDX(i, j + 1)] * vmac[IDX(i, j + 1)] -
                        yzlo[IDX(i, j)] * vmac[IDX(i, j)]);

        uad = umac[IDX(i, j)];
        xedge[IDX(i, j)] =
            (fabs(uad) < 1e-10) ? 0.5 * (stl + sth) : ((uad >= 0) ? stl : sth);
      }

    memset(xzlo, 0, NN * sizeof(double));
    for (int j = 0; j < N; j++)
      for (int i = 0; i < N; i++) {
        uad = umac[IDX(i, j)];
        lo_u = xlo[IDX(i, j)]; hi_u = xhi[IDX(i, j)];
        xzlo[IDX(i, j)] = (fabs(uad) < 1e-10) ? 0.5 * (lo_u + hi_u)
                                              : ((uad >= 0) ? lo_u : hi_u);
      }

    for (int j = 0; j < N; j++)
      for (int i = 0; i < N; i++) {
        qvyl = (vmac[IDX(i, j)] - vmac[IDX(i, j - 1)]) * q[IDX(i, j - 1)];
        stl = ylo[IDX(i, j)] - 0.5 * dtdy * qvyl -
                   0.5 * dtdx *
                       (xzlo[IDX(i + 1, j - 1)] * umac[IDX(i + 1, j - 1)] -
                        xzlo[IDX(i, j - 1)] * umac[IDX(i, j - 1)]);

        qvyh = (vmac[IDX(i, j + 1)] - vmac[IDX(i, j)]) * q[IDX(i, j)];
        sth = yhi[IDX(i, j)] - 0.5 * dtdy * qvyh -
                   0.5 * dtdx *
                       (xzlo[IDX(i + 1, j)] * umac[IDX(i + 1, j)] -
                        xzlo[IDX(i, j)] * umac[IDX(i, j)]);

        vad = vmac[IDX(i, j)];
        yedge[IDX(i, j)] =
            (fabs(vad) < 1e-10) ? 0.5 * (stl + sth) : ((vad >= 0) ? stl : sth);
      }
  }

  div_ad = abuf + 8 * NN; phi_ad = abuf + 9 * NN;
  memset(div_ad, 0, NN * sizeof(double));
  memset(phi_ad, 0, NN * sizeof(double));
  for (int j = 0; j < N; j++)
    for (int i = 0; i < N; i++)
      div_ad[IDX(i, j)] = (xedge_u[IDX(i + 1, j)] - xedge_u[IDX(i, j)] +
                        yedge_v[IDX(i, j + 1)] - yedge_v[IDX(i, j)]) *
                       ih;
  for (int k = 0; k < NN; k++)
    div_ad[k] *= h0 * h0;
  mg_solve_periodic(phi_ad, div_ad, N, 1e-10);
  for (int j = 0; j < N; j++)
    for (int i = 0; i < N; i++) {
      xedge_u[IDX(i, j)] -= (phi_ad[IDX(i, j)] - phi_ad[IDX(i - 1, j)]) * ih;
      yedge_v[IDX(i, j)] -= (phi_ad[IDX(i, j)] - phi_ad[IDX(i, j - 1)]) * ih;
    }

  qp = abuf + 8 * NN; u_new = abuf + 9 * NN; v_new = abuf + 10 * NN;
  memset(qp, 0, 3 * NN * sizeof(double));
  amr_gather(qp, F_P, N, h0);
  for (int gj = 0; gj < N; gj++)
    for (int gi = 0; gi < N; gi++) {
      uc_g = qu[IDX(gi, gj)]; vc_g = qv[IDX(gi, gj)];
      uR = xedge_u[IDX(gi + 1, gj)]; uL = xedge_u[IDX(gi, gj)];
      vT = yedge_v[IDX(gi, gj + 1)]; vB = yedge_v[IDX(gi, gj)];
      uu_R = xedge_u[IDX(gi + 1, gj)]; uu_L = xedge_u[IDX(gi, gj)];
      vv_T = yedge_v[IDX(gi, gj + 1)]; vv_B = yedge_v[IDX(gi, gj)];
      u_yT = yedge_u[IDX(gi, gj + 1)]; u_yB = yedge_u[IDX(gi, gj)];
      v_xR = xedge_v[IDX(gi + 1, gj)]; v_xL = xedge_v[IDX(gi, gj)];
      adv_u = 0.5 * (uR + uL) * (uu_R - uu_L) * ih +
                   0.5 * (vT + vB) * (u_yT - u_yB) * ih;
      adv_v = 0.5 * (uR + uL) * (v_xR - v_xL) * ih +
                   0.5 * (vT + vB) * (vv_T - vv_B) * ih;
      lap_u = (qu[IDX(gi + 1, gj)] + qu[IDX(gi - 1, gj)] +
                    qu[IDX(gi, gj + 1)] + qu[IDX(gi, gj - 1)] - 4 * uc_g) *
                   ih * ih;
      lap_v = (qv[IDX(gi + 1, gj)] + qv[IDX(gi - 1, gj)] +
                    qv[IDX(gi, gj + 1)] + qv[IDX(gi, gj - 1)] - 4 * vc_g) *
                   ih * ih;
      dpx = (qp[IDX(gi + 1, gj)] - qp[IDX(gi - 1, gj)]) * 0.5 * ih;
      dpy = (qp[IDX(gi, gj + 1)] - qp[IDX(gi, gj - 1)]) * 0.5 * ih;
      alpha_g = sim.nu * dt * 0.5;
      u_new[IDX(gi, gj)] = uc_g + alpha_g * lap_u + dt * (-adv_u - dpx);
      v_new[IDX(gi, gj)] = vc_g + alpha_g * lap_v + dt * (-adv_v - dpy);
    }

  amr_scatter(u_new, F_U, N, h0);
  amr_scatter(v_new, F_V, N, h0);
#undef IDX
}
static void helmholtz_solve(Real dt, int field) {
  Real alpha = sim.nu * dt * 0.5;
  Real hf = amr_finest_h();
  int Ng = amr_finest_N(hf);
  int NN = Ng * Ng;
  static double *hbuf; static int hbufn;
  double *flat_f, *flat_x;
  if (NN > hbufn) { hbuf = realloc(hbuf, 2 * NN * sizeof(double)); hbufn = NN; }
  flat_f = hbuf; flat_x = hbuf + NN;
  memset(hbuf, 0, 2 * NN * sizeof(double));
  amr_gather(flat_f, field, Ng, hf);
  memcpy(flat_x, flat_f, NN * sizeof(double));
  mg_solve_helmholtz(flat_x, flat_f, Ng, alpha, hf, 1e-10);
  amr_scatter(flat_x, field, Ng, hf);
}

static void poisson_solve(Real dt) {
  Real hf = amr_finest_h();
  int N = amr_finest_N(hf);
  int NN = N * N;
  static double *pbuf; static int pbufn;
  double *rhs, *phi, *gu, *gv;
  Real fac;
  if (NN > pbufn) { pbuf = realloc(pbuf, 4 * NN * sizeof(double)); pbufn = NN; }
  rhs = pbuf; phi = pbuf + NN; gu = pbuf + 2*NN; gv = pbuf + 3*NN;
  memset(pbuf, 0, 4 * NN * sizeof(double));
  amr_gather(gu, F_U, N, hf);
  amr_gather(gv, F_V, N, hf);
  amr_gather(phi, F_PHI, N, hf);
  /* Solve: (-4φ + Σφ_nb) = (h²/dt) * div(u*)
     where div = (u_{i+1}-u_{i-1})/(2h) + (v_{j+1}-v_{j-1})/(2h).
     φ is the pressure increment; project() applies u -= dt*∇φ, p += φ. */
#define GI(i, j) (((j) + N) % N * N + ((i) + N) % N)
  fac = 0.5 * hf / dt;
  for (int j = 0; j < N; j++)
    for (int i = 0; i < N; i++)
      rhs[j * N + i] = fac * (gu[GI(i + 1, j)] - gu[GI(i - 1, j)] +
                               gv[GI(i, j + 1)] - gv[GI(i, j - 1)]);
#undef GI
  mg_solve_periodic(phi, rhs, N, 1e-10);
  amr_scatter(phi, F_PHI, N, hf);
}

static void project(Real dt) {
#pragma omp parallel
  {
    Real bp[LB_BUF];
    Real *u, *v, *p, *phi;
    int nm;
    Real ih;
#pragma omp for
    for (long long id = 0; id < sim.n; id++) {
      lb_load(bp, 1, F_PHI, 1, id);
      u = BLK(id) + BS * BS * F_U;
      v = BLK(id) + BS * BS * F_V;
      p = BLK(id) + BS * BS * F_P;
      phi = BLK(id) + BS * BS * F_PHI;
      nm = BS + 2;
      ih = 0.5 / sim.blk[id].h;
      for (int j = 0; j < BS; j++)
        for (int i = 0; i < BS; i++) {
          int k = j * BS + i;
#define PH(di, dj) bp[nm * ((j) + (dj) + 1) + (i) + (di) + 1]
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
  int nthreads = 1;
  char *base = (char *)&sim;
  int ntab = sizeof param_tab / sizeof *param_tab;
  int seen[sizeof param_tab / sizeof *param_tab] = {0};
  Real rho_layer, delta;
  Real smax;
  const char *mkey, *mval;
  int mi;
  char *mend;
  int ns;
  long long midx;
  int do_dump;
  char mpath[FILENAME_MAX];
#ifdef _OPENMP
  nthreads = omp_get_max_threads();
#endif
  fprintf(stderr, "main.c: %d threads\n", nthreads);
  argv++;
  while (*argv) {
    if ((*argv)[0] != '-' || !argv[1]) { fprintf(stderr, "usage: main -key val ...\n"); exit(1); }
    mkey = *argv++ + 1; mval = *argv++;
    for (mi = 0; mi < ntab; mi++)
      if (strcmp(mkey, param_tab[mi].name) == 0) break;
    if (mi == ntab) { fprintf(stderr, "unknown: -%s\n", mkey); exit(1); }
    if (param_tab[mi].type == 0)
      *(int *)(base + param_tab[mi].off) = (int)strtol(mval, &mend, 10);
    else
      *(Real *)(base + param_tab[mi].off) = strtod(mval, &mend);
    if (mend == mval || *mend) { fprintf(stderr, "-%s: bad '%s'\n", mkey, mval); exit(1); }
    seen[mi] = 1;
  }
  for (int i = 0; i < ntab; i++)
    if (!seen[i]) { fprintf(stderr, "-%s: not set\n", param_tab[i].name); exit(1); }

  ns = 1 << sim.levelStart;
  midx = 0;
  sim.nb = ns;
  sim.n = (long long)ns * ns;
  sim.blk = calloc(sim.n, sizeof *sim.blk);
  sim.fld = calloc(sim.n * BLK_S, sizeof(Real));
  for (int iy = 0; iy < ns; iy++)
    for (int ix = 0; ix < ns; ix++)
      bl_fill(&sim.blk[midx++], sim.levelStart, ix, iy);
  hm_rebuild();
  lb_init();

  rho_layer = 30.0; delta = 0.05;
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
    do_dump = 0;
    if (sim.dumpTime > 0 && sim.time >= sim.nextDumpTime) {
      sim.nextDumpTime += sim.dumpTime;
      do_dump = 1;
    }
    if (sim.sdump > 0 && sim.step % sim.sdump == 0)
      do_dump = 1;
    if (do_dump) {
      compute_vorticity();
      snprintf(mpath, sizeof mpath, "%08d", sim.dump_count++);
      dump(sim.time, sim.step, mpath);
    }
    if (sim.endTime > 0 && sim.time >= sim.endTime)
      break;

    smax = 0;
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
