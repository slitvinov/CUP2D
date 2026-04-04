#include <assert.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef double Real;
enum { BS = 8, NM = BS + 2, NC = BS / 2 + 2 };
enum {
  LB_BUF =
      ((2 * 4 + BS) * (2 * 4 + BS) + (BS / 2 + 4 + 3) * (BS / 2 + 4 + 3)) * 2
};
enum {
  F_U = 0,
  F_V = 1,
  F_P = 2,
  F_PHI = 3,
  F_W = 4,
  F_TMP = 5,
  F_TMP2 = 6,
  F_TMP3 = 7,
  F_UMAC = 8,
  F_VMAC = 9,
  F_N = 10,
  BLK_S = F_N * BS * BS,
};
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
enum { N_STATUS = 10 };
static struct LbTab (*lb_tab[5][3])[3][2][2][N_STATUS];
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
struct Blk {
  Real h, origin[2];
  int level, n, ix, iy;
};
#define BLK(i) (sim.fld + (long long)(i) * BLK_S)

struct FldDesc {
  int offset;
  int dim;
  char *prefix;
};
static struct FldDesc fld_t[] = {{F_U, 1, "u"},    {F_V, 1, "v"},
                                 {F_P, 1, "p"},    {F_PHI, 1, NULL},
                                 {F_W, 1, "vort"}, {F_TMP, 1, NULL}};
enum { NVARS = sizeof fld_t / sizeof *fld_t };

static int nb_ch_off[9][2][2] = {
    [0] = {{-1, -1}, {0, 0}}, [1] = {{0, -1}, {1, -1}}, [2] = {{2, -1}, {0, 0}},
    [3] = {{-1, 0}, {-1, 1}}, [4] = {{0, 0}, {0, 0}},   [5] = {{2, 0}, {2, 1}},
    [6] = {{-1, 2}, {0, 0}},  [7] = {{0, 2}, {1, 2}},   [8] = {{2, 2}, {0, 0}},
};
static int nb_ch_n[9] = {1, 2, 1, 2, 0, 2, 1, 2, 1};


struct Nb {
  int8_t s;
  int idx;
  int ch[2];
};
static int ad_sib_ic[4] = {-1, 5, 7, 8};
struct Param {
  char *name;
  int type;
  size_t off;
};
static struct Param param_tab[] = {
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

static int hm_slot(struct HMap *m, long long key) {
  unsigned long long h;

  h = (unsigned long long)key * 0x9E3779B97F4A7C15ULL;
  return (int)(h >> 32) & (m->cap - 1);
}

static int hm_get(struct HMap *m, long long key) {
  int i;

  i = hm_slot(m, key);
  while (m->e[i].key >= 0) {
    if (m->e[i].key == key) return m->e[i].val;
    i = (i + 1) & (m->cap - 1);
  }
  return -1;
}
static long long hm_key(int level, int ix, int iy) {
  long long n;

  n = 1LL << level;
  return ((n * n) - 1) / 3 + iy * n + ix;
}

static void bl_fill(struct Blk *b, int level, int ix, int iy) {
  int scale;

  scale = 1 << (level - sim.levelStart);
  b->level = level;
  b->n = 1 << level;
  b->ix = ix;
  b->iy = iy;
  b->h = 1.0 / (BS * sim.nb * scale);
  b->origin[0] = b->h * BS * ix;
  b->origin[1] = b->h * BS * iy;
}

static void hm_rebuild(void) {
  int cap, j, s;
  long long i, key;

  cap = 1;
  while (cap < 4 * sim.n) cap <<= 1;
  if (sim.hm.cap != cap) {
    free(sim.hm.e);
    sim.hm.cap = cap;
    sim.hm.e = malloc(cap * sizeof *sim.hm.e);
  }
  for (j = 0; j < cap; j++) sim.hm.e[j].key = -1;
  for (i = 0; i < sim.n; i++) {
    key = hm_key(sim.blk[i].level, sim.blk[i].ix, sim.blk[i].iy);
    s = hm_slot(&sim.hm, key);
    while (sim.hm.e[s].key >= 0 && sim.hm.e[s].key != key)
      s = (s + 1) & (cap - 1);
    sim.hm.e[s].key = key;
    sim.hm.e[s].val = i;
    if (i < 3)
      fprintf(stderr, "  hm: blk %lld lev=%d ix=%d iy=%d key=%lld slot=%d\n", i,
              sim.blk[i].level, sim.blk[i].ix, sim.blk[i].iy, key, s);
  }
  fprintf(stderr, "  hm: get(21)=%d get(22)=%d\n", hm_get(&sim.hm, 21),
          hm_get(&sim.hm, 22));
}
static struct Nb nb_find(int level, int ix, int iy, int icode) {
  int L1, b, cx, cy, fx, fy, idx, nL1, nd, nx, ny, scale;

  struct Nb r = {0, -1, {-1, -1}};
  cx = icode % 3 - 1;
  cy = icode / 3 - 1;
  scale = 1 << (level - sim.levelStart);
  nd = sim.nb * scale;
  nx = (ix + cx + nd) % nd;
  ny = (iy + cy + nd) % nd;
  idx = hm_get(&sim.hm, hm_key(level, nx, ny));
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
  L1 = level + 1;
  nL1 = 1 << L1;
  for (b = 0; b < nb_ch_n[icode]; b++) {
    fx = (ix * 2 + nb_ch_off[icode][b][0] + nL1) % nL1;
    fy = (iy * 2 + nb_ch_off[icode][b][1] + nL1) % nL1;
    r.ch[b] = hm_get(&sim.hm, hm_key(L1, fx, fy));
  }
  return r;
}

static void lb_exec(Real *blk[], Real *dst[], struct LbOp *ops, int n, int dim,
                    int nm, int nc) {
  Real *avg_d, *avg_q1, *avg_src, *c, *m;
  Real a, b, cv, sum;
  int d, dd, i, ii, jj, k;
  int8_t *w;
  struct LbOp *o;

  m = dst[0];
  c = dst[1];
  for (i = 0; i < n; i++) {
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
      for (k = 0; k < o->p1; k++)
        for (dd = 0; dd < dim; dd++)
          avg_d[k * dim + dd] =
              (avg_src[2 * k * dim + dd] + avg_src[(2 * k + 1) * dim + dd] +
               avg_q1[2 * k * dim + dd] + avg_q1[(2 * k + 1) * dim + dd]) /
              4;
      break;
    case OP_INTERP9: {
      static int8_t W[4][9] = {
          {1, 10, -1, 10, 56, -6, -1, -6, 1},
          {-1, 10, 1, -6, 56, 10, 1, -6, -1},
          {-1, -6, 1, 10, 56, -6, 1, 10, -1},
          {1, -6, -1, -6, 56, 10, -1, 10, 1},
      };
      w = W[o->flags & 3];
      for (d = 0; d < dim; d++) {
        sum = 0;
        for (jj = 0; jj < 3; jj++)
          for (ii = 0; ii < 3; ii++)
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
      static int8_t W[2][3] = {
          {8, 10, -3},
          {24, -15, 6},
      };
      w = W[o->flags & 1];
      for (d = 0; d < dim; d++) {
        a = m[o->src_off + d];
        b = m[o->dst_off + d];
        cv = m[o->p1 + d];
        m[o->src_off + d] = (w[0] * a + w[1] * b + w[2] * cv) / 15.0;
      }
      break;
    }
    }
  }
}

static void lb_init(void) {
  FILE *fp;
  char fname[64];
  int ci, dim, ss;
  size_t sz;
  struct LbTab *tab;

  int configs[][2] = {{1, 1}, {1, 2}, {2, 1}, {3, 1}};

  for (ci = 0; ci < 4; ci++) {
    ss = configs[ci][0];
    dim = configs[ci][1];
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
    lb_tab[ss][dim] = (struct LbTab(*)[3][2][2][N_STATUS])tab;
  }
}

struct LbDir {
  struct LbTab *e;
  Real *blk[2];
};
static void lb_load(Real *m, int dim, int blk_offset, int ss,
                    long long info_idx) {
  Real *c, *dst[2], *lblk[2], *p0;
  int b, cx, cy, i, icode, level, nc, nd, nm, xi, yi;
  struct Blk *info;
  struct LbDir dirs[8];
  struct LbSrc *bs;
  struct Nb nr;

  struct LbTab *te, (*cflb_tab)[3][2][2][N_STATUS];
  info = &sim.blk[info_idx];
  nm = 2 * ss + BS;
  nc = BS / 2 + ss + 3;
  level = info->level;
  xi = info->ix;
  yi = info->iy;
  cflb_tab = lb_tab[ss][dim];
  p0 = BLK(info_idx) + BS * BS * blk_offset;
  nd = 0;

  for (i = 0; i < BS; i++)
    memcpy(m + dim * ((i + ss) * nm + ss), p0 + dim * BS * i,
           BS * dim * sizeof(Real));

  c = m + nm * nm * dim;
  dst[0] = m;
  dst[1] = c;

  for (icode = 0; icode < 9; icode++) {
    cx = icode % 3 - 1;
    cy = icode / 3 - 1;
    lblk[0] = NULL;
    lblk[1] = NULL;
    if (!cx && !cy) continue;
    nr = nb_find(level, xi, yi, icode);
    te = &cflb_tab[cx + 1][cy + 1][xi % 2][yi % 2][nr.s];
    for (b = 0; b < te->n_blk; b++) {
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

  for (i = 0; i < nd; i++)
    lb_exec(dirs[i].blk, dst, dirs[i].e->ops, dirs[i].e->n_pre, dim, nm, nc);
  for (i = 0; i < nd; i++)
    lb_exec(dirs[i].blk, dst, dirs[i].e->ops + MAX_PRE, dirs[i].e->n_post, dim,
            nm, nc);
}

static void compute_vorticity(void) {
  Real *w;
  Real ih;
  Real bu[LB_BUF], bv[LB_BUF];
  int i, j, nm;
  long long id;

  for (id = 0; id < sim.n; id++) {
    lb_load(bu, 1, F_U, 1, id);
    lb_load(bv, 1, F_V, 1, id);
    w = BLK(id) + BS * BS * F_W;
    nm = BS + 2;
    ih = 0.5 / sim.blk[id].h;
    for (j = 0; j < BS; j++)
      for (i = 0; i < BS; i++) {
#define U(di, dj) bu[nm * ((j) + (dj) + 1) + (i) + (di) + 1]
#define V(di, dj) bv[nm * ((j) + (dj) + 1) + (i) + (di) + 1]
        w[j * BS + i] = (V(1, 0) - V(-1, 0)) * ih - (U(0, 1) - U(0, -1)) * ih;
#undef U
#undef V
      }
  }
}

static void restrict_2to1(Real *fine, Real *coarse, int dim, int fs, int cs,
                          int ni, int nj) {
  int d, i, j;

  for (j = 0; j < nj; j++)
    for (i = 0; i < ni; i++)
      for (d = 0; d < dim; d++)
        coarse[dim * (j * cs + i) + d] =
            0.25 * (fine[dim * ((2 * j) * fs + 2 * i) + d] +
                    fine[dim * ((2 * j) * fs + 2 * i + 1) + d] +
                    fine[dim * ((2 * j + 1) * fs + 2 * i) + d] +
                    fine[dim * ((2 * j + 1) * fs + 2 * i + 1) + d]);
}

static Real ad_ref_w[4][9] = {
    {1. / 64, 10. / 64, -1. / 64, 10. / 64, 56. / 64, -6. / 64, -1. / 64,
     -6. / 64, 1. / 64},
    {-1. / 64, 10. / 64, 1. / 64, -6. / 64, 56. / 64, 10. / 64, 1. / 64,
     -6. / 64, -1. / 64},
    {-1. / 64, -6. / 64, 1. / 64, 10. / 64, 56. / 64, -6. / 64, 1. / 64,
     10. / 64, -1. / 64},
    {1. / 64, -6. / 64, -1. / 64, -6. / 64, 56. / 64, 10. / 64, -1. / 64,
     10. / 64, 1. / 64},
};

static void prolong_2to1(Real *coarse, Real *fine, int dim, int cs, int fs,
                         int ni, int nj, int ci0, int cj0) {
  Real val;
  int d, di, dj, i, ic, j, jc, kk, s;

  for (j = 0; j < nj; j++)
    for (i = 0; i < ni; i++) {
      ic = i + ci0;
      jc = j + cj0;
      for (s = 0; s < 4; s++) {
        di = s & 1;
        dj = s >> 1;
        for (d = 0; d < dim; d++) {
          val = 0;
          for (kk = 0; kk < 9; kk++)
            val += ad_ref_w[s][kk] *
                   coarse[dim * ((jc + kk / 3 - 1) * cs + ic + kk % 3 - 1) + d];
          fine[dim * ((2 * j + dj) * fs + 2 * i + di) + d] = val;
        }
      }
    }
}

static void compute_indicator(void) {
  Real *t;
  Real au, av, pu, pv, su, sv;
  Real bu[LB_BUF], bv[LB_BUF], cu[NC * NC], cv[NC * NC];
  int ci, cj, di, di0, dj, dj0, fi, fj, i, ic, ic0, j, jc, jc0, kk, s;
  long long id;

  for (id = 0; id < sim.n; id++) {
    lb_load(bu, 1, F_U, 1, id);
    lb_load(bv, 1, F_V, 1, id);

    for (jc0 = 0; jc0 < NC; jc0++)
      for (ic0 = 0; ic0 < NC; ic0++) {
        fi = 2 * ic0 - 1;
        fj = 2 * jc0 - 1;
        su = 0;
        sv = 0;
        for (dj0 = 0; dj0 < 2; dj0++)
          for (di0 = 0; di0 < 2; di0++) {
            su += bu[NM * (fj + dj0 + 1) + fi + di0 + 1];
            sv += bv[NM * (fj + dj0 + 1) + fi + di0 + 1];
          }
        cu[jc0 * NC + ic0] = su * 0.25;
        cv[jc0 * NC + ic0] = sv * 0.25;
      }

    t = BLK(id) + BS * BS * F_TMP;
    for (j = 0; j < BS; j += 2)
      for (i = 0; i < BS; i += 2) {
        ic = i / 2 + 1;
        jc = j / 2 + 1;
        for (s = 0; s < 4; s++) {
          di = s & 1;
          dj = s >> 1;
          pu = 0;
          pv = 0;
          for (kk = 0; kk < 9; kk++) {
            ci = ic + kk % 3 - 1;
            cj = jc + kk / 3 - 1;
            pu += ad_ref_w[s][kk] * cu[cj * NC + ci];
            pv += ad_ref_w[s][kk] * cv[cj * NC + ci];
          }
          au = bu[NM * (j + dj + 1) + i + di + 1];
          av = bv[NM * (j + dj + 1) + i + di + 1];
          t[BS * (j + dj) + i + di] = fmax(fabs(au - pu), fabs(av - pv));
        }
      }
  }
}

static void dump(Real time, int step, char *path) {
  FILE *file;
  char attr_path[FILENAME_MAX], xyz_path[FILENAME_MAX];
  int dim, offset;
  int32_t blk_info[3];
  long i, j;
  size_t fi;

  snprintf(xyz_path, sizeof xyz_path, "%s.xyz.raw", path);
  file = fopen(xyz_path, "wb");
  for (i = 0; i < sim.n; i++) {
    blk_info[0] = sim.blk[i].ix;
    blk_info[1] = sim.blk[i].iy;
    blk_info[2] = sim.blk[i].level;
    fwrite(blk_info, sizeof(int32_t), 3, file);
  }
  fclose(file);
  for (fi = 0; fi < NVARS; fi++)
    if (fld_t[fi].prefix) {
      dim = fld_t[fi].dim;
      offset = fld_t[fi].offset;
      snprintf(attr_path, sizeof attr_path, "%s.%s.raw", path,
               fld_t[fi].prefix);
      file = fopen(attr_path, "wb");
      for (j = 0; j < sim.n; j++)
        fwrite(BLK(j) + offset * BS * BS, sizeof(Real), dim * BS * BS, file);
      fclose(file);
    }
}

static int ad_run(void) {
  Real *b, *blk_omp[4], *blks[4], *dst_omp;
  Real Linf;
  Real lm[LB_BUF];
  enum AdSt *state;
  int Changed, I, J, More, dim_omp, ic, j, lev, level_omp, nm_ad, off_omp, ok,
      px, py, s, x_omp, y_omp;
  long long *com_idx, *ref_idx;
  long long ci_ad, ci_omp, cnt, i, k, n_com, n_ref, nprev;
  long long sib_ad[4], sib_omp[4];
  size_t m, v;
  struct Blk *bj, *bs_blk, *p0_ad, *p0_omp, *par;
  struct Nb nr;

  n_ref = 0;
  n_com = 0;
  Changed = 0;
  compute_indicator();
  state = calloc(sim.n, sizeof *state);
  ref_idx = malloc(sim.n * sizeof *ref_idx);
  com_idx = malloc(sim.n * sizeof *com_idx);
  for (i = 0; i < sim.n; i++) {
    b = BLK(i) + BS * BS * F_TMP;
    Linf = 0;
    for (j = 0; j < BS * BS; j++) Linf = fmax(Linf, fabs(b[j]));
    lev = sim.blk[i].level;
    state[i] = Linf > sim.Rtol && lev < sim.levelMax           ? Refine
               : Linf < sim.Rtol / 1.5 && lev > sim.levelStart ? Compress
                                                               : Leave;
    Changed |= state[i] != Leave;
  }
  if (!Changed) goto done;
  for (More = 1; More;) {
    More = 0;
    for (j = 0; j < sim.n; j++) {
      if (state[j] != Refine) continue;
      bj = &sim.blk[j];
      for (ic = 0; ic < 9; ic++) {
        if (ic == 4) continue;
        nr = nb_find(bj->level, bj->ix, bj->iy, ic);
        if (nr.s == 1 || nr.idx < 0) continue;
        if (nr.s == 2 && state[nr.idx] != Refine) {
          state[nr.idx] = Refine;
          More = 1;
        } else if (nr.s == 0 && state[nr.idx] == Compress)
          state[nr.idx] = Leave;
      }
    }
  }
  for (j = 0; j < sim.n; j++) {
    if (state[j] != Compress) continue;
    bj = &sim.blk[j];
    if ((bj->ix | bj->iy) & 1) continue;
    sib_ad[0] = j;
    sib_ad[1] = 0;
    sib_ad[2] = 0;
    sib_ad[3] = 0;
    ok = 1;
    for (s = 1; s < 4 && ok; s++) {
      nr = nb_find(bj->level, bj->ix, bj->iy, ad_sib_ic[s]);
      ok = nr.s == 0 && nr.idx >= 0 && state[nr.idx] == Compress;
      sib_ad[s] = nr.idx;
    }
    for (s = 0; s < 4 && ok; s++) {
      bs_blk = &sim.blk[sib_ad[s]];
      for (ic = 0; ic < 9 && ok; ic++)
        if (ic != 4)
          ok = nb_find(bs_blk->level, bs_blk->ix, bs_blk->iy, ic).s != 1;
    }
    if (!ok) state[j] = Leave;
  }
  for (j = 0; j < sim.n; j++)
    if (state[j] == Refine) ref_idx[n_ref++] = j;
    else if (state[j] == Compress && !((sim.blk[j].ix | sim.blk[j].iy) & 1))
      com_idx[n_com++] = j;
  fprintf(stderr, "  ad: com/ref %lld/%lld\n", n_com, n_ref);
  if (n_ref == 0 && n_com == 0) goto done;
  nprev = sim.n;
  sim.n += 4 * n_ref;
  sim.blk = realloc(sim.blk, sim.n * sizeof *sim.blk);
  sim.fld = realloc(sim.fld, sim.n * BLK_S * sizeof(Real));
  memset(BLK(nprev), 0, 4 * n_ref * BLK_S * sizeof(Real));
  state = realloc(state, sim.n * sizeof *state);
  for (i = nprev; i < sim.n; i++) state[i] = Leave;
  for (k = 0; k < n_com; k++) {
    ci_ad = com_idx[k];
    p0_ad = &sim.blk[ci_ad];
    for (s = 1; s < 4; s++)
      state[nb_find(p0_ad->level, p0_ad->ix, p0_ad->iy, ad_sib_ic[s]).idx] =
          Dealloc;
  }
  for (k = 0; k < n_ref; k++) {
    par = &sim.blk[ref_idx[k]];
    px = par->ix;
    py = par->iy;
    nm_ad = 2 + BS;
    for (J = 0; J < 2; J++)
      for (I = 0; I < 2; I++) {
        ci_omp = nprev + 4 * k + 2 * J + I;
        bl_fill(&sim.blk[ci_omp], par->level + 1, 2 * px + I, 2 * py + J);
        blks[2 * J + I] = BLK(ci_omp);
      }
    for (m = 0; m < NVARS; m++) {
      dim_omp = fld_t[m].dim;
      off_omp = fld_t[m].offset;
      lb_load(lm, dim_omp, off_omp, 1, ref_idx[k]);
      for (J = 0; J < 2; J++)
        for (I = 0; I < 2; I++)
          prolong_2to1(lm, blks[J * 2 + I] + off_omp * BS * BS, dim_omp, nm_ad,
                       BS, BS / 2, BS / 2, I * (BS / 2) + 1, J * (BS / 2) + 1);
    }
    state[ref_idx[k]] = Dealloc;
  }
  for (k = 0; k < n_com; k++) {
    ci_omp = com_idx[k];
    p0_omp = &sim.blk[ci_omp];
    level_omp = p0_omp->level;
    x_omp = p0_omp->ix;
    y_omp = p0_omp->iy;
    sib_omp[0] = ci_omp;
    sib_omp[1] = 0;
    sib_omp[2] = 0;
    sib_omp[3] = 0;
    for (s = 1; s < 4; s++)
      sib_omp[s] = nb_find(level_omp, x_omp, y_omp, ad_sib_ic[s]).idx;
    for (s = 0; s < 4; s++) blk_omp[s] = BLK(sib_omp[s]);
    for (v = 0; v < NVARS; v++) {
      dim_omp = fld_t[v].dim;
      off_omp = fld_t[v].offset;
      dst_omp = blk_omp[0] + off_omp * BS * BS;
      for (J = 0; J < 2; J++)
        for (I = 0; I < 2; I++)
          restrict_2to1(blk_omp[J * 2 + I] + off_omp * BS * BS,
                        dst_omp + dim_omp * (J * (BS / 2) * BS + I * (BS / 2)),
                        dim_omp, BS, BS, BS / 2, BS / 2);
    }
    bl_fill(p0_omp, level_omp - 1, x_omp / 2, y_omp / 2);
  }
  cnt = 0;
  for (i = 0; i < sim.n; i++) {
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

static inline Real mcp(Real l, Real r) {
  return l > 0 && r > 0 ? fmin(0.5 * (l + r), 2 * fmin(l, r)) : 0;
}

static inline Real slope4(Real a, Real b, Real c, Real d, Real e) {
  Real l2, l1, r1, r2, s, sp, sm, q, sgn;
  l2 = a - c; l1 = b - c; r1 = d - c; r2 = e - c;
  if (l1 > 0 && r1 < 0) {
    l2 = -l2; l1 = -l1; r1 = -r1; r2 = -r2; sgn = -1;
  } else if (l1 < 0 && r1 > 0) {
    sgn = 1;
  } else {
    return 0;
  }
  s = fmin(-2 * l1, 2 * r1);
  sp = mcp(r1, r2 - r1);
  sm = mcp(-l1, l1 - l2);
  q = 2.0 / 3.0 * (r1 - l1) - (sp + sm) / 6.0;
  return fmin(fabs(q), s) * sgn;
}

static void mac_project(Real dt);

/* Riemann solve for the normal component (Burgers): Brown Eq 21 */
static Real riemann(Real uL, Real uR) {
  if (uL > 0 && uR > 0) return uL;
  if (uL < 0 && uR < 0) return uR;
  if (uL >= 0 && uR <= 0) return (uL + uR > 0) ? uL : (uL + uR < 0) ? uR : 0;
  return 0.5 * (uL + uR);
}

/* Upwind for passively advected component: Bell Eq 3.7b */
static Real upwind(Real qL, Real qR, Real umac) {
  return umac > 0 ? qL : umac < 0 ? qR : 0.5 * (qL + qR);
}

static void advect_diffuse(Real dt) {
  Real *um, *vm, *eu, *ev, *eut, *evr;
  Real adv_u, adv_v, alpha, dpx, dpy, dtdx, dth, h, ih, lap_u, lap_v, nu,
      su, su1, sv, sv1, uc, vc, un, vn, uL, uR, vL, vR, sL, sR;
  Real bu[LB_BUF], bv[LB_BUF], bp[LB_BUF], bum[LB_BUF], bvm[LB_BUF],
      beu[LB_BUF], bev[LB_BUF], beut[LB_BUF], bevr[LB_BUF];
  int i, j, nm;
  long long id;

  alpha = sim.nu * dt * 0.5;
  dth = 0.5 * dt;
  nu = sim.nu;

  /*------------------------------------------------------------
   * PASS 1: normal-only prediction + Riemann  (Brown Eq 20-21)
   * Produces:
   *   F_UMAC = u_mac at x-face (i+1/2,j)   [MAC velocity]
   *   F_VMAC = v_mac at y-face (i,j+1/2)   [MAC velocity]
   *   F_TMP3 = u at y-face (i,j+1/2)       [for transverse]
   *   F_W    = v at x-face (i+1/2,j)       [for transverse]
   *------------------------------------------------------------*/
  nm = BS + 6;
#define Q(b, i, j) b[nm * ((j) + 3) + (i) + 3]
  for (id = 0; id < sim.n; id++) {
    um = BLK(id) + BS * BS * F_UMAC;
    vm = BLK(id) + BS * BS * F_VMAC;
    eut = BLK(id) + BS * BS * F_TMP3;
    evr = BLK(id) + BS * BS * F_W;
    h = sim.blk[id].h;
    dtdx = dt / h;
    lb_load(bu, 1, F_U, 3, id);
    lb_load(bv, 1, F_V, 3, id);
    for (j = 0; j < BS; j++)
      for (i = 0; i < BS; i++) {
        /* --- x-face (i+1/2, j) --- */
        uc = Q(bu, i, j);
        un = Q(bu, i + 1, j);
        su = slope4(Q(bu, i - 2, j), Q(bu, i - 1, j), uc,
                    Q(bu, i + 1, j), Q(bu, i + 2, j));
        su1 = slope4(Q(bu, i - 1, j), Q(bu, i, j), un,
                     Q(bu, i + 2, j), Q(bu, i + 3, j));
        sL = uc > 0 ? 1 : 0;
        sR = un < 0 ? 1 : 0;
        uL = uc + (0.5 - sL * 0.5 * dtdx * uc) * su;
        uR = un + (-0.5 - sR * 0.5 * dtdx * un) * su1;
        um[j * BS + i] = riemann(uL, uR);

        /* v at x-face: extrapolate v using x-slope, speed = u */
        vc = Q(bv, i, j);
        vn = Q(bv, i + 1, j);
        sv = slope4(Q(bv, i - 2, j), Q(bv, i - 1, j), vc,
                    Q(bv, i + 1, j), Q(bv, i + 2, j));
        sv1 = slope4(Q(bv, i - 1, j), Q(bv, i, j), vn,
                     Q(bv, i + 2, j), Q(bv, i + 3, j));
        vL = vc + (0.5 - sL * 0.5 * dtdx * uc) * sv;
        vR = vn + (-0.5 - sR * 0.5 * dtdx * un) * sv1;
        evr[j * BS + i] = upwind(vL, vR, um[j * BS + i]);

        /* --- y-face (i, j+1/2) --- */
        vc = Q(bv, i, j);
        vn = Q(bv, i, j + 1);
        sv = slope4(Q(bv, i, j - 2), Q(bv, i, j - 1), vc,
                    Q(bv, i, j + 1), Q(bv, i, j + 2));
        sv1 = slope4(Q(bv, i, j - 1), Q(bv, i, j), vn,
                     Q(bv, i, j + 2), Q(bv, i, j + 3));
        sL = vc > 0 ? 1 : 0;
        sR = vn < 0 ? 1 : 0;
        vL = vc + (0.5 - sL * 0.5 * dtdx * vc) * sv;
        vR = vn + (-0.5 - sR * 0.5 * dtdx * vn) * sv1;
        vm[j * BS + i] = riemann(vL, vR);

        /* u at y-face: extrapolate u using y-slope, speed = v */
        uc = Q(bu, i, j);
        un = Q(bu, i, j + 1);
        su = slope4(Q(bu, i, j - 2), Q(bu, i, j - 1), uc,
                    Q(bu, i, j + 1), Q(bu, i, j + 2));
        su1 = slope4(Q(bu, i, j - 1), Q(bu, i, j), un,
                     Q(bu, i, j + 2), Q(bu, i, j + 3));
        uL = uc + (0.5 - (vc > 0 ? 1 : 0) * 0.5 * dtdx * vc) * su;
        uR = un + (-0.5 - (vn < 0 ? 1 : 0) * 0.5 * dtdx * vn) * su1;
        eut[j * BS + i] = upwind(uL, uR, vm[j * BS + i]);
      }
  }
#undef Q

  /*------------------------------------------------------------
   * MAC projection  (Brown Eq 23)
   *------------------------------------------------------------*/
  mac_project(dt);

  /*------------------------------------------------------------
   * PASS 2: full edge values  (Brown Eq 19/22)
   * Add transverse + viscous + pressure to get time-centered
   * edge values, then upwind using MAC velocities.
   * Stores results in F_TMP (u at x-face), F_TMP2 (v at y-face).
   *------------------------------------------------------------*/
  {
    int nm3 = BS + 6, nm1 = BS + 2, nm2 = BS + 4;
#define U(b, di, dj) b[nm3 * ((j) + (dj) + 3) + (i) + (di) + 3]
#define M(b, di, dj) b[nm1 * ((j) + (dj) + 1) + (i) + (di) + 1]
#define P(b, di, dj) b[nm2 * ((j) + (dj) + 2) + (i) + (di) + 2]
    for (id = 0; id < sim.n; id++) {
      eu = BLK(id) + BS * BS * F_TMP;
      ev = BLK(id) + BS * BS * F_TMP2;
      h = sim.blk[id].h;
      ih = 1.0 / h;
      lb_load(bu, 1, F_U, 3, id);
      lb_load(bv, 1, F_V, 3, id);
      lb_load(bp, 1, F_P, 2, id);
      lb_load(bum, 1, F_UMAC, 1, id);
      lb_load(bvm, 1, F_VMAC, 1, id);
      for (j = 0; j < BS; j++)
        for (i = 0; i < BS; i++) {
          Real umR, umL, vmT, vmB;
          Real euR, euL, evT, evB, eutT, eutB, evrR, evrL;
          Real q, s, qn, sn;

          umR = M(bum, 0, 0);
          umL = M(bum, -1, 0);
          vmT = M(bvm, 0, 0);
          vmB = M(bvm, 0, -1);

          q = U(bu, 0, 0);
          s = slope4(U(bu, -2, 0), U(bu, -1, 0), q, U(bu, 1, 0), U(bu, 2, 0));
          qn = U(bu, 1, 0);
          sn = slope4(U(bu, -1, 0), q, qn, U(bu, 2, 0), U(bu, 3, 0));
          euR = riemann(q + (0.5 - (q > 0) * 0.5 * dtdx * q) * s,
                        qn + (-0.5 - (qn < 0) * 0.5 * dtdx * qn) * sn);
          qn = U(bu, -1, 0);
          sn = slope4(U(bu, -3, 0), U(bu, -2, 0), qn, q, U(bu, 1, 0));
          euL = riemann(qn + (0.5 - (qn > 0) * 0.5 * dtdx * qn) * sn,
                        q + (-0.5 - (q < 0) * 0.5 * dtdx * q) * s);

          s = slope4(U(bu, 0, -2), U(bu, 0, -1), q, U(bu, 0, 1), U(bu, 0, 2));
          qn = U(bu, 0, 1);
          sn = slope4(U(bu, 0, -1), q, qn, U(bu, 0, 2), U(bu, 0, 3));
          vc = U(bv, 0, 0);
          vn = U(bv, 0, 1);
          eutT = upwind(q + (0.5 - (vc > 0) * 0.5 * dtdx * vc) * s,
                        qn + (-0.5 - (vn < 0) * 0.5 * dtdx * vn) * sn, vmT);
          qn = U(bu, 0, -1);
          sn = slope4(U(bu, 0, -3), U(bu, 0, -2), qn, q, U(bu, 0, 1));
          vn = U(bv, 0, -1);
          eutB = upwind(qn + (0.5 - (vn > 0) * 0.5 * dtdx * vn) * sn,
                        q + (-0.5 - (vc < 0) * 0.5 * dtdx * vc) * s, vmB);

          q = U(bv, 0, 0);
          s = slope4(U(bv, 0, -2), U(bv, 0, -1), q, U(bv, 0, 1), U(bv, 0, 2));
          qn = U(bv, 0, 1);
          sn = slope4(U(bv, 0, -1), q, qn, U(bv, 0, 2), U(bv, 0, 3));
          evT = riemann(q + (0.5 - (q > 0) * 0.5 * dtdx * q) * s,
                        qn + (-0.5 - (qn < 0) * 0.5 * dtdx * qn) * sn);
          qn = U(bv, 0, -1);
          sn = slope4(U(bv, 0, -3), U(bv, 0, -2), qn, q, U(bv, 0, 1));
          evB = riemann(qn + (0.5 - (qn > 0) * 0.5 * dtdx * qn) * sn,
                        q + (-0.5 - (q < 0) * 0.5 * dtdx * q) * s);

          uc = U(bu, 0, 0);
          s = slope4(U(bv, -2, 0), U(bv, -1, 0), q, U(bv, 1, 0), U(bv, 2, 0));
          qn = U(bv, 1, 0);
          sn = slope4(U(bv, -1, 0), q, qn, U(bv, 2, 0), U(bv, 3, 0));
          un = U(bu, 1, 0);
          evrR = upwind(q + (0.5 - (uc > 0) * 0.5 * dtdx * uc) * s,
                        qn + (-0.5 - (un < 0) * 0.5 * dtdx * un) * sn, umR);
          qn = U(bv, -1, 0);
          sn = slope4(U(bv, -3, 0), U(bv, -2, 0), qn, q, U(bv, 1, 0));
          un = U(bu, -1, 0);
          evrL = upwind(qn + (0.5 - (un > 0) * 0.5 * dtdx * un) * sn,
                        q + (-0.5 - (uc < 0) * 0.5 * dtdx * uc) * s, umL);

          adv_u = 0.5 * (umR + umL) * (euR - euL) * ih +
                  0.5 * (vmT + vmB) * (eutT - eutB) * ih;
          adv_v = 0.5 * (umR + umL) * (evrR - evrL) * ih +
                  0.5 * (vmT + vmB) * (evT - evB) * ih;
          uc = U(bu, 0, 0);
          vc = U(bv, 0, 0);
          lap_u = (U(bu, 1, 0) + U(bu, -1, 0) + U(bu, 0, 1) + U(bu, 0, -1) -
                   4 * uc) *
                  ih * ih;
          lap_v = (U(bv, 1, 0) + U(bv, -1, 0) + U(bv, 0, 1) + U(bv, 0, -1) -
                   4 * vc) *
                  ih * ih;
          dpx = (P(bp, 1, 0) - P(bp, -1, 0)) * 0.5 * ih;
          dpy = (P(bp, 0, 1) - P(bp, 0, -1)) * 0.5 * ih;
          eu[j * BS + i] = uc + alpha * lap_u + dt * (-adv_u - dpx);
          ev[j * BS + i] = vc + alpha * lap_v + dt * (-adv_v - dpy);
        }
    }
#undef U
#undef M
#undef P
  }
  for (id = 0; id < sim.n; id++) {
    memcpy(BLK(id) + BS * BS * F_U, BLK(id) + BS * BS * F_TMP,
           BS * BS * sizeof(Real));
    memcpy(BLK(id) + BS * BS * F_V, BLK(id) + BS * BS * F_TMP2,
           BS * BS * sizeof(Real));
  }
}

static void helmholtz_solve(Real dt, int field) {
  Real *f, *u;
  Real ah2, alpha, h, ia;
  Real buf[LB_BUF];
  int i, iter, j, nm;
  long long id;

  alpha = sim.nu * dt * 0.5;

  for (id = 0; id < sim.n; id++)
    memcpy(BLK(id) + BS * BS * F_TMP, BLK(id) + BS * BS * field,
           BS * BS * sizeof(Real));

  for (iter = 0; iter < 10; iter++) {
    {
      nm = BS + 2;
      for (id = 0; id < sim.n; id++) {
        u = BLK(id) + BS * BS * field;
        f = BLK(id) + BS * BS * F_TMP;
        h = sim.blk[id].h;
        ah2 = alpha / (h * h);
        ia = 1.0 / (1.0 + 4.0 * ah2);
        lb_load(buf, 1, field, 1, id);
        for (j = 0; j < BS; j++)
          for (i = 0; i < BS; i++) {
#define HB(di, dj) buf[nm * ((j) + (dj) + 1) + (i) + (di) + 1]
            u[j * BS + i] = (f[j * BS + i] + ah2 * (HB(1, 0) + HB(-1, 0) +
                                                    HB(0, 1) + HB(0, -1))) *
                            ia;
#undef HB
          }
      }
    }
  }
}

static void blk_laplacian(int src, int dst_field) {
  Real *out;
  Real c, h;
  Real buf[LB_BUF];
  int i, j, nm;
  long long id;

  nm = BS + 4;
  for (id = 0; id < sim.n; id++) {
    out = BLK(id) + BS * BS * dst_field;
    h = sim.blk[id].h;
    c = 1.0 / (4.0 * h * h);
    lb_load(buf, 1, src, 2, id);
    for (j = 0; j < BS; j++)
      for (i = 0; i < BS; i++) {
#define PH(di, dj) buf[nm * ((j) + (dj) + 2) + (i) + (di) + 2]
        out[j * BS + i] =
            (PH(2, 0) + PH(-2, 0) + PH(0, 2) + PH(0, -2) - 4 * PH(0, 0)) * c;
#undef PH
      }
  }
}

static Real blk_dot(int fa, int fb) {
  Real *a, *b;
  Real s;
  int k;
  long long id;

  s = 0;
  for (id = 0; id < sim.n; id++) {
    a = BLK(id) + BS * BS * fa;
    b = BLK(id) + BS * BS * fb;
    for (k = 0; k < BS * BS; k++) s += a[k] * b[k];
  }
  return s;
}

static void blk_axpy(Real alpha, int src, int dst_field) {
  Real *d, *s;
  int k;
  long long id;

  for (id = 0; id < sim.n; id++) {
    d = BLK(id) + BS * BS * dst_field;
    s = BLK(id) + BS * BS * src;
    for (k = 0; k < BS * BS; k++) d[k] += alpha * s[k];
  }
}

static void blk_copy(int src, int dst_field) {
  long long id;

  for (id = 0; id < sim.n; id++)
    memcpy(BLK(id) + BS * BS * dst_field, BLK(id) + BS * BS * src,
           BS * BS * sizeof(Real));
}

static void blk_mean_sub(int field) {
  Real *f;
  Real s;
  int k;
  long long id, ntot;

  s = 0;
  ntot = sim.n * BS * BS;
  for (id = 0; id < sim.n; id++) {
    f = BLK(id) + BS * BS * field;
    for (k = 0; k < BS * BS; k++) s += f[k];
  }
  s /= ntot;
  for (id = 0; id < sim.n; id++) {
    f = BLK(id) + BS * BS * field;
    for (k = 0; k < BS * BS; k++) f[k] -= s;
  }
}

static void blk_smooth_poisson(int rhs_field, int sol_field, int niter) {
  Real *f, *u;
  Real buf[LB_BUF];
  int i, it, j, nm;
  long long id;

  for (it = 0; it < niter; it++) {
    nm = BS + 4;
    for (id = 0; id < sim.n; id++) {
      u = BLK(id) + BS * BS * sol_field;
      f = BLK(id) + BS * BS * rhs_field;
      lb_load(buf, 1, sol_field, 2, id);
      for (j = 0; j < BS; j++)
        for (i = 0; i < BS; i++) {
#define PB(di, dj) buf[nm * ((j) + (dj) + 2) + (i) + (di) + 2]
          u[j * BS + i] = 0.25 * (PB(2, 0) + PB(-2, 0) + PB(0, 2) + PB(0, -2) -
                                  f[j * BS + i]);
#undef PB
        }
    }
    blk_mean_sub(sol_field);
  }
}

static void blk_laplacian_std(int src, int dst_field) {
  Real *out;
  Real c, h;
  Real buf[LB_BUF];
  int i, j, nm;
  long long id;

  nm = BS + 2;
  for (id = 0; id < sim.n; id++) {
    out = BLK(id) + BS * BS * dst_field;
    h = sim.blk[id].h;
    c = 1.0 / (h * h);
    lb_load(buf, 1, src, 1, id);
    for (j = 0; j < BS; j++)
      for (i = 0; i < BS; i++) {
#define PH(di, dj) buf[nm * ((j) + (dj) + 1) + (i) + (di) + 1]
        out[j * BS + i] =
            (PH(1, 0) + PH(-1, 0) + PH(0, 1) + PH(0, -1) - 4 * PH(0, 0)) * c;
#undef PH
      }
  }
}

static void blk_smooth_poisson_std(int rhs_field, int sol_field, int niter) {
  Real *f, *u;
  Real buf[LB_BUF];
  int i, it, j, nm;
  long long id;

  for (it = 0; it < niter; it++) {
    nm = BS + 2;
    for (id = 0; id < sim.n; id++) {
      u = BLK(id) + BS * BS * sol_field;
      f = BLK(id) + BS * BS * rhs_field;
      lb_load(buf, 1, sol_field, 1, id);
      for (j = 0; j < BS; j++)
        for (i = 0; i < BS; i++) {
#define PB(di, dj) buf[nm * ((j) + (dj) + 1) + (i) + (di) + 1]
          u[j * BS + i] = 0.25 * (PB(1, 0) + PB(-1, 0) + PB(0, 1) + PB(0, -1) -
                                  f[j * BS + i]);
#undef PB
        }
    }
    blk_mean_sub(sol_field);
  }
}

static void mac_project(Real dt) {
  Real *f, *p, *r, *rhs, *um, *vm, *z;
  Real al, beta, h, ih, pAp, rmax, rz, rz2;
  Real bphi[LB_BUF], bum[LB_BUF], bvm[LB_BUF];
  int i, iter, j, k, nm;
  long long id;

  nm = BS + 2;
  for (id = 0; id < sim.n; id++) {
    rhs = BLK(id) + BS * BS * F_TMP;
    ih = 1.0 / sim.blk[id].h;
    lb_load(bum, 1, F_UMAC, 1, id);
    lb_load(bvm, 1, F_VMAC, 1, id);
#define UM(di, dj) bum[nm * ((j) + (dj) + 1) + (i) + (di) + 1]
#define VM(di, dj) bvm[nm * ((j) + (dj) + 1) + (i) + (di) + 1]
    for (j = 0; j < BS; j++)
      for (i = 0; i < BS; i++)
        rhs[j * BS + i] = (UM(0, 0) - UM(-1, 0) + VM(0, 0) - VM(0, -1)) * ih;
#undef UM
#undef VM
  }

  for (id = 0; id < sim.n; id++)
    memset(BLK(id) + BS * BS * F_TMP2, 0, BS * BS * sizeof(Real));

  blk_laplacian_std(F_TMP2, F_W);
  for (id = 0; id < sim.n; id++) {
    r = BLK(id) + BS * BS * F_W;
    f = BLK(id) + BS * BS * F_TMP;
    for (k = 0; k < BS * BS; k++) r[k] = f[k] - r[k];
  }
  blk_mean_sub(F_W);
  for (id = 0; id < sim.n; id++)
    memset(BLK(id) + BS * BS * F_PHI, 0, BS * BS * sizeof(Real));
  blk_smooth_poisson_std(F_W, F_PHI, 4);
  blk_copy(F_PHI, F_TMP3);
  rz = blk_dot(F_W, F_PHI);
  for (iter = 0; iter < 200; iter++) {
    blk_laplacian_std(F_TMP3, F_PHI);
    pAp = blk_dot(F_TMP3, F_PHI);
    if (fabs(pAp) < 1e-30) break;
    al = rz / pAp;
    blk_axpy(al, F_TMP3, F_TMP2);
    blk_axpy(-al, F_PHI, F_W);
    blk_mean_sub(F_W);
    blk_mean_sub(F_TMP2);
    rmax = 0;
    for (id = 0; id < sim.n; id++) {
      r = BLK(id) + BS * BS * F_W;
      for (k = 0; k < BS * BS; k++)
        if (fabs(r[k]) > rmax) rmax = fabs(r[k]);
    }
    if (rmax < 1e-10) break;
    for (id = 0; id < sim.n; id++)
      memset(BLK(id) + BS * BS * F_PHI, 0, BS * BS * sizeof(Real));
    blk_smooth_poisson_std(F_W, F_PHI, 4);
    rz2 = blk_dot(F_W, F_PHI);
    beta = rz2 / (rz + 1e-30);
    for (id = 0; id < sim.n; id++) {
      p = BLK(id) + BS * BS * F_TMP3;
      z = BLK(id) + BS * BS * F_PHI;
      for (k = 0; k < BS * BS; k++) p[k] = z[k] + beta * p[k];
    }
    rz = rz2;
  }
  blk_mean_sub(F_TMP2);

  nm = BS + 2;
  for (id = 0; id < sim.n; id++) {
    um = BLK(id) + BS * BS * F_UMAC;
    vm = BLK(id) + BS * BS * F_VMAC;
    ih = 1.0 / sim.blk[id].h;
    lb_load(bphi, 1, F_TMP2, 1, id);
#define PH(di, dj) bphi[nm * ((j) + (dj) + 1) + (i) + (di) + 1]
    for (j = 0; j < BS; j++)
      for (i = 0; i < BS; i++) {
        um[j * BS + i] -= (PH(1, 0) - PH(0, 0)) * ih;
        vm[j * BS + i] -= (PH(0, 1) - PH(0, 0)) * ih;
      }
#undef PH
  }
}

static void poisson_solve(Real dt) {
  Real *f, *p, *r, *rhs, *z;
  Real al, beta, fac, h, pAp, rmax, rz, rz2;
  Real bu[LB_BUF], bv[LB_BUF];
  int i, iter, j, k, nm;
  long long id;

  nm = BS + 2;
  for (id = 0; id < sim.n; id++) {
    rhs = BLK(id) + BS * BS * F_TMP;
    h = sim.blk[id].h;
    fac = 2.0 * h / dt;
    lb_load(bu, 1, F_U, 1, id);
    lb_load(bv, 1, F_V, 1, id);
    for (j = 0; j < BS; j++)
      for (i = 0; i < BS; i++) {
#define UB(di, dj) bu[nm * ((j) + (dj) + 1) + (i) + (di) + 1]
#define VB(di, dj) bv[nm * ((j) + (dj) + 1) + (i) + (di) + 1]
        rhs[j * BS + i] = fac * (UB(1, 0) - UB(-1, 0) + VB(0, 1) - VB(0, -1));
#undef UB
#undef VB
      }
  }

  blk_laplacian_std(F_PHI, F_W);
  for (id = 0; id < sim.n; id++) {
    r = BLK(id) + BS * BS * F_W;
    f = BLK(id) + BS * BS * F_TMP;
    for (k = 0; k < BS * BS; k++) r[k] = f[k] - r[k];
  }
  blk_mean_sub(F_W);

  for (id = 0; id < sim.n; id++)
    memset(BLK(id) + BS * BS * F_TMP3, 0, BS * BS * sizeof(Real));
  blk_smooth_poisson_std(F_W, F_TMP3, 4);
  blk_copy(F_TMP3, F_TMP2);
  rz = blk_dot(F_W, F_TMP3);

  for (iter = 0; iter < 200; iter++) {
    blk_laplacian_std(F_TMP2, F_TMP3);
    pAp = blk_dot(F_TMP2, F_TMP3);
    if (fabs(pAp) < 1e-30) break;
    al = rz / pAp;
    blk_axpy(al, F_TMP2, F_PHI);
    blk_axpy(-al, F_TMP3, F_W);
    blk_mean_sub(F_W);
    blk_mean_sub(F_PHI);
    rmax = 0;
    for (id = 0; id < sim.n; id++) {
      r = BLK(id) + BS * BS * F_W;
      for (k = 0; k < BS * BS; k++)
        if (fabs(r[k]) > rmax) rmax = fabs(r[k]);
    }
    if (rmax < 1e-10) break;

    for (id = 0; id < sim.n; id++)
      memset(BLK(id) + BS * BS * F_TMP3, 0, BS * BS * sizeof(Real));
    blk_smooth_poisson_std(F_W, F_TMP3, 4);
    rz2 = blk_dot(F_W, F_TMP3);
    beta = rz2 / (rz + 1e-30);

    for (id = 0; id < sim.n; id++) {
      p = BLK(id) + BS * BS * F_TMP2;
      z = BLK(id) + BS * BS * F_TMP3;
      for (k = 0; k < BS * BS; k++) p[k] = z[k] + beta * p[k];
    }
    rz = rz2;
  }
  blk_mean_sub(F_PHI);
}

static void project(Real dt) {
  Real *p, *phi, *u, *v;
  Real ih;
  Real bp[LB_BUF];
  int i, j, k, nm;
  long long id;

  for (id = 0; id < sim.n; id++) {
    lb_load(bp, 1, F_PHI, 1, id);
    u = BLK(id) + BS * BS * F_U;
    v = BLK(id) + BS * BS * F_V;
    p = BLK(id) + BS * BS * F_P;
    phi = BLK(id) + BS * BS * F_PHI;
    nm = BS + 2;
    ih = 0.5 / sim.blk[id].h;
    for (j = 0; j < BS; j++)
      for (i = 0; i < BS; i++) {
        k = j * BS + i;
#define PH(di, dj) bp[nm * ((j) + (dj) + 1) + (i) + (di) + 1]
        u[k] -= dt * (PH(1, 0) - PH(-1, 0)) * ih;
        v[k] -= dt * (PH(0, 1) - PH(0, -1)) * ih;
        p[k] += phi[k];
#undef PH
      }
  }
}

int main(int argc, char **argv) {
  Real *u, *v;
  Real delta, h, ih, rho_layer, smax, x, y;
  char *base, *mend, *mkey, *mval;
  char mpath[FILENAME_MAX];
  int do_dump, i, ix, iy, j, mi, ns, ntab;
  int seen[16];
  long long midx;
  struct Blk *info;

  base = (char *)&sim;
  ntab = sizeof param_tab / sizeof *param_tab;
  memset(seen, 0, sizeof seen);
  argv++;
  while (*argv) {
    if ((*argv)[0] != '-' || !argv[1]) {
      fprintf(stderr, "usage: main -key val ...\n");
      exit(1);
    }
    mkey = *argv++ + 1;
    mval = *argv++;
    for (mi = 0; mi < ntab; mi++)
      if (strcmp(mkey, param_tab[mi].name) == 0) break;
    if (mi == ntab) {
      fprintf(stderr, "unknown: -%s\n", mkey);
      exit(1);
    }
    if (param_tab[mi].type == 0)
      *(int *)(base + param_tab[mi].off) = (int)strtol(mval, &mend, 10);
    else
      *(Real *)(base + param_tab[mi].off) = strtod(mval, &mend);
    if (mend == mval || *mend) {
      fprintf(stderr, "-%s: bad '%s'\n", mkey, mval);
      exit(1);
    }
    seen[mi] = 1;
  }
  for (i = 0; i < ntab; i++)
    if (!seen[i]) {
      fprintf(stderr, "-%s: not set\n", param_tab[i].name);
      exit(1);
    }

  ns = 1 << sim.levelStart;
  midx = 0;
  sim.nb = ns;
  sim.n = (long long)ns * ns;
  sim.blk = calloc(sim.n, sizeof *sim.blk);
  sim.fld = calloc(sim.n * BLK_S, sizeof(Real));
  for (iy = 0; iy < ns; iy++)
    for (ix = 0; ix < ns; ix++)
      bl_fill(&sim.blk[midx++], sim.levelStart, ix, iy);
  hm_rebuild();
  lb_init();

  rho_layer = 30.0;
  delta = 0.05;
  fprintf(stderr, "main.c: IC rho=%g delta=%g nu=%g\n", rho_layer, delta,
          sim.nu);
  for (i = 0; i < sim.n; i++) {
    info = &sim.blk[i];
    u = BLK(i) + BS * BS * F_U;
    v = BLK(i) + BS * BS * F_V;
    h = info->h;
    for (iy = 0; iy < BS; iy++)
      for (ix = 0; ix < BS; ix++) {
        j = BS * iy + ix;
        x = info->origin[0] + (ix + 0.5) * h;
        y = info->origin[1] + (iy + 0.5) * h;
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
    if (sim.sdump > 0 && sim.step % sim.sdump == 0) do_dump = 1;
    if (do_dump) {
      compute_vorticity();
      snprintf(mpath, sizeof mpath, "%08d", sim.dump_count++);
      dump(sim.time, sim.step, mpath);
    }
    if (sim.endTime > 0 && sim.time >= sim.endTime) break;
    if (sim.step > 0 && !(sim.dt > 0)) {
      fprintf(stderr, "main.c: dt=%e, aborting\n", sim.dt);
      break;
    }

    smax = 0;
    for (i = 0; i < sim.n; i++) {
      u = BLK(i) + BS * BS * F_U;
      v = BLK(i) + BS * BS * F_V;
      ih = 1.0 / sim.blk[i].h;
      for (j = 0; j < BS * BS; j++)
        smax = fmax(smax, fmax(fabs(u[j]), fabs(v[j])) * ih);
    }
    sim.dt = sim.CFL / (smax + 1e-30);

    if (sim.AdaptSteps > 0 && sim.step > 0 && sim.step % sim.AdaptSteps == 0) {
      ad_run();
      poisson_solve(sim.dt);
      project(sim.dt);
    }

    advect_diffuse(sim.dt);
    helmholtz_solve(sim.dt, F_U);
    helmholtz_solve(sim.dt, F_V);
    poisson_solve(sim.dt);
    project(sim.dt);

    sim.time += sim.dt;
    sim.step++;
    if (smax > 1e10) {
      fprintf(stderr, "main.c: blowup smax=%e at step %d t=%e\n", smax,
              sim.step, sim.time);
      break;
    }
  }
  fprintf(stderr, "main.c: end\n");
}
