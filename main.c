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
#define GAMMA 1.4
enum {
  F_RHO = 0,
  F_MOM = 1,
  F_ENE = 3,
  F_DRHO = 4,
  F_DMOM = 5,
  F_DENE = 7,
  F_TMP = 8,
  F_N = 9,
  BLK_S = F_N *BS *BS,
};

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
  long long n;
  struct HMap hm;
  struct Blk *blk;
  Real *fld;
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
} fld_t[] = {{F_RHO, 1, "rho"}, {F_MOM, 2, "mom"}, {F_ENE, 1, NULL},
            {F_DRHO, 1, NULL}, {F_DMOM, 2, NULL}, {F_DENE, 1, NULL},
            {F_TMP, 1, "pres"}};
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

static void compute_indicator() {
#pragma omp parallel
  {
    Real ur[LB_BUF], ue[LB_BUF];
#pragma omp for nowait
    for (long long id = 0; id < sim.n; ++id) {
      lb_load(ur, 1, F_RHO, 1, id);
      lb_load(ue, 1, F_ENE, 1, id);
      Real *TMP = BLK(id) + BS * BS * F_TMP;
      int ss = 1, nm = 2 * ss + BS;
      for (int j = 0; j < BS; ++j)
        for (int i = 0; i < BS; ++i) {
#define R(dx, dy) ur[nm * (j + ss + (dy)) + i + ss + (dx)]
#define E(dx, dy) ue[nm * (j + ss + (dy)) + i + ss + (dx)]
          Real r = R(0,0), e = E(0,0);
          Real grx = R(1,0)-R(-1,0), gry = R(0,1)-R(0,-1);
          Real gex = E(1,0)-E(-1,0), gey = E(0,1)-E(0,-1);
          Real ir = sqrt(grx*grx+gry*gry) / (r+1e-30);
          Real ie = sqrt(gex*gex+gey*gey) / (e+1e-30);
          TMP[j * BS + i] = fmax(ir, ie);
#undef R
#undef E
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
  compute_indicator();
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
static void euler_rhs() {
#pragma omp parallel for
  for (long long i = 0; i < sim.n; i++) {
    memset(BLK(i) + BS*BS*F_DRHO, 0, BS*BS*sizeof(Real));
    memset(BLK(i) + BS*BS*F_DMOM, 0, 2*BS*BS*sizeof(Real));
    memset(BLK(i) + BS*BS*F_DENE, 0, BS*BS*sizeof(Real));
  }
#pragma omp parallel
  {
    Real br[LB_BUF], bm[LB_BUF], be[LB_BUF];
#pragma omp for
    for (long long id = 0; id < sim.n; ++id) {
      lb_load(br, 1, F_RHO, 1, id);
      lb_load(bm, 2, F_MOM, 1, id);
      lb_load(be, 1, F_ENE, 1, id);
      int ss = 1, nm = 2*ss + BS;
      Real ih = 1.0 / sim.blk[id].h;
      Real *drho = BLK(id) + BS*BS*F_DRHO;
      Real *dmom = BLK(id) + BS*BS*F_DMOM;
      Real *dene = BLK(id) + BS*BS*F_DENE;
#define RHO(di,dj) br[nm*((dj)+ss) + (di)+ss]
#define MU(di,dj)  bm[2*(nm*((dj)+ss) + (di)+ss)]
#define MV(di,dj)  bm[2*(nm*((dj)+ss) + (di)+ss) + 1]
#define ENE(di,dj) be[nm*((dj)+ss) + (di)+ss]
      for (int iy = 0; iy < BS; iy++)
        for (int ix = 0; ix <= BS; ix++) {
          int iL = ix-1, iR = ix;
          Real rL=RHO(iL,iy), rR=RHO(iR,iy);
          Real muL=MU(iL,iy), muR=MU(iR,iy);
          Real mvL=MV(iL,iy), mvR=MV(iR,iy);
          Real eL=ENE(iL,iy), eR=ENE(iR,iy);
          Real uL=muL/rL, vL=mvL/rL, uR=muR/rR, vR=mvR/rR;
          Real pL=fmax(0,(GAMMA-1)*(eL-0.5*(muL*uL+mvL*vL)));
          Real pR=fmax(0,(GAMMA-1)*(eR-0.5*(muR*uR+mvR*vR)));
          Real aL=sqrt(GAMMA*pL/rL), aR=sqrt(GAMMA*pR/rR);
          Real SL=fmin(uL-aL,uR-aR), SR=fmax(uL+aL,uR+aR);
          Real fD,fU,fV,fE;
          if (SL >= 0) {
            fD=rL*uL; fU=muL*uL+pL; fV=mvL*uL; fE=(eL+pL)*uL;
          } else if (SR <= 0) {
            fD=rR*uR; fU=muR*uR+pR; fV=mvR*uR; fE=(eR+pR)*uR;
          } else {
            Real s=1.0/(SR-SL);
            fD=(SR*rL*uL      -SL*rR*uR      +SL*SR*(rR-rL))*s;
            fU=(SR*(muL*uL+pL)-SL*(muR*uR+pR)+SL*SR*(muR-muL))*s;
            fV=(SR*mvL*uL     -SL*mvR*uR     +SL*SR*(mvR-mvL))*s;
            fE=(SR*(eL+pL)*uL -SL*(eR+pR)*uR +SL*SR*(eR-eL))*s;
          }
          if (ix > 0) {
            int k=BS*iy+(ix-1);
            drho[k]-=fD*ih; dmom[2*k]-=fU*ih; dmom[2*k+1]-=fV*ih; dene[k]-=fE*ih;
          }
          if (ix < BS) {
            int k=BS*iy+ix;
            drho[k]+=fD*ih; dmom[2*k]+=fU*ih; dmom[2*k+1]+=fV*ih; dene[k]+=fE*ih;
          }
        }
      for (int iy = 0; iy <= BS; iy++)
        for (int ix = 0; ix < BS; ix++) {
          int jL = iy-1, jR = iy;
          Real rL=RHO(ix,jL), rR=RHO(ix,jR);
          Real muL=MU(ix,jL), muR=MU(ix,jR);
          Real mvL=MV(ix,jL), mvR=MV(ix,jR);
          Real eL=ENE(ix,jL), eR=ENE(ix,jR);
          Real uL=muL/rL, vL=mvL/rL, uR=muR/rR, vR=mvR/rR;
          Real pL=fmax(0,(GAMMA-1)*(eL-0.5*(muL*uL+mvL*vL)));
          Real pR=fmax(0,(GAMMA-1)*(eR-0.5*(muR*uR+mvR*vR)));
          Real aL=sqrt(GAMMA*pL/rL), aR=sqrt(GAMMA*pR/rR);
          Real SL=fmin(vL-aL,vR-aR), SR=fmax(vL+aL,vR+aR);
          Real gD,gU,gV,gE;
          if (SL >= 0) {
            gD=rL*vL; gU=muL*vL; gV=mvL*vL+pL; gE=(eL+pL)*vL;
          } else if (SR <= 0) {
            gD=rR*vR; gU=muR*vR; gV=mvR*vR+pR; gE=(eR+pR)*vR;
          } else {
            Real s=1.0/(SR-SL);
            gD=(SR*rL*vL      -SL*rR*vR      +SL*SR*(rR-rL))*s;
            gU=(SR*muL*vL     -SL*muR*vR     +SL*SR*(muR-muL))*s;
            gV=(SR*(mvL*vL+pL)-SL*(mvR*vR+pR)+SL*SR*(mvR-mvL))*s;
            gE=(SR*(eL+pL)*vL -SL*(eR+pR)*vR +SL*SR*(eR-eL))*s;
          }
          if (iy > 0) {
            int k=BS*(iy-1)+ix;
            drho[k]-=gD*ih; dmom[2*k]-=gU*ih; dmom[2*k+1]-=gV*ih; dene[k]-=gE*ih;
          }
          if (iy < BS) {
            int k=BS*iy+ix;
            drho[k]+=gD*ih; dmom[2*k]+=gU*ih; dmom[2*k+1]+=gV*ih; dene[k]+=gE*ih;
          }
        }
#undef RHO
#undef MU
#undef MV
#undef ENE
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
    Real *rho = BLK(i) + BS*BS*F_RHO;
    Real *mom = BLK(i) + BS*BS*F_MOM;
    Real *ene = BLK(i) + BS*BS*F_ENE;
    for (int iy = 0; iy < BS; iy++)
      for (int ix = 0; ix < BS; ix++) {
        Real px = info->origin[0] + info->h * (ix + 0.5);
        Real py = info->origin[1] + info->h * (iy + 0.5);
        int j = BS * iy + ix;
        Real r = 1.0;
        Real dx = px - 0.35, dy = py - 0.2;
        Real p = (dx*dx + dy*dy < 0.03*0.03) ? 100.0 : 1.0;
        rho[j] = r;
        mom[2*j] = 0;
        mom[2*j+1] = 0;
        ene[j] = p / (GAMMA - 1);
      }
  }
  for (int i = 0; i < sim.levelMax; i++)
    ad_run();
  while (1) {
    if (sim.step % 10 == 0)
      fprintf(stderr, "main.c: %08d %.6e dt=%.3e blk=%lld\n",
              sim.step, sim.time, sim.dt, sim.n);
    if (sim.dumpTime > 0 && sim.time >= sim.nextDumpTime) {
      sim.nextDumpTime += sim.dumpTime;
#pragma omp parallel for
      for (long long i = 0; i < sim.n; i++) {
        Real *r = BLK(i)+BS*BS*F_RHO;
        Real *m = BLK(i)+BS*BS*F_MOM;
        Real *e = BLK(i)+BS*BS*F_ENE;
        Real *t = BLK(i)+BS*BS*F_TMP;
        for (int j = 0; j < BS*BS; j++)
          t[j] = (GAMMA-1)*(e[j] - 0.5*(m[2*j]*m[2*j]+m[2*j+1]*m[2*j+1])/r[j]);
      }
      char path[FILENAME_MAX];
      snprintf(path, sizeof path, "vel.%08d", sim.dump_count++);
      dump(sim.time, path);
    }
    if (sim.endTime > 0 && sim.time >= sim.endTime)
      break;
    Real smax = 0;
#pragma omp parallel for reduction(max:smax)
    for (long long i = 0; i < sim.n; i++) {
      Real *rho = BLK(i)+BS*BS*F_RHO;
      Real *mom = BLK(i)+BS*BS*F_MOM;
      Real *ene = BLK(i)+BS*BS*F_ENE;
      Real ih = 1.0 / sim.blk[i].h;
      for (int j = 0; j < BS*BS; j++) {
        Real r=rho[j], u=mom[2*j]/r, v=mom[2*j+1]/r;
        Real p=fmax(0,(GAMMA-1)*(ene[j]-0.5*r*(u*u+v*v)));
        Real a=sqrt(GAMMA*p/r);
        smax = fmax(smax, (fabs(u)+fabs(v)+2*a)*ih);
      }
    }
    sim.dt = sim.CFL / (smax + 1e-30);
    if (sim.step % sim.AdaptSteps == 0)
      ad_run();
    euler_rhs();
#pragma omp parallel for
    for (long long i = 0; i < sim.n; i++) {
      Real *rho=BLK(i)+BS*BS*F_RHO, *drho=BLK(i)+BS*BS*F_DRHO;
      Real *mom=BLK(i)+BS*BS*F_MOM, *dmom=BLK(i)+BS*BS*F_DMOM;
      Real *ene=BLK(i)+BS*BS*F_ENE, *dene=BLK(i)+BS*BS*F_DENE;
      for (int j = 0; j < BS*BS; j++) {
        rho[j] += sim.dt * drho[j];
        mom[2*j] += sim.dt * dmom[2*j];
        mom[2*j+1] += sim.dt * dmom[2*j+1];
        ene[j] += sim.dt * dene[j];
      }
    }
    sim.time += sim.dt;
    sim.step++;
  }
  fprintf(stderr, "main.c: end\n");
}
