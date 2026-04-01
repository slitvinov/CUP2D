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
  F_MOM = 1,   /* 3 components: mx, my, mz */
  F_ENE = 4,
  F_DRHO = 5,
  F_DMOM = 6,  /* 3 components */
  F_DENE = 9,
  F_TMP = 10,
  F_N = 11,
  BLK_S = F_N * BS * BS * BS,
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
enum { BC_WALL=0, BC_SYMMETRY=1, BC_INFLOW=2, BC_OUTFLOW=3 };
static struct Sim {
  int AdaptSteps;
  int levelMax;
  int levelStart;
  int step;
  int dump_count;
  int nb[3];        /* base blocks per direction at levelStart */
  int bc[6];        /* BC type: x-,x+,y-,y+,z-,z+ */
  Real CFL;
  Real Ctol;
  Real dt;
  Real dumpTime;
  Real endTime;
  Real nextDumpTime;
  Real Rtol;
  Real time;
  Real L[3];        /* domain size */
  Real inflow[5];   /* inflow state: rho, mx, my, mz, E */
  long long n;
  struct HMap hm;
  struct Blk *blk;
  Real *fld;
} sim;
static long long hm_key(int level, int ix, int iy, int iz) {
  long long n = 1LL << level;
  return ((n * n * n) - 1) / 7 + iz * n * n + iy * n + ix;
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
  double h, origin[3];
  int level, n, ix, iy, iz;
};
#define BLK(i) (sim.fld + (long long)(i) * BLK_S)
static void bl_fill(struct Blk *b, int level, int ix, int iy, int iz) {
  int n = 1 << level;
  b->level = level;
  b->n = n;
  b->ix = ix;
  b->iy = iy;
  b->iz = iz;
  /* h = smallest domain extent / (BS * n_blocks_in_that_dir * 2^(level-levelStart)) */
  /* For cubic cells: h is the same in all directions */
  int s = 1 << sim.levelStart;
  b->h = sim.L[0] / (BS * sim.nb[0]) / (1 << (level - sim.levelStart));
  b->origin[0] = b->h * BS * ix;
  b->origin[1] = b->h * BS * iy;
  b->origin[2] = b->h * BS * iz;
}
struct {
  int offset;
  int dim;
  const char *prefix;
} fld_t[] = {{F_RHO, 1, "rho"}, {F_MOM, 3, "mom"}, {F_ENE, 1, NULL},
            {F_DRHO, 1, NULL}, {F_DMOM, 3, NULL}, {F_DENE, 1, NULL},
            {F_TMP, 1, "pres"}};
enum { NVARS = sizeof fld_t / sizeof *fld_t };

static inline int nb_skin(int c, int coord, int n) {
  int skin = coord == 0 || coord == n - 1;
  int skip = coord == 0 ? -1 : 1;
  return c == skip && skin;
}
/* 3D child offsets for finer neighbors: nb_ch_off[icode][child_idx][3] */
/* nb_ch_n[icode] = number of children for this neighbor direction */
/* For face neighbors: 4 children, edge: 2, corner: 1 */
static int nb_ch_n_3d(int cx, int cy, int cz) {
  int a = abs(cx) + abs(cy) + abs(cz);
  if (a == 1) return 4;
  if (a == 2) return 2;
  return 1;
}
static void nb_ch_off_3d(int cx, int cy, int cz, int b, int out[3]) {
  /* compute the child offset for child b of the finer neighbor at (cx,cy,cz) */
  int dirs[3] = {cx, cy, cz};
  int zero_axes[3], nz = 0;
  for (int d = 0; d < 3; d++)
    if (dirs[d] == 0) zero_axes[nz++] = d;
  out[0] = cx == 0 ? 0 : cx * 2;
  out[1] = cy == 0 ? 0 : cy * 2;
  out[2] = cz == 0 ? 0 : cz * 2;
  if (nz >= 1) out[zero_axes[0]] = (b & 1);
  if (nz >= 2) out[zero_axes[1]] = ((b >> 1) & 1);
}

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
    long long key = hm_key(sim.blk[i].level, sim.blk[i].ix,
                           sim.blk[i].iy, sim.blk[i].iz);
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
  int ch[4];
};
/* Wall BC status encoding for 3D:
   s=0: same level, s=1: finer, s=2: coarser
   s=3: x-wall, s=4: y-wall, s=5: z-wall
   s=6: xy-edge, s=7: xz-edge, s=8: yz-edge
   s=9: xyz-corner */
static struct Nb nb_find(int level, int ix, int iy, int iz, int icode) {
  struct Nb r = {0, -1, {-1, -1, -1, -1}};
  int cx = icode % 3 - 1, cy = (icode / 3) % 3 - 1, cz = icode / 9 - 1;
  int scale = 1 << (level - sim.levelStart);
  int nd[3] = {sim.nb[0]*scale, sim.nb[1]*scale, sim.nb[2]*scale};
  int xskin = nb_skin(cx, ix, nd[0]);
  int yskin = nb_skin(cy, iy, nd[1]);
  int zskin = nb_skin(cz, iz, nd[2]);
  int nbc = xskin + yskin + zskin;
  if (nbc == 3) { r.s = 9; return r; }
  if (nbc == 2) {
    if (!xskin) r.s = 8;      /* yz-edge */
    else if (!yskin) r.s = 7;  /* xz-edge */
    else r.s = 6;              /* xy-edge */
    return r;
  }
  if (nbc == 1) {
    if (xskin) r.s = 3;
    else if (yskin) r.s = 4;
    else r.s = 5;
    return r;
  }
  int nnx = (ix + cx + nd[0]) % nd[0], nny = (iy + cy + nd[1]) % nd[1], nnz = (iz + cz + nd[2]) % nd[2];
  int idx = hm_get(&sim.hm, hm_key(level, nnx, nny, nnz));
  if (idx >= 0) {
    r.s = 0;
    r.idx = idx;
    return r;
  }
  if (level > 0) {
    idx = hm_get(&sim.hm, hm_key(level - 1, nnx / 2, nny / 2, nnz / 2));
    if (idx >= 0) {
      r.s = 2;
      r.idx = idx;
      return r;
    }
  }
  if (level > 1) {
    idx = hm_get(&sim.hm, hm_key(level - 2, nnx / 4, nny / 4, nnz / 4));
    if (idx >= 0) {
      r.s = 2;
      r.idx = idx;
      return r;
    }
  }
  r.s = 1;
  int nch = nb_ch_n_3d(cx, cy, cz);
  int L1 = level + 1, nL1 = 1 << L1;
  for (int b = 0; b < nch; b++) {
    int off[3];
    nb_ch_off_3d(cx, cy, cz, b, off);
    int fx = (ix * 2 + off[0] + nL1) % nL1;
    int fy = (iy * 2 + off[1] + nL1) % nL1;
    int fz = (iz * 2 + off[2] + nL1) % nL1;
    r.ch[b] = hm_get(&sim.hm, hm_key(L1, fx, fy, fz));
  }
  return r;
}

/* ---- Ghost fill tables ---- */
enum {
  OP_COPY,
  OP_AVG,
  OP_INTERP27,
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
  int8_t xi_mul, yi_mul, zi_mul;
  int8_t xi_add, yi_add, zi_add;
  int8_t xi_shift, yi_shift, zi_shift;
  int8_t is_self;
  int8_t self_idx;
};
enum { MAX_PRE = 400, MAX_POST = 64, MAX_OPS = MAX_PRE + MAX_POST };
struct LbTab {
  int8_t n_blk;
  struct LbSrc blk_src[4];
  int8_t _pad[3];
  int32_t n_pre;
  int32_t n_post;
  struct LbOp ops[MAX_OPS];
};
/* Table indexing: lb_tab[ss][dim][cx+1][cy+1][cz+1][xp%2][yp%2][zp%2][status]
   ss in {1,4}, dim in {1,3}
   cx,cy,cz in {-1,0,1}, xp,yp,zp parity, status in {0..9} */
static const struct LbTab *lb_tab[5][4];
static void lb_init(void) {
  int configs[][2] = {{1, 1}, {1, 3}, {4, 1}};
  for (int ci = 0; ci < 3; ci++) {
    int ss = configs[ci][0], dim = configs[ci][1];
    char fname[64];
    snprintf(fname, sizeof fname, "tab3d_ss%d_dim%d.bin", ss, dim);
    FILE *fp = fopen(fname, "rb");
    if (!fp) {
      fprintf(stderr, "main.c: cannot open %s\n", fname);
      exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    void *tab = malloc(sz);
    if (fread(tab, 1, sz, fp) != sz) {
      fprintf(stderr, "main.c: short read from %s\n", fname);
      exit(1);
    }
    fclose(fp);
    lb_tab[ss][dim] = (const struct LbTab *)tab;
  }
}

enum { LB_BUF3 = (2*4+BS)*(2*4+BS)*(2*4+BS)*3 + (BS/2+4+3)*(BS/2+4+3)*(BS/2+4+3)*3 };
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
      int stride1 = o->p1 * dim;        /* row stride in source */
      int stride2 = o->p2 * dim;        /* plane stride in source */
      int cols = o->flags;              /* number of output cells */
      for (int k = 0; k < cols; k++)
        for (int dd = 0; dd < dim; dd++) {
          d[k * dim + dd] =
            (src[2*k*dim+dd] + src[(2*k+1)*dim+dd]
           + src[2*k*dim+dd+stride1] + src[(2*k+1)*dim+dd+stride1]
           + src[2*k*dim+dd+stride2] + src[(2*k+1)*dim+dd+stride2]
           + src[2*k*dim+dd+stride1+stride2] + src[(2*k+1)*dim+dd+stride1+stride2]) / 8;
        }
      break;
    }
    case OP_INTERP27: {
      /* Trilinear-cubic 27-point interpolation from coarse buffer to fine cell.
         flags = parity index px|(py<<1)|(pz<<2)
         src_off = center of 3x3x3 stencil in coarse buffer (c)
         dst_off = destination in fine buffer (m)
         1D weights: sub-cell at -1/4 -> {5,30,-3}/32
                     sub-cell at +1/4 -> {-3,30,5}/32 */
      static const int W1[2][3] = {{5,30,-3},{-3,30,5}};
      int px = o->flags & 1, py = (o->flags>>1) & 1, pz = (o->flags>>2) & 1;
      for (int d = 0; d < dim; d++) {
        Real sum = 0;
        for (int kk = -1; kk <= 1; kk++)
          for (int jj = -1; jj <= 1; jj++)
            for (int ii = -1; ii <= 1; ii++)
              sum += (Real)W1[px][ii+1] * W1[py][jj+1] * W1[pz][kk+1]
                   * c[o->src_off + d + dim*(ii + nc*jj + nc*nc*kk)];
        m[o->dst_off + d] = sum / 32768.0;
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
      /* flags encodes which axes to negate as a bitmask: bit0=x, bit1=y, bit2=z */
      int mask = o->flags;
      for (int d = 0; d < 3; d++)
        buf[o->dst_off + d] = (mask & (1 << d)) ? -buf[o->src_off + d]
                                                 : buf[o->src_off + d];
      break;
    }
    }
  }
}

static void lb_load(Real *m, int dim, int blk_offset, int ss, long long info_idx) {
  struct Blk *info = &sim.blk[info_idx];
  int nm = 2 * ss + BS;
  int nc = BS / 2 + ss + 3;
  int level = info->level;
  int xi = info->ix, yi = info->iy, zi = info->iz;
  const struct LbTab *tab = lb_tab[ss][dim];

  Real *p0 = BLK(info_idx) + BS * BS * BS * blk_offset;
  for (int k = 0; k < BS; k++)
    for (int j = 0; j < BS; j++)
      memcpy(m + dim * (((k + ss) * nm + j + ss) * nm + ss),
             p0 + dim * ((k * BS + j) * BS),
             BS * dim * sizeof(Real));

  Real *c = m + nm * nm * nm * dim;
  Real *dst[2] = {m, c};

  struct {
    const struct LbTab *e;
    Real *blk[4];
  } dirs[26];
  int nd = 0;
  for (int icode = 0; icode < 27; icode++) {
    int cx = icode % 3 - 1, cy = (icode / 3) % 3 - 1, cz = icode / 9 - 1;
    if (!cx && !cy && !cz)
      continue;
    struct Nb nr = nb_find(level, xi, yi, zi, icode);
    int tidx = ((((((cx+1)*3+(cy+1))*3+(cz+1))*2+(xi%2))*2+(yi%2))*2+(zi%2))*10 + nr.s;
    const struct LbTab *te = &tab[tidx];
    /* Skip neighbors with missing block data */
    if (nr.s == 0 && nr.idx < 0) continue;
    if (nr.s == 2 && nr.idx < 0) continue;
    if (nr.s == 1) {
      int nch = nb_ch_n_3d(cx, cy, cz);
      int skip = 0;
      for (int b = 0; b < nch; b++) if (nr.ch[b] < 0) skip = 1;
      if (skip) continue;
    }
    Real *blk[4] = {NULL, NULL, NULL, NULL};
    for (int b = 0; b < te->n_blk; b++) {
      const struct LbSrc *bs = &te->blk_src[b];
      if (bs->is_self) {
        blk[b] = dst[bs->self_idx];
      } else if (bs->level_delta == 1) {
        blk[b] = BLK(nr.ch[b]) + BS * BS * BS * blk_offset;
      } else {
        blk[b] = BLK(nr.idx) + BS * BS * BS * blk_offset;
      }
    }
    dirs[nd].e = te;
    for (int b = 0; b < 4; b++) dirs[nd].blk[b] = blk[b];
    nd++;
  }

  for (int i = 0; i < nd; i++)
    lb_exec(dirs[i].blk, dst, dirs[i].e->ops, dirs[i].e->n_pre,
            dim, nm, nc);
  for (int i = 0; i < nd; i++)
    lb_exec(dirs[i].blk, dst, dirs[i].e->ops + MAX_PRE,
            dirs[i].e->n_post, dim, nm, nc);

  /* Fixup inflow/outflow BCs (override table's wall reflection) */
  {
    int scl = 1 << (level - sim.levelStart);
    int nd3[3] = {sim.nb[0]*scl, sim.nb[1]*scl, sim.nb[2]*scl};
    int coords[3] = {xi, yi, zi};
    /* 6 faces: axis 0,1,2 × side 0,1 */
    for (int axis = 0; axis < 3; axis++) {
      for (int side = 0; side < 2; side++) {
        int face = 2*axis + side;
        int bc = sim.bc[face];
        if (bc != BC_INFLOW && bc != BC_OUTFLOW) continue;
        int at_bnd = (side == 0) ? (coords[axis] == 0) : (coords[axis] == nd3[axis]-1);
        if (!at_bnd) continue;
        /* Iterate over ghost cells on this face */
        int gs[3] = {0, 0, 0}, ge[3] = {BS, BS, BS};
        gs[axis] = side == 0 ? -ss : BS;
        ge[axis] = side == 0 ? 0 : BS + ss;
        for (int iz2 = gs[2]; iz2 < ge[2]; iz2++)
          for (int iy2 = gs[1]; iy2 < ge[1]; iy2++)
            for (int ix2 = gs[0]; ix2 < ge[0]; ix2++) {
              int gi = dim * ((iz2+ss)*nm*nm + (iy2+ss)*nm + (ix2+ss));
              if (bc == BC_INFLOW) {
                /* Fixed inflow state */
                if (blk_offset == F_RHO) m[gi] = sim.inflow[0];
                else if (blk_offset == F_MOM) {
                  m[gi] = sim.inflow[1]; m[gi+1] = sim.inflow[2]; m[gi+2] = sim.inflow[3];
                } else if (blk_offset == F_ENE) m[gi] = sim.inflow[4];
                else {
                  /* For indicator/scratch fields, copy from interior */
                  int ic[3] = {ix2, iy2, iz2};
                  ic[axis] = side == 0 ? 0 : BS-1;
                  int ii = dim * ((ic[2]+ss)*nm*nm + (ic[1]+ss)*nm + (ic[0]+ss));
                  for (int d = 0; d < dim; d++) m[gi+d] = m[ii+d];
                }
              } else { /* BC_OUTFLOW: zero-gradient (copy from interior) */
                int ic[3] = {ix2, iy2, iz2};
                ic[axis] = side == 0 ? 0 : BS-1;
                int ii = dim * ((ic[2]+ss)*nm*nm + (ic[1]+ss)*nm + (ic[0]+ss));
                for (int d = 0; d < dim; d++) m[gi+d] = m[ii+d];
              }
            }
      }
    }
  }
}

/* ---- Physics ---- */
static inline Real minmod(Real a, Real b) {
  return a * b <= 0 ? 0 : fabs(a) < fabs(b) ? a : b;
}
static inline Real pres(Real r, Real mx, Real my, Real mz, Real e) {
  return fmax(0, (GAMMA - 1) * (e - 0.5 * (mx * mx + my * my + mz * mz) / r));
}

static void hll_x(Real rL, Real mxL, Real myL, Real mzL, Real eL,
                  Real rR, Real mxR, Real myR, Real mzR, Real eR,
                  Real *fD, Real *fU, Real *fV, Real *fW, Real *fE) {
  Real uL = mxL / rL, uR = mxR / rR;
  Real pL = pres(rL, mxL, myL, mzL, eL), pR = pres(rR, mxR, myR, mzR, eR);
  Real aL = sqrt(GAMMA * pL / rL), aR = sqrt(GAMMA * pR / rR);
  Real SL = fmin(uL - aL, uR - aR), SR = fmax(uL + aL, uR + aR);
  if (SL >= 0) {
    *fD = rL*uL; *fU = mxL*uL+pL; *fV = myL*uL; *fW = mzL*uL; *fE = (eL+pL)*uL;
  } else if (SR <= 0) {
    *fD = rR*uR; *fU = mxR*uR+pR; *fV = myR*uR; *fW = mzR*uR; *fE = (eR+pR)*uR;
  } else {
    Real s = 1.0 / (SR - SL);
    *fD = (SR*rL*uL - SL*rR*uR + SL*SR*(rR-rL))*s;
    *fU = (SR*(mxL*uL+pL) - SL*(mxR*uR+pR) + SL*SR*(mxR-mxL))*s;
    *fV = (SR*myL*uL - SL*myR*uR + SL*SR*(myR-myL))*s;
    *fW = (SR*mzL*uL - SL*mzR*uR + SL*SR*(mzR-mzL))*s;
    *fE = (SR*(eL+pL)*uL - SL*(eR+pR)*uR + SL*SR*(eR-eL))*s;
  }
}
static void hll_y(Real rL, Real mxL, Real myL, Real mzL, Real eL,
                  Real rR, Real mxR, Real myR, Real mzR, Real eR,
                  Real *gD, Real *gU, Real *gV, Real *gW, Real *gE) {
  Real vL = myL / rL, vR = myR / rR;
  Real pL = pres(rL, mxL, myL, mzL, eL), pR = pres(rR, mxR, myR, mzR, eR);
  Real aL = sqrt(GAMMA * pL / rL), aR = sqrt(GAMMA * pR / rR);
  Real SL = fmin(vL - aL, vR - aR), SR = fmax(vL + aL, vR + aR);
  if (SL >= 0) {
    *gD = rL*vL; *gU = mxL*vL; *gV = myL*vL+pL; *gW = mzL*vL; *gE = (eL+pL)*vL;
  } else if (SR <= 0) {
    *gD = rR*vR; *gU = mxR*vR; *gV = myR*vR+pR; *gW = mzR*vR; *gE = (eR+pR)*vR;
  } else {
    Real s = 1.0 / (SR - SL);
    *gD = (SR*rL*vL - SL*rR*vR + SL*SR*(rR-rL))*s;
    *gU = (SR*mxL*vL - SL*mxR*vR + SL*SR*(mxR-mxL))*s;
    *gV = (SR*(myL*vL+pL) - SL*(myR*vR+pR) + SL*SR*(myR-myL))*s;
    *gW = (SR*mzL*vL - SL*mzR*vR + SL*SR*(mzR-mzL))*s;
    *gE = (SR*(eL+pL)*vL - SL*(eR+pR)*vR + SL*SR*(eR-eL))*s;
  }
}
static void hll_z(Real rL, Real mxL, Real myL, Real mzL, Real eL,
                  Real rR, Real mxR, Real myR, Real mzR, Real eR,
                  Real *hD, Real *hU, Real *hV, Real *hW, Real *hE) {
  Real wL = mzL / rL, wR = mzR / rR;
  Real pL = pres(rL, mxL, myL, mzL, eL), pR = pres(rR, mxR, myR, mzR, eR);
  Real aL = sqrt(GAMMA * pL / rL), aR = sqrt(GAMMA * pR / rR);
  Real SL = fmin(wL - aL, wR - aR), SR = fmax(wL + aL, wR + aR);
  if (SL >= 0) {
    *hD = rL*wL; *hU = mxL*wL; *hV = myL*wL; *hW = mzL*wL+pL; *hE = (eL+pL)*wL;
  } else if (SR <= 0) {
    *hD = rR*wR; *hU = mxR*wR; *hV = myR*wR; *hW = mzR*wR+pR; *hE = (eR+pR)*wR;
  } else {
    Real s = 1.0 / (SR - SL);
    *hD = (SR*rL*wL - SL*rR*wR + SL*SR*(rR-rL))*s;
    *hU = (SR*mxL*wL - SL*mxR*wR + SL*SR*(mxR-mxL))*s;
    *hV = (SR*myL*wL - SL*myR*wR + SL*SR*(myR-myL))*s;
    *hW = (SR*(mzL*wL+pL) - SL*(mzR*wR+pR) + SL*SR*(mzR-mzL))*s;
    *hE = (SR*(eL+pL)*wL - SL*(eR+pR)*wR + SL*SR*(eR-eL))*s;
  }
}

/* X-sweep: update all cells in blocks at given level using x-direction fluxes */
static void euler_x_sweep(Real dt, int level) {
#pragma omp parallel
  {
    Real br[LB_BUF3], bm[LB_BUF3], be[LB_BUF3];
#pragma omp for
    for (long long id = 0; id < sim.n; ++id) {
      if (sim.blk[id].level != level) continue;
      lb_load(br, 1, F_RHO, 1, id);
      lb_load(bm, 3, F_MOM, 1, id);
      lb_load(be, 1, F_ENE, 1, id);
      int ss = 1, nm = 2 * ss + BS;
      Real dth = dt / sim.blk[id].h;
#define RH(di,dj,dk) br[nm*nm*((dk)+ss)+nm*((dj)+ss)+(di)+ss]
#define MX(di,dj,dk) bm[3*(nm*nm*((dk)+ss)+nm*((dj)+ss)+(di)+ss)]
#define MY(di,dj,dk) bm[3*(nm*nm*((dk)+ss)+nm*((dj)+ss)+(di)+ss)+1]
#define MZ(di,dj,dk) bm[3*(nm*nm*((dk)+ss)+nm*((dj)+ss)+(di)+ss)+2]
#define EN(di,dj,dk) be[nm*nm*((dk)+ss)+nm*((dj)+ss)+(di)+ss]
      Real *rho = BLK(id)+BS*BS*BS*F_RHO;
      Real *mom = BLK(id)+BS*BS*BS*F_MOM;
      Real *ene = BLK(id)+BS*BS*BS*F_ENE;
      for (int k = 0; k < BS; k++)
        for (int j = 0; j < BS; j++)
          for (int i = 0; i <= BS; i++) {
            Real rL=RH(i-1,j,k),rR=RH(i,j,k);
            Real mxL=MX(i-1,j,k),mxR=MX(i,j,k);
            Real myL=MY(i-1,j,k),myR=MY(i,j,k);
            Real mzL=MZ(i-1,j,k),mzR=MZ(i,j,k);
            Real eL=EN(i-1,j,k),eR=EN(i,j,k);
            Real sr0=0,sx0=0,sy0=0,sz0=0,se0=0;
            Real sr1=0,sx1=0,sy1=0,sz1=0,se1=0;
            if (i >= 1) {
              sr0=minmod(rL-RH(i-2,j,k),rR-rL);
              sx0=minmod(mxL-MX(i-2,j,k),mxR-mxL);
              sy0=minmod(myL-MY(i-2,j,k),myR-myL);
              sz0=minmod(mzL-MZ(i-2,j,k),mzR-mzL);
              se0=minmod(eL-EN(i-2,j,k),eR-eL);
            }
            if (i < BS) {
              sr1=minmod(rR-rL,RH(i+1,j,k)-rR);
              sx1=minmod(mxR-mxL,MX(i+1,j,k)-mxR);
              sy1=minmod(myR-myL,MY(i+1,j,k)-myR);
              sz1=minmod(mzR-mzL,MZ(i+1,j,k)-mzR);
              se1=minmod(eR-eL,EN(i+1,j,k)-eR);
            }
            Real rl=rL+.5*sr0,rr=rR-.5*sr1;
            Real xl=mxL+.5*sx0,xr=mxR-.5*sx1;
            Real yl=myL+.5*sy0,yr=myR-.5*sy1;
            Real zl=mzL+.5*sz0,zr=mzR-.5*sz1;
            Real el=eL+.5*se0,er=eR-.5*se1;
            if(rl<=0){rl=rL;xl=mxL;yl=myL;zl=mzL;el=eL;}
            if(rr<=0){rr=rR;xr=mxR;yr=myR;zr=mzR;er=eR;}
            Real fD,fU,fV,fW,fE;
            hll_x(rl,xl,yl,zl,el,rr,xr,yr,zr,er,&fD,&fU,&fV,&fW,&fE);
            if (i > 0) {
              int c = (k*BS+j)*BS+(i-1);
              rho[c]-=fD*dth; mom[3*c]-=fU*dth; mom[3*c+1]-=fV*dth;
              mom[3*c+2]-=fW*dth; ene[c]-=fE*dth;
            }
            if (i < BS) {
              int c = (k*BS+j)*BS+i;
              rho[c]+=fD*dth; mom[3*c]+=fU*dth; mom[3*c+1]+=fV*dth;
              mom[3*c+2]+=fW*dth; ene[c]+=fE*dth;
            }
          }
#undef RH
#undef MX
#undef MY
#undef MZ
#undef EN
    }
  }
}
/* Y-sweep and Z-sweep follow the same pattern */
static void euler_y_sweep(Real dt, int level) {
#pragma omp parallel
  {
    Real br[LB_BUF3], bm[LB_BUF3], be[LB_BUF3];
#pragma omp for
    for (long long id = 0; id < sim.n; ++id) {
      if (sim.blk[id].level != level) continue;
      lb_load(br, 1, F_RHO, 1, id);
      lb_load(bm, 3, F_MOM, 1, id);
      lb_load(be, 1, F_ENE, 1, id);
      int ss = 1, nm = 2 * ss + BS;
      Real dth = dt / sim.blk[id].h;
#define RH(di,dj,dk) br[nm*nm*((dk)+ss)+nm*((dj)+ss)+(di)+ss]
#define MX(di,dj,dk) bm[3*(nm*nm*((dk)+ss)+nm*((dj)+ss)+(di)+ss)]
#define MY(di,dj,dk) bm[3*(nm*nm*((dk)+ss)+nm*((dj)+ss)+(di)+ss)+1]
#define MZ(di,dj,dk) bm[3*(nm*nm*((dk)+ss)+nm*((dj)+ss)+(di)+ss)+2]
#define EN(di,dj,dk) be[nm*nm*((dk)+ss)+nm*((dj)+ss)+(di)+ss]
      Real *rho = BLK(id)+BS*BS*BS*F_RHO;
      Real *mom = BLK(id)+BS*BS*BS*F_MOM;
      Real *ene = BLK(id)+BS*BS*BS*F_ENE;
      for (int k = 0; k < BS; k++)
        for (int j = 0; j <= BS; j++)
          for (int i = 0; i < BS; i++) {
            Real rL=RH(i,j-1,k),rR=RH(i,j,k);
            Real mxL=MX(i,j-1,k),mxR=MX(i,j,k);
            Real myL=MY(i,j-1,k),myR=MY(i,j,k);
            Real mzL=MZ(i,j-1,k),mzR=MZ(i,j,k);
            Real eL=EN(i,j-1,k),eR=EN(i,j,k);
            Real sr0=0,sx0=0,sy0=0,sz0=0,se0=0;
            Real sr1=0,sx1=0,sy1=0,sz1=0,se1=0;
            if (j >= 1) {
              sr0=minmod(rL-RH(i,j-2,k),rR-rL);
              sx0=minmod(mxL-MX(i,j-2,k),mxR-mxL);
              sy0=minmod(myL-MY(i,j-2,k),myR-myL);
              sz0=minmod(mzL-MZ(i,j-2,k),mzR-mzL);
              se0=minmod(eL-EN(i,j-2,k),eR-eL);
            }
            if (j < BS) {
              sr1=minmod(rR-rL,RH(i,j+1,k)-rR);
              sx1=minmod(mxR-mxL,MX(i,j+1,k)-mxR);
              sy1=minmod(myR-myL,MY(i,j+1,k)-myR);
              sz1=minmod(mzR-mzL,MZ(i,j+1,k)-mzR);
              se1=minmod(eR-eL,EN(i,j+1,k)-eR);
            }
            Real rl=rL+.5*sr0,rr=rR-.5*sr1;
            Real xl=mxL+.5*sx0,xr=mxR-.5*sx1;
            Real yl=myL+.5*sy0,yr=myR-.5*sy1;
            Real zl=mzL+.5*sz0,zr=mzR-.5*sz1;
            Real el=eL+.5*se0,er=eR-.5*se1;
            if(rl<=0){rl=rL;xl=mxL;yl=myL;zl=mzL;el=eL;}
            if(rr<=0){rr=rR;xr=mxR;yr=myR;zr=mzR;er=eR;}
            Real gD,gU,gV,gW,gE;
            hll_y(rl,xl,yl,zl,el,rr,xr,yr,zr,er,&gD,&gU,&gV,&gW,&gE);
            if (j > 0) {
              int c = (k*BS+(j-1))*BS+i;
              rho[c]-=gD*dth; mom[3*c]-=gU*dth; mom[3*c+1]-=gV*dth;
              mom[3*c+2]-=gW*dth; ene[c]-=gE*dth;
            }
            if (j < BS) {
              int c = (k*BS+j)*BS+i;
              rho[c]+=gD*dth; mom[3*c]+=gU*dth; mom[3*c+1]+=gV*dth;
              mom[3*c+2]+=gW*dth; ene[c]+=gE*dth;
            }
          }
#undef RH
#undef MX
#undef MY
#undef MZ
#undef EN
    }
  }
}
static void euler_z_sweep(Real dt, int level) {
#pragma omp parallel
  {
    Real br[LB_BUF3], bm[LB_BUF3], be[LB_BUF3];
#pragma omp for
    for (long long id = 0; id < sim.n; ++id) {
      if (sim.blk[id].level != level) continue;
      lb_load(br, 1, F_RHO, 1, id);
      lb_load(bm, 3, F_MOM, 1, id);
      lb_load(be, 1, F_ENE, 1, id);
      int ss = 1, nm = 2 * ss + BS;
      Real dth = dt / sim.blk[id].h;
#define RH(di,dj,dk) br[nm*nm*((dk)+ss)+nm*((dj)+ss)+(di)+ss]
#define MX(di,dj,dk) bm[3*(nm*nm*((dk)+ss)+nm*((dj)+ss)+(di)+ss)]
#define MY(di,dj,dk) bm[3*(nm*nm*((dk)+ss)+nm*((dj)+ss)+(di)+ss)+1]
#define MZ(di,dj,dk) bm[3*(nm*nm*((dk)+ss)+nm*((dj)+ss)+(di)+ss)+2]
#define EN(di,dj,dk) be[nm*nm*((dk)+ss)+nm*((dj)+ss)+(di)+ss]
      Real *rho = BLK(id)+BS*BS*BS*F_RHO;
      Real *mom = BLK(id)+BS*BS*BS*F_MOM;
      Real *ene = BLK(id)+BS*BS*BS*F_ENE;
      for (int k = 0; k <= BS; k++)
        for (int j = 0; j < BS; j++)
          for (int i = 0; i < BS; i++) {
            Real rL=RH(i,j,k-1),rR=RH(i,j,k);
            Real mxL=MX(i,j,k-1),mxR=MX(i,j,k);
            Real myL=MY(i,j,k-1),myR=MY(i,j,k);
            Real mzL=MZ(i,j,k-1),mzR=MZ(i,j,k);
            Real eL=EN(i,j,k-1),eR=EN(i,j,k);
            Real sr0=0,sx0=0,sy0=0,sz0=0,se0=0;
            Real sr1=0,sx1=0,sy1=0,sz1=0,se1=0;
            if (k >= 1) {
              sr0=minmod(rL-RH(i,j,k-2),rR-rL);
              sx0=minmod(mxL-MX(i,j,k-2),mxR-mxL);
              sy0=minmod(myL-MY(i,j,k-2),myR-myL);
              sz0=minmod(mzL-MZ(i,j,k-2),mzR-mzL);
              se0=minmod(eL-EN(i,j,k-2),eR-eL);
            }
            if (k < BS) {
              sr1=minmod(rR-rL,RH(i,j,k+1)-rR);
              sx1=minmod(mxR-mxL,MX(i,j,k+1)-mxR);
              sy1=minmod(myR-myL,MY(i,j,k+1)-myR);
              sz1=minmod(mzR-mzL,MZ(i,j,k+1)-mzR);
              se1=minmod(eR-eL,EN(i,j,k+1)-eR);
            }
            Real rl=rL+.5*sr0,rr=rR-.5*sr1;
            Real xl=mxL+.5*sx0,xr=mxR-.5*sx1;
            Real yl=myL+.5*sy0,yr=myR-.5*sy1;
            Real zl=mzL+.5*sz0,zr=mzR-.5*sz1;
            Real el=eL+.5*se0,er=eR-.5*se1;
            if(rl<=0){rl=rL;xl=mxL;yl=myL;zl=mzL;el=eL;}
            if(rr<=0){rr=rR;xr=mxR;yr=myR;zr=mzR;er=eR;}
            Real hD,hU,hV,hW,hE;
            hll_z(rl,xl,yl,zl,el,rr,xr,yr,zr,er,&hD,&hU,&hV,&hW,&hE);
            if (k > 0) {
              int c = ((k-1)*BS+j)*BS+i;
              rho[c]-=hD*dth; mom[3*c]-=hU*dth; mom[3*c+1]-=hV*dth;
              mom[3*c+2]-=hW*dth; ene[c]-=hE*dth;
            }
            if (k < BS) {
              int c = (k*BS+j)*BS+i;
              rho[c]+=hD*dth; mom[3*c]+=hU*dth; mom[3*c+1]+=hV*dth;
              mom[3*c+2]+=hW*dth; ene[c]+=hE*dth;
            }
          }
#undef RH
#undef MX
#undef MY
#undef MZ
#undef EN
    }
  }
}

/* ---- Refinement indicator ---- */
static void compute_indicator() {
#pragma omp parallel
  {
    Real ur[LB_BUF3], um[LB_BUF3], ue[LB_BUF3];
#pragma omp for nowait
    for (long long id = 0; id < sim.n; ++id) {
      lb_load(ur, 1, F_RHO, 1, id);
      lb_load(um, 3, F_MOM, 1, id);
      lb_load(ue, 1, F_ENE, 1, id);
      Real *TMP = BLK(id) + BS*BS*BS * F_TMP;
      int ss = 1, nm = 2*ss + BS;
      for (int k = 0; k < BS; k++)
        for (int j = 0; j < BS; j++)
          for (int i = 0; i < BS; i++) {
#define R(di,dj,dk)  ur[nm*nm*((k)+(dk)+ss)+nm*((j)+(dj)+ss)+(i)+(di)+ss]
#define MX(di,dj,dk) um[3*(nm*nm*((k)+(dk)+ss)+nm*((j)+(dj)+ss)+(i)+(di)+ss)]
#define MY(di,dj,dk) um[3*(nm*nm*((k)+(dk)+ss)+nm*((j)+(dj)+ss)+(i)+(di)+ss)+1]
#define MZ(di,dj,dk) um[3*(nm*nm*((k)+(dk)+ss)+nm*((j)+(dj)+ss)+(i)+(di)+ss)+2]
#define E(di,dj,dk)  ue[nm*nm*((k)+(dk)+ss)+nm*((j)+(dj)+ss)+(i)+(di)+ss]
            Real r0 = R(0,0,0), p0 = pres(r0,MX(0,0,0),MY(0,0,0),MZ(0,0,0),E(0,0,0));
            Real ux0=MX(0,0,0)/r0, uy0=MY(0,0,0)/r0, uz0=MZ(0,0,0)/r0;
            Real xi = 0;
            int di[]={-1,1,0,0,0,0}, dj[]={0,0,-1,1,0,0}, dk[]={0,0,0,0,-1,1};
            for (int nb = 0; nb < 6; nb++) {
              int ii=di[nb], jj=dj[nb], kk=dk[nb];
              Real rn=R(ii,jj,kk);
              Real pn=pres(rn,MX(ii,jj,kk),MY(ii,jj,kk),MZ(ii,jj,kk),E(ii,jj,kk));
              Real pmin=fmin(pn,p0)+1e-30;
              Real dp=fabs(pn-p0)/pmin;
              int dir = nb < 2 ? 0 : (nb < 4 ? 1 : 2);
              Real un0 = dir==0 ? ux0 : (dir==1 ? uy0 : uz0);
              Real unn_v = dir==0 ? MX(ii,jj,kk)/rn : (dir==1 ? MY(ii,jj,kk)/rn : MZ(ii,jj,kk)/rn);
              int sign = (nb % 2 == 0) ? -1 : 1;
              int compressing = sign*(unn_v - un0) < 0;
              if (dp > 0.2 && compressing) xi = 1;
              Real rmin=fmin(rn,r0)+1e-30;
              Real dr=fabs(rn-r0)/rmin;
              if (dp <= 0.2 && dr > 0.2) xi = 1;
              Real amax=fmax(fabs(pn),fabs(p0))+1e-30;
              Real grad=(fabs(pn)-fabs(p0))/amax;
              xi = fmax(xi, grad);
            }
            TMP[(k*BS+j)*BS+i] = fmax(0, fmin(1, xi));
#undef R
#undef MX
#undef MY
#undef MZ
#undef E
          }
    }
  }
  /* Save raw indicator, then smooth */
#pragma omp parallel for
  for (long long id = 0; id < sim.n; ++id)
    memcpy(BLK(id)+BS*BS*BS*F_DENE, BLK(id)+BS*BS*BS*F_TMP, BS*BS*BS*sizeof(Real));
  for (int iter = 0; iter < 3; iter++) {
#pragma omp parallel for
    for (long long id = 0; id < sim.n; ++id) {
      Real *TMP = BLK(id)+BS*BS*BS*F_TMP;
      Real *RAW = BLK(id)+BS*BS*BS*F_DENE;
      Real *D = BLK(id)+BS*BS*BS*F_DRHO;
      for (int k = 0; k < BS; k++)
        for (int j = 0; j < BS; j++)
          for (int i = 0; i < BS; i++) {
            int c = (k*BS+j)*BS+i;
            Real xi = TMP[c];
            Real xim = (i>0) ? TMP[c-1] : xi;
            Real xip = (i<BS-1) ? TMP[c+1] : xi;
            Real xjm = (j>0) ? TMP[c-BS] : xi;
            Real xjp = (j<BS-1) ? TMP[c+BS] : xi;
            Real xkm = (k>0) ? TMP[c-BS*BS] : xi;
            Real xkp = (k<BS-1) ? TMP[c+BS*BS] : xi;
            Real lap = xim+xip+xjm+xjp+xkm+xkp - 6*xi;
            Real Q = (xi > sim.Rtol) ? 1 : 0;
            D[c] = xi + (1.0/6.0)*(lap + Q);
            D[c] = fmax(D[c], RAW[c]);
            D[c] = fmax(0, fmin(1, D[c]));
          }
    }
#pragma omp parallel for
    for (long long id = 0; id < sim.n; ++id)
      memcpy(BLK(id)+BS*BS*BS*F_TMP, BLK(id)+BS*BS*BS*F_DRHO, BS*BS*BS*sizeof(Real));
  }
}

/* ---- Output ---- */
static void dump(Real time, int step, char *path) {
  char xyz_path[FILENAME_MAX], attr_path[FILENAME_MAX], xdmf_path[FILENAME_MAX];
  snprintf(xyz_path, sizeof xyz_path, "%s.xyz.raw", path);
  snprintf(xdmf_path, sizeof xdmf_path, "%s.xdmf2", path);
  char *xyz_base = xyz_path;
  for (int j = 0; xyz_path[j]; j++)
    if (xyz_path[j] == '/' && xyz_path[j+1]) xyz_base = &xyz_path[j+1];
  FILE *xdmf = fopen(xdmf_path, "w");
  fprintf(xdmf,
    "<Xdmf Version=\"2.0\">\n<Domain><Grid>\n"
    "  <Time Value=\"%.16e\"/>\n"
    "  <Information Name=\"Step\" Value=\"%d\"/>\n"
    "  <Topology Dimensions=\"%lld\" TopologyType=\"Hexahedron\"/>\n"
    "  <Geometry GeometryType=\"XYZ\">\n"
    "    <DataItem Dimensions=\"%lld 3\" Format=\"Binary\">%s</DataItem>\n"
    "  </Geometry>\n",
    time, step, BS*BS*BS*sim.n, 8LL*BS*BS*BS*sim.n, xyz_base);
  for (size_t i = 0; i < NVARS; i++)
    if (fld_t[i].prefix) {
      int dim = fld_t[i].dim;
      snprintf(attr_path, sizeof attr_path, "%s.%s.raw", path, fld_t[i].prefix);
      fprintf(xdmf,
        "  <Attribute AttributeType=\"%s\" Name=\"%s\" Center=\"Cell\">\n"
        "    <DataItem Dimensions=\"%lld %d\" Precision=\"%ld\" Format=\"Binary\">%s</DataItem>\n"
        "  </Attribute>\n",
        dim > 1 ? "Vector" : "Scalar", fld_t[i].prefix,
        BS*BS*BS*sim.n, dim, sizeof(Real),
        attr_path + (xyz_path - xyz_base));
    }
  fprintf(xdmf, "</Grid></Domain></Xdmf>\n");
  fclose(xdmf);

  FILE *file = fopen(xyz_path, "wb");
  float xyz[24 * BS * BS * BS]; /* 8 verts * 3 coords per cell */
  for (long long bi = 0; bi < sim.n; bi++) {
    struct Blk *info = &sim.blk[bi];
    int c = 0;
    Real h = info->h;
    for (int iz = 0; iz < BS; iz++)
      for (int iy = 0; iy < BS; iy++)
        for (int ix = 0; ix < BS; ix++) {
          Real x0=info->origin[0]+h*ix, x1=x0+h;
          Real y0=info->origin[1]+h*iy, y1=y0+h;
          Real z0=info->origin[2]+h*iz, z1=z0+h;
          /* VTK hexahedron order */
          float v[8][3] = {
            {x0,y0,z0},{x1,y0,z0},{x1,y1,z0},{x0,y1,z0},
            {x0,y0,z1},{x1,y0,z1},{x1,y1,z1},{x0,y1,z1}};
          memcpy(xyz + c, v, sizeof v);
          c += 24;
        }
    fwrite(xyz, sizeof xyz, 1, file);
  }
  fclose(file);

  for (size_t i = 0; i < NVARS; i++)
    if (fld_t[i].prefix) {
      int dim = fld_t[i].dim, off = fld_t[i].offset;
      snprintf(attr_path, sizeof attr_path, "%s.%s.raw", path, fld_t[i].prefix);
      file = fopen(attr_path, "wb");
      for (long long j = 0; j < sim.n; j++)
        fwrite(BLK(j) + off * BS*BS*BS, sizeof(Real), dim * BS*BS*BS, file);
      fclose(file);
    }
}

/* ---- AMR ---- */
static const int ad_sib_ic_3d[8] = {-1, 14, 16, 17, 22, 23, 25, 26};
static int ad_run(void) {
  compute_indicator();
  enum AdSt *state = calloc(sim.n, sizeof *state);
  long long *ref_idx = malloc(sim.n * sizeof *ref_idx);
  long long *com_idx = malloc(sim.n * sizeof *com_idx);
  long long n_ref = 0, n_com = 0;
  int Changed = 0;

#pragma omp parallel for reduction(|| : Changed)
  for (long long i = 0; i < sim.n; i++) {
    Real *b = BLK(i) + BS*BS*BS * F_TMP;
    double Linf = 0;
    for (int j = 0; j < BS*BS*BS; j++) Linf = fmax(Linf, fabs(b[j]));
    int lev = sim.blk[i].level;
    state[i] = Linf > sim.Rtol && lev < sim.levelMax - 1 ? Refine
             : Linf < sim.Ctol && lev > sim.levelStart    ? Compress
             : Leave;
    Changed |= state[i] != Leave;
  }
  if (!Changed) goto done;

  /* 2:1 balance propagation */
  for (int More = 1; More;) {
    More = 0;
    for (long long j = 0; j < sim.n; j++) {
      if (state[j] != Refine) continue;
      struct Blk *bj = &sim.blk[j];
      for (int ic = 0; ic < 27; ic++) {
        if (ic == 13) continue;
        struct Nb nr = nb_find(bj->level, bj->ix, bj->iy, bj->iz, ic);
        if (nr.s >= 3 || nr.idx < 0) continue;
        if (nr.s == 2 && state[nr.idx] != Refine)
          { state[nr.idx] = Refine; More = 1; }
        else if (nr.s == 0 && state[nr.idx] == Compress)
          state[nr.idx] = Leave;
      }
    }
  }

  /* Compression: check 8 siblings */
  for (long long j = 0; j < sim.n; j++) {
    if (state[j] != Compress) continue;
    struct Blk *bj = &sim.blk[j];
    if ((bj->ix | bj->iy | bj->iz) & 1) continue;
    long long sib[8] = {j};
    int ok = 1;
    for (int s = 1; s < 8 && ok; s++) {
      struct Nb nr = nb_find(bj->level, bj->ix, bj->iy, bj->iz, ad_sib_ic_3d[s]);
      ok = nr.s == 0 && nr.idx >= 0 && state[nr.idx] == Compress;
      sib[s] = nr.idx;
    }
    for (int s = 0; s < 8 && ok; s++) {
      struct Blk *bs = &sim.blk[sib[s]];
      for (int ic = 0; ic < 27 && ok; ic++)
        if (ic != 13) ok = nb_find(bs->level, bs->ix, bs->iy, bs->iz, ic).s != 1;
    }
    if (!ok) state[j] = Leave;
  }

  for (long long j = 0; j < sim.n; j++)
    if (state[j] == Refine)
      ref_idx[n_ref++] = j;
    else if (state[j] == Compress && !((sim.blk[j].ix|sim.blk[j].iy|sim.blk[j].iz) & 1))
      com_idx[n_com++] = j;
  fprintf(stderr, "%s:%d: com/ref: %lld %lld (n=%lld)\n", __FILE__, __LINE__, n_com, n_ref, sim.n);
  if (n_ref == 0 && n_com == 0) goto done;

  /* Refinement: 1 block -> 8 children */
  long long nprev = sim.n;
  sim.n += 8 * n_ref;
  sim.blk = realloc(sim.blk, sim.n * sizeof *sim.blk);
  sim.fld = realloc(sim.fld, sim.n * BLK_S * sizeof(Real));
  memset(BLK(nprev), 0, 8 * n_ref * BLK_S * sizeof(Real));
  state = realloc(state, sim.n * sizeof *state);
  for (long long i = nprev; i < sim.n; i++) state[i] = Leave;

#pragma omp parallel for
  for (long long r = 0; r < n_ref; r++) {
    struct Blk *par = &sim.blk[ref_idx[r]];
    int px=par->ix, py=par->iy, pz=par->iz;
    for (int K = 0; K < 2; K++)
      for (int J = 0; J < 2; J++)
        for (int I = 0; I < 2; I++) {
          long long ci = nprev + 8*r + 4*K + 2*J + I;
          bl_fill(&sim.blk[ci], par->level+1, 2*px+I, 2*py+J, 2*pz+K);
          /* Piecewise constant interpolation (simple but conservative) */
          for (size_t v = 0; v < NVARS; v++) {
            int dim = fld_t[v].dim, off = fld_t[v].offset;
            Real *src = BLK(ref_idx[r]) + off * BS*BS*BS;
            Real *dst = BLK(ci) + off * BS*BS*BS;
            for (int kk = 0; kk < BS; kk++)
              for (int jj = 0; jj < BS; jj++)
                for (int ii = 0; ii < BS; ii++) {
                  int si = I*(BS/2)+ii/2, sj = J*(BS/2)+jj/2, sk = K*(BS/2)+kk/2;
                  for (int d = 0; d < dim; d++)
                    dst[dim*((kk*BS+jj)*BS+ii)+d] = src[dim*((sk*BS+sj)*BS+si)+d];
                }
          }
        }
    state[ref_idx[r]] = Dealloc;
  }

  /* Compression: 8 siblings -> 1 parent */
#pragma omp parallel for
  for (long long r = 0; r < n_com; r++) {
    long long ci = com_idx[r];
    struct Blk *p0 = &sim.blk[ci];
    int level=p0->level, x=p0->ix, y=p0->iy, z=p0->iz;
    long long sib_idx[8] = {ci};
    for (int s = 1; s < 8; s++) {
      sib_idx[s] = nb_find(level, x, y, z, ad_sib_ic_3d[s]).idx;
      state[sib_idx[s]] = Dealloc;
    }
    for (size_t v = 0; v < NVARS; v++) {
      int dim = fld_t[v].dim, off = fld_t[v].offset;
      Real *dst = BLK(ci) + off * BS*BS*BS;
      for (int K = 0; K < 2; K++)
        for (int J = 0; J < 2; J++)
          for (int I = 0; I < 2; I++) {
            Real *src = BLK(sib_idx[4*K+2*J+I]) + off * BS*BS*BS;
            for (int kk = 0; kk < BS; kk += 2)
              for (int jj = 0; jj < BS; jj += 2)
                for (int ii = 0; ii < BS; ii += 2) {
                  int oc = (kk/2+K*(BS/2))*BS*BS + (jj/2+J*(BS/2))*BS + ii/2+I*(BS/2);
                  for (int d = 0; d < dim; d++) {
                    Real s = 0;
                    for (int dk=0;dk<2;dk++) for (int dj=0;dj<2;dj++) for (int di=0;di<2;di++)
                      s += src[dim*(((kk+dk)*BS+(jj+dj))*BS+ii+di)+d];
                    dst[dim*oc+d] = s / 8;
                  }
                }
          }
    }
    bl_fill(p0, level-1, x/2, y/2, z/2);
  }

  fprintf(stderr, "  refine done, n=%lld\n", sim.n);
  fflush(stderr);
  /* Compact */
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
  fprintf(stderr, "  compact: %lld blocks\n", sim.n); fflush(stderr);
  hm_rebuild();
  fprintf(stderr, "  hm_rebuild done\n"); fflush(stderr);

done:
  free(state); free(ref_idx); free(com_idx);
  return Changed;
}

/* ---- Time stepping ---- */
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

static void subcycle(int level, int lmax, Real dt, int order) {
  if (level < lmax)
    subcycle(level + 1, lmax, dt / 2, order);
  if (order) {
    euler_z_sweep(dt, level);
    euler_y_sweep(dt, level);
    euler_x_sweep(dt, level);
  } else {
    euler_x_sweep(dt, level);
    euler_y_sweep(dt, level);
    euler_z_sweep(dt, level);
  }
  if (level < lmax)
    subcycle(level + 1, lmax, dt / 2, order ^ 1);
}

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
  /* Domain: 1 x 1/4 x 1/4 (Khokhlov Section 7.5) */
  sim.L[0] = 1.0; sim.L[1] = 0.25; sim.L[2] = 0.25;
  /* BCs: x-: inflow, x+: outflow, y-: wall, y+: symmetry, z-: wall, z+: symmetry */
  sim.bc[0]=BC_INFLOW; sim.bc[1]=BC_OUTFLOW;
  sim.bc[2]=BC_WALL; sim.bc[3]=BC_SYMMETRY;
  sim.bc[4]=BC_WALL; sim.bc[5]=BC_SYMMETRY;
  /* Blocks per direction at levelStart */
  {
    Real h0 = sim.L[0] / (BS * (1 << sim.levelStart)); /* cell size */
    /* Use Lx to determine h, then compute block counts */
    sim.nb[0] = (int)(sim.L[0] / (h0 * BS) + 0.5);
    sim.nb[1] = (int)(sim.L[1] / (h0 * BS) + 0.5);
    sim.nb[2] = (int)(sim.L[2] / (h0 * BS) + 0.5);
    if (sim.nb[1] < 1) sim.nb[1] = 1;
    if (sim.nb[2] < 1) sim.nb[2] = 1;
    fprintf(stderr, "main.c: domain [%.2f x %.2f x %.2f] blocks %d x %d x %d\n",
            sim.L[0], sim.L[1], sim.L[2], sim.nb[0], sim.nb[1], sim.nb[2]);
    sim.n = (long long)sim.nb[0] * sim.nb[1] * sim.nb[2];
    sim.blk = calloc(sim.n, sizeof *sim.blk);
    sim.fld = calloc(sim.n * BLK_S, sizeof(Real));
    long long idx = 0;
    for (int iz = 0; iz < sim.nb[2]; iz++)
      for (int iy = 0; iy < sim.nb[1]; iy++)
        for (int ix = 0; ix < sim.nb[0]; ix++)
          bl_fill(&sim.blk[idx++], sim.levelStart, ix, iy, iz);
  }
  hm_rebuild();
  lb_init();
  /* IC: Shock-bubble interaction (Khokhlov Section 7.5)
     M=1.25 shock from left, bubble at (0.25, 0.25, 0.25) R=0.125
     Pre-shock: rho=1, P=1, u=0
     Post-shock (Rankine-Hugoniot M=1.25, gamma=1.4):
       rho2 = 1.4286, P2 = 1.6563, u2 = 0.4437
     Bubble: rho_b=0.166, P=1 (pressure equilibrium) */
  {
    Real M = 1.25, g = GAMMA;
    Real rho1=1, P1=1;
    Real P2 = P1*(1 + 2*g/(g+1)*(M*M-1));
    Real rho2 = rho1*((g+1)*M*M)/((g-1)*M*M+2);
    Real Us = M*sqrt(g*P1/rho1);
    Real u2 = (1-rho1/rho2)*Us;
    Real E2 = P2/(g-1) + 0.5*rho2*u2*u2;
    Real E1 = P1/(g-1);
    Real x_shock = 0.05; /* initial shock position */
    Real xb=0.25, yb=0.25, zb=0.25, Rb=0.125;
    Real rho_b=0.166, E_b=P1/(g-1); /* bubble: low density, same pressure */
    /* Save inflow state for BC */
    sim.inflow[0]=rho2; sim.inflow[1]=rho2*u2; sim.inflow[2]=0;
    sim.inflow[3]=0; sim.inflow[4]=E2;
    fprintf(stderr, "main.c: shock M=%.2f rho2=%.4f P2=%.4f u2=%.4f\n", M,rho2,P2,u2);
    fprintf(stderr, "main.c: bubble at (%.2f,%.2f,%.2f) R=%.3f rho_b=%.3f\n", xb,yb,zb,Rb,rho_b);
#pragma omp parallel for
    for (long long i = 0; i < sim.n; i++) {
      struct Blk *info = &sim.blk[i];
      Real *rho = BLK(i)+BS*BS*BS*F_RHO;
      Real *mom = BLK(i)+BS*BS*BS*F_MOM;
      Real *ene = BLK(i)+BS*BS*BS*F_ENE;
      Real h = info->h;
      for (int iz = 0; iz < BS; iz++)
        for (int iy = 0; iy < BS; iy++)
          for (int ix = 0; ix < BS; ix++) {
            Real x=info->origin[0]+h*(ix+0.5);
            Real y=info->origin[1]+h*(iy+0.5);
            Real z=info->origin[2]+h*(iz+0.5);
            int j = (iz*BS+iy)*BS+ix;
            Real dr = sqrt((x-xb)*(x-xb)+(y-yb)*(y-yb)+(z-zb)*(z-zb));
            if (dr < Rb) {
              /* inside bubble */
              rho[j] = rho_b;
              mom[3*j]=0; mom[3*j+1]=0; mom[3*j+2]=0;
              ene[j] = E_b;
            } else if (x < x_shock) {
              /* post-shock */
              rho[j] = rho2;
              mom[3*j]=rho2*u2; mom[3*j+1]=0; mom[3*j+2]=0;
              ene[j] = E2;
            } else {
              /* pre-shock */
              rho[j] = rho1;
              mom[3*j]=0; mom[3*j+1]=0; mom[3*j+2]=0;
              ene[j] = E1;
            }
          }
    }
  }
  for (int i = 0; i < sim.levelMax; i++) ad_run();
  /* Main loop */
  while (1) {
    if (sim.step % 10 == 0)
      fprintf(stderr, "main.c: %08d %.6e dt=%.3e blk=%lld\n",
              sim.step, sim.time, sim.dt, sim.n);
    {
      int do_dump = 0;
      if (sim.dumpTime > 0 && sim.time >= sim.nextDumpTime) {
        sim.nextDumpTime += sim.dumpTime; do_dump = 1;
      }
      if (dumpSteps > 0 && sim.step % dumpSteps == 0) do_dump = 1;
      if (do_dump) {
#pragma omp parallel for
        for (long long i = 0; i < sim.n; i++) {
          Real *r=BLK(i)+BS*BS*BS*F_RHO, *m=BLK(i)+BS*BS*BS*F_MOM;
          Real *e=BLK(i)+BS*BS*BS*F_ENE, *t=BLK(i)+BS*BS*BS*F_TMP;
          for (int j = 0; j < BS*BS*BS; j++)
            t[j] = (GAMMA-1)*(e[j]-0.5*(m[3*j]*m[3*j]+m[3*j+1]*m[3*j+1]+m[3*j+2]*m[3*j+2])/r[j]);
        }
        char path[FILENAME_MAX];
        snprintf(path, sizeof path, "vel.%08d", sim.dump_count++);
        dump(sim.time, sim.step, path);
      }
    }
    if (sim.endTime > 0 && sim.time >= sim.endTime) break;
    int lmin = sim.levelMax, lmax = 0;
    for (long long i = 0; i < sim.n; i++) {
      int l = sim.blk[i].level;
      if (l < lmin) lmin = l;
      if (l > lmax) lmax = l;
    }
    Real smax = 0;
#pragma omp parallel for reduction(max:smax)
    for (long long i = 0; i < sim.n; i++) {
      Real *rho=BLK(i)+BS*BS*BS*F_RHO, *mom=BLK(i)+BS*BS*BS*F_MOM;
      Real *ene=BLK(i)+BS*BS*BS*F_ENE;
      int l = sim.blk[i].level;
      Real ih = 1.0/sim.blk[i].h;
      Real scale = 1.0 / (1 << (l - lmin));
      for (int j = 0; j < BS*BS*BS; j++) {
        Real r=rho[j], u=mom[3*j]/r, v=mom[3*j+1]/r, w=mom[3*j+2]/r;
        Real p=fmax(0,(GAMMA-1)*(ene[j]-0.5*r*(u*u+v*v+w*w)));
        Real a=sqrt(GAMMA*p/r);
        smax = fmax(smax, fmax(fabs(u)+a, fmax(fabs(v)+a, fabs(w)+a))*ih*scale);
      }
    }
    sim.dt = sim.CFL / (smax + 1e-30);
    if (sim.step > 0 && sim.step % sim.AdaptSteps == 0) ad_run();
    subcycle(lmin, lmax, sim.dt, sim.step & 1);
    sim.time += sim.dt;
    sim.step++;
  }
  fprintf(stderr, "main.c: end\n");
}
