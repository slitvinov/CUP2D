/* mesh2iso.c — extract AMR isolines and render to PNG (no image buffer).
   Uses marching squares on dual AMR grid (from iso2d.c), streams PNG scanlines.
   Usage: ./mesh2iso -g vel.xyz.raw -s vel.vort.raw -o iso.png
                     -n 1024 -w 1 -a -36 -b 36 -d 6 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

enum { BS = 8, ZBUF = 64 * 1024 };

/* ---- arg parsing (plan9 style, no defaults) ---- */
static const char *arg_find(int argc, char **argv, const char *key) {
  for (int i = 1; i < argc; i++)
    if (argv[i][0] == '-' && strcmp(argv[i] + 1, key) == 0) {
      if (i + 1 < argc)
        return argv[i + 1];
      fprintf(stderr, "mesh2iso: error: option -%s has no value\n", key);
      exit(1);
    }
  fprintf(stderr, "mesh2iso: error: option -%s is not set\n", key);
  exit(1);
}
static int arg_i(int argc, char **argv, const char *key) {
  const char *s = arg_find(argc, argv, key);
  char *end;
  long v = strtol(s, &end, 10);
  if (end == s || *end != '\0') {
    fprintf(stderr, "mesh2iso: error: -%s: bad integer '%s'\n", key, s);
    exit(1);
  }
  return (int)v;
}

/* ---- marching squares tables (from iso2d.c) ---- */
static const int8_t msq_cases[16][5] = {
    {-1,-1,-1,-1,-1}, {0,3,-1,-1,-1}, {0,1,-1,-1,-1},
    {1,3,-1,-1,-1},   {1,2,-1,-1,-1}, {0,3,1,2,-1},
    {0,2,-1,-1,-1},   {2,3,-1,-1,-1}, {2,3,-1,-1,-1},
    {0,2,-1,-1,-1},   {0,1,2,3,-1},   {1,2,-1,-1,-1},
    {1,3,-1,-1,-1},   {0,1,-1,-1,-1}, {0,3,-1,-1,-1},
    {-1,-1,-1,-1,-1},
};
static const int8_t msq_edges[4][2] = {{0,1},{1,2},{3,2},{0,3}};

/* ---- cell / segment types ---- */
struct vec2i { int x, y; };
struct vec2f { float x, y; };
struct Cell {
  struct vec2i lower;
  int level;
  uint64_t morton;
  float scalar;
};
struct Seg { float x0, y0, x1, y1; };

static struct vec2i vec2i_shr(struct vec2i v, int s) {
  struct vec2i u = {v.x >> s, v.y >> s};
  return u;
}
static int vec2i_eq(struct vec2i a, struct vec2i b) {
  return a.x == b.x && a.y == b.y;
}
static int vec2i_lt(struct vec2i a, struct vec2i b) {
  return (a.x < b.x) || (a.x == b.x && a.y < b.y);
}

static long leftShift2(long x) {
  x = (x | x << 16) & 0x0000FFFF0000FFFFull;
  x = (x | x << 8)  & 0x00FF00FF00FF00FFull;
  x = (x | x << 4)  & 0x0F0F0F0F0F0F0F0Full;
  x = (x | x << 2)  & 0x3333333333333333ull;
  x = (x | x << 1)  & 0x5555555555555555ull;
  return x;
}
static uint64_t morton_code(int x, int y) {
  return (leftShift2((uint32_t)y) << 1) | leftShift2((uint32_t)x);
}

static int comp_cell(const void *a, const void *b) {
  uint64_t am = ((const struct Cell *)a)->morton;
  uint64_t bm = ((const struct Cell *)b)->morton;
  return (am > bm) - (am < bm);
}

/* ---- neighbor lookup (binary search on Morton-sorted cells) ---- */
static int findActual(struct Cell *cells, long long ncell,
                      struct Cell *result, struct vec2i lower, int level) {
  uint64_t target = morton_code(lower.x, lower.y);
  long long lo = 0, hi = ncell;
  while (lo < hi) {
    long long mid = lo + (hi - lo) / 2;
    if (cells[mid].morton < target)
      lo = mid + 1;
    else
      hi = mid;
  }
  if (lo == ncell) return 0;
  *result = cells[lo];
  int f = level > result->level ? level : result->level;
  if (vec2i_eq(vec2i_shr(result->lower, f), vec2i_shr(lower, f)))
    return 1;
  if (lo > 0) {
    *result = cells[lo - 1];
    f = level > result->level ? level : result->level;
    if (vec2i_eq(vec2i_shr(result->lower, f), vec2i_shr(lower, f)))
      return 1;
  }
  return 0;
}

/* ---- isoline extraction (marching squares on dual AMR grid) ---- */
static void extract(struct Cell *cells, long long ncell, float iso,
                    struct Seg **segs, long long *nseg, long long *cap) {
  for (long long wid = 0; wid < ncell; wid++) {
    for (int did = 0; did < 4; did++) {
      struct Cell cell = cells[wid];
      int dy = (did & 2) ? 1 : -1;
      int dx = (did & 1) ? 1 : -1;
      struct Cell corner[2][2];
      int skip = 0;
      for (int iy = 0; iy < 2 && !skip; iy++)
        for (int ix = 0; ix < 2 && !skip; ix++) {
          struct vec2i lower;
          lower.x = cell.lower.x + dx * ix * (1 << cell.level);
          lower.y = cell.lower.y + dy * iy * (1 << cell.level);
          if (!findActual(cells, ncell, &corner[iy][ix], lower, cell.level)) {
            skip = 1; break;
          }
          if (corner[iy][ix].level < cell.level) {
            skip = 1; break;
          }
          if (corner[iy][ix].level == cell.level &&
              vec2i_lt(corner[iy][ix].lower, cell.lower)) {
            skip = 1; break;
          }
        }
      if (skip) continue;
      int x = dx == -1, y = dy == -1;
      struct { float x, y, s; } v[4];
      struct Cell *cs[4] = {&corner[0+y][0+x], &corner[0+y][1-x],
                            &corner[1-y][1-x], &corner[1-y][0+x]};
      for (int i = 0; i < 4; i++) {
        v[i].x = cs[i]->lower.x + 0.5f * (1 << cs[i]->level);
        v[i].y = cs[i]->lower.y + 0.5f * (1 << cs[i]->level);
        v[i].s = cs[i]->scalar;
      }
      int index = 0;
      for (int i = 0; i < 4; i++)
        if (v[i].s > iso) index += (1 << i);
      if (index == 0 || index == 0xf) continue;
      for (const int8_t *e = &msq_cases[index][0]; e[0] > -1; e += 2) {
        float px[2], py[2];
        for (int ii = 0; ii < 2; ii++) {
          int i0 = msq_edges[e[ii]][0], i1 = msq_edges[e[ii]][1];
          float t = (iso - v[i0].s) / (v[i1].s - v[i0].s);
          px[ii] = (1-t)*v[i0].x + t*v[i1].x;
          py[ii] = (1-t)*v[i0].y + t*v[i1].y;
        }
        if (px[0]==px[1] && py[0]==py[1]) continue;
        if (*nseg >= *cap) {
          *cap = *cap ? *cap * 2 : 4096;
          *segs = realloc(*segs, *cap * sizeof(struct Seg));
        }
        struct Seg *s = &(*segs)[(*nseg)++];
        s->x0 = px[0]; s->y0 = py[0];
        s->x1 = px[1]; s->y1 = py[1];
      }
    }
  }
}

/* ---- PNG writing (from mesh2png.c) ---- */
static void put32(unsigned char *p, unsigned v) {
  p[0]=v>>24; p[1]=v>>16; p[2]=v>>8; p[3]=v;
}
static void png_chunk(FILE *f, const char *type, const unsigned char *d, unsigned n) {
  unsigned char h[8]; put32(h,n); memcpy(h+4,type,4);
  fwrite(h,1,8,f);
  unsigned long c = crc32(0,(const unsigned char*)type,4);
  if (n) { fwrite(d,1,n,f); c=crc32(c,d,n); }
  unsigned char t[4]; put32(t,c); fwrite(t,1,4,f);
}

static long fsize(FILE *f) {
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  return n;
}

int main(int argc, char **argv) {
  if (argc < 11) {
    fprintf(stderr, "usage: %s -g coords.xyz.raw -s scalar.raw -o out.png"
                    " -n size -w linewidth\n"
                    "reads iso levels from stdin (whitespace-separated)\n", argv[0]);
    return 1;
  }
  const char *geo_path = arg_find(argc, argv, "g");
  const char *sc_path  = arg_find(argc, argv, "s");
  const char *out_path = arg_find(argc, argv, "o");
  int N   = arg_i(argc, argv, "n");
  int lw  = arg_i(argc, argv, "w");

  /* read iso levels from stdin */
  float *isovals = NULL;
  int niso = 0, isocap = 0;
  { float v; char *end; char buf[64];
    while (scanf("%63s", buf) == 1) {
      v = strtof(buf, &end);
      if (end == buf || *end != '\0') {
        fprintf(stderr, "mesh2iso: error: bad iso level '%s'\n", buf);
        return 1;
      }
      if (niso >= isocap) {
        isocap = isocap ? isocap * 2 : 64;
        isovals = realloc(isovals, isocap * sizeof *isovals);
      }
      isovals[niso++] = v;
    }
  }
  if (niso == 0) {
    fprintf(stderr, "mesh2iso: error: no iso levels on stdin\n");
    return 1;
  }

  /* read geometry: (ix, iy, level) per block, [0,1]^2 domain, BS=8 */
  FILE *fp = fopen(geo_path, "rb");
  if (!fp) { perror(geo_path); return 1; }
  long geo_sz = fsize(fp);
  int nblk = geo_sz / (3 * sizeof(int32_t));
  long long ncell = (long long)nblk * BS * BS;
  int32_t *binfo = malloc(3 * nblk * sizeof(int32_t));
  fread(binfo, sizeof(int32_t), 3 * nblk, fp);
  fclose(fp);

  /* find finest level */
  int lmax = 0;
  for (int b = 0; b < nblk; b++)
    if (binfo[3*b+2] > lmax) lmax = binfo[3*b+2];

  /* build cells from blocks */
  struct Cell *cells = malloc(ncell * sizeof *cells);
  long long ci = 0;
  for (int b = 0; b < nblk; b++) {
    int bix = binfo[3*b], biy = binfo[3*b+1], lev = binfo[3*b+2];
    int ratio = 1 << (lmax - lev);
    int bx = bix * BS * ratio, by = biy * BS * ratio;
    for (int j = 0; j < BS; j++)
      for (int i = 0; i < BS; i++) {
        int cx = bx + i * ratio, cy = by + j * ratio;
        cells[ci].lower.x = cx;
        cells[ci].lower.y = cy;
        cells[ci].level = lmax - lev;
        cells[ci].morton = morton_code(cx, cy);
        ci++;
      }
  }
  free(binfo);

  /* read scalar field (float64 or float32) */
  fp = fopen(sc_path, "rb");
  if (!fp) { perror(sc_path); return 1; }
  long sc_sz = fsize(fp);
  if (sc_sz == (long)(ncell * sizeof(double))) {
    double *buf = malloc(ncell * sizeof(double));
    fread(buf, sizeof(double), ncell, fp);
    for (long long i = 0; i < ncell; i++) cells[i].scalar = (float)buf[i];
    free(buf);
  } else if (sc_sz == (long)(ncell * sizeof(float))) {
    float *buf = malloc(ncell * sizeof(float));
    fread(buf, sizeof(float), ncell, fp);
    for (long long i = 0; i < ncell; i++) cells[i].scalar = buf[i];
    free(buf);
  } else {
    fprintf(stderr, "mesh2iso: error: scalar size %ld does not match %lld cells\n", sc_sz, ncell);
    return 1;
  }
  fclose(fp);

  /* sort by Morton code */
  qsort(cells, ncell, sizeof *cells, comp_cell);

  /* extract isolines for all levels */
  struct Seg *segs = NULL;
  long long nseg = 0, cap = 0;
  for (int il = 0; il < niso; il++)
    extract(cells, ncell, isovals[il], &segs, &nseg, &cap);
  free(cells);
  free(isovals);

  /* convert segment coords from integer grid to world [0,1] */
  { float hmin = 1.0f / (BS << lmax);
    for (long long i = 0; i < nseg; i++) {
      segs[i].x0 *= hmin;
      segs[i].y0 *= hmin;
      segs[i].x1 *= hmin;
      segs[i].y1 *= hmin;
    }
  }

  /* rasterize segments with Bresenham into framebuffer */
  unsigned char *fb = calloc(N * N, 1);
  float sc = (float)N;
  for (long long i = 0; i < nseg; i++) {
    int x0 = (int)(segs[i].x0 * sc);
    int y0 = (int)(segs[i].y0 * sc);
    int x1 = (int)(segs[i].x1 * sc);
    int y1 = (int)(segs[i].y1 * sc);
    /* flip y for image coordinates */
    y0 = N - 1 - y0; y1 = N - 1 - y1;
    int dx = abs(x1-x0), dy = abs(y1-y0);
    int sx = x0<x1 ? 1 : -1, sy = y0<y1 ? 1 : -1;
    int err = dx - dy;
    for (;;) {
      /* draw pixel with line width */
      for (int dj = -(lw/2); dj <= (lw-1)/2; dj++)
        for (int di = -(lw/2); di <= (lw-1)/2; di++) {
          int px = x0+di, py = y0+dj;
          if (px>=0 && px<N && py>=0 && py<N) fb[py*N+px] = 1;
        }
      if (x0==x1 && y0==y1) break;
      int e2 = 2*err;
      if (e2 > -dy) { err -= dy; x0 += sx; }
      if (e2 <  dx) { err += dx; y0 += sy; }
    }
  }
  free(segs);

  /* write PNG from framebuffer */
  FILE *out = fopen(out_path, "wb");
  unsigned char sig[] = {137,80,78,71,13,10,26,10};
  fwrite(sig,1,8,out);
  unsigned char ihdr[13]; put32(ihdr,N); put32(ihdr+4,N);
  ihdr[8]=8; ihdr[9]=0; ihdr[10]=0; ihdr[11]=0; ihdr[12]=0;
  png_chunk(out, "IHDR", ihdr, 13);

  z_stream z = {0};
  deflateInit(&z, Z_DEFAULT_COMPRESSION);
  unsigned char *row = malloc(N+1);
  unsigned char zbuf[ZBUF];
  for (int y = 0; y < N; y++) {
    row[0] = 0; /* filter: none */
    for (int x = 0; x < N; x++)
      row[1+x] = fb[y*N+x] ? 0 : 255;
    z.next_in = row; z.avail_in = N+1;
    int flush = y==N-1 ? Z_FINISH : Z_NO_FLUSH;
    do {
      z.next_out = zbuf; z.avail_out = ZBUF;
      deflate(&z, flush);
      unsigned have = ZBUF - z.avail_out;
      if (have) png_chunk(out, "IDAT", zbuf, have);
    } while (z.avail_out==0);
  }
  deflateEnd(&z);
  free(fb);
  png_chunk(out, "IEND", NULL, 0);
  fclose(out); free(row);
  return 0;
}
