/* mesh2png.c — render AMR block boundaries to PNG.
   Reads xyz.raw as (ix, iy, level) per block, assumes [0,1]^2 domain, BS=8.
   Usage: ./mesh2png -i file.xyz.raw -o mesh.png -s 1024 -w 2 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

enum { BS = 8, ZBUF = 64 * 1024 };
struct Blk {
  float ox, oy, bw;
};

static const char *arg_find(int argc, char **argv, const char *key) {
  int i;
  for (i = 1; i < argc; i++)
    if (argv[i][0] == '-' && strcmp(argv[i] + 1, key) == 0) {
      if (i + 1 < argc) return argv[i + 1];
      fprintf(stderr, "mesh2png: -%s: no value\n", key);
      exit(1);
    }
  fprintf(stderr, "mesh2png: -%s: not set\n", key);
  exit(1);
}
static int arg_i(int argc, char **argv, const char *key) {
  const char *s = arg_find(argc, argv, key);
  char *end;
  long v = strtol(s, &end, 10);
  if (end == s || *end) {
    fprintf(stderr, "mesh2png: -%s: bad '%s'\n", key, s);
    exit(1);
  }
  return (int)v;
}

static void put32(unsigned char *p, unsigned v) {
  p[0] = v >> 24;
  p[1] = v >> 16;
  p[2] = v >> 8;
  p[3] = v;
}
static void chunk(FILE *f, const char *type, const unsigned char *d,
                  unsigned n) {
  unsigned char h[8];
  unsigned long c;
  unsigned char t[4];
  put32(h, n);
  memcpy(h + 4, type, 4);
  fwrite(h, 1, 8, f);
  c = crc32(0, (const unsigned char *)type, 4);
  if (n) {
    fwrite(d, 1, n, f);
    c = crc32(c, d, n);
  }
  put32(t, c);
  fwrite(t, 1, 4, f);
}

int main(int argc, char **argv) {
  const char *inf, *outf;
  int N, lw, nblk, b, y;
  FILE *fp, *out;
  long fsz;
  struct Blk *blk;
  int32_t info[3];
  float h, bw, sc, wy, tol, htol;
  unsigned char sig[] = {137, 80, 78, 71, 13, 10, 26, 10};
  unsigned char ihdr[13];
  z_stream z = {0};
  unsigned char *row;
  unsigned char zbuf[ZBUF];

  if (argc < 9) {
    fprintf(stderr, "usage: %s -i file.xyz.raw -o out.png -s N -w W\n",
            argv[0]);
    return 1;
  }
  inf = arg_find(argc, argv, "i");
  outf = arg_find(argc, argv, "o");
  N = arg_i(argc, argv, "s");
  lw = arg_i(argc, argv, "w");

  fp = fopen(inf, "rb");
  if (!fp) {
    perror(inf);
    return 1;
  }
  fseek(fp, 0, SEEK_END);
  fsz = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  nblk = fsz / (3 * sizeof(int32_t));
  blk = malloc(nblk * sizeof *blk);
  for (b = 0; b < nblk; b++) {
    fread(info, sizeof(int32_t), 3, fp);
    h = 1.0f / (BS << info[2]);
    bw = BS * h;
    blk[b].ox = info[0] * bw;
    blk[b].oy = info[1] * bw;
    blk[b].bw = bw;
  }
  fclose(fp);

  out = fopen(outf, "wb");
  fwrite(sig, 1, 8, out);
  put32(ihdr, N);
  put32(ihdr + 4, N);
  ihdr[8] = 8;
  ihdr[9] = 0;
  ihdr[10] = 0;
  ihdr[11] = 0;
  ihdr[12] = 0;
  chunk(out, "IHDR", ihdr, 13);

  deflateInit(&z, Z_DEFAULT_COMPRESSION);
  row = malloc(N + 1);
  sc = (float)N;

  for (y = 0; y < N; y++) {
    float x0, y0, x1, y1;
    int flush, pl, pr, px0, px1, d;
    unsigned have;
    wy = (N - 1 - y + 0.5f) / sc;
    row[0] = 0;
    memset(row + 1, 255, N);
    tol = 0.6f / sc;
    htol = lw * 0.5f / sc;
    for (b = 0; b < nblk; b++) {
      x0 = blk[b].ox;
      y0 = blk[b].oy;
      x1 = x0 + blk[b].bw;
      y1 = y0 + blk[b].bw;
      if (fabsf(wy - y0) <= htol || fabsf(wy - y1) <= htol) {
        px0 = (int)(x0 * sc);
        px1 = (int)(x1 * sc);
        if (px0 < 0) px0 = 0;
        if (px1 > N) px1 = N;
        for (d = px0; d < px1; d++) row[1 + d] = 0;
      }
      if (wy >= y0 - tol && wy <= y1 + tol) {
        for (d = -(lw / 2); d <= (lw - 1) / 2; d++) {
          pl = (int)(x0 * sc + 0.5f) + d;
          pr = (int)(x1 * sc + 0.5f) + d;
          if (pl >= 0 && pl < N) row[1 + pl] = 0;
          if (pr >= 0 && pr < N) row[1 + pr] = 0;
        }
      }
    }
    z.next_in = row;
    z.avail_in = N + 1;
    flush = y == N - 1 ? Z_FINISH : Z_NO_FLUSH;
    do {
      z.next_out = zbuf;
      z.avail_out = ZBUF;
      deflate(&z, flush);
      have = ZBUF - z.avail_out;
      if (have) chunk(out, "IDAT", zbuf, have);
    } while (z.avail_out == 0);
  }
  deflateEnd(&z);
  chunk(out, "IEND", NULL, 0);
  fclose(out);
  free(row);
  free(blk);
  return 0;
}
