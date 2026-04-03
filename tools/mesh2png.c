/* mesh2png.c — render AMR block boundaries from xyz.raw to PNG.
   Streams scanlines, no image buffer.
   Usage: ./mesh2png -i vel.xyz.raw -o mesh.png -s 1024 -w 2 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

enum { BS = 8, ZBUF = 64 * 1024 };
struct Blk { float ox, oy, bw; };

static const char *arg_find(int argc, char **argv, const char *key) {
  for (int i = 1; i < argc; i++)
    if (argv[i][0] == '-' && strcmp(argv[i] + 1, key) == 0) {
      if (i + 1 < argc)
        return argv[i + 1];
      fprintf(stderr, "mesh2png: error: option -%s has no value\n", key);
      exit(1);
    }
  fprintf(stderr, "mesh2png: error: option -%s is not set\n", key);
  exit(1);
}
static int arg_i(int argc, char **argv, const char *key) {
  const char *s = arg_find(argc, argv, key);
  char *end;
  long v = strtol(s, &end, 10);
  if (end == s || *end != '\0') {
    fprintf(stderr, "mesh2png: error: -%s: bad integer '%s'\n", key, s);
    exit(1);
  }
  return (int)v;
}

static void put32(unsigned char *p, unsigned v) {
  p[0]=v>>24; p[1]=v>>16; p[2]=v>>8; p[3]=v;
}
static void chunk(FILE *f, const char *type, const unsigned char *d, unsigned n) {
  unsigned char h[8]; put32(h,n); memcpy(h+4,type,4);
  fwrite(h,1,8,f);
  unsigned long c = crc32(0,(const unsigned char*)type,4);
  if (n) { fwrite(d,1,n,f); c=crc32(c,d,n); }
  unsigned char t[4]; put32(t,c); fwrite(t,1,4,f);
}

int main(int argc, char **argv) {
  if (argc < 9) {
    fprintf(stderr, "usage: %s -i file.xyz.raw -o out.png -s N -w W\n", argv[0]);
    return 1;
  }
  const char *inf = arg_find(argc, argv, "i");
  const char *outf = arg_find(argc, argv, "o");
  int N = arg_i(argc, argv, "s");
  int lw = arg_i(argc, argv, "w");

  /* read block info: first cell of each BS*BS block */
  FILE *fp = fopen(inf, "rb");
  if (!fp) { perror(inf); return 1; }
  fseek(fp, 0, SEEK_END);
  long fsz = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  int ncell = fsz / (4*2*sizeof(float));
  int nblk = ncell / (BS*BS);
  struct Blk *blk = malloc(nblk * sizeof *blk);
  float hmin = 1, hmax = 0;
  for (int b = 0; b < nblk; b++) {
    float c[4][2];
    fread(c, sizeof(float), 8, fp);
    float h = c[3][0] - c[0][0];
    blk[b].ox = c[0][0]; blk[b].oy = c[0][1]; blk[b].bw = BS*h;
    if (h < hmin) hmin = h;
    if (h > hmax) hmax = h;
    fseek(fp, (long)(BS*BS-1)*8*sizeof(float), SEEK_CUR);
  }
  fclose(fp);
  fprintf(stderr, "%d blocks, h=[%g,%g]\n", nblk, hmin, hmax);

  /* write PNG */
  FILE *out = fopen(outf, "wb");
  unsigned char sig[] = {137,80,78,71,13,10,26,10};
  fwrite(sig,1,8,out);
  unsigned char ihdr[13]; put32(ihdr,N); put32(ihdr+4,N);
  ihdr[8]=8; ihdr[9]=0; ihdr[10]=0; ihdr[11]=0; ihdr[12]=0;
  chunk(out, "IHDR", ihdr, 13);

  /* stream IDAT: one scanline at a time */
  z_stream z = {0};
  deflateInit(&z, Z_DEFAULT_COMPRESSION);
  unsigned char *row = malloc(N+1);
  unsigned char zbuf[ZBUF];
  float sc = (float)N;

  for (int y = 0; y < N; y++) {
    float wy = (N-1-y+0.5f) / sc;
    row[0] = 0; /* filter: none */
    memset(row+1, 255, N);
    float tol = 0.6f / sc;
    for (int b = 0; b < nblk; b++) {
      float x0=blk[b].ox, y0=blk[b].oy, bw=blk[b].bw;
      float x1=x0+bw, y1=y0+bw;
      /* horizontal edges */
      float htol = (lw-0.5f) / sc;
      if (fabsf(wy-y0)<htol || fabsf(wy-y1)<htol) {
        int px0=(int)(x0*sc), px1=(int)(x1*sc);
        if (px0<0) px0=0; if (px1>N) px1=N;
        for (int p=px0;p<px1;p++) row[1+p]=0;
      }
      /* vertical edges */
      if (wy>=y0-tol && wy<=y1+tol) {
        for (int d=-(lw/2);d<=(lw-1)/2;d++) {
          int pl=(int)(x0*sc+0.5f)+d, pr=(int)(x1*sc+0.5f)+d;
          if (pl>=0&&pl<N) row[1+pl]=0;
          if (pr>=0&&pr<N) row[1+pr]=0;
        }
      }
    }
    z.next_in=row; z.avail_in=N+1;
    int flush = y==N-1 ? Z_FINISH : Z_NO_FLUSH;
    do {
      z.next_out=zbuf; z.avail_out=ZBUF;
      deflate(&z, flush);
      unsigned have = ZBUF - z.avail_out;
      if (have) chunk(out, "IDAT", zbuf, have);
    } while (z.avail_out==0);
  }
  deflateEnd(&z);
  chunk(out, "IEND", NULL, 0);
  fclose(out); free(row); free(blk);
  fprintf(stderr, "wrote %s (%dx%d)\n", outf, N, N);
  return 0;
}
