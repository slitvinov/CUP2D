#!/bin/sh

set -eu
cd "$(dirname "$0")"

python3 gen_table.py

cc -O2 -o main main.c -lm

args_uni="-AdaptSteps 0 -CFL 0.8 -Rtol 1 -levelMax 3 -levelStart 3 -nu 1e-4 -sdump 0 -tdump 0.15 -tend 0.15"
args_amr="-AdaptSteps 2 -CFL 0.8 -Rtol 0.5 -levelMax 4 -levelStart 2 -nu 1e-4 -sdump 0 -tdump 0.15 -tend 0.15"

rm -f *.raw
./main $args_uni 2>/dev/null
mkdir -p _test_uni
mv *.raw _test_uni/

rm -f *.raw
./main $args_amr 2>_test_amr.log
mkdir -p _test_amr
mv *.raw _test_amr/

cc -O2 -x c -o check_dumps - -lm <<'EOF'
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

enum { BS = 8 };

static long fsize(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "check: cannot open %s\n", path); exit(1); }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fclose(f);
  return n;
}

static int nblocks(const char *path) {
  return (int)(fsize(path) / (3 * sizeof(int32_t)));
}

int main(void) {
  int ok = 1;

  {
    long n = fsize("_test_uni/00000001.vort.raw") / sizeof(double);
    FILE *f = fopen("_test_uni/00000001.vort.raw", "rb");
    double wmin = 1e30, wmax = -1e30;
    long i;
    for (i = 0; i < n; i++) {
      double v;
      fread(&v, sizeof(double), 1, f);
      if (v < wmin) wmin = v;
      if (v > wmax) wmax = v;
    }
    fclose(f);
    fprintf(stderr, "test1 (uniform): vort [%.1f, %.1f]", wmin, wmax);
    if (wmax < 5 || wmin > -5) {
      fprintf(stderr, " FAIL (vorticity too small)\n");
      ok = 0;
    } else {
      fprintf(stderr, " PASS\n");
    }
  }

  {
    int nb0 = nblocks("_test_amr/00000000.xyz.raw");
    int nb1 = nblocks("_test_amr/00000001.xyz.raw");
    fprintf(stderr, "test2 (AMR): %d -> %d blocks", nb0, nb1);
    if (nb0 == nb1) {
      fprintf(stderr, " FAIL\n");
      ok = 0;
    } else {
      fprintf(stderr, " PASS\n");
    }
  }

  {
    FILE *f = fopen("_test_amr.log", "r");
    int saw_ref = 0, saw_com = 0;
    char line[256];
    int c, r;
    while (fgets(line, sizeof line, f)) {
      if (sscanf(line, "  ad: com/ref %d/%d", &c, &r) == 2) {
        if (r > 0) saw_ref = 1;
        if (c > 0) saw_com = 1;
      }
    }
    fclose(f);
    fprintf(stderr, "test3 (refine=%d coarsen=%d)", saw_ref, saw_com);
    if (!saw_ref || !saw_com) {
      fprintf(stderr, " FAIL\n");
      ok = 0;
    } else {
      fprintf(stderr, " PASS\n");
    }
  }

  return ok ? 0 : 1;
}
EOF

./check_dumps
rm -f check_dumps main _test_amr.log
rm -rf _test_uni _test_amr
