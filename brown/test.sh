#!/bin/sh

set -eu
cd "$(dirname "$0")"

python3 gen_table.py

gcc-15 -O2 -o main_omp main.c -fopenmp -lm
gcc-15 -O2 -o main_seq main.c -lm

# --- Test 1: uniform grid, seq vs omp must match ---
args_uni="-AdaptSteps 0 -CFL 0.8 -Rtol 1 -levelMax 3 -levelStart 3 -nu 1e-4 -sdump 0 -tdump 0.15 -tend 0.15"

rm -f *.raw
./main_seq $args_uni 2>/dev/null
mkdir -p _test_seq
mv *.raw _test_seq/

rm -f *.raw
./main_omp $args_uni 2>/dev/null
mkdir -p _test_omp
mv *.raw _test_omp/

# --- Test 2: AMR run, check refine+coarsen happens ---
args_amr="-AdaptSteps 2 -CFL 0.8 -Rtol 0.1 -levelMax 5 -levelStart 2 -nu 1e-4 -sdump 0 -tdump 0.15 -tend 0.15"

rm -f *.raw
./main_omp $args_amr 2>_test_amr.log
mkdir -p _test_amr
mv *.raw _test_amr/

# --- Check ---
cc -O2 -x c -o check_dumps - -lm <<'EOF'
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
  long sz = fsize(path);
  return (int)(sz / (3 * sizeof(int32_t)));
}

int main(void) {
  int ok = 1;

  /* Test 1: seq vs omp on uniform grid */
  {
    long n_seq = fsize("_test_seq/00000001.vort.raw") / sizeof(double);
    long n_omp = fsize("_test_omp/00000001.vort.raw") / sizeof(double);
    if (n_seq != n_omp) {
      fprintf(stderr, "FAIL test1: cell count seq=%ld omp=%ld\n", n_seq, n_omp);
      return 1;
    }
    FILE *fs = fopen("_test_seq/00000001.vort.raw", "rb");
    FILE *fo = fopen("_test_omp/00000001.vort.raw", "rb");
    double maxerr = 0;
    for (long i = 0; i < n_seq; i++) {
      double vs, vo;
      fread(&vs, sizeof(double), 1, fs);
      fread(&vo, sizeof(double), 1, fo);
      double e = fabs(vs - vo);
      if (e > maxerr) maxerr = e;
    }
    fclose(fs); fclose(fo);
    fprintf(stderr, "test1 (seq vs omp uniform): maxerr=%.6e", maxerr);
    if (maxerr > 0.1) {
      fprintf(stderr, " FAIL\n");
      ok = 0;
    } else {
      fprintf(stderr, " PASS\n");
    }
  }

  /* Test 2: AMR triggers refine and coarsen */
  {
    int nb0 = nblocks("_test_amr/00000000.xyz.raw");
    int nb1 = nblocks("_test_amr/00000001.xyz.raw");
    fprintf(stderr, "test2 (AMR): %d -> %d blocks", nb0, nb1);
    if (nb0 == nb1) {
      fprintf(stderr, " FAIL (no adaptation)\n");
      ok = 0;
    } else {
      fprintf(stderr, " PASS\n");
    }
  }

  /* Test 3: check refine and coarsen both happened */
  {
    FILE *f = fopen("_test_amr.log", "r");
    int saw_ref = 0, saw_com = 0;
    char line[256];
    while (fgets(line, sizeof line, f)) {
      int c, r;
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
rm -f check_dumps main_seq main_omp _test_amr.log
rm -rf _test_seq _test_omp _test_amr
