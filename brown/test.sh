#!/bin/sh
set -eu
cd "$(dirname "$0")"

UPDATE=0
test "${1:-}" = "--update" && UPDATE=1

python3 gen_table.py

D=/tmp/brown_test_$$
mkdir -p $D/uni $D/amr $D/san_uni $D/san_amr $D/warn

ARGS_UNI="-levelStart 3 -levelMax 3 -AdaptSteps 0 -Rtol 1 -nu 1e-4 -CFL 0.4 -tend 0.05 -tdump 0.05 -sdump 0"
ARGS_AMR="-levelStart 2 -levelMax 4 -AdaptSteps 2 -Rtol 0.5 -nu 1e-4 -CFL 0.4 -tend 0.05 -tdump 0.05 -sdump 0"

for f in tab_*.bin; do for d in $D/uni $D/amr $D/san_uni $D/san_amr; do cp $f $d/; done; done

cc -O2 -o $D/uni/main main.c -lm
cp $D/uni/main $D/amr/main
cc -O0 -g -fsanitize=address,undefined -o $D/san_uni/main main.c -lm
cp $D/san_uni/main $D/san_amr/main

(cd $D/uni && ./main $ARGS_UNI 2>log && echo OK >status) &
(cd $D/amr && ./main $ARGS_AMR 2>log && echo OK >status) &
(cd $D/san_uni && ./main $ARGS_UNI 2>log && echo OK >status) &
(cd $D/san_amr && ./main $ARGS_AMR 2>log && echo OK >status) &
(cc -O1 -Wall -Wextra -Wuninitialized -Wsometimes-uninitialized \
   -Wno-unused-parameter -Werror \
   -o /dev/null main.c -lm 2>$D/warn/log && echo OK >$D/warn/status) &
wait

if test $UPDATE -eq 1; then
  cp $D/uni/00000001.vort.raw ref_uni.vort.raw
  cp $D/amr/log ref_amr.log
  echo "references updated"
  rm -rf $D
  exit 0
fi

cc -O2 -x c -o $D/check - -lm <<'EOF'
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static long fsize(const char *p) {
  FILE *f = fopen(p, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", p); exit(1); }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fclose(f);
  return n;
}

int main(int argc, char **argv) {
  const char *uni = argv[1], *amr = argv[2], *amr_log = argv[3], *ref_dir = argv[4];
  char p[512], r[512];
  int ok = 1;

  snprintf(p, sizeof p, "%s/00000001.vort.raw", uni);
  snprintf(r, sizeof r, "%s/ref_uni.vort.raw", ref_dir);

  if (fsize(r) > 0) {
    long n = fsize(p) / sizeof(double);
    long nr = fsize(r) / sizeof(double);
    if (n != nr) {
      fprintf(stderr, "test1: cell count mismatch %ld vs ref %ld FAIL\n", n, nr);
      ok = 0;
    } else {
      FILE *fp = fopen(p, "rb"), *fr = fopen(r, "rb");
      double maxerr = 0;
      for (long i = 0; i < n; i++) {
        double v, vr;
        fread(&v, sizeof v, 1, fp);
        fread(&vr, sizeof vr, 1, fr);
        double e = fabs(v - vr);
        if (e > maxerr) maxerr = e;
      }
      fclose(fp); fclose(fr);
      fprintf(stderr, "test1 (uniform vs ref): maxerr=%.6e", maxerr);
      if (maxerr > 1e-10) { fprintf(stderr, " FAIL\n"); ok = 0; }
      else fprintf(stderr, " PASS\n");
    }
  } else {
    long n = fsize(p) / sizeof(double);
    FILE *f = fopen(p, "rb");
    double wmin = 1e30, wmax = -1e30;
    for (long i = 0; i < n; i++) {
      double v; fread(&v, sizeof v, 1, f);
      if (v < wmin) wmin = v; if (v > wmax) wmax = v;
    }
    fclose(f);
    fprintf(stderr, "test1 (uniform, no ref): vort [%.1f, %.1f]", wmin, wmax);
    if (wmax < 5 || wmin > -5) { fprintf(stderr, " FAIL\n"); ok = 0; }
    else fprintf(stderr, " PASS\n");
  }

  { char p0[512], p1[512];
    snprintf(p0, sizeof p0, "%s/00000000.xyz.raw", amr);
    snprintf(p1, sizeof p1, "%s/00000001.xyz.raw", amr);
    int nb0 = fsize(p0) / (3 * sizeof(int32_t));
    int nb1 = fsize(p1) / (3 * sizeof(int32_t));
    fprintf(stderr, "test2 (AMR): %d -> %d blocks", nb0, nb1);
    if (nb0 == nb1) { fprintf(stderr, " FAIL\n"); ok = 0; }
    else fprintf(stderr, " PASS\n");
  }

  { snprintf(p, sizeof p, "%s", amr_log);
    FILE *f = fopen(p, "r");
    int saw_ref = 0, saw_com = 0, c, rv;
    char line[256];
    while (fgets(line, sizeof line, f))
      if (sscanf(line, "  ad: com/ref %d/%d", &c, &rv) == 2) {
        if (rv > 0) saw_ref = 1;
        if (c > 0) saw_com = 1;
      }
    fclose(f);
    fprintf(stderr, "test3 (refine=%d coarsen=%d)", saw_ref, saw_com);
    if (!saw_ref || !saw_com) { fprintf(stderr, " FAIL\n"); ok = 0; }
    else fprintf(stderr, " PASS\n");
  }

  return ok ? 0 : 1;
}
EOF

fail=0

test -f $D/uni/status || { echo "FAIL: uniform run crashed"; fail=1; }
test -f $D/amr/status || { echo "FAIL: AMR run crashed"; fail=1; }
test -f $D/san_uni/status || { echo "FAIL: ASAN uniform crashed"; fail=1; }
test -f $D/san_amr/status || { echo "FAIL: ASAN AMR crashed"; fail=1; }
test -f $D/warn/status || { echo "FAIL: compiler warnings"; cat $D/warn/log; fail=1; }

grep -q 'ERROR\|runtime error' $D/san_uni/log && { echo "FAIL: ASAN uniform errors"; fail=1; } || echo "san_uni: clean"
grep -q 'ERROR\|runtime error' $D/san_amr/log && { echo "FAIL: ASAN AMR errors"; fail=1; } || echo "san_amr: clean"

SRCDIR=$(pwd)
test $fail -eq 0 && $D/check $D/uni $D/amr $D/amr/log $SRCDIR

rm -rf $D
exit $fail
