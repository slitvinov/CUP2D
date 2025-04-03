#include <algorithm>
#include <array>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#define _BS_ 8
typedef double Real;
static struct {
  int levelMax;
} sim;
#include "../utils.h"

uint64_t splitBy3(unsigned int a) {
  uint64_t x = a & 0x1fffff;
  x = (x | x << 32) & 0x1f00000000ffff;
  x = (x | x << 16) & 0x1f0000ff0000ff;
  x = (x | x << 8) & 0x100f00f00f00f00f;
  x = (x | x << 4) & 0x10c30c30c30c30c3;
  x = (x | x << 2) & 0x1249249249249249;
  return x;
}

uint64_t morton(unsigned int x, unsigned int y) {
  uint64_t answer = 0;
  answer |= splitBy3(x) | splitBy3(y) << 1;
  return answer;
}

int main(int argc, char **argv) {
  char *end;
  int i, j, level, LevelFlag;
  long long Z;
  sim.levelMax = 12;
  LevelFlag = 0;
  while (*++argv != NULL && argv[0][0] == '-') {
    switch (argv[0][1]) {
    case 'h':
      fprintf(stderr, "usage: curve -l z\n");
      exit(1);
      break;
    case 'l':
      argv++;
      if (argv[0] == NULL) {
	fprintf(stderr, "curve: error: -l needs an argument\n");
	exit(1);
      }
      level = strtol(argv[0], &end, 10);
      if (*end != '\0' || level <= 0) {
        fprintf(
            stderr,
            "curve: error: -l argument is not a positive integer: '%s'\n",
            argv[0]);
        exit(1);
      }
      LevelFlag = 1;
      break;
    default:
      fprintf(stderr, "curve: error: unknown option '%s'\n", *argv);
      exit(1);
    }
  }

  if (LevelFlag == 0) {
    fprintf(stderr, "curve: error: -l (level) must be set\n");
    exit(1);
  }
  if (argv[0] == NULL) {
    for (Z = 0; Z < 1 << (2 * level); Z++) {
      sfc_inverse(Z, level, &i, &j);      
      printf("%d %d\n", i, j);
    }
  } else if (argv[1] == NULL) {
    Z = atoi(argv[0]);
    sfc_inverse(Z, level, &i, &j);
    printf("%d %d %d\n", i, j, (int)Z);
  } else if (argv[2] == NULL) {
    i = atoi(argv[0]);
    j = atoi(argv[1]);
    Z = forward(level, i, j);
    printf("[%d %d] %d %d\n", i, j, (int)Z, (int)morton(i, j));
  }
}
