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
  int i, j, level;
  long long Z;
  sim.levelMax = 8;
  level = 5;
  if (argc == 3) {
    i = atoi(argv[1]);
    j = atoi(argv[2]);
    Z = forward(level, i, j);
    printf("[%d %d] %d %d\n", i, j, (int)Z, (int)morton(i, j));
  } else {
    Z = atoi(argv[1]);
    sfc_inverse(Z, level, &i, &j);
    printf("%d %d %d\n", i, j, (int)Z);
  }
}
