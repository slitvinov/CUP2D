#include "stdio.h"

int abs(int i) { return i > 0 ? i : -i; }

int max(int x, int y) { return x > y ? x : y; }

int min(int x, int y) { return x > y ? y : x; }

typedef double Real;
enum { BS = 8 };
struct ChildNeighborPattern {
  int cx, cy;
  int Bstep;
  int ys;
  struct {
    int x, y;
  } child_offset[2];
  int count;
};
static struct ChildNeighborPattern childNeighborTable[] = {
    // cx, cy, Bstep, ys, children[], count
    {-1, -1, 3, 1, {{-1, -1}, {}}, 1},    // SW
    {0, -1, 1, 1, {{0, -1}, {1, -1}}, 2}, // S
    {1, -1, 3, 1, {{2, -1}, {}}, 1},      // SE
    {-1, 0, 1, 2, {{-1, 0}, {-1, 1}}, 2}, // W
    {1, 0, 1, 2, {{2, 0}, {2, 1}}, 2},    // E
    {-1, 1, 3, 1, {{-1, 2}, {}}, 1},      // NW
    {0, 1, 1, 1, {{0, 2}, {1, 2}}, 2},    // N
    {1, 1, 3, 1, {{2, 2}, {}}, 1},        // NE
};
const struct ChildNeighborPattern *get_child_pattern(int cx, int cy) {
  for (int i = 0; i < sizeof childNeighborTable / sizeof childNeighborTable[0];
       i++) {
    if (childNeighborTable[i].cx == cx && childNeighborTable[i].cy == cy)
      return &childNeighborTable[i];
  }
  return NULL;
}
int main() {
  int nm[2], e[2], s[2], dim, sx, sy;
  int cx = 1;
  int cy = 1;
  Real *m;
  const struct ChildNeighborPattern *pattern = get_child_pattern(cx, cy);
  int ys = pattern->ys;
  int mod = ((e[1] - s[1]) / ys) % 4;
  for (int B = 0; B <= 3; B += pattern->Bstep) {
    int aux = (abs(cx) == 1) ? (B % 2) : (B / 2);
    int ix = /* 2 * xi */ + max(cx, 0) + cx + (B % 2) * max(0, 1 - abs(cx));
    int iy = /* 2 * yi */ + max(cy, 0) + cy + aux * max(0, 1 - abs(cy));
    // const long long Z = forward(info->level + 1, ix, iy);
    Real *b; // = getf0(all, info->level + 1, Z)->block;
    int i = abs(cx) * (s[0] - sx) +
            (1 - abs(cx)) * (s[0] - sx + (B % 2) * (e[0] - s[0]) / 2);
    int x = s[0] - cx * BS + min(0, cx) * (e[0] - s[0]);

    for (int iy = s[1]; iy < e[1] - mod; iy += 4 * ys) {
      int k0 = i + (abs(cy) * (iy + 0 * ys - sy) +
                    (1 - abs(cy)) * ((iy + 0 * ys) / 2 - sy +
                                     aux * (e[1] - s[1]) / 2)) *
                       nm[0];
      int k1 = i + (abs(cy) * (iy + 1 * ys - sy) +
                    (1 - abs(cy)) * ((iy + 1 * ys) / 2 - sy +
                                     aux * (e[1] - s[1]) / 2)) *
                       nm[0];
      int k2 = i + (abs(cy) * (iy + 2 * ys - sy) +
                    (1 - abs(cy)) * ((iy + 2 * ys) / 2 - sy +
                                     aux * (e[1] - s[1]) / 2)) *
                       nm[0];
      int k3 = i + (abs(cy) * (iy + 3 * ys - sy) +
                    (1 - abs(cy)) * ((iy + 3 * ys) / 2 - sy +
                                     aux * (e[1] - s[1]) / 2)) *
                       nm[0];
      int y0 = (abs(cy) == 1) ? 2 * (iy + 0 * ys - cy * BS) + min(0, cy) * BS
                              : iy + 0 * ys;
      int y1 = (abs(cy) == 1) ? 2 * (iy + 1 * ys - cy * BS) + min(0, cy) * BS
                              : iy + 1 * ys;
      int y2 = (abs(cy) == 1) ? 2 * (iy + 2 * ys - cy * BS) + min(0, cy) * BS
                              : iy + 2 * ys;
      int y3 = (abs(cy) == 1) ? 2 * (iy + 3 * ys - cy * BS) + min(0, cy) * BS
                              : iy + 3 * ys;
      /* int z0 = y0 + 1; */
      int z1 = y1 + 1;
      int z2 = y2 + 1;
      int z3 = y3 + 1;
      Real *p0 = m + dim * k0;
      Real *p1 = m + dim * k1;
      Real *p2 = m + dim * k2;
      Real *p3 = m + dim * k3;
      Real *q00 = b + dim * (BS * y0 + x);
      // Real *q10 = b + dim * (BS * z0 + x);
      Real *q01 = b + dim * (BS * y1 + x);
      Real *q11 = b + dim * (BS * z1 + x);
      Real *q02 = b + dim * (BS * y2 + x);
      Real *q12 = b + dim * (BS * z2 + x);
      Real *q03 = b + dim * (BS * y3 + x);
      Real *q13 = b + dim * (BS * z3 + x);
      for (int ee = 0;
           ee < (abs(cx) * (e[0] - s[0]) + (1 - abs(cx)) * ((e[0] - s[0]) / 2));
           ee++) {
        Real *q000 = q00 + dim * 2 * ee;
        Real *q001 = q00 + dim * (2 * ee + 1);
        Real *q010 = q01 + dim * 2 * ee;
        Real *q011 = q01 + dim * (2 * ee + 1);
        Real *q020 = q02 + dim * 2 * ee;
        Real *q021 = q02 + dim * (2 * ee + 1);
        Real *q030 = q03 + dim * 2 * ee;
        Real *q031 = q03 + dim * (2 * ee + 1);
        Real *q110 = q11 + dim * 2 * ee;
        Real *q111 = q11 + dim * (2 * ee + 1);
        Real *q120 = q12 + dim * 2 * ee;
        Real *q121 = q12 + dim * (2 * ee + 1);
        Real *q130 = q13 + dim * 2 * ee;
        Real *q131 = q13 + dim * (2 * ee + 1);
        for (int d = 0; d < dim; d++) {
          *(p0 + dim * ee + d) =
              (*(q000 + d) + *(q010 + d) + *(q001 + d) + *(q011 + d)) / 4;
          *(p1 + dim * ee + d) =
              (*(q010 + d) + *(q110 + d) + *(q011 + d) + *(q111 + d)) / 4;
          *(p2 + dim * ee + d) =
              (*(q020 + d) + *(q120 + d) + *(q021 + d) + *(q121 + d)) / 4;
          *(p3 + dim * ee + d) =
              (*(q030 + d) + *(q130 + d) + *(q031 + d) + *(q131 + d)) / 4;
        }
      }
    }
    for (int iy = e[1] - mod; iy < e[1]; iy += ys) {
      int k = i + (abs(cy) * (iy - sy) +
                   (1 - abs(cy)) *
                       (iy / 2 - sy + aux * (e[1] - s[1]) / 2)) *
                      nm[0];
      int y = (abs(cy) == 1) ? 2 * (iy - cy * BS) + min(0, cy) * BS : iy;
      int z = y + 1;
      Real *p = m + dim * k;
      Real *q0 = b + dim * (BS * y + x);
      Real *q1 = b + dim * (BS * z + x);
      for (int ee = 0;
           ee < (abs(cx) * (e[0] - s[0]) + (1 - abs(cx)) * ((e[0] - s[0]) / 2));
           ee++) {
        Real *q00 = q0 + dim * 2 * ee;
        Real *q01 = q0 + dim * (2 * ee + 1);
        Real *q10 = q1 + dim * 2 * ee;
        Real *q11 = q1 + dim * (2 * ee + 1);
        for (int d = 0; d < dim; d++)
          *(p + dim * ee + d) =
              (*(q00 + d) + *(q10 + d) + *(q01 + d) + *(q11 + d)) / 4;
      }
    }
  }
}
