#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <gsl/gsl_linalg.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mpi.h>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "cuda.h"
#define OMPI_SKIP_MPICXX 1
enum { DIMENSION = 2 };
typedef double Real;
#define MPI_Real MPI_DOUBLE
static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
static Real dist(Real a[2], Real b[2]) {
  return std::pow(a[0] - b[0], 2) + std::pow(a[1] - b[1], 2);
}
static void rotate2D(const Real Rmatrix2D[2][2], Real &x, Real &y) {
  Real p[2] = {x, y};
  x = Rmatrix2D[0][0] * p[0] + Rmatrix2D[0][1] * p[1];
  y = Rmatrix2D[1][0] * p[0] + Rmatrix2D[1][1] * p[1];
}
static Real dds(int i, int m, Real *a, Real *b) {
  if (i == 0)
    return (a[i + 1] - a[i]) / (b[i + 1] - b[i]);
  else if (i == m - 1)
    return (a[i] - a[i - 1]) / (b[i] - b[i - 1]);
  else
    return ((a[i + 1] - a[i]) / (b[i + 1] - b[i]) +
            (a[i] - a[i - 1]) / (b[i] - b[i - 1])) /
           2;
}
static double getA_local(int I1, int I2) {
  int j1 = I1 / _BS_;
  int i1 = I1 % _BS_;
  int j2 = I2 / _BS_;
  int i2 = I2 % _BS_;
  if (i1 == i2 && j1 == j2)
    return 4.0;
  else if (abs(i1 - i2) + abs(j1 - j2) == 1)
    return -1.0;
  else
    return 0.0;
}
struct Value {
  std::string content;
  Value() = default;
  Value(const std::string &content_) : content(content_) {}
  Value(const Value &c) = default;
  Value &operator=(const Value &rhs) {
    if (this != &rhs)
      content = rhs.content;
    return *this;
  }
  Value &operator+=(const Value &rhs) {
    content += " " + rhs.content;
    return *this;
  }
  double asDouble(double def = 0) {
    if (content == "") {
      std::ostringstream sbuf;
      sbuf << def;
      content = sbuf.str();
    }
    return (double)atof(content.c_str());
  }
  int asInt(int def = 0) {
    if (content == "") {
      std::ostringstream sbuf;
      sbuf << def;
      content = sbuf.str();
    }
    return atoi(content.c_str());
  }
  bool asBool(bool def) {
    if (content == "") {
      if (def)
        content = "true";
      else
        content = "false";
    }
    if (content == "0")
      return false;
    if (content == "false")
      return false;
    return true;
  }
  std::string asString(const std::string &def) {
    if (content == "")
      content = def;
    return content;
  }
};
struct CommandlineParser {
  bool bStrictMode;
  bool _isnumber(const std::string &s) const;
  std::map<std::string, Value> mapArguments;
  CommandlineParser(int argc, char **argv);
  Value &operator()(std::string key);
  void set_strict_mode() { bStrictMode = true; }
  void unset_strict_mode() { bStrictMode = false; }
};
static bool _existKey(const std::string &key,
                      const std::map<std::string, Value> &container) {
  return container.find(key) != container.end();
}
Value &CommandlineParser::operator()(std::string key) {
  if (key[0] == '-')
    key.erase(0, 1);
  if (key[0] == '+')
    key.erase(0, 1);
  if (bStrictMode) {
    if (mapArguments.find(key) == mapArguments.end()) {
      printf("runtime %s is not set\n", key.data());
      abort();
    }
  }
  return mapArguments[key];
}
CommandlineParser::CommandlineParser(const int argc, char **argv)
    : bStrictMode(false) {
  for (int i = 1; i < argc; i++)
    if (argv[i][0] == '-') {
      std::string values = "";
      int itemCount = 0;
      for (int j = i + 1; j < argc; j++) {
        const bool leadingDash = (argv[j][0] == '-');
        char *end = NULL;
        strtod(argv[j], &end);
        const bool isNumeric = end != argv[j];
        if (leadingDash && !isNumeric)
          break;
        else {
          if (std::strcmp(values.c_str(), ""))
            values += ' ';
          values += argv[j];
          itemCount++;
        }
      }
      if (itemCount == 0)
        values = "true";
      std::string key(argv[i]);
      key.erase(0, 1);
      if (key[0] == '+') {
        key.erase(0, 1);
        if (!_existKey(key, mapArguments))
          mapArguments[key] = Value(values);
        else
          mapArguments[key] += Value(values);
      } else {
        if (!_existKey(key, mapArguments))
          mapArguments[key] = Value(values);
      }
      i += itemCount;
    }
}
struct SpaceFillingCurve2D {
  int BX;
  int BY;
  int levelMax;
  bool isRegular;
  int base_level;
  std::vector<std::vector<long long>> Zsave;
  std::vector<std::vector<int>> i_inverse;
  std::vector<std::vector<int>> j_inverse;
  long long AxestoTranspose(const int *X_in, int b) const {
    int x = X_in[0];
    int y = X_in[1];
    int n = 1 << b;
    int rx, ry, s, d = 0;
    for (s = n / 2; s > 0; s /= 2) {
      rx = (x & s) > 0;
      ry = (y & s) > 0;
      d += s * s * ((3 * rx) ^ ry);
      rot(n, &x, &y, rx, ry);
    }
    return d;
  }
  void TransposetoAxes(long long index, int *X, int b) const {
    int n = 1 << b;
    long long rx, ry, s, t = index;
    X[0] = 0;
    X[1] = 0;
    for (s = 1; s < n; s *= 2) {
      rx = 1 & (t / 2);
      ry = 1 & (t ^ rx);
      rot(s, &X[0], &X[1], rx, ry);
      X[0] += s * rx;
      X[1] += s * ry;
      t /= 4;
    }
  }
  void rot(long long n, int *x, int *y, long long rx, long long ry) const {
    if (ry == 0) {
      if (rx == 1) {
        *x = n - 1 - *x;
        *y = n - 1 - *y;
      }
      int t = *x;
      *x = *y;
      *y = t;
    }
  }
  SpaceFillingCurve2D(int a_BX, int a_BY, int lmax)
      : BX(a_BX), BY(a_BY), levelMax(lmax) {
    const int n_max = std::max(BX, BY);
    base_level = (log(n_max) / log(2));
    if (base_level < (double)(log(n_max) / log(2)))
      base_level++;
    i_inverse.resize(lmax);
    j_inverse.resize(lmax);
    Zsave.resize(lmax);
    {
      const int l = 0;
      const int aux = pow(pow(2, l), 2);
      i_inverse[l].resize(BX * BY * aux, -1);
      j_inverse[l].resize(BX * BY * aux, -1);
      Zsave[l].resize(BX * BY * aux, -1);
    }
    isRegular = true;
#pragma omp parallel for collapse(2)
    for (int j = 0; j < BY; j++)
      for (int i = 0; i < BX; i++) {
        const int c[2] = {i, j};
        long long index = AxestoTranspose(c, base_level);
        long long substract = 0;
        for (long long h = 0; h < index; h++) {
          int X[2] = {0, 0};
          TransposetoAxes(h, X, base_level);
          if (X[0] >= BX || X[1] >= BY)
            substract++;
        }
        index -= substract;
        if (substract > 0)
          isRegular = false;
        i_inverse[0][index] = i;
        j_inverse[0][index] = j;
        Zsave[0][j * BX + i] = index;
      }
  }
  long long forward(const int l, const int i, const int j) {
    const int aux = 1 << l;
    if (l >= levelMax)
      return 0;
    long long retval;
    if (!isRegular) {
      const int I = i / aux;
      const int J = j / aux;
      const int c2_a[2] = {i - I * aux, j - J * aux};
      retval = AxestoTranspose(c2_a, l);
      retval += IJ_to_index(I, J) * aux * aux;
    } else {
      const int c2_a[2] = {i, j};
      retval = AxestoTranspose(c2_a, l + base_level);
    }
    return retval;
  }
  void inverse(long long Z, int l, int &i, int &j) {
    if (isRegular) {
      int X[2] = {0, 0};
      TransposetoAxes(Z, X, l + base_level);
      i = X[0];
      j = X[1];
    } else {
      int aux = 1 << l;
      long long Zloc = Z % (aux * aux);
      int X[2] = {0, 0};
      TransposetoAxes(Zloc, X, l);
      long long index = Z / (aux * aux);
      int I, J;
      index_to_IJ(index, I, J);
      i = X[0] + I * aux;
      j = X[1] + J * aux;
    }
    return;
  }
  long long IJ_to_index(int I, int J) {
    long long index = Zsave[0][J * BX + I];
    return index;
  }
  void index_to_IJ(long long index, int &I, int &J) {
    I = i_inverse[0][index];
    J = j_inverse[0][index];
    return;
  }
  long long Encode(int level, int index[2]) {
    int lmax = levelMax;
    long long retval = 0;
    int ix = index[0];
    int iy = index[1];
    for (int l = level; l >= 0; l--) {
      long long Zp = forward(l, ix, iy);
      retval += Zp;
      ix /= 2;
      iy /= 2;
    }
    ix = 2 * index[0];
    iy = 2 * index[1];
    for (int l = level + 1; l < lmax; l++) {
      long long Zc = forward(l, ix, iy);
      Zc -= Zc % 4;
      retval += Zc;
      int ix1, iy1;
      inverse(Zc, l, ix1, iy1);
      ix = 2 * ix1;
      iy = 2 * iy1;
    }
    retval += level;
    return retval;
  }
};
enum State : signed char { Leave = 0, Refine = 1, Compress = -1 };
struct TreePosition {
  int position{-3};
  bool CheckCoarser() const { return position == -2; }
  bool CheckFiner() const { return position == -1; }
  bool Exists() const { return position >= 0; }
  int rank() const { return position; }
  void setrank(const int r) { position = r; }
  void setCheckCoarser() { position = -2; }
  void setCheckFiner() { position = -1; }
};
struct BlockInfo {
  bool changed2;
  double h;
  double origin[3];
  int index[3];
  int level;
  long long blockID, blockID_2, halo_block_id, Z, Zchild[2][2][2],
      Znei[3][3][3], Zparent;
  State state;
  void *auxiliary;
  void *ptrBlock{nullptr};
  static int levelMax(int l = 0) {
    static int lmax = l;
    return lmax;
  }
  static int blocks_per_dim(int i, int nx = 0, int ny = 0) {
    static int a[2] = {nx, ny};
    return a[i];
  }
  static SpaceFillingCurve2D *SFC() {
    static SpaceFillingCurve2D Zcurve(blocks_per_dim(0), blocks_per_dim(1),
                                      levelMax());
    return &Zcurve;
  }
  static long long forward(int level, int ix, int iy) {
    return (*SFC()).forward(level, ix, iy);
  }
  static long long Encode(int level, int index[2]) {
    return (*SFC()).Encode(level, index);
  }
  static void inverse(long long Z, int l, int &i, int &j) {
    (*SFC()).inverse(Z, l, i, j);
  }
  void pos(Real p[2], int ix, int iy) const {
    p[0] = origin[0] + h * (ix + 0.5);
    p[1] = origin[1] + h * (iy + 0.5);
  }
  std::array<Real, 2> pos(int ix, int iy) const {
    std::array<Real, 2> result;
    pos(result.data(), ix, iy);
    return result;
  }
  bool operator<(const BlockInfo &other) const {
    return (blockID_2 < other.blockID_2);
  }
  BlockInfo(){};
  void setup(const int a_level, const double a_h, const double a_origin[3],
             const long long a_Z) {
    level = a_level;
    Z = a_Z;
    state = Leave;
    level = a_level;
    h = a_h;
    origin[0] = a_origin[0];
    origin[1] = a_origin[1];
    origin[2] = a_origin[2];
    changed2 = true;
    auxiliary = nullptr;
    const int TwoPower = 1 << level;
    inverse(Z, level, index[0], index[1]);
    index[2] = 0;
    const int Bmax[3] = {blocks_per_dim(0) * TwoPower,
                         blocks_per_dim(1) * TwoPower, 1};
    for (int i = -1; i < 2; i++)
      for (int j = -1; j < 2; j++)
        for (int k = -1; k < 2; k++)
          Znei[i + 1][j + 1][k + 1] =
              forward(level, (index[0] + i + Bmax[0]) % Bmax[0],
                      (index[1] + j + Bmax[1]) % Bmax[1]);
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++)
          Zchild[i][j][k] =
              forward(level + 1, 2 * index[0] + i, 2 * index[1] + j);
    Zparent = (level == 0)
                  ? 0
                  : forward(level - 1, (index[0] / 2 + Bmax[0]) % Bmax[0],
                            (index[1] / 2 + Bmax[1]) % Bmax[1]);
    blockID_2 = Encode(level, index);
    blockID = blockID_2;
  }
  long long Znei_(const int i, const int j, const int k) const {
    assert(abs(i) <= 1);
    assert(abs(j) <= 1);
    assert(abs(k) <= 1);
    return Znei[1 + i][1 + j][1 + k];
  }
};
template <typename BlockType, typename ElementType> struct BlockCase {
  std::vector<ElementType> m_pData[6];
  static constexpr unsigned int m_vSize[] = {_BS_, _BS_, 1};
  bool storedFace[6];
  int level;
  long long Z;
  BlockCase(bool _storedFace[6], int _level, long long _Z) {
    storedFace[0] = _storedFace[0];
    storedFace[1] = _storedFace[1];
    storedFace[2] = _storedFace[2];
    storedFace[3] = _storedFace[3];
    storedFace[4] = _storedFace[4];
    storedFace[5] = _storedFace[5];
    for (int d = 0; d < 3; d++) {
      int d1 = (d + 1) % 3;
      int d2 = (d + 2) % 3;
      if (storedFace[2 * d])
        m_pData[2 * d].resize(m_vSize[d1] * m_vSize[d2]);
      if (storedFace[2 * d + 1])
        m_pData[2 * d + 1].resize(m_vSize[d1] * m_vSize[d2]);
    }
    level = _level;
    Z = _Z;
  }
};
template <typename TGrid, typename ElementType> struct FluxCorrection {
  typedef TGrid GridType;
  typedef typename GridType::BlockType BlockType;
  typedef BlockCase<BlockType, ElementType> Case;
  int rank{0};
  std::map<std::array<long long, 2>, Case *> MapOfCases;
  TGrid *grid;
  std::vector<Case> Cases;
  void FillCase(BlockInfo &info, const int *const code) {
    const int myFace = abs(code[0]) * std::max(0, code[0]) +
                       abs(code[1]) * (std::max(0, code[1]) + 2) +
                       abs(code[2]) * (std::max(0, code[2]) + 4);
    const int otherFace = abs(-code[0]) * std::max(0, -code[0]) +
                          abs(-code[1]) * (std::max(0, -code[1]) + 2) +
                          abs(-code[2]) * (std::max(0, -code[2]) + 4);
    std::array<long long, 2> temp = {(long long)info.level, info.Z};
    auto search = MapOfCases.find(temp);
    Case &CoarseCase = (*search->second);
    std::vector<ElementType> &CoarseFace = CoarseCase.m_pData[myFace];
    assert(myFace / 2 == otherFace / 2);
    assert(search != MapOfCases.end());
    assert(CoarseCase.Z == info.Z);
    assert(CoarseCase.level == info.level);
    for (int B = 0; B <= 1; B++) {
      const int aux = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
      const long long Z = (*grid).getZforward(
          info.level + 1,
          2 * info.index[0] + std::max(code[0], 0) + code[0] +
              (B % 2) * std::max(0, 1 - abs(code[0])),
          2 * info.index[1] + std::max(code[1], 0) + code[1] +
              aux * std::max(0, 1 - abs(code[1])));
      const int other_rank = grid->Tree(info.level + 1, Z).rank();
      if (other_rank != rank)
        continue;
      auto search1 = MapOfCases.find({info.level + 1, Z});
      Case &FineCase = (*search1->second);
      std::vector<ElementType> &FineFace = FineCase.m_pData[otherFace];
      const int d = myFace / 2;
      const int d1 = std::max((d + 1) % 3, (d + 2) % 3);
      const int d2 = std::min((d + 1) % 3, (d + 2) % 3);
      const int N1F = FineCase.m_vSize[d1];
      const int N2F = FineCase.m_vSize[d2];
      const int N1 = N1F;
      const int N2 = N2F;
      int base = 0;
      if (B == 1)
        base = (N2 / 2) + (0) * N2;
      else if (B == 2)
        base = (0) + (N1 / 2) * N2;
      else if (B == 3)
        base = (N2 / 2) + (N1 / 2) * N2;
      assert(search1 != MapOfCases.end());
      assert(N1F == (int)CoarseCase.m_vSize[d1]);
      assert(N2F == (int)CoarseCase.m_vSize[d2]);
      assert(FineFace.size() == CoarseFace.size());
      for (int i2 = 0; i2 < N2; i2 += 2) {
        CoarseFace[base + i2 / 2] += FineFace[i2] + FineFace[i2 + 1];
        FineFace[i2].clear();
        FineFace[i2 + 1].clear();
      }
    }
  }
  virtual void prepare(TGrid &_grid) {
    if (_grid.UpdateFluxCorrection == false)
      return;
    _grid.UpdateFluxCorrection = false;
    Cases.clear();
    MapOfCases.clear();
    grid = &_grid;
    std::vector<BlockInfo> &B = (*grid).m_vInfo;
    std::array<int, 3> blocksPerDim = (*grid).getMaxBlocks();
    std::array<int, 6> icode = {1 * 2 + 3 * 1 + 9 * 1, 1 * 0 + 3 * 1 + 9 * 1,
                                1 * 1 + 3 * 2 + 9 * 1, 1 * 1 + 3 * 0 + 9 * 1,
                                1 * 1 + 3 * 1 + 9 * 2, 1 * 1 + 3 * 1 + 9 * 0};
    for (auto &info : B) {
      grid->getBlockInfoAll(info.level, info.Z).auxiliary = nullptr;
      const int aux = 1 << info.level;
      const bool xskin =
          info.index[0] == 0 || info.index[0] == blocksPerDim[0] * aux - 1;
      const bool yskin =
          info.index[1] == 0 || info.index[1] == blocksPerDim[1] * aux - 1;
      const bool zskin =
          info.index[2] == 0 || info.index[2] == blocksPerDim[2] * aux - 1;
      const int xskip = info.index[0] == 0 ? -1 : 1;
      const int yskip = info.index[1] == 0 ? -1 : 1;
      const int zskip = info.index[2] == 0 ? -1 : 1;
      bool storeFace[6] = {false, false, false, false, false, false};
      bool stored = false;
      for (int f = 0; f < 6; f++) {
        const int code[3] = {icode[f] % 3 - 1, (icode[f] / 3) % 3 - 1,
                             (icode[f] / 9) % 3 - 1};
        if (!_grid.xperiodic && code[0] == xskip && xskin)
          continue;
        if (!_grid.yperiodic && code[1] == yskip && yskin)
          continue;
        if (!_grid.zperiodic && code[2] == zskip && zskin)
          continue;
        if (code[2] != 0)
          continue;
        if (!grid->Tree(info.level, info.Znei_(code[0], code[1], code[2]))
                 .Exists()) {
          storeFace[abs(code[0]) * std::max(0, code[0]) +
                    abs(code[1]) * (std::max(0, code[1]) + 2) +
                    abs(code[2]) * (std::max(0, code[2]) + 4)] = true;
          stored = true;
        }
      }
      if (stored) {
        Cases.push_back(Case(storeFace, info.level, info.Z));
      }
    }
    size_t Cases_index = 0;
    if (Cases.size() > 0)
      for (auto &info : B) {
        if (Cases_index == Cases.size())
          break;
        if (Cases[Cases_index].level == info.level &&
            Cases[Cases_index].Z == info.Z) {
          MapOfCases.insert(std::pair<std::array<long long, 2>, Case *>(
              {Cases[Cases_index].level, Cases[Cases_index].Z},
              &Cases[Cases_index]));
          grid->getBlockInfoAll(Cases[Cases_index].level, Cases[Cases_index].Z)
              .auxiliary = &Cases[Cases_index];
          info.auxiliary = &Cases[Cases_index];
          Cases_index++;
        }
      }
  }
  virtual void FillBlockCases() {
    std::vector<BlockInfo> &B = (*grid).m_vInfo;
    std::array<int, 3> blocksPerDim = (*grid).getMaxBlocks();
    std::array<int, 6> icode = {1 * 2 + 3 * 1 + 9 * 1, 1 * 0 + 3 * 1 + 9 * 1,
                                1 * 1 + 3 * 2 + 9 * 1, 1 * 1 + 3 * 0 + 9 * 1,
                                1 * 1 + 3 * 1 + 9 * 2, 1 * 1 + 3 * 1 + 9 * 0};
#pragma omp parallel for
    for (size_t i = 0; i < B.size(); i++) {
      BlockInfo &info = B[i];
      const int aux = 1 << info.level;
      const bool xskin =
          info.index[0] == 0 || info.index[0] == blocksPerDim[0] * aux - 1;
      const bool yskin =
          info.index[1] == 0 || info.index[1] == blocksPerDim[1] * aux - 1;
      const bool zskin =
          info.index[2] == 0 || info.index[2] == blocksPerDim[2] * aux - 1;
      const int xskip = info.index[0] == 0 ? -1 : 1;
      const int yskip = info.index[1] == 0 ? -1 : 1;
      const int zskip = info.index[2] == 0 ? -1 : 1;
      for (int f = 0; f < 6; f++) {
        const int code[3] = {icode[f] % 3 - 1, (icode[f] / 3) % 3 - 1,
                             (icode[f] / 9) % 3 - 1};
        if (!grid->xperiodic && code[0] == xskip && xskin)
          continue;
        if (!grid->yperiodic && code[1] == yskip && yskin)
          continue;
        if (!grid->zperiodic && code[2] == zskip && zskin)
          continue;
        if (code[2] != 0)
          continue;
        bool checkFiner =
            grid->Tree(info.level, info.Znei_(code[0], code[1], code[2]))
                .CheckFiner();
        if (checkFiner) {
          FillCase(info, code);
          const int myFace = abs(code[0]) * std::max(0, code[0]) +
                             abs(code[1]) * (std::max(0, code[1]) + 2) +
                             abs(code[2]) * (std::max(0, code[2]) + 4);
          std::array<long long, 2> temp = {(long long)info.level, info.Z};
          auto search = MapOfCases.find(temp);
          assert(search != MapOfCases.end());
          Case &CoarseCase = (*search->second);
          std::vector<ElementType> &CoarseFace = CoarseCase.m_pData[myFace];
          const int d = myFace / 2;
          const int d2 = std::min((d + 1) % 3, (d + 2) % 3);
          const int N2 = CoarseCase.m_vSize[d2];
          BlockType &block = *(BlockType *)info.ptrBlock;
          assert(d != 2);
          if (d == 0) {
            const int j = (myFace % 2 == 0) ? 0 : _BS_ - 1;
            for (int i2 = 0; i2 < N2; i2++) {
              block(j, i2) += CoarseFace[i2];
              CoarseFace[i2].clear();
            }
          } else {
            const int j = (myFace % 2 == 0) ? 0 : _BS_ - 1;
            for (int i2 = 0; i2 < N2; i2++) {
              block(i2, j) += CoarseFace[i2];
              CoarseFace[i2].clear();
            }
          }
        }
      }
    }
  }
};
struct BlockGroup {
  int i_min[3];
  int i_max[3];
  int level;
  std::vector<long long> Z;
  size_t ID;
  double origin[3];
  double h;
  int NXX;
  int NYY;
  int NZZ;
};
template <typename Block, typename ElementType> struct Grid {
  typedef Block BlockType;
  std::unordered_map<long long, BlockInfo *> BlockInfoAll;
  std::unordered_map<long long, TreePosition> Octree;
  std::vector<BlockInfo> m_vInfo;
  const int NX;
  const int NY;
  const int NZ;
  const double maxextent;
  const int levelMax;
  const int levelStart;
  const bool xperiodic;
  const bool yperiodic;
  const bool zperiodic;
  std::vector<BlockGroup> MyGroups;
  std::vector<long long> level_base;
  bool UpdateFluxCorrection{true};
  bool UpdateGroups{true};
  bool FiniteDifferences{true};
  FluxCorrection<Grid, ElementType> CorrectorGrid;
  TreePosition &Tree(const int m, const long long n) {
    const long long aux = level_base[m] + n;
    const auto retval = Octree.find(aux);
    if (retval == Octree.end()) {
#pragma omp critical
      {
        const auto retval1 = Octree.find(aux);
        if (retval1 == Octree.end()) {
          TreePosition dum;
          Octree[aux] = dum;
        }
      }
      return Tree(m, n);
    } else {
      return retval->second;
    }
  }
  TreePosition &Tree(BlockInfo &info) { return Tree(info.level, info.Z); }
  TreePosition &Tree(const BlockInfo &info) { return Tree(info.level, info.Z); }
  void _alloc(const int m, const long long n) {
    std::allocator<Block> alloc;
    BlockInfo &new_info = getBlockInfoAll(m, n);
    new_info.ptrBlock = alloc.allocate(1);
#pragma omp critical
    { m_vInfo.push_back(new_info); }
    Tree(m, n).setrank(rank());
  }
  void _dealloc(const int m, const long long n) {
    std::allocator<Block> alloc;
    alloc.deallocate((Block *)getBlockInfoAll(m, n).ptrBlock, 1);
    for (size_t j = 0; j < m_vInfo.size(); j++) {
      if (m_vInfo[j].level == m && m_vInfo[j].Z == n) {
        m_vInfo.erase(m_vInfo.begin() + j);
        return;
      }
    }
  }
  void dealloc_many(const std::vector<long long> &dealloc_IDs) {
    for (size_t j = 0; j < m_vInfo.size(); j++)
      m_vInfo[j].changed2 = false;
    std::allocator<Block> alloc;
    for (size_t i = 0; i < dealloc_IDs.size(); i++)
      for (size_t j = 0; j < m_vInfo.size(); j++) {
        if (m_vInfo[j].blockID_2 == dealloc_IDs[i]) {
          const int m = m_vInfo[j].level;
          const long long n = m_vInfo[j].Z;
          m_vInfo[j].changed2 = true;
          alloc.deallocate((Block *)getBlockInfoAll(m, n).ptrBlock, 1);
          break;
        }
      }
    m_vInfo.erase(std::remove_if(m_vInfo.begin(), m_vInfo.end(),
                                 [](const BlockInfo &x) { return x.changed2; }),
                  m_vInfo.end());
  }
  void FindBlockInfo(const int m, const long long n, const int m_new,
                     const long long n_new) {
    for (size_t j = 0; j < m_vInfo.size(); j++)
      if (m == m_vInfo[j].level && n == m_vInfo[j].Z) {
        BlockInfo &correct_info = getBlockInfoAll(m_new, n_new);
        correct_info.state = Leave;
        m_vInfo[j] = correct_info;
        return;
      }
  }
  virtual void FillPos(bool CopyInfos = true) {
    std::sort(m_vInfo.begin(), m_vInfo.end());
    Octree.reserve(Octree.size() + m_vInfo.size() / 8);
    if (CopyInfos)
      for (size_t j = 0; j < m_vInfo.size(); j++) {
        const int m = m_vInfo[j].level;
        const long long n = m_vInfo[j].Z;
        BlockInfo &correct_info = getBlockInfoAll(m, n);
        correct_info.blockID = j;
        m_vInfo[j] = correct_info;
        assert(Tree(m, n).Exists());
      }
    else
      for (size_t j = 0; j < m_vInfo.size(); j++) {
        const int m = m_vInfo[j].level;
        const long long n = m_vInfo[j].Z;
        BlockInfo &correct_info = getBlockInfoAll(m, n);
        correct_info.blockID = j;
        m_vInfo[j].blockID = j;
        m_vInfo[j].state = correct_info.state;
        assert(Tree(m, n).Exists());
      }
  }
  Grid(unsigned int _NX, unsigned int _NY, unsigned int _NZ, double _maxextent,
       unsigned int _levelStart, unsigned int _levelMax, bool a_xperiodic,
       bool a_yperiodic, bool a_zperiodic)
      : NX(_NX), NY(_NY), NZ(_NZ), maxextent(_maxextent), levelMax(_levelMax),
        levelStart(_levelStart), xperiodic(a_xperiodic), yperiodic(a_yperiodic),
        zperiodic(a_zperiodic) {
    BlockInfo dummy;
    const int nx = dummy.blocks_per_dim(0, NX, NY);
    const int ny = dummy.blocks_per_dim(1, NX, NY);
    const int nz = 1;
    const int lvlMax = dummy.levelMax(levelMax);
    for (int m = 0; m < lvlMax; m++) {
      const int TwoPower = 1 << m;
      const long long Ntot = nx * ny * nz * pow(TwoPower, (Real)DIMENSION);
      if (m == 0)
        level_base.push_back(Ntot);
      if (m > 0)
        level_base.push_back(level_base[m - 1] + Ntot);
    }
  }
  virtual Block *avail(const int m, const long long n) {
    return (Block *)getBlockInfoAll(m, n).ptrBlock;
  }
  virtual int rank() const { return 0; }
  virtual void initialize_blocks(const std::vector<long long> &blocksZ,
                                 const std::vector<short int> &blockslevel) {
    std::allocator<Block> alloc;
    for (size_t i = 0; i < m_vInfo.size(); i++) {
      const int m = m_vInfo[i].level;
      const long long n = m_vInfo[i].Z;
      alloc.deallocate((Block *)getBlockInfoAll(m, n).ptrBlock, 1);
    }
    std::vector<long long> aux;
    for (auto &m : BlockInfoAll)
      aux.push_back(m.first);
    for (size_t i = 0; i < aux.size(); i++) {
      const auto retval = BlockInfoAll.find(aux[i]);
      if (retval != BlockInfoAll.end()) {
        delete retval->second;
      }
    }
    m_vInfo.clear();
    BlockInfoAll.clear();
    Octree.clear();
    for (size_t i = 0; i < blocksZ.size(); i++) {
      const int level = blockslevel[i];
      const long long Z = blocksZ[i];
      _alloc(level, Z);
      Tree(level, Z).setrank(rank());
      int p[2];
      BlockInfo::inverse(Z, level, p[0], p[1]);
      if (level < levelMax - 1)
        for (int j1 = 0; j1 < 2; j1++)
          for (int i1 = 0; i1 < 2; i1++) {
            const long long nc =
                getZforward(level + 1, 2 * p[0] + i1, 2 * p[1] + j1);
            Tree(level + 1, nc).setCheckCoarser();
          }
      if (level > 0) {
        const long long nf = getZforward(level - 1, p[0] / 2, p[1] / 2);
        Tree(level - 1, nf).setCheckFiner();
      }
    }
    FillPos();
    UpdateFluxCorrection = true;
    UpdateGroups = true;
  }
  long long getZforward(const int level, const int i, const int j) const {
    const int TwoPower = 1 << level;
    const int ix = (i + TwoPower * NX) % (NX * TwoPower);
    const int iy = (j + TwoPower * NY) % (NY * TwoPower);
    return BlockInfo::forward(level, ix, iy);
  }
  Block *avail1(const int ix, const int iy, const int m) {
    const long long n = getZforward(m, ix, iy);
    return avail(m, n);
  }
  Block &operator()(const long long ID) {
    return *(Block *)m_vInfo[ID].ptrBlock;
  }
  std::array<int, 3> getMaxBlocks() const { return {NX, NY, NZ}; }
  std::array<int, 3> getMaxMostRefinedBlocks() const {
    return {
        NX << (levelMax - 1),
        NY << (levelMax - 1),
        1,
    };
  }
  std::array<int, 3> getMaxMostRefinedCells() const {
    const auto b = getMaxMostRefinedBlocks();
    return {b[0] * _BS_, b[1] * _BS_, b[2] * 1};
  }
  int getlevelMax() const { return levelMax; }
  BlockInfo &getBlockInfoAll(const int m, const long long n) {
    const long long aux = level_base[m] + n;
    const auto retval = BlockInfoAll.find(aux);
    if (retval != BlockInfoAll.end()) {
      return *retval->second;
    } else {
#pragma omp critical
      {
        const auto retval1 = BlockInfoAll.find(aux);
        if (retval1 == BlockInfoAll.end()) {
          BlockInfo *dumm = new BlockInfo();
          const int TwoPower = 1 << m;
          const double h0 =
              (maxextent / std::max(NX * _BS_, std::max(NY * _BS_, NZ * 1)));
          const double h = h0 / TwoPower;
          double origin[3];
          int i, j, k;
          BlockInfo::inverse(n, m, i, j);
          k = 0;
          origin[0] = i * _BS_ * h;
          origin[1] = j * _BS_ * h;
          origin[2] = k * 1 * h;
          dumm->setup(m, h, origin, n);
          BlockInfoAll[aux] = dumm;
        }
      }
      return getBlockInfoAll(m, n);
    }
  }
  virtual int get_world_size() const { return 1; }
  virtual void UpdateBoundary(bool) {}
};
struct StencilInfo {
  int sx;
  int sy;
  int sz;
  int ex;
  int ey;
  int ez;
  std::vector<int> selcomponents;
  bool tensorial;
  StencilInfo() {}
  StencilInfo(int _sx, int _sy, int _sz, int _ex, int _ey, int _ez,
              bool _tensorial, const std::vector<int> &components)
      : sx(_sx), sy(_sy), sz(_sz), ex(_ex), ey(_ey), ez(_ez),
        selcomponents(components), tensorial(_tensorial) {
    assert(selcomponents.size() > 0);
    if (!isvalid()) {
      std::cout << "Stencilinfo instance not valid. Aborting\n";
      abort();
    }
  }
  StencilInfo(const StencilInfo &c)
      : sx(c.sx), sy(c.sy), sz(c.sz), ex(c.ex), ey(c.ey), ez(c.ez),
        selcomponents(c.selcomponents), tensorial(c.tensorial) {}
  std::vector<int> _all() const {
    int extra[] = {sx, sy, sz, ex, ey, ez, (int)tensorial};
    std::vector<int> all(selcomponents);
    all.insert(all.end(), extra, extra + sizeof(extra) / sizeof(int));
    return all;
  }
  bool operator<(StencilInfo s) const {
    std::vector<int> me = _all(), you = s._all();
    const int N = std::min(me.size(), you.size());
    for (int i = 0; i < N; ++i)
      if (me[i] < you[i])
        return true;
      else if (me[i] > you[i])
        return false;
    return me.size() < you.size();
  }
  bool isvalid() const {
    const bool not0 = selcomponents.size() == 0;
    const bool not1 = sx > 0 || ex <= 0 || sx > ex;
    const bool not2 = sy > 0 || ey <= 0 || sy > ey;
    const bool not3 = sz > 0 || ez <= 0 || sz > ez;
    return !(not0 || not1 || not2 || not3);
  }
};
void pack(Real *srcbase, Real *dst, unsigned int gptfloats,
          int *selected_components, int ncomponents, int xstart, int ystart,
          int zstart, int xend, int yend, int zend, int BSX, int BSY) {
  if (gptfloats == 1) {
    const int mod = (xend - xstart) % 4;
    for (int idst = 0, iz = zstart; iz < zend; ++iz)
      for (int iy = ystart; iy < yend; ++iy) {
        for (int ix = xstart; ix < xend - mod; ix += 4, idst += 4) {
          dst[idst + 0] = srcbase[ix + 0 + BSX * (iy + BSY * iz)];
          dst[idst + 1] = srcbase[ix + 1 + BSX * (iy + BSY * iz)];
          dst[idst + 2] = srcbase[ix + 2 + BSX * (iy + BSY * iz)];
          dst[idst + 3] = srcbase[ix + 3 + BSX * (iy + BSY * iz)];
        }
        for (int ix = xend - mod; ix < xend; ix++, idst++) {
          dst[idst] = srcbase[ix + BSX * (iy + BSY * iz)];
        }
      }
  } else {
    for (int idst = 0, iz = zstart; iz < zend; ++iz)
      for (int iy = ystart; iy < yend; ++iy)
        for (int ix = xstart; ix < xend; ++ix) {
          const Real *src = srcbase + gptfloats * (ix + BSX * (iy + BSY * iz));
          for (int ic = 0; ic < ncomponents; ic++, idst++)
            dst[idst] = src[selected_components[ic]];
        }
  }
}
static void unpack_subregion(Real *pack, Real *dstbase, unsigned int gptfloats,
                             int *selected_components, int ncomponents,
                             int srcxstart, int srcystart, int srczstart,
                             int LX, int LY, int dstxstart, int dstystart,
                             int dstzstart, int dstxend, int dstyend,
                             int dstzend, int xsize, int ysize) {
  if (gptfloats == 1) {
    const int mod = (dstxend - dstxstart) % 4;
    for (int zd = dstzstart; zd < dstzend; ++zd)
      for (int yd = dstystart; yd < dstyend; ++yd) {
        const int offset = -dstxstart + srcxstart +
                           LX * (yd - dstystart + srcystart +
                                 LY * (zd - dstzstart + srczstart));
        const int offset_dst = xsize * (yd + ysize * zd);
        for (int xd = dstxstart; xd < dstxend - mod; xd += 4) {
          dstbase[xd + 0 + offset_dst] = pack[xd + 0 + offset];
          dstbase[xd + 1 + offset_dst] = pack[xd + 1 + offset];
          dstbase[xd + 2 + offset_dst] = pack[xd + 2 + offset];
          dstbase[xd + 3 + offset_dst] = pack[xd + 3 + offset];
        }
        for (int xd = dstxend - mod; xd < dstxend; ++xd) {
          dstbase[xd + offset_dst] = pack[xd + offset];
        }
      }
  } else {
    for (int zd = dstzstart; zd < dstzend; ++zd)
      for (int yd = dstystart; yd < dstyend; ++yd)
        for (int xd = dstxstart; xd < dstxend; ++xd) {
          Real *const dst =
              dstbase + gptfloats * (xd + xsize * (yd + ysize * zd));
          const Real *src =
              pack + ncomponents * (xd - dstxstart + srcxstart +
                                    LX * (yd - dstystart + srcystart +
                                          LY * (zd - dstzstart + srczstart)));
          for (int c = 0; c < ncomponents; ++c)
            dst[selected_components[c]] = src[c];
        }
  }
}
struct Interface {
  BlockInfo *infos[2];
  int icode[2];
  bool CoarseStencil;
  bool ToBeKept;
  int dis;
  Interface(BlockInfo &i0, BlockInfo &i1, const int a_icode0,
            const int a_icode1) {
    infos[0] = &i0;
    infos[1] = &i1;
    icode[0] = a_icode0;
    icode[1] = a_icode1;
    CoarseStencil = false;
    ToBeKept = true;
    dis = 0;
  }
  bool operator<(const Interface &other) const {
    if (infos[0]->blockID_2 == other.infos[0]->blockID_2) {
      if (icode[0] == other.icode[0]) {
        if (infos[1]->blockID_2 == other.infos[1]->blockID_2) {
          return (icode[1] < other.icode[1]);
        }
        return (infos[1]->blockID_2 < other.infos[1]->blockID_2);
      }
      return (icode[0] < other.icode[0]);
    }
    return (infos[0]->blockID_2 < other.infos[0]->blockID_2);
  }
};
struct MyRange {
  std::vector<int> removedIndices;
  int index;
  int sx;
  int sy;
  int sz;
  int ex;
  int ey;
  int ez;
  bool needed{true};
  bool avg_down{true};
  bool contains(MyRange &r) const {
    if (avg_down != r.avg_down)
      return false;
    int V = (ez - sz) * (ey - sy) * (ex - sx);
    int Vr = (r.ez - r.sz) * (r.ey - r.sy) * (r.ex - r.sx);
    return (sx <= r.sx && r.ex <= ex) && (sy <= r.sy && r.ey <= ey) &&
           (sz <= r.sz && r.ez <= ez) && (Vr < V);
  }
  void Remove(const MyRange &other) {
    size_t s = removedIndices.size();
    removedIndices.resize(s + other.removedIndices.size());
    for (size_t i = 0; i < other.removedIndices.size(); i++)
      removedIndices[s + i] = other.removedIndices[i];
  }
};
struct UnPackInfo {
  int offset;
  int lx;
  int ly;
  int lz;
  int srcxstart;
  int srcystart;
  int srczstart;
  int LX;
  int LY;
  int CoarseVersionOffset;
  int CoarseVersionLX;
  int CoarseVersionLY;
  int CoarseVersionsrcxstart;
  int CoarseVersionsrcystart;
  int CoarseVersionsrczstart;
  int level;
  int icode;
  int rank;
  int index_0;
  int index_1;
  int index_2;
  long long IDreceiver;
};
struct StencilManager {
  const StencilInfo stencil;
  const StencilInfo Cstencil;
  int nX;
  int nY;
  int nZ;
  int sLength[3 * 27 * 3];
  std::array<MyRange, 3 * 27> AllStencils;
  MyRange Coarse_Range;
  StencilManager(StencilInfo a_stencil, StencilInfo a_Cstencil, int a_nX,
                 int a_nY, int a_nZ)
      : stencil(a_stencil), Cstencil(a_Cstencil), nX(a_nX), nY(a_nY), nZ(a_nZ) {
    const int sC[3] = {(stencil.sx - 1) / 2 + Cstencil.sx,
                       (stencil.sy - 1) / 2 + Cstencil.sy,
                       (stencil.sz - 1) / 2 + Cstencil.sz};
    const int eC[3] = {stencil.ex / 2 + Cstencil.ex,
                       stencil.ey / 2 + Cstencil.ey,
                       stencil.ez / 2 + Cstencil.ez};
    for (int icode = 0; icode < 27; icode++) {
      const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                           (icode / 9) % 3 - 1};
      MyRange &range0 = AllStencils[icode];
      range0.sx = code[0] < 1 ? (code[0] < 0 ? nX + stencil.sx : 0) : 0;
      range0.sy = code[1] < 1 ? (code[1] < 0 ? nY + stencil.sy : 0) : 0;
      range0.sz = code[2] < 1 ? (code[2] < 0 ? nZ + stencil.sz : 0) : 0;
      range0.ex = code[0] < 1 ? nX : stencil.ex - 1;
      range0.ey = code[1] < 1 ? nY : stencil.ey - 1;
      range0.ez = code[2] < 1 ? nZ : stencil.ez - 1;
      sLength[3 * icode + 0] = range0.ex - range0.sx;
      sLength[3 * icode + 1] = range0.ey - range0.sy;
      sLength[3 * icode + 2] = range0.ez - range0.sz;
      MyRange &range1 = AllStencils[icode + 27];
      range1.sx = code[0] < 1 ? (code[0] < 0 ? nX + 2 * stencil.sx : 0) : 0;
      range1.sy = code[1] < 1 ? (code[1] < 0 ? nY + 2 * stencil.sy : 0) : 0;
      range1.sz = code[2] < 1 ? (code[2] < 0 ? nZ + 2 * stencil.sz : 0) : 0;
      range1.ex = code[0] < 1 ? nX : 2 * (stencil.ex - 1);
      range1.ey = code[1] < 1 ? nY : 2 * (stencil.ey - 1);
      range1.ez = code[2] < 1 ? nZ : 2 * (stencil.ez - 1);
      sLength[3 * (icode + 27) + 0] = (range1.ex - range1.sx) / 2;
      sLength[3 * (icode + 27) + 1] = (range1.ey - range1.sy) / 2;
      sLength[3 * (icode + 27) + 2] = 1;
      MyRange &range2 = AllStencils[icode + 2 * 27];
      range2.sx = code[0] < 1 ? (code[0] < 0 ? nX / 2 + sC[0] : 0) : 0;
      range2.sy = code[1] < 1 ? (code[1] < 0 ? nY / 2 + sC[1] : 0) : 0;
      range2.ex = code[0] < 1 ? nX / 2 : eC[0] - 1;
      range2.ey = code[1] < 1 ? nY / 2 : eC[1] - 1;
      range2.sz = 0;
      range2.ez = 1;
      sLength[3 * (icode + 2 * 27) + 0] = range2.ex - range2.sx;
      sLength[3 * (icode + 2 * 27) + 1] = range2.ey - range2.sy;
      sLength[3 * (icode + 2 * 27) + 2] = range2.ez - range2.sz;
    }
  }
  void CoarseStencilLength(const int icode, int *L) const {
    L[0] = sLength[3 * (icode + 2 * 27) + 0];
    L[1] = sLength[3 * (icode + 2 * 27) + 1];
    L[2] = sLength[3 * (icode + 2 * 27) + 2];
  }
  void DetermineStencilLength(const int level_sender, const int level_receiver,
                              const int icode, int *L) {
    if (level_sender == level_receiver) {
      L[0] = sLength[3 * icode + 0];
      L[1] = sLength[3 * icode + 1];
      L[2] = sLength[3 * icode + 2];
    } else if (level_sender > level_receiver) {
      L[0] = sLength[3 * (icode + 27) + 0];
      L[1] = sLength[3 * (icode + 27) + 1];
      L[2] = sLength[3 * (icode + 27) + 2];
    } else {
      L[0] = sLength[3 * (icode + 2 * 27) + 0];
      L[1] = sLength[3 * (icode + 2 * 27) + 1];
      L[2] = sLength[3 * (icode + 2 * 27) + 2];
    }
  }
  MyRange &DetermineStencil(const Interface &f, bool CoarseVersion = false) {
    if (CoarseVersion) {
      AllStencils[f.icode[1] + 2 * 27].needed = true;
      return AllStencils[f.icode[1] + 2 * 27];
    } else {
      if (f.infos[0]->level == f.infos[1]->level) {
        AllStencils[f.icode[1]].needed = true;
        return AllStencils[f.icode[1]];
      } else if (f.infos[0]->level > f.infos[1]->level) {
        AllStencils[f.icode[1] + 27].needed = true;
        return AllStencils[f.icode[1] + 27];
      } else {
        Coarse_Range.needed = true;
        const int code[3] = {f.icode[1] % 3 - 1, (f.icode[1] / 3) % 3 - 1,
                             (f.icode[1] / 9) % 3 - 1};
        const int s[3] = {
            code[0] < 1
                ? (code[0] < 0 ? ((stencil.sx - 1) / 2 + Cstencil.sx) : 0)
                : nX / 2,
            code[1] < 1
                ? (code[1] < 0 ? ((stencil.sy - 1) / 2 + Cstencil.sy) : 0)
                : nY / 2,
            code[2] < 1
                ? (code[2] < 0 ? ((stencil.sz - 1) / 2 + Cstencil.sz) : 0)
                : nZ / 2};
        const int e[3] = {
            code[0] < 1 ? (code[0] < 0 ? 0 : nX / 2)
                        : nX / 2 + stencil.ex / 2 + Cstencil.ex - 1,
            code[1] < 1 ? (code[1] < 0 ? 0 : nY / 2)
                        : nY / 2 + stencil.ey / 2 + Cstencil.ey - 1,
            code[2] < 1 ? (code[2] < 0 ? 0 : nZ / 2)
                        : nZ / 2 + stencil.ez / 2 + Cstencil.ez - 1};
        const int base[3] = {(f.infos[1]->index[0] + code[0]) % 2,
                             (f.infos[1]->index[1] + code[1]) % 2,
                             (f.infos[1]->index[2] + code[2]) % 2};
        int Cindex_true[3];
        for (int d = 0; d < 3; d++)
          Cindex_true[d] = f.infos[1]->index[d] + code[d];
        int CoarseEdge[3];
        CoarseEdge[0] = (code[0] == 0) ? 0
                        : (((f.infos[1]->index[0] % 2 == 0) &&
                            (Cindex_true[0] > f.infos[1]->index[0])) ||
                           ((f.infos[1]->index[0] % 2 == 1) &&
                            (Cindex_true[0] < f.infos[1]->index[0])))
                            ? 1
                            : 0;
        CoarseEdge[1] = (code[1] == 0) ? 0
                        : (((f.infos[1]->index[1] % 2 == 0) &&
                            (Cindex_true[1] > f.infos[1]->index[1])) ||
                           ((f.infos[1]->index[1] % 2 == 1) &&
                            (Cindex_true[1] < f.infos[1]->index[1])))
                            ? 1
                            : 0;
        CoarseEdge[2] = (code[2] == 0) ? 0
                        : (((f.infos[1]->index[2] % 2 == 0) &&
                            (Cindex_true[2] > f.infos[1]->index[2])) ||
                           ((f.infos[1]->index[2] % 2 == 1) &&
                            (Cindex_true[2] < f.infos[1]->index[2])))
                            ? 1
                            : 0;
        Coarse_Range.sx = s[0] + std::max(code[0], 0) * nX / 2 +
                          (1 - abs(code[0])) * base[0] * nX / 2 - code[0] * nX +
                          CoarseEdge[0] * code[0] * nX / 2;
        Coarse_Range.sy = s[1] + std::max(code[1], 0) * nY / 2 +
                          (1 - abs(code[1])) * base[1] * nY / 2 - code[1] * nY +
                          CoarseEdge[1] * code[1] * nY / 2;
        Coarse_Range.sz = 0;
        Coarse_Range.ex = e[0] + std::max(code[0], 0) * nX / 2 +
                          (1 - abs(code[0])) * base[0] * nX / 2 - code[0] * nX +
                          CoarseEdge[0] * code[0] * nX / 2;
        Coarse_Range.ey = e[1] + std::max(code[1], 0) * nY / 2 +
                          (1 - abs(code[1])) * base[1] * nY / 2 - code[1] * nY +
                          CoarseEdge[1] * code[1] * nY / 2;
        Coarse_Range.ez = 1;
        return Coarse_Range;
      }
    }
  }
  void __FixDuplicates(const Interface &f, const Interface &f_dup, int lx,
                       int ly, int lz, int lx_dup, int ly_dup, int lz_dup,
                       int &sx, int &sy, int &sz) {
    const BlockInfo &receiver = *f.infos[1];
    const BlockInfo &receiver_dup = *f_dup.infos[1];
    if (receiver.level >= receiver_dup.level) {
      int icode_dup = f_dup.icode[1];
      const int code_dup[3] = {icode_dup % 3 - 1, (icode_dup / 3) % 3 - 1,
                               (icode_dup / 9) % 3 - 1};
      sx = (lx == lx_dup || code_dup[0] != -1) ? 0 : lx - lx_dup;
      sy = (ly == ly_dup || code_dup[1] != -1) ? 0 : ly - ly_dup;
      sz = (lz == lz_dup || code_dup[2] != -1) ? 0 : lz - lz_dup;
    } else {
      MyRange &range = DetermineStencil(f);
      MyRange &range_dup = DetermineStencil(f_dup);
      sx = range_dup.sx - range.sx;
      sy = range_dup.sy - range.sy;
      sz = range_dup.sz - range.sz;
    }
  }
  void __FixDuplicates2(const Interface &f, const Interface &f_dup, int &sx,
                        int &sy, int &sz) {
    if (f.infos[0]->level != f.infos[1]->level ||
        f_dup.infos[0]->level != f_dup.infos[1]->level)
      return;
    MyRange &range = DetermineStencil(f, true);
    MyRange &range_dup = DetermineStencil(f_dup, true);
    sx = range_dup.sx - range.sx;
    sy = range_dup.sy - range.sy;
    sz = range_dup.sz - range.sz;
  }
};
struct HaloBlockGroup {
  std::vector<BlockInfo *> myblocks;
  std::set<int> myranks;
  bool ready = false;
};
template <typename TGrid> struct SynchronizerMPI_AMR {
  int rank;
  int size;
  StencilInfo stencil;
  StencilInfo Cstencil;
  TGrid *grid;
  int nX;
  int nY;
  int nZ;
  MPI_Datatype MPIREAL;
  std::vector<BlockInfo *> inner_blocks;
  std::vector<BlockInfo *> halo_blocks;
  std::vector<std::vector<Real>> send_buffer;
  std::vector<std::vector<Real>> recv_buffer;
  std::vector<MPI_Request> requests;
  std::vector<int> send_buffer_size;
  std::vector<int> recv_buffer_size;
  std::set<int> Neighbors;
  std::vector<std::vector<UnPackInfo>> myunpacks;
  StencilManager SM;
  const unsigned int gptfloats;
  const int NC;
  struct PackInfo {
    Real *block;
    Real *pack;
    int sx;
    int sy;
    int sz;
    int ex;
    int ey;
    int ez;
  };
  std::vector<std::vector<PackInfo>> send_packinfos;
  std::vector<std::vector<Interface>> send_interfaces;
  std::vector<std::vector<Interface>> recv_interfaces;
  std::vector<std::vector<int>> ToBeAveragedDown;
  bool use_averages;
  std::unordered_map<std::string, HaloBlockGroup> mapofHaloBlockGroups;
  std::unordered_map<int, MPI_Request *> mapofrequests;
  struct DuplicatesManager {
    struct cube {
      std::vector<MyRange> compass[27];
      void clear() {
        for (int i = 0; i < 27; i++)
          compass[i].clear();
      }
      cube() {}
      std::vector<MyRange *> keepEl() {
        std::vector<MyRange *> retval;
        for (int i = 0; i < 27; i++)
          for (size_t j = 0; j < compass[i].size(); j++)
            if (compass[i][j].needed)
              retval.push_back(&compass[i][j]);
        return retval;
      }
      void __needed(std::vector<int> &v) {
        static constexpr std::array<int, 3> faces_and_edges[18] = {
            {0, 1, 1}, {2, 1, 1}, {1, 0, 1}, {1, 2, 1}, {1, 1, 0}, {1, 1, 2},
            {0, 0, 1}, {0, 2, 1}, {2, 0, 1}, {2, 2, 1}, {1, 0, 0}, {1, 0, 2},
            {1, 2, 0}, {1, 2, 2}, {0, 1, 0}, {0, 1, 2}, {2, 1, 0}, {2, 1, 2}};
        for (auto &f : faces_and_edges)
          if (compass[f[0] + f[1] * 3 + f[2] * 9].size() != 0) {
            bool needme = false;
            auto &me = compass[f[0] + f[1] * 3 + f[2] * 9];
            for (size_t j1 = 0; j1 < me.size(); j1++)
              if (me[j1].needed) {
                needme = true;
                for (size_t j2 = 0; j2 < me.size(); j2++)
                  if (me[j2].needed && me[j2].contains(me[j1])) {
                    me[j1].needed = false;
                    me[j2].removedIndices.push_back(me[j1].index);
                    me[j2].Remove(me[j1]);
                    v.push_back(me[j1].index);
                    break;
                  }
              }
            if (!needme)
              continue;
            const int imax = (f[0] == 1) ? 2 : f[0];
            const int imin = (f[0] == 1) ? 0 : f[0];
            const int jmax = (f[1] == 1) ? 2 : f[1];
            const int jmin = (f[1] == 1) ? 0 : f[1];
            const int kmax = (f[2] == 1) ? 2 : f[2];
            const int kmin = (f[2] == 1) ? 0 : f[2];
            for (int k = kmin; k <= kmax; k++)
              for (int j = jmin; j <= jmax; j++)
                for (int i = imin; i <= imax; i++) {
                  if (i == f[0] && j == f[1] && k == f[2])
                    continue;
                  auto &other = compass[i + j * 3 + k * 9];
                  for (size_t j1 = 0; j1 < other.size(); j1++) {
                    auto &o = other[j1];
                    if (o.needed)
                      for (size_t k1 = 0; k1 < me.size(); k1++) {
                        auto &m = me[k1];
                        if (m.needed && m.contains(o)) {
                          o.needed = false;
                          m.removedIndices.push_back(o.index);
                          m.Remove(o);
                          v.push_back(o.index);
                          break;
                        }
                      }
                  }
                }
          }
      }
    };
    cube C;
    std::vector<int> offsets;
    std::vector<int> offsets_recv;
    SynchronizerMPI_AMR *Synch_ptr;
    std::vector<int> positions;
    std::vector<size_t> sizes;
    DuplicatesManager(SynchronizerMPI_AMR &Synch) {
      positions.resize(Synch.size);
      sizes.resize(Synch.size);
      offsets.resize(Synch.size, 0);
      offsets_recv.resize(Synch.size, 0);
      Synch_ptr = &Synch;
    }
    void Add(const int r, const int index) {
      if (sizes[r] == 0)
        positions[r] = index;
      sizes[r]++;
    }
    void RemoveDuplicates(const int r, std::vector<Interface> &f,
                          int &total_size) {
      if (sizes[r] == 0)
        return;
      bool skip_needed = false;
      const int nc = Synch_ptr->stencil.selcomponents.size();
      std::sort(f.begin() + positions[r], f.begin() + sizes[r] + positions[r]);
      C.clear();
      for (size_t i = 0; i < sizes[r]; i++) {
        C.compass[f[i + positions[r]].icode[0]].push_back(
            Synch_ptr->SM.DetermineStencil(f[i + positions[r]]));
        C.compass[f[i + positions[r]].icode[0]].back().index = i + positions[r];
        C.compass[f[i + positions[r]].icode[0]].back().avg_down =
            (f[i + positions[r]].infos[0]->level >
             f[i + positions[r]].infos[1]->level);
        if (skip_needed == false)
          skip_needed = f[i + positions[r]].CoarseStencil;
      }
      if (skip_needed == false) {
        std::vector<int> remEl;
        C.__needed(remEl);
        for (size_t k = 0; k < remEl.size(); k++)
          f[remEl[k]].ToBeKept = false;
      }
      int L[3] = {0, 0, 0};
      int Lc[3] = {0, 0, 0};
      for (auto &i : C.keepEl()) {
        const int k = i->index;
        Synch_ptr->SM.DetermineStencilLength(
            f[k].infos[0]->level, f[k].infos[1]->level, f[k].icode[1], L);
        const int V = L[0] * L[1] * L[2];
        total_size += V;
        f[k].dis = offsets[r];
        if (f[k].CoarseStencil) {
          Synch_ptr->SM.CoarseStencilLength(f[k].icode[1], Lc);
          const int Vc = Lc[0] * Lc[1] * Lc[2];
          total_size += Vc;
          offsets[r] += Vc * nc;
        }
        offsets[r] += V * nc;
        for (size_t kk = 0; kk < (*i).removedIndices.size(); kk++)
          f[i->removedIndices[kk]].dis = f[k].dis;
      }
    }
    void RemoveDuplicates_recv(std::vector<Interface> &f, int &total_size,
                               const int otherrank, const size_t start,
                               const size_t finish) {
      bool skip_needed = false;
      const int nc = Synch_ptr->stencil.selcomponents.size();
      C.clear();
      for (size_t i = start; i < finish; i++) {
        C.compass[f[i].icode[0]].push_back(
            Synch_ptr->SM.DetermineStencil(f[i]));
        C.compass[f[i].icode[0]].back().index = i;
        C.compass[f[i].icode[0]].back().avg_down =
            (f[i].infos[0]->level > f[i].infos[1]->level);
        if (skip_needed == false)
          skip_needed = f[i].CoarseStencil;
      }
      if (skip_needed == false) {
        std::vector<int> remEl;
        C.__needed(remEl);
        for (size_t k = 0; k < remEl.size(); k++)
          f[remEl[k]].ToBeKept = false;
      }
      for (auto &i : C.keepEl()) {
        const int k = i->index;
        int L[3] = {0, 0, 0};
        int Lc[3] = {0, 0, 0};
        Synch_ptr->SM.DetermineStencilLength(
            f[k].infos[0]->level, f[k].infos[1]->level, f[k].icode[1], L);
        const int V = L[0] * L[1] * L[2];
        int Vc = 0;
        total_size += V;
        f[k].dis = offsets_recv[otherrank];
        UnPackInfo info = {f[k].dis,
                           L[0],
                           L[1],
                           L[2],
                           0,
                           0,
                           0,
                           L[0],
                           L[1],
                           -1,
                           0,
                           0,
                           0,
                           0,
                           0,
                           f[k].infos[0]->level,
                           f[k].icode[1],
                           otherrank,
                           f[k].infos[0]->index[0],
                           f[k].infos[0]->index[1],
                           f[k].infos[0]->index[2],
                           f[k].infos[1]->blockID_2};
        if (f[k].CoarseStencil) {
          Synch_ptr->SM.CoarseStencilLength(f[k].icode[1], Lc);
          Vc = Lc[0] * Lc[1] * Lc[2];
          total_size += Vc;
          offsets_recv[otherrank] += Vc * nc;
          info.CoarseVersionOffset = V * nc;
          info.CoarseVersionLX = Lc[0];
          info.CoarseVersionLY = Lc[1];
        }
        offsets_recv[otherrank] += V * nc;
        Synch_ptr->myunpacks[f[k].infos[1]->halo_block_id].push_back(info);
        for (size_t kk = 0; kk < (*i).removedIndices.size(); kk++) {
          const int remEl1 = i->removedIndices[kk];
          Synch_ptr->SM.DetermineStencilLength(f[remEl1].infos[0]->level,
                                               f[remEl1].infos[1]->level,
                                               f[remEl1].icode[1], &L[0]);
          int srcx, srcy, srcz;
          Synch_ptr->SM.__FixDuplicates(f[k], f[remEl1], info.lx, info.ly,
                                        info.lz, L[0], L[1], L[2], srcx, srcy,
                                        srcz);
          int Csrcx = 0;
          int Csrcy = 0;
          int Csrcz = 0;
          if (f[k].CoarseStencil)
            Synch_ptr->SM.__FixDuplicates2(f[k], f[remEl1], Csrcx, Csrcy,
                                           Csrcz);
          Synch_ptr->myunpacks[f[remEl1].infos[1]->halo_block_id].push_back(
              {info.offset,
               L[0],
               L[1],
               L[2],
               srcx,
               srcy,
               srcz,
               info.LX,
               info.LY,
               info.CoarseVersionOffset,
               info.CoarseVersionLX,
               info.CoarseVersionLY,
               Csrcx,
               Csrcy,
               Csrcz,
               f[remEl1].infos[0]->level,
               f[remEl1].icode[1],
               otherrank,
               f[remEl1].infos[0]->index[0],
               f[remEl1].infos[0]->index[1],
               f[remEl1].infos[0]->index[2],
               f[remEl1].infos[1]->blockID_2});
          f[remEl1].dis = info.offset;
        }
      }
    }
  };
  bool UseCoarseStencil(const Interface &f) {
    BlockInfo &a = *f.infos[0];
    BlockInfo &b = *f.infos[1];
    if (a.level == 0 || (!use_averages))
      return false;
    int imin[3];
    int imax[3];
    const int aux = 1 << a.level;
    const bool periodic[3] = {grid->xperiodic, grid->yperiodic,
                              grid->zperiodic};
    const int blocks[3] = {grid->getMaxBlocks()[0] * aux - 1,
                           grid->getMaxBlocks()[1] * aux - 1,
                           grid->getMaxBlocks()[2] * aux - 1};
    for (int d = 0; d < 3; d++) {
      imin[d] = (a.index[d] < b.index[d]) ? 0 : -1;
      imax[d] = (a.index[d] > b.index[d]) ? 0 : +1;
      if (periodic[d]) {
        if (a.index[d] == 0 && b.index[d] == blocks[d])
          imin[d] = -1;
        if (b.index[d] == 0 && a.index[d] == blocks[d])
          imax[d] = +1;
      } else {
        if (a.index[d] == 0 && b.index[d] == 0)
          imin[d] = 0;
        if (a.index[d] == blocks[d] && b.index[d] == blocks[d])
          imax[d] = 0;
      }
    }
    bool retval = false;
    for (int i2 = imin[2]; i2 <= imax[2]; i2++)
      for (int i1 = imin[1]; i1 <= imax[1]; i1++)
        for (int i0 = imin[0]; i0 <= imax[0]; i0++) {
          if ((grid->Tree(a.level, a.Znei_(i0, i1, i2))).CheckCoarser()) {
            retval = true;
            break;
          }
        }
    return retval;
  }
  void AverageDownAndFill(Real *__restrict__ dst, const BlockInfo *const info,
                          const int code[3]) {
    const int s[3] = {code[0] < 1 ? (code[0] < 0 ? stencil.sx : 0) : nX,
                      code[1] < 1 ? (code[1] < 0 ? stencil.sy : 0) : nY,
                      code[2] < 1 ? (code[2] < 0 ? stencil.sz : 0) : nZ};
    const int e[3] = {
        code[0] < 1 ? (code[0] < 0 ? 0 : nX) : nX + stencil.ex - 1,
        code[1] < 1 ? (code[1] < 0 ? 0 : nY) : nY + stencil.ey - 1,
        code[2] < 1 ? (code[2] < 0 ? 0 : nZ) : nZ + stencil.ez - 1};
    Real *src = (Real *)(*info).ptrBlock;
    const int xStep = (code[0] == 0) ? 2 : 1;
    const int yStep = (code[1] == 0) ? 2 : 1;
    int pos = 0;
    for (int iy = s[1]; iy < e[1]; iy += yStep) {
      const int YY = (abs(code[1]) == 1)
                         ? 2 * (iy - code[1] * nY) + std::min(0, code[1]) * nY
                         : iy;
      for (int ix = s[0]; ix < e[0]; ix += xStep) {
        const int XX = (abs(code[0]) == 1)
                           ? 2 * (ix - code[0] * nX) + std::min(0, code[0]) * nX
                           : ix;
        for (int c = 0; c < NC; c++) {
          int comp = stencil.selcomponents[c];
          dst[pos] =
              0.25 * (((*(src + gptfloats * (XX + (YY)*nX) + comp)) +
                       (*(src + gptfloats * (XX + 1 + (YY + 1) * nX) + comp))) +
                      ((*(src + gptfloats * (XX + (YY + 1) * nX) + comp)) +
                       (*(src + gptfloats * (XX + 1 + (YY)*nX) + comp))));
          pos++;
        }
      }
    }
  }
  void AverageDownAndFill2(Real *dst, const BlockInfo *const info,
                           const int code[3]) {
    const int eC[3] = {(stencil.ex) / 2 + Cstencil.ex,
                       (stencil.ey) / 2 + Cstencil.ey,
                       (stencil.ez) / 2 + Cstencil.ez};
    const int sC[3] = {(stencil.sx - 1) / 2 + Cstencil.sx,
                       (stencil.sy - 1) / 2 + Cstencil.sy,
                       (stencil.sz - 1) / 2 + Cstencil.sz};
    const int s[3] = {code[0] < 1 ? (code[0] < 0 ? sC[0] : 0) : nX / 2,
                      code[1] < 1 ? (code[1] < 0 ? sC[1] : 0) : nY / 2,
                      code[2] < 1 ? (code[2] < 0 ? sC[2] : 0) : nZ / 2};
    const int e[3] = {
        code[0] < 1 ? (code[0] < 0 ? 0 : nX / 2) : nX / 2 + eC[0] - 1,
        code[1] < 1 ? (code[1] < 0 ? 0 : nY / 2) : nY / 2 + eC[1] - 1,
        code[2] < 1 ? (code[2] < 0 ? 0 : nZ / 2) : nZ / 2 + eC[2] - 1};
    Real *src = (Real *)(*info).ptrBlock;
    int pos = 0;
    for (int iy = s[1]; iy < e[1]; iy++) {
      const int YY = 2 * (iy - s[1]) + s[1] + std::max(code[1], 0) * nY / 2 -
                     code[1] * nY + std::min(0, code[1]) * (e[1] - s[1]);
      for (int ix = s[0]; ix < e[0]; ix++) {
        const int XX = 2 * (ix - s[0]) + s[0] + std::max(code[0], 0) * nX / 2 -
                       code[0] * nX + std::min(0, code[0]) * (e[0] - s[0]);
        for (int c = 0; c < NC; c++) {
          int comp = stencil.selcomponents[c];
          dst[pos] =
              0.25 * (((*(src + gptfloats * (XX + (YY)*nX) + comp)) +
                       (*(src + gptfloats * (XX + 1 + (YY + 1) * nX) + comp))) +
                      ((*(src + gptfloats * (XX + (YY + 1) * nX) + comp)) +
                       (*(src + gptfloats * (XX + 1 + (YY)*nX) + comp))));
          pos++;
        }
      }
    }
  }
  std::string EncodeSet(const std::set<int> &ranks) {
    std::string retval;
    for (auto r : ranks) {
      std::stringstream ss;
      ss << std::setw(size) << std::setfill('0') << r;
      std::string s = ss.str();
      retval += s;
    }
    return retval;
  }
  void _Setup() {
    Neighbors.clear();
    inner_blocks.clear();
    halo_blocks.clear();
    for (int r = 0; r < size; r++) {
      send_interfaces[r].clear();
      recv_interfaces[r].clear();
      send_buffer_size[r] = 0;
    }
    for (size_t i = 0; i < myunpacks.size(); i++)
      myunpacks[i].clear();
    myunpacks.clear();
    DuplicatesManager DM(*(this));
    for (BlockInfo &info : grid->m_vInfo) {
      info.halo_block_id = -1;
      const bool xskin =
          info.index[0] == 0 ||
          info.index[0] == ((grid->getMaxBlocks()[0] << info.level) - 1);
      const bool yskin =
          info.index[1] == 0 ||
          info.index[1] == ((grid->getMaxBlocks()[1] << info.level) - 1);
      const bool zskin =
          info.index[2] == 0 ||
          info.index[2] == ((grid->getMaxBlocks()[2] << info.level) - 1);
      const int xskip = info.index[0] == 0 ? -1 : 1;
      const int yskip = info.index[1] == 0 ? -1 : 1;
      const int zskip = info.index[2] == 0 ? -1 : 1;
      bool isInner = true;
      std::vector<int> ToBeChecked;
      bool Coarsened = false;
      for (int icode = 0; icode < 27; icode++) {
        if (icode == 1 * 1 + 3 * 1 + 9 * 1)
          continue;
        const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                             (icode / 9) % 3 - 1};
        if (code[2] != 0)
          continue;
        if (!grid->xperiodic && code[0] == xskip && xskin)
          continue;
        if (!grid->yperiodic && code[1] == yskip && yskin)
          continue;
        if (!grid->zperiodic && code[2] == zskip && zskin)
          continue;
        const TreePosition &infoNeiTree =
            grid->Tree(info.level, info.Znei_(code[0], code[1], code[2]));
        if (infoNeiTree.Exists() && infoNeiTree.rank() != rank) {
          isInner = false;
          Neighbors.insert(infoNeiTree.rank());
          BlockInfo &infoNei = grid->getBlockInfoAll(
              info.level, info.Znei_(code[0], code[1], code[2]));
          const int icode2 =
              (-code[0] + 1) + (-code[1] + 1) * 3 + (-code[2] + 1) * 9;
          send_interfaces[infoNeiTree.rank()].push_back(
              {info, infoNei, icode, icode2});
          recv_interfaces[infoNeiTree.rank()].push_back(
              {infoNei, info, icode2, icode});
          ToBeChecked.push_back(infoNeiTree.rank());
          ToBeChecked.push_back(
              (int)send_interfaces[infoNeiTree.rank()].size() - 1);
          ToBeChecked.push_back(
              (int)recv_interfaces[infoNeiTree.rank()].size() - 1);
          DM.Add(infoNeiTree.rank(),
                 (int)send_interfaces[infoNeiTree.rank()].size() - 1);
        } else if (infoNeiTree.CheckCoarser()) {
          Coarsened = true;
          BlockInfo &infoNei = grid->getBlockInfoAll(
              info.level, info.Znei_(code[0], code[1], code[2]));
          const int infoNeiCoarserrank =
              grid->Tree(info.level - 1, infoNei.Zparent).rank();
          if (infoNeiCoarserrank != rank) {
            isInner = false;
            Neighbors.insert(infoNeiCoarserrank);
            BlockInfo &infoNeiCoarser =
                grid->getBlockInfoAll(infoNei.level - 1, infoNei.Zparent);
            const int icode2 =
                (-code[0] + 1) + (-code[1] + 1) * 3 + (-code[2] + 1) * 9;
            const int Bmax[3] = {grid->getMaxBlocks()[0] << (info.level - 1),
                                 grid->getMaxBlocks()[1] << (info.level - 1),
                                 grid->getMaxBlocks()[2] << (info.level - 1)};
            const int test_idx[3] = {
                (infoNeiCoarser.index[0] - code[0] + Bmax[0]) % Bmax[0],
                (infoNeiCoarser.index[1] - code[1] + Bmax[1]) % Bmax[1],
                (infoNeiCoarser.index[2] - code[2] + Bmax[2]) % Bmax[2]};
            if (info.index[0] / 2 == test_idx[0] &&
                info.index[1] / 2 == test_idx[1] &&
                info.index[2] / 2 == test_idx[2]) {
              send_interfaces[infoNeiCoarserrank].push_back(
                  {info, infoNeiCoarser, icode, icode2});
              recv_interfaces[infoNeiCoarserrank].push_back(
                  {infoNeiCoarser, info, icode2, icode});
              DM.Add(infoNeiCoarserrank,
                     (int)send_interfaces[infoNeiCoarserrank].size() - 1);
              if (abs(code[0]) + abs(code[1]) + abs(code[2]) == 1) {
                const int d0 = abs(code[1] + 2 * code[2]);
                const int d1 = (d0 + 1) % 3;
                const int d2 = (d0 + 2) % 3;
                int code3[3];
                code3[d0] = code[d0];
                code3[d1] = -2 * (info.index[d1] % 2) + 1;
                code3[d2] = -2 * (info.index[d2] % 2) + 1;
                const int icode3 =
                    (code3[0] + 1) + (code3[1] + 1) * 3 + (code3[2] + 1) * 9;
                int code4[3];
                code4[d0] = code[d0];
                code4[d1] = code3[d1];
                code4[d2] = 0;
                const int icode4 =
                    (code4[0] + 1) + (code4[1] + 1) * 3 + (code4[2] + 1) * 9;
                int code5[3];
                code5[d0] = code[d0];
                code5[d1] = 0;
                code5[d2] = code3[d2];
                const int icode5 =
                    (code5[0] + 1) + (code5[1] + 1) * 3 + (code5[2] + 1) * 9;
                if (code3[2] == 0)
                  recv_interfaces[infoNeiCoarserrank].push_back(
                      {infoNeiCoarser, info, icode2, icode3});
                if (code4[2] == 0)
                  recv_interfaces[infoNeiCoarserrank].push_back(
                      {infoNeiCoarser, info, icode2, icode4});
                if (code5[2] == 0)
                  recv_interfaces[infoNeiCoarserrank].push_back(
                      {infoNeiCoarser, info, icode2, icode5});
              }
            }
          }
        } else if (infoNeiTree.CheckFiner()) {
          BlockInfo &infoNei = grid->getBlockInfoAll(
              info.level, info.Znei_(code[0], code[1], code[2]));
          int Bstep = 1;
          if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 2))
            Bstep = 3;
          else if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 3))
            Bstep = 4;
          for (int B = 0; B <= 3; B += Bstep) {
            if (Bstep == 1 && B >= 2)
              continue;
            if (Bstep > 1 && B >= 1)
              continue;
            const int temp = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
            const long long nFine =
                infoNei.Zchild[std::max(-code[0], 0) +
                               (B % 2) * std::max(0, 1 - abs(code[0]))]
                              [std::max(-code[1], 0) +
                               temp * std::max(0, 1 - abs(code[1]))]
                              [std::max(-code[2], 0) +
                               (B / 2) * std::max(0, 1 - abs(code[2]))];
            const int infoNeiFinerrank =
                grid->Tree(info.level + 1, nFine).rank();
            if (infoNeiFinerrank != rank) {
              isInner = false;
              Neighbors.insert(infoNeiFinerrank);
              BlockInfo &infoNeiFiner =
                  grid->getBlockInfoAll(info.level + 1, nFine);
              const int icode2 =
                  (-code[0] + 1) + (-code[1] + 1) * 3 + (-code[2] + 1) * 9;
              send_interfaces[infoNeiFinerrank].push_back(
                  {info, infoNeiFiner, icode, icode2});
              recv_interfaces[infoNeiFinerrank].push_back(
                  {infoNeiFiner, info, icode2, icode});
              DM.Add(infoNeiFinerrank,
                     (int)send_interfaces[infoNeiFinerrank].size() - 1);
              if (Bstep == 1) {
                const int d0 = abs(code[1] + 2 * code[2]);
                const int d1 = (d0 + 1) % 3;
                const int d2 = (d0 + 2) % 3;
                int code3[3];
                code3[d0] = -code[d0];
                code3[d1] = -2 * (infoNeiFiner.index[d1] % 2) + 1;
                code3[d2] = -2 * (infoNeiFiner.index[d2] % 2) + 1;
                const int icode3 =
                    (code3[0] + 1) + (code3[1] + 1) * 3 + (code3[2] + 1) * 9;
                int code4[3];
                code4[d0] = -code[d0];
                code4[d1] = code3[d1];
                code4[d2] = 0;
                const int icode4 =
                    (code4[0] + 1) + (code4[1] + 1) * 3 + (code4[2] + 1) * 9;
                int code5[3];
                code5[d0] = -code[d0];
                code5[d1] = 0;
                code5[d2] = code3[d2];
                const int icode5 =
                    (code5[0] + 1) + (code5[1] + 1) * 3 + (code5[2] + 1) * 9;
                if (code3[2] == 0) {
                  send_interfaces[infoNeiFinerrank].push_back(
                      Interface(info, infoNeiFiner, icode, icode3));
                  DM.Add(infoNeiFinerrank,
                         (int)send_interfaces[infoNeiFinerrank].size() - 1);
                }
                if (code4[2] == 0) {
                  send_interfaces[infoNeiFinerrank].push_back(
                      Interface(info, infoNeiFiner, icode, icode4));
                  DM.Add(infoNeiFinerrank,
                         (int)send_interfaces[infoNeiFinerrank].size() - 1);
                }
                if (code5[2] == 0) {
                  send_interfaces[infoNeiFinerrank].push_back(
                      Interface(info, infoNeiFiner, icode, icode5));
                  DM.Add(infoNeiFinerrank,
                         (int)send_interfaces[infoNeiFinerrank].size() - 1);
                }
              }
            }
          }
        }
      }
      if (isInner) {
        info.halo_block_id = -1;
        inner_blocks.push_back(&info);
      } else {
        info.halo_block_id = halo_blocks.size();
        halo_blocks.push_back(&info);
        if (Coarsened) {
          for (size_t j = 0; j < ToBeChecked.size(); j += 3) {
            const int r = ToBeChecked[j];
            const int send = ToBeChecked[j + 1];
            const int recv = ToBeChecked[j + 2];
            const bool tmp = UseCoarseStencil(send_interfaces[r][send]);
            send_interfaces[r][send].CoarseStencil = tmp;
            recv_interfaces[r][recv].CoarseStencil = tmp;
          }
        }
        for (int r = 0; r < size; r++)
          if (DM.sizes[r] > 0) {
            DM.RemoveDuplicates(r, send_interfaces[r], send_buffer_size[r]);
            DM.sizes[r] = 0;
          }
      }
      grid->getBlockInfoAll(info.level, info.Z).halo_block_id =
          info.halo_block_id;
    }
    myunpacks.resize(halo_blocks.size());
    for (int r = 0; r < size; r++) {
      recv_buffer_size[r] = 0;
      std::sort(recv_interfaces[r].begin(), recv_interfaces[r].end());
      size_t counter = 0;
      while (counter < recv_interfaces[r].size()) {
        const long long ID = recv_interfaces[r][counter].infos[0]->blockID_2;
        const size_t start = counter;
        size_t finish = start + 1;
        counter++;
        size_t j;
        for (j = counter; j < recv_interfaces[r].size(); j++) {
          if (recv_interfaces[r][j].infos[0]->blockID_2 == ID)
            finish++;
          else
            break;
        }
        counter = j;
        DM.RemoveDuplicates_recv(recv_interfaces[r], recv_buffer_size[r], r,
                                 start, finish);
      }
      send_buffer[r].resize(send_buffer_size[r] * NC);
      recv_buffer[r].resize(recv_buffer_size[r] * NC);
      send_packinfos[r].clear();
      ToBeAveragedDown[r].clear();
      for (int i = 0; i < (int)send_interfaces[r].size(); i++) {
        const Interface &f = send_interfaces[r][i];
        if (!f.ToBeKept)
          continue;
        if (f.infos[0]->level <= f.infos[1]->level) {
          const MyRange &range = SM.DetermineStencil(f);
          send_packinfos[r].push_back(
              {(Real *)f.infos[0]->ptrBlock, &send_buffer[r][f.dis], range.sx,
               range.sy, range.sz, range.ex, range.ey, range.ez});
          if (f.CoarseStencil) {
            const int V = (range.ex - range.sx) * (range.ey - range.sy) *
                          (range.ez - range.sz);
            ToBeAveragedDown[r].push_back(i);
            ToBeAveragedDown[r].push_back(f.dis + V * NC);
          }
        } else {
          ToBeAveragedDown[r].push_back(i);
          ToBeAveragedDown[r].push_back(f.dis);
        }
      }
    }
    mapofHaloBlockGroups.clear();
    for (auto &info : halo_blocks) {
      const int id = info->halo_block_id;
      UnPackInfo *unpacks = myunpacks[id].data();
      std::set<int> ranks;
      for (size_t jj = 0; jj < myunpacks[id].size(); jj++) {
        const UnPackInfo &unpack = unpacks[jj];
        ranks.insert(unpack.rank);
      }
      auto set_ID = EncodeSet(ranks);
      const auto retval = mapofHaloBlockGroups.find(set_ID);
      if (retval == mapofHaloBlockGroups.end()) {
        HaloBlockGroup temporary;
        temporary.myranks = ranks;
        temporary.myblocks.push_back(info);
        mapofHaloBlockGroups[set_ID] = temporary;
      } else {
        (retval->second).myblocks.push_back(info);
      }
    }
  }
  SynchronizerMPI_AMR(StencilInfo a_stencil, StencilInfo a_Cstencil,
                      TGrid *_grid)
      : stencil(a_stencil), Cstencil(a_Cstencil),
        SM(a_stencil, a_Cstencil, _BS_, _BS_, 1),
        gptfloats(sizeof(typename TGrid::Block::ElementType) / sizeof(Real)),
        NC(a_stencil.selcomponents.size()) {
    grid = _grid;
    use_averages = (grid->FiniteDifferences == false || stencil.tensorial ||
                    stencil.sx < -2 || stencil.sy < -2 || stencil.sz < -2 ||
                    stencil.ex > 3 || stencil.ey > 3 || stencil.ez > 3);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    nX = _BS_;
    nY = _BS_;
    nZ = 1;
    send_interfaces.resize(size);
    recv_interfaces.resize(size);
    send_packinfos.resize(size);
    send_buffer_size.resize(size);
    recv_buffer_size.resize(size);
    send_buffer.resize(size);
    recv_buffer.resize(size);
    ToBeAveragedDown.resize(size);
    std::sort(stencil.selcomponents.begin(), stencil.selcomponents.end());
    if (sizeof(Real) == sizeof(double)) {
      MPIREAL = MPI_DOUBLE;
    } else if (sizeof(Real) == sizeof(long double)) {
      MPIREAL = MPI_LONG_DOUBLE;
    } else {
      MPIREAL = MPI_FLOAT;
      assert(sizeof(Real) == sizeof(float));
    }
  }
  std::vector<BlockInfo *> &avail_inner() { return inner_blocks; }
  std::vector<BlockInfo *> &avail_halo() {
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    return halo_blocks;
  }
  std::vector<BlockInfo *> &avail_halo_nowait() { return halo_blocks; }
  std::vector<BlockInfo *> dummy_vector;
  std::vector<BlockInfo *> &avail_next() {
    bool done = false;
    auto it = mapofHaloBlockGroups.begin();
    while (done == false) {
      done = true;
      it = mapofHaloBlockGroups.begin();
      while (it != mapofHaloBlockGroups.end()) {
        if ((it->second).ready == false) {
          std::set<int> ranks = (it->second).myranks;
          int flag = 0;
          for (auto r : ranks) {
            const auto retval = mapofrequests.find(r);
            MPI_Test(retval->second, &flag, MPI_STATUS_IGNORE);
            if (flag == false)
              break;
          }
          if (flag == 1) {
            (it->second).ready = true;
            return (it->second).myblocks;
          }
        }
        done = done && (it->second).ready;
        it++;
      }
    }
    return dummy_vector;
  }
  void sync() {
    auto it = mapofHaloBlockGroups.begin();
    while (it != mapofHaloBlockGroups.end()) {
      (it->second).ready = false;
      it++;
    }
    const int timestamp = grid->timestamp;
    mapofrequests.clear();
    requests.clear();
    requests.reserve(2 * size);
    for (auto r : Neighbors)
      if (recv_buffer_size[r] > 0) {
        requests.resize(requests.size() + 1);
        mapofrequests[r] = &requests.back();
        MPI_Irecv(&recv_buffer[r][0], recv_buffer_size[r] * NC, MPIREAL, r,
                  timestamp, MPI_COMM_WORLD, &requests.back());
      }
    for (int r = 0; r < size; r++)
      if (send_buffer_size[r] != 0) {
#pragma omp parallel
        {
#pragma omp for
          for (size_t j = 0; j < ToBeAveragedDown[r].size(); j += 2) {
            const int i = ToBeAveragedDown[r][j];
            const int d = ToBeAveragedDown[r][j + 1];
            const Interface &f = send_interfaces[r][i];
            const int code[3] = {-(f.icode[0] % 3 - 1),
                                 -((f.icode[0] / 3) % 3 - 1),
                                 -((f.icode[0] / 9) % 3 - 1)};
            if (f.CoarseStencil)
              AverageDownAndFill2(send_buffer[r].data() + d, f.infos[0], code);
            else
              AverageDownAndFill(send_buffer[r].data() + d, f.infos[0], code);
          }
#pragma omp for
          for (size_t i = 0; i < send_packinfos[r].size(); i++) {
            const PackInfo &info = send_packinfos[r][i];
            pack(info.block, info.pack, gptfloats,
                 &stencil.selcomponents.front(), NC, info.sx, info.sy, info.sz,
                 info.ex, info.ey, info.ez, nX, nY);
          }
        }
      }
    for (auto r : Neighbors)
      if (send_buffer_size[r] > 0) {
        requests.resize(requests.size() + 1);
        MPI_Isend(&send_buffer[r][0], send_buffer_size[r] * NC, MPIREAL, r,
                  timestamp, MPI_COMM_WORLD, &requests.back());
      }
  }
  void fetch(const BlockInfo &info, const unsigned int Length[3],
             const unsigned int CLength[3], Real *cacheBlock,
             Real *coarseBlock) {
    const int id = info.halo_block_id;
    if (id < 0)
      return;
    UnPackInfo *unpacks = myunpacks[id].data();
    for (size_t jj = 0; jj < myunpacks[id].size(); jj++) {
      const UnPackInfo &unpack = unpacks[jj];
      const int code[3] = {unpack.icode % 3 - 1, (unpack.icode / 3) % 3 - 1,
                           (unpack.icode / 9) % 3 - 1};
      const int otherrank = unpack.rank;
      const int s[3] = {code[0] < 1 ? (code[0] < 0 ? stencil.sx : 0) : nX,
                        code[1] < 1 ? (code[1] < 0 ? stencil.sy : 0) : nY,
                        code[2] < 1 ? (code[2] < 0 ? stencil.sz : 0) : nZ};
      const int e[3] = {
          code[0] < 1 ? (code[0] < 0 ? 0 : nX) : nX + stencil.ex - 1,
          code[1] < 1 ? (code[1] < 0 ? 0 : nY) : nY + stencil.ey - 1,
          code[2] < 1 ? (code[2] < 0 ? 0 : nZ) : nZ + stencil.ez - 1};
      if (unpack.level == info.level) {
        Real *dst =
            cacheBlock + ((s[2] - stencil.sz) * Length[0] * Length[1] +
                          (s[1] - stencil.sy) * Length[0] + s[0] - stencil.sx) *
                             gptfloats;
        unpack_subregion(&recv_buffer[otherrank][unpack.offset], &dst[0],
                         gptfloats, &stencil.selcomponents[0],
                         stencil.selcomponents.size(), unpack.srcxstart,
                         unpack.srcystart, unpack.srczstart, unpack.LX,
                         unpack.LY, 0, 0, 0, unpack.lx, unpack.ly, unpack.lz,
                         Length[0], Length[1]);
        if (unpack.CoarseVersionOffset >= 0) {
          const int offset[3] = {(stencil.sx - 1) / 2 + Cstencil.sx,
                                 (stencil.sy - 1) / 2 + Cstencil.sy,
                                 (stencil.sz - 1) / 2 + Cstencil.sz};
          const int sC[3] = {
              code[0] < 1 ? (code[0] < 0 ? offset[0] : 0) : nX / 2,
              code[1] < 1 ? (code[1] < 0 ? offset[1] : 0) : nY / 2,
              code[2] < 1 ? (code[2] < 0 ? offset[2] : 0) : nZ / 2};
          Real *dst1 = coarseBlock +
                       ((sC[2] - offset[2]) * CLength[0] * CLength[1] +
                        (sC[1] - offset[1]) * CLength[0] + sC[0] - offset[0]) *
                           gptfloats;
          int L[3];
          SM.CoarseStencilLength(
              (-code[0] + 1) + 3 * (-code[1] + 1) + 9 * (-code[2] + 1), L);
          unpack_subregion(
              &recv_buffer[otherrank]
                          [unpack.offset + unpack.CoarseVersionOffset],
              &dst1[0], gptfloats, &stencil.selcomponents[0],
              stencil.selcomponents.size(), unpack.CoarseVersionsrcxstart,
              unpack.CoarseVersionsrcystart, unpack.CoarseVersionsrczstart,
              unpack.CoarseVersionLX, unpack.CoarseVersionLY, 0, 0, 0, L[0],
              L[1], L[2], CLength[0], CLength[1]);
        }
      } else if (unpack.level < info.level) {
        const int offset[3] = {(stencil.sx - 1) / 2 + Cstencil.sx,
                               (stencil.sy - 1) / 2 + Cstencil.sy,
                               (stencil.sz - 1) / 2 + Cstencil.sz};
        const int sC[3] = {code[0] < 1 ? (code[0] < 0 ? offset[0] : 0) : nX / 2,
                           code[1] < 1 ? (code[1] < 0 ? offset[1] : 0) : nY / 2,
                           code[2] < 1 ? (code[2] < 0 ? offset[2] : 0)
                                       : nZ / 2};
        Real *dst = coarseBlock +
                    ((sC[2] - offset[2]) * CLength[0] * CLength[1] + sC[0] -
                     offset[0] + (sC[1] - offset[1]) * CLength[0]) *
                        gptfloats;
        unpack_subregion(&recv_buffer[otherrank][unpack.offset], &dst[0],
                         gptfloats, &stencil.selcomponents[0],
                         stencil.selcomponents.size(), unpack.srcxstart,
                         unpack.srcystart, unpack.srczstart, unpack.LX,
                         unpack.LY, 0, 0, 0, unpack.lx, unpack.ly, unpack.lz,
                         CLength[0], CLength[1]);
      } else {
        int B;
        if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 3))
          B = 0;
        else if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 2)) {
          int t;
          if (code[0] == 0)
            t = unpack.index_0 - 2 * info.index[0];
          else if (code[1] == 0)
            t = unpack.index_1 - 2 * info.index[1];
          else
            t = unpack.index_2 - 2 * info.index[2];
          assert(t == 0 || t == 1);
          B = (t == 1) ? 3 : 0;
        } else {
          int Bmod, Bdiv;
          if (abs(code[0]) == 1) {
            Bmod = unpack.index_1 - 2 * info.index[1];
            Bdiv = unpack.index_2 - 2 * info.index[2];
          } else if (abs(code[1]) == 1) {
            Bmod = unpack.index_0 - 2 * info.index[0];
            Bdiv = unpack.index_2 - 2 * info.index[2];
          } else {
            Bmod = unpack.index_0 - 2 * info.index[0];
            Bdiv = unpack.index_1 - 2 * info.index[1];
          }
          B = 2 * Bdiv + Bmod;
        }
        const int aux1 = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
        Real *dst =
            cacheBlock +
            ((abs(code[2]) * (s[2] - stencil.sz) +
              (1 - abs(code[2])) *
                  (-stencil.sz + (B / 2) * (e[2] - s[2]) / 2)) *
                 Length[0] * Length[1] +
             (abs(code[1]) * (s[1] - stencil.sy) +
              (1 - abs(code[1])) * (-stencil.sy + aux1 * (e[1] - s[1]) / 2)) *
                 Length[0] +
             abs(code[0]) * (s[0] - stencil.sx) +
             (1 - abs(code[0])) * (-stencil.sx + (B % 2) * (e[0] - s[0]) / 2)) *
                gptfloats;
        unpack_subregion(&recv_buffer[otherrank][unpack.offset], &dst[0],
                         gptfloats, &stencil.selcomponents[0],
                         stencil.selcomponents.size(), unpack.srcxstart,
                         unpack.srcystart, unpack.srczstart, unpack.LX,
                         unpack.LY, 0, 0, 0, unpack.lx, unpack.ly, unpack.lz,
                         Length[0], Length[1]);
      }
    }
  }
};
template <typename TFluxCorrection, typename ElementType>
struct FluxCorrectionMPI : public TFluxCorrection {
  using TGrid = typename TFluxCorrection::GridType;
  typedef typename TFluxCorrection::BlockType BlockType;
  typedef BlockCase<BlockType, ElementType> Case;
  int size;
  struct face {
    BlockInfo *infos[2];
    int icode[2];
    int offset;
    face(BlockInfo &i0, BlockInfo &i1, int a_icode0, int a_icode1) {
      infos[0] = &i0;
      infos[1] = &i1;
      icode[0] = a_icode0;
      icode[1] = a_icode1;
    }
    bool operator<(const face &other) const {
      if (infos[0]->blockID_2 == other.infos[0]->blockID_2) {
        return (icode[0] < other.icode[0]);
      } else {
        return (infos[0]->blockID_2 < other.infos[0]->blockID_2);
      }
    }
  };
  std::vector<std::vector<Real>> send_buffer;
  std::vector<std::vector<Real>> recv_buffer;
  std::vector<std::vector<face>> send_faces;
  std::vector<std::vector<face>> recv_faces;
  void FillCase(face &F) {
    BlockInfo &info = *F.infos[1];
    const int icode = F.icode[1];
    const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                         (icode / 9) % 3 - 1};
    const int myFace = abs(code[0]) * std::max(0, code[0]) +
                       abs(code[1]) * (std::max(0, code[1]) + 2) +
                       abs(code[2]) * (std::max(0, code[2]) + 4);
    std::array<long long, 2> temp = {(long long)info.level, info.Z};
    auto search = TFluxCorrection::MapOfCases.find(temp);
    assert(search != TFluxCorrection::MapOfCases.end());
    Case &CoarseCase = (*search->second);
    std::vector<ElementType> &CoarseFace = CoarseCase.m_pData[myFace];
    for (int B = 0; B <= 1; B++) {
      const int aux = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
      const long long Z =
          (*TFluxCorrection::grid)
              .getZforward(info.level + 1,
                           2 * info.index[0] + std::max(code[0], 0) + code[0] +
                               (B % 2) * std::max(0, 1 - abs(code[0])),
                           2 * info.index[1] + std::max(code[1], 0) + code[1] +
                               aux * std::max(0, 1 - abs(code[1])));
      if (Z != F.infos[0]->Z)
        continue;
      const int d = myFace / 2;
      const int d1 = std::max((d + 1) % 3, (d + 2) % 3);
      const int d2 = std::min((d + 1) % 3, (d + 2) % 3);
      const int N1 = CoarseCase.m_vSize[d1];
      const int N2 = CoarseCase.m_vSize[d2];
      int base = 0;
      if (B == 1)
        base = (N2 / 2) + (0) * N2;
      else if (B == 2)
        base = (0) + (N1 / 2) * N2;
      else if (B == 3)
        base = (N2 / 2) + (N1 / 2) * N2;
      int r = (*TFluxCorrection::grid)
                  .Tree(F.infos[0]->level, F.infos[0]->Z)
                  .rank();
      int dis = 0;
      for (int i2 = 0; i2 < N2; i2 += 2) {
        for (int j = 0; j < ElementType::DIM; j++)
          CoarseFace[base + (i2 / 2)].member(j) +=
              recv_buffer[r][F.offset + dis + j];
        dis += ElementType::DIM;
      }
    }
  }
  void FillCase_2(face &F, int codex, int codey, int codez) {
    BlockInfo &info = *F.infos[1];
    const int icode = F.icode[1];
    const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                         (icode / 9) % 3 - 1};
    if (abs(code[0]) != codex)
      return;
    if (abs(code[1]) != codey)
      return;
    if (abs(code[2]) != codez)
      return;
    const int myFace = abs(code[0]) * std::max(0, code[0]) +
                       abs(code[1]) * (std::max(0, code[1]) + 2) +
                       abs(code[2]) * (std::max(0, code[2]) + 4);
    std::array<long long, 2> temp = {(long long)info.level, info.Z};
    auto search = TFluxCorrection::MapOfCases.find(temp);
    assert(search != TFluxCorrection::MapOfCases.end());
    Case &CoarseCase = (*search->second);
    std::vector<ElementType> &CoarseFace = CoarseCase.m_pData[myFace];
    const int d = myFace / 2;
    const int d2 = std::min((d + 1) % 3, (d + 2) % 3);
    const int N2 = CoarseCase.m_vSize[d2];
    BlockType &block = *(BlockType *)info.ptrBlock;
    assert(d != 2);
    if (d == 0) {
      const int j = (myFace % 2 == 0) ? 0 : _BS_ - 1;
      for (int i2 = 0; i2 < N2; i2++) {
        block(j, i2) += CoarseFace[i2];
        CoarseFace[i2].clear();
      }
    } else {
      const int j = (myFace % 2 == 0) ? 0 : _BS_ - 1;
      for (int i2 = 0; i2 < N2; i2++) {
        block(i2, j) += CoarseFace[i2];
        CoarseFace[i2].clear();
      }
    }
  }
  virtual void prepare(TGrid &_grid) override {
    if (_grid.UpdateFluxCorrection == false)
      return;
    _grid.UpdateFluxCorrection = false;
    int temprank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &temprank);
    TFluxCorrection::rank = temprank;
    send_buffer.resize(size);
    recv_buffer.resize(size);
    send_faces.resize(size);
    recv_faces.resize(size);
    for (int r = 0; r < size; r++) {
      send_faces[r].clear();
      recv_faces[r].clear();
    }
    std::vector<int> send_buffer_size(size, 0);
    std::vector<int> recv_buffer_size(size, 0);
    const int NC = ElementType::DIM;
    int blocksize[3];
    blocksize[0] = _BS_;
    blocksize[1] = _BS_;
    blocksize[2] = 1;
    TFluxCorrection::Cases.clear();
    TFluxCorrection::MapOfCases.clear();
    TFluxCorrection::grid = &_grid;
    std::vector<BlockInfo> &BB = (*TFluxCorrection::grid).m_vInfo;
    std::array<int, 3> blocksPerDim = _grid.getMaxBlocks();
    std::array<int, 6> icode = {1 * 2 + 3 * 1 + 9 * 1, 1 * 0 + 3 * 1 + 9 * 1,
                                1 * 1 + 3 * 2 + 9 * 1, 1 * 1 + 3 * 0 + 9 * 1,
                                1 * 1 + 3 * 1 + 9 * 2, 1 * 1 + 3 * 1 + 9 * 0};
    for (auto &info : BB) {
      (*TFluxCorrection::grid).getBlockInfoAll(info.level, info.Z).auxiliary =
          nullptr;
      info.auxiliary = nullptr;
      const int aux = 1 << info.level;
      const bool xskin =
          info.index[0] == 0 || info.index[0] == blocksPerDim[0] * aux - 1;
      const bool yskin =
          info.index[1] == 0 || info.index[1] == blocksPerDim[1] * aux - 1;
      const bool zskin =
          info.index[2] == 0 || info.index[2] == blocksPerDim[2] * aux - 1;
      const int xskip = info.index[0] == 0 ? -1 : 1;
      const int yskip = info.index[1] == 0 ? -1 : 1;
      const int zskip = info.index[2] == 0 ? -1 : 1;
      bool storeFace[6] = {false, false, false, false, false, false};
      bool stored = false;
      for (int f = 0; f < 6; f++) {
        const int code[3] = {icode[f] % 3 - 1, (icode[f] / 3) % 3 - 1,
                             (icode[f] / 9) % 3 - 1};
        if (!_grid.xperiodic && code[0] == xskip && xskin)
          continue;
        if (!_grid.yperiodic && code[1] == yskip && yskin)
          continue;
        if (!_grid.zperiodic && code[2] == zskip && zskin)
          continue;
        if (code[2] != 0)
          continue;
        if (!(*TFluxCorrection::grid)
                 .Tree(info.level, info.Znei_(code[0], code[1], code[2]))
                 .Exists()) {
          storeFace[abs(code[0]) * std::max(0, code[0]) +
                    abs(code[1]) * (std::max(0, code[1]) + 2) +
                    abs(code[2]) * (std::max(0, code[2]) + 4)] = true;
          stored = true;
        }
        int L[3];
        L[0] = (code[0] == 0) ? blocksize[0] / 2 : 1;
        L[1] = (code[1] == 0) ? blocksize[1] / 2 : 1;
        L[2] = 1;
        int V = L[0] * L[1] * L[2];
        if ((*TFluxCorrection::grid)
                .Tree(info.level, info.Znei_(code[0], code[1], code[2]))
                .CheckCoarser()) {
          BlockInfo &infoNei =
              (*TFluxCorrection::grid)
                  .getBlockInfoAll(info.level,
                                   info.Znei_(code[0], code[1], code[2]));
          const long long nCoarse = infoNei.Zparent;
          BlockInfo &infoNeiCoarser =
              (*TFluxCorrection::grid).getBlockInfoAll(info.level - 1, nCoarse);
          const int infoNeiCoarserrank =
              (*TFluxCorrection::grid).Tree(info.level - 1, nCoarse).rank();
          {
            int code2[3] = {-code[0], -code[1], -code[2]};
            int icode2 =
                (code2[0] + 1) + (code2[1] + 1) * 3 + (code2[2] + 1) * 9;
            send_faces[infoNeiCoarserrank].push_back(
                face(info, infoNeiCoarser, icode[f], icode2));
            send_buffer_size[infoNeiCoarserrank] += V;
          }
        } else if ((*TFluxCorrection::grid)
                       .Tree(info.level, info.Znei_(code[0], code[1], code[2]))
                       .CheckFiner()) {
          BlockInfo &infoNei =
              (*TFluxCorrection::grid)
                  .getBlockInfoAll(info.level,
                                   info.Znei_(code[0], code[1], code[2]));
          int Bstep = 1;
          for (int B = 0; B <= 1; B += Bstep) {
            const int temp = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
            const long long nFine =
                infoNei.Zchild[std::max(-code[0], 0) +
                               (B % 2) * std::max(0, 1 - abs(code[0]))]
                              [std::max(-code[1], 0) +
                               temp * std::max(0, 1 - abs(code[1]))]
                              [std::max(-code[2], 0) +
                               (B / 2) * std::max(0, 1 - abs(code[2]))];
            const int infoNeiFinerrank =
                (*TFluxCorrection::grid).Tree(infoNei.level + 1, nFine).rank();
            {
              BlockInfo &infoNeiFiner =
                  (*TFluxCorrection::grid)
                      .getBlockInfoAll(infoNei.level + 1, nFine);
              int icode2 =
                  (-code[0] + 1) + (-code[1] + 1) * 3 + (-code[2] + 1) * 9;
              recv_faces[infoNeiFinerrank].push_back(
                  face(infoNeiFiner, info, icode2, icode[f]));
              recv_buffer_size[infoNeiFinerrank] += V;
            }
          }
        }
      }
      if (stored) {
        TFluxCorrection::Cases.push_back(Case(storeFace, info.level, info.Z));
      }
    }
    size_t Cases_index = 0;
    if (TFluxCorrection::Cases.size() > 0)
      for (auto &info : BB) {
        if (Cases_index == TFluxCorrection::Cases.size())
          break;
        if (TFluxCorrection::Cases[Cases_index].level == info.level &&
            TFluxCorrection::Cases[Cases_index].Z == info.Z) {
          TFluxCorrection::MapOfCases.insert(
              std::pair<std::array<long long, 2>, Case *>(
                  {TFluxCorrection::Cases[Cases_index].level,
                   TFluxCorrection::Cases[Cases_index].Z},
                  &TFluxCorrection::Cases[Cases_index]));
          TFluxCorrection::grid
              ->getBlockInfoAll(TFluxCorrection::Cases[Cases_index].level,
                                TFluxCorrection::Cases[Cases_index].Z)
              .auxiliary = &TFluxCorrection::Cases[Cases_index];
          info.auxiliary = &TFluxCorrection::Cases[Cases_index];
          Cases_index++;
        }
      }
    for (int r = 0; r < size; r++) {
      std::sort(send_faces[r].begin(), send_faces[r].end());
      std::sort(recv_faces[r].begin(), recv_faces[r].end());
    }
    for (int r = 0; r < size; r++) {
      send_buffer[r].resize(send_buffer_size[r] * NC);
      recv_buffer[r].resize(recv_buffer_size[r] * NC);
      int offset = 0;
      for (int k = 0; k < (int)recv_faces[r].size(); k++) {
        face &f = recv_faces[r][k];
        const int code[3] = {f.icode[1] % 3 - 1, (f.icode[1] / 3) % 3 - 1,
                             (f.icode[1] / 9) % 3 - 1};
        int L[3];
        L[0] = (code[0] == 0) ? blocksize[0] / 2 : 1;
        L[1] = (code[1] == 0) ? blocksize[1] / 2 : 1;
        L[2] = 1;
        int V = L[0] * L[1] * L[2];
        f.offset = offset;
        offset += V * NC;
      }
    }
  }
  virtual void FillBlockCases() override {
    auto MPI_real =
        (sizeof(Real) == sizeof(float))
            ? MPI_FLOAT
            : ((sizeof(Real) == sizeof(double)) ? MPI_DOUBLE : MPI_LONG_DOUBLE);
    for (int r = 0; r < size; r++) {
      int displacement = 0;
      for (int k = 0; k < (int)send_faces[r].size(); k++) {
        face &f = send_faces[r][k];
        BlockInfo &info = *(f.infos[0]);
        auto search =
            TFluxCorrection::MapOfCases.find({(long long)info.level, info.Z});
        assert(search != TFluxCorrection::MapOfCases.end());
        Case &FineCase = (*search->second);
        int icode = f.icode[0];
        int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, (icode / 9) % 3 - 1};
        int myFace = abs(code[0]) * std::max(0, code[0]) +
                     abs(code[1]) * (std::max(0, code[1]) + 2) +
                     abs(code[2]) * (std::max(0, code[2]) + 4);
        std::vector<ElementType> &FineFace = FineCase.m_pData[myFace];
        int d = myFace / 2;
        int d2 = std::min((d + 1) % 3, (d + 2) % 3);
        int N2 = FineCase.m_vSize[d2];
        for (int i2 = 0; i2 < N2; i2 += 2) {
          ElementType avg = FineFace[i2] + FineFace[i2 + 1];
          for (int j = 0; j < ElementType::DIM; j++)
            send_buffer[r][displacement + j] = avg.member(j);
          displacement += ElementType::DIM;
          FineFace[i2].clear();
          FineFace[i2 + 1].clear();
        }
      }
    }
    std::vector<MPI_Request> send_requests;
    std::vector<MPI_Request> recv_requests;
    int me = TFluxCorrection::rank;
    for (int r = 0; r < size; r++)
      if (r != me) {
        if (recv_buffer[r].size() != 0) {
          MPI_Request req{};
          recv_requests.push_back(req);
          MPI_Irecv(&recv_buffer[r][0], recv_buffer[r].size(), MPI_real, r,
                    123456, MPI_COMM_WORLD, &recv_requests.back());
        }
        if (send_buffer[r].size() != 0) {
          MPI_Request req{};
          send_requests.push_back(req);
          MPI_Isend(&send_buffer[r][0], send_buffer[r].size(), MPI_real, r,
                    123456, MPI_COMM_WORLD, &send_requests.back());
        }
      }
    MPI_Request me_send_request;
    MPI_Request me_recv_request;
    if (recv_buffer[me].size() != 0) {
      MPI_Irecv(&recv_buffer[me][0], recv_buffer[me].size(), MPI_real, me,
                123456, MPI_COMM_WORLD, &me_recv_request);
    }
    if (send_buffer[me].size() != 0) {
      MPI_Isend(&send_buffer[me][0], send_buffer[me].size(), MPI_real, me,
                123456, MPI_COMM_WORLD, &me_send_request);
    }
    if (recv_buffer[me].size() > 0)
      MPI_Waitall(1, &me_recv_request, MPI_STATUSES_IGNORE);
    if (send_buffer[me].size() > 0)
      MPI_Waitall(1, &me_send_request, MPI_STATUSES_IGNORE);
    for (int index = 0; index < (int)recv_faces[me].size(); index++)
      FillCase(recv_faces[me][index]);
    if (recv_requests.size() > 0)
      MPI_Waitall(recv_requests.size(), &recv_requests[0], MPI_STATUSES_IGNORE);
    for (int r = 0; r < size; r++)
      if (r != me)
        for (int index = 0; index < (int)recv_faces[r].size(); index++)
          FillCase(recv_faces[r][index]);
    for (int r = 0; r < size; r++)
      for (int index = 0; index < (int)recv_faces[r].size(); index++)
        FillCase_2(recv_faces[r][index], 1, 0, 0);
    for (int r = 0; r < size; r++)
      for (int index = 0; index < (int)recv_faces[r].size(); index++)
        FillCase_2(recv_faces[r][index], 0, 1, 0);
    if (send_requests.size() > 0)
      MPI_Waitall(send_requests.size(), &send_requests[0], MPI_STATUSES_IGNORE);
  }
};
template <typename TGrid, typename ElementType> struct GridMPI : public TGrid {
  typedef typename TGrid::BlockType Block;
  typedef SynchronizerMPI_AMR<GridMPI<TGrid, ElementType>> SynchronizerMPIType;
  size_t timestamp;
  int myrank;
  int world_size;
  std::map<StencilInfo, SynchronizerMPIType *> SynchronizerMPIs;
  FluxCorrectionMPI<FluxCorrection<GridMPI<TGrid, ElementType>, ElementType>,
                    ElementType>
      Corrector;
  std::vector<BlockInfo *> boundary;
  GridMPI(int nX, int nY, int nZ, double a_maxextent, int a_levelStart,
          int a_levelMax, bool a_xperiodic, bool a_yperiodic, bool a_zperiodic)
      : TGrid(nX, nY, nZ, a_maxextent, a_levelStart, a_levelMax, a_xperiodic,
              a_yperiodic, a_zperiodic),
        timestamp(0) {
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    const long long total_blocks =
        nX * nY * nZ * pow(pow(2, a_levelStart), DIMENSION);
    long long my_blocks = total_blocks / world_size;
    if ((long long)myrank < total_blocks % world_size)
      my_blocks++;
    long long n_start = myrank * (total_blocks / world_size);
    if (total_blocks % world_size > 0) {
      if ((long long)myrank < total_blocks % world_size)
        n_start += myrank;
      else
        n_start += total_blocks % world_size;
    }
    std::vector<short int> levels(my_blocks, a_levelStart);
    std::vector<long long> Zs(my_blocks);
    for (long long n = n_start; n < n_start + my_blocks; n++)
      Zs[n - n_start] = n;
    initialize_blocks(Zs, levels);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  virtual Block *avail(const int m, const long long n) override {
    return (TGrid::Tree(m, n).rank() == myrank)
               ? (Block *)TGrid::getBlockInfoAll(m, n).ptrBlock
               : nullptr;
  }
  virtual void UpdateBoundary(bool clean = false) override {
    const auto blocksPerDim = TGrid::getMaxBlocks();
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::vector<std::vector<long long>> send_buffer(size);
    std::vector<BlockInfo *> &bbb = boundary;
    std::set<int> Neighbors;
    for (size_t jjj = 0; jjj < bbb.size(); jjj++) {
      BlockInfo &info = *bbb[jjj];
      std::set<int> receivers;
      const int aux = 1 << info.level;
      const bool xskin =
          info.index[0] == 0 || info.index[0] == blocksPerDim[0] * aux - 1;
      const bool yskin =
          info.index[1] == 0 || info.index[1] == blocksPerDim[1] * aux - 1;
      const bool zskin =
          info.index[2] == 0 || info.index[2] == blocksPerDim[2] * aux - 1;
      const int xskip = info.index[0] == 0 ? -1 : 1;
      const int yskip = info.index[1] == 0 ? -1 : 1;
      const int zskip = info.index[2] == 0 ? -1 : 1;
      for (int icode = 0; icode < 27; icode++) {
        if (icode == 1 * 1 + 3 * 1 + 9 * 1)
          continue;
        const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                             (icode / 9) % 3 - 1};
        if (!TGrid::xperiodic && code[0] == xskip && xskin)
          continue;
        if (!TGrid::yperiodic && code[1] == yskip && yskin)
          continue;
        if (!TGrid::zperiodic && code[2] == zskip && zskin)
          continue;
        if (code[2] != 0)
          continue;
        BlockInfo &infoNei = TGrid::getBlockInfoAll(
            info.level, info.Znei_(code[0], code[1], code[2]));
        const TreePosition &infoNeiTree = TGrid::Tree(infoNei.level, infoNei.Z);
        if (infoNeiTree.Exists() && infoNeiTree.rank() != rank) {
          if (infoNei.state != Refine || clean)
            infoNei.state = Leave;
          receivers.insert(infoNeiTree.rank());
          Neighbors.insert(infoNeiTree.rank());
        } else if (infoNeiTree.CheckCoarser()) {
          const long long nCoarse = infoNei.Zparent;
          BlockInfo &infoNeiCoarser =
              TGrid::getBlockInfoAll(infoNei.level - 1, nCoarse);
          const int infoNeiCoarserrank =
              TGrid::Tree(infoNei.level - 1, nCoarse).rank();
          if (infoNeiCoarserrank != rank) {
            assert(infoNeiCoarserrank >= 0);
            if (infoNeiCoarser.state != Refine || clean)
              infoNeiCoarser.state = Leave;
            receivers.insert(infoNeiCoarserrank);
            Neighbors.insert(infoNeiCoarserrank);
          }
        } else if (infoNeiTree.CheckFiner()) {
          int Bstep = 1;
          if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 2))
            Bstep = 3;
          else if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 3))
            Bstep = 4;
          for (int B = 0; B <= 1; B += Bstep) {
            const int temp = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
            const long long nFine =
                infoNei.Zchild[std::max(-code[0], 0) +
                               (B % 2) * std::max(0, 1 - abs(code[0]))]
                              [std::max(-code[1], 0) +
                               temp * std::max(0, 1 - abs(code[1]))]
                              [std::max(-code[2], 0) +
                               (B / 2) * std::max(0, 1 - abs(code[2]))];
            BlockInfo &infoNeiFiner =
                TGrid::getBlockInfoAll(infoNei.level + 1, nFine);
            const int infoNeiFinerrank =
                TGrid::Tree(infoNei.level + 1, nFine).rank();
            if (infoNeiFinerrank != rank) {
              if (infoNeiFiner.state != Refine || clean)
                infoNeiFiner.state = Leave;
              receivers.insert(infoNeiFinerrank);
              Neighbors.insert(infoNeiFinerrank);
            }
          }
        }
      }
      if (info.changed2 && info.state != Leave) {
        if (info.state == Refine)
          info.changed2 = false;
        std::set<int>::iterator it = receivers.begin();
        while (it != receivers.end()) {
          int temp = (info.state == Compress) ? 1 : 2;
          send_buffer[*it].push_back(info.level);
          send_buffer[*it].push_back(info.Z);
          send_buffer[*it].push_back(temp);
          it++;
        }
      }
    }
    std::vector<MPI_Request> requests;
    long long dummy = 0;
    for (int r : Neighbors)
      if (r != rank) {
        requests.resize(requests.size() + 1);
        if (send_buffer[r].size() != 0)
          MPI_Isend(&send_buffer[r][0], send_buffer[r].size(), MPI_LONG_LONG, r,
                    123, MPI_COMM_WORLD, &requests[requests.size() - 1]);
        else {
          MPI_Isend(&dummy, 1, MPI_LONG_LONG, r, 123, MPI_COMM_WORLD,
                    &requests[requests.size() - 1]);
        }
      }
    std::vector<std::vector<long long>> recv_buffer(size);
    for (int r : Neighbors)
      if (r != rank) {
        int recv_size;
        MPI_Status status;
        MPI_Probe(r, 123, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_LONG_LONG, &recv_size);
        if (recv_size > 0) {
          recv_buffer[r].resize(recv_size);
          requests.resize(requests.size() + 1);
          MPI_Irecv(&recv_buffer[r][0], recv_buffer[r].size(), MPI_LONG_LONG, r,
                    123, MPI_COMM_WORLD, &requests[requests.size() - 1]);
        }
      }
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    for (int r = 0; r < size; r++)
      if (recv_buffer[r].size() > 1)
        for (int index = 0; index < (int)recv_buffer[r].size(); index += 3) {
          int level = recv_buffer[r][index];
          long long Z = recv_buffer[r][index + 1];
          TGrid::getBlockInfoAll(level, Z).state =
              (recv_buffer[r][index + 2] == 1) ? Compress : Refine;
        }
  };
  void UpdateBlockInfoAll_States(bool UpdateIDs = false) {
    std::vector<int> myNeighbors = FindMyNeighbors();
    const auto blocksPerDim = TGrid::getMaxBlocks();
    std::vector<long long> myData;
    for (auto &info : TGrid::m_vInfo) {
      bool myflag = false;
      const int aux = 1 << info.level;
      const bool xskin =
          info.index[0] == 0 || info.index[0] == blocksPerDim[0] * aux - 1;
      const bool yskin =
          info.index[1] == 0 || info.index[1] == blocksPerDim[1] * aux - 1;
      const bool zskin =
          info.index[2] == 0 || info.index[2] == blocksPerDim[2] * aux - 1;
      const int xskip = info.index[0] == 0 ? -1 : 1;
      const int yskip = info.index[1] == 0 ? -1 : 1;
      const int zskip = info.index[2] == 0 ? -1 : 1;
      for (int icode = 0; icode < 27; icode++) {
        if (icode == 1 * 1 + 3 * 1 + 9 * 1)
          continue;
        const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                             (icode / 9) % 3 - 1};
        if (!TGrid::xperiodic && code[0] == xskip && xskin)
          continue;
        if (!TGrid::yperiodic && code[1] == yskip && yskin)
          continue;
        if (!TGrid::zperiodic && code[2] == zskip && zskin)
          continue;
        if (code[2] != 0)
          continue;
        BlockInfo &infoNei = TGrid::getBlockInfoAll(
            info.level, info.Znei_(code[0], code[1], code[2]));
        const TreePosition &infoNeiTree = TGrid::Tree(infoNei.level, infoNei.Z);
        if (infoNeiTree.Exists() && infoNeiTree.rank() != myrank) {
          myflag = true;
          break;
        } else if (infoNeiTree.CheckCoarser()) {
          long long nCoarse = infoNei.Zparent;
          const int infoNeiCoarserrank =
              TGrid::Tree(infoNei.level - 1, nCoarse).rank();
          if (infoNeiCoarserrank != myrank) {
            myflag = true;
            break;
          }
        } else if (infoNeiTree.CheckFiner()) {
          int Bstep = 1;
          if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 2))
            Bstep = 3;
          else if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 3))
            Bstep = 4;
          for (int B = 0; B <= 3; B += Bstep) {
            const int temp = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
            const long long nFine =
                infoNei.Zchild[std::max(-code[0], 0) +
                               (B % 2) * std::max(0, 1 - abs(code[0]))]
                              [std::max(-code[1], 0) +
                               temp * std::max(0, 1 - abs(code[1]))]
                              [std::max(-code[2], 0) +
                               (B / 2) * std::max(0, 1 - abs(code[2]))];
            const int infoNeiFinerrank =
                TGrid::Tree(infoNei.level + 1, nFine).rank();
            if (infoNeiFinerrank != myrank) {
              myflag = true;
              break;
            }
          }
        } else if (infoNeiTree.rank() < 0) {
          myflag = true;
          break;
        }
      }
      if (myflag) {
        myData.push_back(info.level);
        myData.push_back(info.Z);
        if (UpdateIDs)
          myData.push_back(info.blockID);
      }
    }
    std::vector<std::vector<long long>> recv_buffer(myNeighbors.size());
    std::vector<std::vector<long long>> send_buffer(myNeighbors.size());
    std::vector<int> recv_size(myNeighbors.size());
    std::vector<MPI_Request> size_requests(2 * myNeighbors.size());
    int mysize = (int)myData.size();
    int kk = 0;
    for (auto r : myNeighbors) {
      MPI_Irecv(&recv_size[kk], 1, MPI_INT, r, timestamp, MPI_COMM_WORLD,
                &size_requests[2 * kk]);
      MPI_Isend(&mysize, 1, MPI_INT, r, timestamp, MPI_COMM_WORLD,
                &size_requests[2 * kk + 1]);
      kk++;
    }
    kk = 0;
    for (size_t j = 0; j < myNeighbors.size(); j++) {
      send_buffer[kk].resize(myData.size());
      for (size_t i = 0; i < myData.size(); i++)
        send_buffer[kk][i] = myData[i];
      kk++;
    }
    MPI_Waitall(size_requests.size(), size_requests.data(),
                MPI_STATUSES_IGNORE);
    std::vector<MPI_Request> requests(2 * myNeighbors.size());
    kk = 0;
    for (auto r : myNeighbors) {
      recv_buffer[kk].resize(recv_size[kk]);
      MPI_Irecv(recv_buffer[kk].data(), recv_buffer[kk].size(), MPI_LONG_LONG,
                r, timestamp, MPI_COMM_WORLD, &requests[2 * kk]);
      MPI_Isend(send_buffer[kk].data(), send_buffer[kk].size(), MPI_LONG_LONG,
                r, timestamp, MPI_COMM_WORLD, &requests[2 * kk + 1]);
      kk++;
    }
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    kk = -1;
    const int increment = UpdateIDs ? 3 : 2;
    for (auto r : myNeighbors) {
      kk++;
      for (size_t index__ = 0; index__ < recv_buffer[kk].size();
           index__ += increment) {
        const int level = (int)recv_buffer[kk][index__];
        const long long Z = recv_buffer[kk][index__ + 1];
        TGrid::Tree(level, Z).setrank(r);
        if (UpdateIDs)
          TGrid::getBlockInfoAll(level, Z).blockID =
              recv_buffer[kk][index__ + 2];
        int p[2];
        BlockInfo::inverse(Z, level, p[0], p[1]);
        if (level < TGrid::levelMax - 1)
          for (int j = 0; j < 2; j++)
            for (int i = 0; i < 2; i++) {
              const long long nc =
                  TGrid::getZforward(level + 1, 2 * p[0] + i, 2 * p[1] + j);
              TGrid::Tree(level + 1, nc).setCheckCoarser();
            }
        if (level > 0) {
          const long long nf =
              TGrid::getZforward(level - 1, p[0] / 2, p[1] / 2);
          TGrid::Tree(level - 1, nf).setCheckFiner();
        }
      }
    }
  }
  std::vector<int> FindMyNeighbors() {
    std::vector<int> myNeighbors;
    double low[3] = {+1e20, +1e20, +1e20};
    double high[3] = {-1e20, -1e20, -1e20};
    double p_low[3];
    double p_high[3];
    for (auto &info : TGrid::m_vInfo) {
      const double h = 2 * info.h;
      info.pos(p_low, 0, 0);
      info.pos(p_high, _BS_ - 1, _BS_ - 1);
      p_low[0] -= h;
      p_low[1] -= h;
      p_low[2] = 0;
      p_high[0] += h;
      p_high[1] += h;
      p_high[2] = 0;
      low[0] = std::min(low[0], p_low[0]);
      low[1] = std::min(low[1], p_low[1]);
      low[2] = 0;
      high[0] = std::max(high[0], p_high[0]);
      high[1] = std::max(high[1], p_high[1]);
      high[2] = 0;
    }
    std::vector<double> all_boxes(world_size * 6);
    double my_box[6] = {low[0], low[1], low[2], high[0], high[1], high[2]};
    MPI_Allgather(my_box, 6, MPI_DOUBLE, all_boxes.data(), 6, MPI_DOUBLE,
                  MPI_COMM_WORLD);
    for (int i = 0; i < world_size; i++) {
      if (i == myrank)
        continue;
      if (Intersect(low, high, &all_boxes[i * 6], &all_boxes[i * 6 + 3]))
        myNeighbors.push_back(i);
    }
    return myNeighbors;
  }
  bool Intersect(double *l1, double *h1, double *l2, double *h2) {
    const double h0 =
        (TGrid::maxextent /
         std::max(TGrid::NX * _BS_, std::max(TGrid::NY * _BS_, TGrid::NZ * 1)));
    const double extent[3] = {TGrid::NX * _BS_ * h0, TGrid::NY * _BS_ * h0,
                              TGrid::NZ * 1 * h0};
    const Real intersect[3][2] = {
        {std::max(l1[0], l2[0]), std::min(h1[0], h2[0])},
        {std::max(l1[1], l2[1]), std::min(h1[1], h2[1])},
        {std::max(l1[2], l2[2]), std::min(h1[2], h2[2])}};
    bool intersection[3];
    intersection[0] = intersect[0][1] - intersect[0][0] > 0.0;
    intersection[1] = intersect[1][1] - intersect[1][0] > 0.0;
    intersection[2] =
        DIMENSION == 3 ? (intersect[2][1] - intersect[2][0] > 0.0) : true;
    const bool isperiodic[3] = {TGrid::xperiodic, TGrid::yperiodic,
                                TGrid::zperiodic};
    for (int d = 0; d < DIMENSION; d++) {
      if (isperiodic[d]) {
        if (h2[d] > extent[d])
          intersection[d] = std::min(h1[d], h2[d] - extent[d]) -
                            std::max(l1[d], l2[d] - extent[d]);
        else if (h1[d] > extent[d])
          intersection[d] = std::min(h2[d], h1[d] - extent[d]) -
                            std::max(l2[d], l1[d] - extent[d]);
      }
      if (!intersection[d])
        return false;
    }
    return true;
  }
  SynchronizerMPIType *sync(const StencilInfo &stencil) {
    assert(stencil.isvalid());
    StencilInfo Cstencil(-1, -1, DIMENSION == 3 ? -1 : 0, 2, 2,
                         DIMENSION == 3 ? 2 : 1, true, stencil.selcomponents);
    SynchronizerMPIType *queryresult = nullptr;
    typename std::map<StencilInfo, SynchronizerMPIType *>::iterator
        itSynchronizerMPI = SynchronizerMPIs.find(stencil);
    if (itSynchronizerMPI == SynchronizerMPIs.end()) {
      queryresult = new SynchronizerMPIType(stencil, Cstencil, this);
      queryresult->_Setup();
      SynchronizerMPIs[stencil] = queryresult;
    } else {
      queryresult = itSynchronizerMPI->second;
    }
    queryresult->sync();
    timestamp = (timestamp + 1) % 32768;
    return queryresult;
  }
  virtual void
  initialize_blocks(const std::vector<long long> &blocksZ,
                    const std::vector<short int> &blockslevel) override {
    TGrid::initialize_blocks(blocksZ, blockslevel);
    UpdateBlockInfoAll_States(false);
    for (auto it = SynchronizerMPIs.begin(); it != SynchronizerMPIs.end(); ++it)
      (*it->second)._Setup();
  }
  virtual int rank() const override { return myrank; }
  virtual int get_world_size() const override { return world_size; }
};
template <class DataType> struct Matrix3D {
  DataType *m_pData{nullptr};
  unsigned int m_vSize[3]{0, 0, 0};
  unsigned int m_nElements{0};
  unsigned int m_nElementsPerSlice{0};
  void _Release() {
    if (m_pData != nullptr) {
      free(m_pData);
      m_pData = nullptr;
    }
  }
  void _Setup(unsigned int nSizeX, unsigned int nSizeY, unsigned int nSizeZ) {
    _Release();
    m_vSize[0] = nSizeX;
    m_vSize[1] = nSizeY;
    m_vSize[2] = nSizeZ;
    m_nElementsPerSlice = nSizeX * nSizeY;
    m_nElements = nSizeX * nSizeY * nSizeZ;
    posix_memalign((void **)&m_pData, std::max(8, CUBISM_ALIGNMENT),
                   sizeof(DataType) * m_nElements);
    assert(m_pData != nullptr);
  }
  ~Matrix3D() { _Release(); }
  Matrix3D(unsigned int nSizeX, unsigned int nSizeY, unsigned int nSizeZ)
      : m_pData(nullptr), m_nElements(0), m_nElementsPerSlice(0) {
    _Setup(nSizeX, nSizeY, nSizeZ);
  }
  Matrix3D() : m_pData(nullptr), m_nElements(-1), m_nElementsPerSlice(-1) {}
  Matrix3D(const Matrix3D &m) = delete;
  Matrix3D(Matrix3D &&m)
      : m_pData{m.m_pData}, m_vSize{m.m_vSize[0], m.m_vSize[1], m.m_vSize[2]},
        m_nElements{m.m_nElements}, m_nElementsPerSlice{m.m_nElementsPerSlice} {
    m.m_pData = nullptr;
  }
  Matrix3D &operator=(const Matrix3D &m) {
#ifndef NDEBUG
    assert(m_vSize[0] == m.m_vSize[0]);
    assert(m_vSize[1] == m.m_vSize[1]);
    assert(m_vSize[2] == m.m_vSize[2]);
#endif
    for (unsigned int i = 0; i < m_nElements; i++)
      m_pData[i] = m.m_pData[i];
    return *this;
  }
  Matrix3D &operator=(DataType d) {
    for (unsigned int i = 0; i < m_nElements; i++)
      m_pData[i] = d;
    return *this;
  }
  Matrix3D &operator=(const double a) {
    for (unsigned int i = 0; i < m_nElements; i++)
      m_pData[i].set(a);
    return *this;
  }
  DataType &Access(unsigned int ix, unsigned int iy, unsigned int iz) const {
#ifndef NDEBUG
    assert(ix < m_vSize[0]);
    assert(iy < m_vSize[1]);
    assert(iz < m_vSize[2]);
#endif
    return m_pData[iz * m_nElementsPerSlice + iy * m_vSize[0] + ix];
  }
  const DataType &Read(unsigned int ix, unsigned int iy,
                       unsigned int iz) const {
#ifndef NDEBUG
    assert(ix < m_vSize[0]);
    assert(iy < m_vSize[1]);
    assert(iz < m_vSize[2]);
#endif
    return m_pData[iz * m_nElementsPerSlice + iy * m_vSize[0] + ix];
  }
  DataType &LinAccess(unsigned int i) const {
#ifndef NDEBUG
    assert(i < m_nElements);
#endif
    return m_pData[i];
  }
  unsigned int getNumberOfElements() const { return m_nElements; }
  unsigned int getNumberOfElementsPerSlice() const {
    return m_nElementsPerSlice;
  }
  unsigned int *getSize() const { return (unsigned int *)m_vSize; }
  unsigned int getSize(int dim) const { return m_vSize[dim]; }
};
constexpr int default_start[3] = {-1, -1, 0};
constexpr int default_end[3] = {2, 2, 1};
template <typename TGrid, typename ElementType> struct BlockLab {
  using GridType = TGrid;
  using BlockType = typename GridType::BlockType;
  Matrix3D<ElementType> *m_cacheBlock;
  int m_stencilStart[3];
  int m_stencilEnd[3];
  bool istensorial;
  bool use_averages;
  GridType *m_refGrid;
  int NX;
  int NY;
  int NZ;
  std::array<BlockType *, 27> myblocks;
  std::array<int, 27> coarsened_nei_codes;
  int coarsened_nei_codes_size;
  int offset[3];
  Matrix3D<ElementType> *m_CoarsenedBlock;
  int m_InterpStencilStart[3];
  int m_InterpStencilEnd[3];
  bool coarsened;
  int CoarseBlockSize[3];
  const double d_coef_plus[9] = {-0.09375, 0.4375,   0.15625, 0.15625, -0.5625,
                                 0.90625,  -0.09375, 0.4375,  0.15625};
  const double d_coef_minus[9] = {0.15625, -0.5625, 0.90625, -0.09375, 0.4375,
                                  0.15625, 0.15625, 0.4375,  -0.09375};
  BlockLab()
      : m_cacheBlock(nullptr), m_refGrid(nullptr), m_CoarsenedBlock(nullptr) {
    m_stencilStart[0] = m_stencilStart[1] = m_stencilStart[2] = 0;
    m_stencilEnd[0] = m_stencilEnd[1] = m_stencilEnd[2] = 0;
    m_InterpStencilStart[0] = m_InterpStencilStart[1] =
        m_InterpStencilStart[2] = 0;
    m_InterpStencilEnd[0] = m_InterpStencilEnd[1] = m_InterpStencilEnd[2] = 0;
    CoarseBlockSize[0] = _BS_ / 2;
    CoarseBlockSize[1] = _BS_ / 2;
    CoarseBlockSize[2] = 1 / 2;
    if (CoarseBlockSize[0] == 0)
      CoarseBlockSize[0] = 1;
    if (CoarseBlockSize[1] == 0)
      CoarseBlockSize[1] = 1;
    if (CoarseBlockSize[2] == 0)
      CoarseBlockSize[2] = 1;
  }
  virtual bool is_xperiodic() { return true; }
  virtual bool is_yperiodic() { return true; }
  virtual bool is_zperiodic() { return true; }
  ~BlockLab() {
    _release(m_cacheBlock);
    _release(m_CoarsenedBlock);
  }
  ElementType &operator()(int ix, int iy = 0, int iz = 0) {
    assert(ix - m_stencilStart[0] >= 0 &&
           ix - m_stencilStart[0] < (int)m_cacheBlock->getSize()[0]);
    assert(iy - m_stencilStart[1] >= 0 &&
           iy - m_stencilStart[1] < (int)m_cacheBlock->getSize()[1]);
    assert(iz - m_stencilStart[2] >= 0 &&
           iz - m_stencilStart[2] < (int)m_cacheBlock->getSize()[2]);
    return m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1],
                                iz - m_stencilStart[2]);
  }
  const ElementType &operator()(int ix, int iy = 0, int iz = 0) const {
    assert(ix - m_stencilStart[0] >= 0 &&
           ix - m_stencilStart[0] < (int)m_cacheBlock->getSize()[0]);
    assert(iy - m_stencilStart[1] >= 0 &&
           iy - m_stencilStart[1] < (int)m_cacheBlock->getSize()[1]);
    assert(iz - m_stencilStart[2] >= 0 &&
           iz - m_stencilStart[2] < (int)m_cacheBlock->getSize()[2]);
    return m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1],
                                iz - m_stencilStart[2]);
  }
  virtual void prepare(GridType &grid, const StencilInfo &stencil,
                       const int Istencil_start[3] = default_start,
                       const int Istencil_end[3] = default_end) {
    istensorial = stencil.tensorial;
    coarsened = false;
    m_stencilStart[0] = stencil.sx;
    m_stencilStart[1] = stencil.sy;
    m_stencilStart[2] = stencil.sz;
    m_stencilEnd[0] = stencil.ex;
    m_stencilEnd[1] = stencil.ey;
    m_stencilEnd[2] = stencil.ez;
    m_InterpStencilStart[0] = Istencil_start[0];
    m_InterpStencilStart[1] = Istencil_start[1];
    m_InterpStencilStart[2] = Istencil_start[2];
    m_InterpStencilEnd[0] = Istencil_end[0];
    m_InterpStencilEnd[1] = Istencil_end[1];
    m_InterpStencilEnd[2] = Istencil_end[2];
    assert(m_InterpStencilStart[0] <= m_InterpStencilEnd[0]);
    assert(m_InterpStencilStart[1] <= m_InterpStencilEnd[1]);
    assert(m_InterpStencilStart[2] <= m_InterpStencilEnd[2]);
    assert(stencil.sx <= stencil.ex);
    assert(stencil.sy <= stencil.ey);
    assert(stencil.sz <= stencil.ez);
    assert(stencil.sx >= -_BS_);
    assert(stencil.sy >= -_BS_);
    assert(stencil.sz >= -1);
    assert(stencil.ex < 2 * _BS_);
    assert(stencil.ey < 2 * _BS_);
    assert(stencil.ez < 2 * 1);
    m_refGrid = &grid;
    if (m_cacheBlock == NULL ||
        (int)m_cacheBlock->getSize()[0] !=
            _BS_ + m_stencilEnd[0] - m_stencilStart[0] - 1 ||
        (int)m_cacheBlock->getSize()[1] !=
            _BS_ + m_stencilEnd[1] - m_stencilStart[1] - 1 ||
        (int)m_cacheBlock->getSize()[2] !=
            1 + m_stencilEnd[2] - m_stencilStart[2] - 1) {
      if (m_cacheBlock != NULL)
        _release(m_cacheBlock);
      m_cacheBlock = new Matrix3D<ElementType>;
      m_cacheBlock->_Setup(_BS_ + m_stencilEnd[0] - m_stencilStart[0] - 1,
                           _BS_ + m_stencilEnd[1] - m_stencilStart[1] - 1,
                           1 + m_stencilEnd[2] - m_stencilStart[2] - 1);
    }
    offset[0] = (m_stencilStart[0] - 1) / 2 + m_InterpStencilStart[0];
    offset[1] = (m_stencilStart[1] - 1) / 2 + m_InterpStencilStart[1];
    offset[2] = (m_stencilStart[2] - 1) / 2 + m_InterpStencilStart[2];
    const int e[3] = {(m_stencilEnd[0]) / 2 + 1 + m_InterpStencilEnd[0] - 1,
                      (m_stencilEnd[1]) / 2 + 1 + m_InterpStencilEnd[1] - 1,
                      (m_stencilEnd[2]) / 2 + 1 + m_InterpStencilEnd[2] - 1};
    if (m_CoarsenedBlock == NULL ||
        (int)m_CoarsenedBlock->getSize()[0] !=
            CoarseBlockSize[0] + e[0] - offset[0] - 1 ||
        (int)m_CoarsenedBlock->getSize()[1] !=
            CoarseBlockSize[1] + e[1] - offset[1] - 1 ||
        (int)m_CoarsenedBlock->getSize()[2] !=
            CoarseBlockSize[2] + e[2] - offset[2] - 1) {
      if (m_CoarsenedBlock != NULL)
        _release(m_CoarsenedBlock);
      m_CoarsenedBlock = new Matrix3D<ElementType>;
      m_CoarsenedBlock->_Setup(CoarseBlockSize[0] + e[0] - offset[0] - 1,
                               CoarseBlockSize[1] + e[1] - offset[1] - 1,
                               CoarseBlockSize[2] + e[2] - offset[2] - 1);
    }
    use_averages = (m_refGrid->FiniteDifferences == false || istensorial ||
                    m_stencilStart[0] < -2 || m_stencilStart[1] < -2 ||
                    m_stencilEnd[0] > 3 || m_stencilEnd[1] > 3);
  }
  virtual void load(const BlockInfo &info, const Real t = 0,
                    const bool applybc = true) {
    const int nX = _BS_;
    const int nY = _BS_;
    const int nZ = 1;
    const bool xperiodic = is_xperiodic();
    const bool yperiodic = is_yperiodic();
    const bool zperiodic = is_zperiodic();
    std::array<int, 3> blocksPerDim = m_refGrid->getMaxBlocks();
    const int aux = 1 << info.level;
    NX = blocksPerDim[0] * aux;
    NY = blocksPerDim[1] * aux;
    NZ = blocksPerDim[2] * aux;
    assert(m_cacheBlock != NULL);
    {
      BlockType &block = *(BlockType *)info.ptrBlock;
      ElementType *ptrSource = &block(0);
      const int nbytes = sizeof(ElementType) * nX;
      const int _iz0 = -m_stencilStart[2];
      const int _iz1 = _iz0 + nZ;
      const int _iy0 = -m_stencilStart[1];
      const int _iy1 = _iy0 + nY;
      const int m_vSize0 = m_cacheBlock->getSize(0);
      const int m_nElemsPerSlice = m_cacheBlock->getNumberOfElementsPerSlice();
      const int my_ix = -m_stencilStart[0];
#pragma GCC ivdep
      for (int iz = _iz0; iz < _iz1; iz++) {
        const int my_izx = iz * m_nElemsPerSlice + my_ix;
#pragma GCC ivdep
        for (int iy = _iy0; iy < _iy1; iy += 4) {
          ElementType *__restrict__ ptrDestination0 =
              &m_cacheBlock->LinAccess(my_izx + (iy)*m_vSize0);
          ElementType *__restrict__ ptrDestination1 =
              &m_cacheBlock->LinAccess(my_izx + (iy + 1) * m_vSize0);
          ElementType *__restrict__ ptrDestination2 =
              &m_cacheBlock->LinAccess(my_izx + (iy + 2) * m_vSize0);
          ElementType *__restrict__ ptrDestination3 =
              &m_cacheBlock->LinAccess(my_izx + (iy + 3) * m_vSize0);
          memcpy(ptrDestination0, (ptrSource), nbytes);
          memcpy(ptrDestination1, (ptrSource + nX), nbytes);
          memcpy(ptrDestination2, (ptrSource + 2 * nX), nbytes);
          memcpy(ptrDestination3, (ptrSource + 3 * nX), nbytes);
          ptrSource += 4 * nX;
        }
      }
    }
    {
      coarsened = false;
      const bool xskin = info.index[0] == 0 || info.index[0] == NX - 1;
      const bool yskin = info.index[1] == 0 || info.index[1] == NY - 1;
      const bool zskin = info.index[2] == 0 || info.index[2] == NZ - 1;
      const int xskip = info.index[0] == 0 ? -1 : 1;
      const int yskip = info.index[1] == 0 ? -1 : 1;
      const int zskip = info.index[2] == 0 ? -1 : 1;
      int icodes[8];
      int k = 0;
      coarsened_nei_codes_size = 0;
      for (int icode = (DIMENSION == 2 ? 9 : 0);
           icode < (DIMENSION == 2 ? 18 : 27); icode++) {
        myblocks[icode] = nullptr;
        if (icode == 1 * 1 + 3 * 1 + 9 * 1)
          continue;
        const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, icode / 9 - 1};
        if (!xperiodic && code[0] == xskip && xskin)
          continue;
        if (!yperiodic && code[1] == yskip && yskin)
          continue;
        if (!zperiodic && code[2] == zskip && zskin)
          continue;
        const auto &TreeNei =
            m_refGrid->Tree(info.level, info.Znei_(code[0], code[1], code[2]));
        if (TreeNei.Exists()) {
          icodes[k++] = icode;
        } else if (TreeNei.CheckCoarser()) {
          coarsened_nei_codes[coarsened_nei_codes_size++] = icode;
          CoarseFineExchange(info, code);
        }
        if (!istensorial && !use_averages &&
            abs(code[0]) + abs(code[1]) + abs(code[2]) > 1)
          continue;
        const int s[3] = {
            code[0] < 1 ? (code[0] < 0 ? m_stencilStart[0] : 0) : nX,
            code[1] < 1 ? (code[1] < 0 ? m_stencilStart[1] : 0) : nY,
            code[2] < 1 ? (code[2] < 0 ? m_stencilStart[2] : 0) : nZ};
        const int e[3] = {
            code[0] < 1 ? (code[0] < 0 ? 0 : nX) : nX + m_stencilEnd[0] - 1,
            code[1] < 1 ? (code[1] < 0 ? 0 : nY) : nY + m_stencilEnd[1] - 1,
            code[2] < 1 ? (code[2] < 0 ? 0 : nZ) : nZ + m_stencilEnd[2] - 1};
        if (TreeNei.Exists())
          SameLevelExchange(info, code, s, e);
        else if (TreeNei.CheckFiner())
          FineToCoarseExchange(info, code, s, e);
      }
      if (coarsened_nei_codes_size > 0)
        for (int i = 0; i < k; ++i) {
          const int icode = icodes[i];
          const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                               icode / 9 - 1};
          const int infoNei_index[3] = {(info.index[0] + code[0] + NX) % NX,
                                        (info.index[1] + code[1] + NY) % NY,
                                        (info.index[2] + code[2] + NZ) % NZ};
          if (UseCoarseStencil(info, infoNei_index)) {
            FillCoarseVersion(code);
            coarsened = true;
          }
        }
      if (m_refGrid->get_world_size() == 1) {
        post_load(info, t, applybc);
      }
    }
  }
  void post_load(const BlockInfo &info, const Real t = 0, bool applybc = true) {
    const int nX = _BS_;
    const int nY = _BS_;
    if (coarsened) {
#pragma GCC ivdep
      for (int j = 0; j < nY / 2; j++) {
#pragma GCC ivdep
        for (int i = 0; i < nX / 2; i++) {
          if (i > -m_InterpStencilStart[0] &&
              i < nX / 2 - m_InterpStencilEnd[0] &&
              j > -m_InterpStencilStart[1] &&
              j < nY / 2 - m_InterpStencilEnd[1])
            continue;
          const int ix = 2 * i - m_stencilStart[0];
          const int iy = 2 * j - m_stencilStart[1];
          ElementType &coarseElement =
              m_CoarsenedBlock->Access(i - offset[0], j - offset[1], 0);
          coarseElement = AverageDown(m_cacheBlock->Read(ix, iy, 0),
                                      m_cacheBlock->Read(ix + 1, iy, 0),
                                      m_cacheBlock->Read(ix, iy + 1, 0),
                                      m_cacheBlock->Read(ix + 1, iy + 1, 0));
        }
      }
    }
    if (applybc)
      _apply_bc(info, t, true);
    CoarseFineInterpolation(info);
    if (applybc)
      _apply_bc(info, t);
  }
  bool UseCoarseStencil(const BlockInfo &a, const int *b_index) {
    if (a.level == 0 || (!use_averages))
      return false;
    std::array<int, 3> blocksPerDim = m_refGrid->getMaxBlocks();
    int imin[3];
    int imax[3];
    const int aux = 1 << a.level;
    const bool periodic[3] = {is_xperiodic(), is_yperiodic(), is_zperiodic()};
    const int blocks[3] = {blocksPerDim[0] * aux - 1, blocksPerDim[1] * aux - 1,
                           blocksPerDim[2] * aux - 1};
    for (int d = 0; d < 3; d++) {
      imin[d] = (a.index[d] < b_index[d]) ? 0 : -1;
      imax[d] = (a.index[d] > b_index[d]) ? 0 : +1;
      if (periodic[d]) {
        if (a.index[d] == 0 && b_index[d] == blocks[d])
          imin[d] = -1;
        if (b_index[d] == 0 && a.index[d] == blocks[d])
          imax[d] = +1;
      } else {
        if (a.index[d] == 0 && b_index[d] == 0)
          imin[d] = 0;
        if (a.index[d] == blocks[d] && b_index[d] == blocks[d])
          imax[d] = 0;
      }
    }
    for (int itest = 0; itest < coarsened_nei_codes_size; itest++)
      for (int i2 = imin[2]; i2 <= imax[2]; i2++)
        for (int i1 = imin[1]; i1 <= imax[1]; i1++)
          for (int i0 = imin[0]; i0 <= imax[0]; i0++) {
            const int icode_test = (i0 + 1) + 3 * (i1 + 1) + 9 * (i2 + 1);
            if (coarsened_nei_codes[itest] == icode_test)
              return true;
          }
    return false;
  }
  void SameLevelExchange(const BlockInfo &info, const int *const code,
                         const int *const s, const int *const e) {
    const int bytes = (e[0] - s[0]) * sizeof(ElementType);
    if (!bytes)
      return;
    const int icode = (code[0] + 1) + 3 * (code[1] + 1) + 9 * (code[2] + 1);
    myblocks[icode] =
        m_refGrid->avail(info.level, info.Znei_(code[0], code[1], code[2]));
    if (myblocks[icode] == nullptr)
      return;
    const BlockType &b = *myblocks[icode];
    const int nX = _BS_;
    const int nY = _BS_;
    const int nZ = 1;
    const int m_vSize0 = m_cacheBlock->getSize(0);
    const int m_nElemsPerSlice = m_cacheBlock->getNumberOfElementsPerSlice();
    const int my_ix = s[0] - m_stencilStart[0];
    const int mod = (e[1] - s[1]) % 4;
#pragma GCC ivdep
    for (int iz = s[2]; iz < e[2]; iz++) {
      const int my_izx = (iz - m_stencilStart[2]) * m_nElemsPerSlice + my_ix;
#pragma GCC ivdep
      for (int iy = s[1]; iy < e[1] - mod; iy += 4) {
        ElementType *__restrict__ ptrDest0 = &m_cacheBlock->LinAccess(
            my_izx + (iy - m_stencilStart[1]) * m_vSize0);
        ElementType *__restrict__ ptrDest1 = &m_cacheBlock->LinAccess(
            my_izx + (iy + 1 - m_stencilStart[1]) * m_vSize0);
        ElementType *__restrict__ ptrDest2 = &m_cacheBlock->LinAccess(
            my_izx + (iy + 2 - m_stencilStart[1]) * m_vSize0);
        ElementType *__restrict__ ptrDest3 = &m_cacheBlock->LinAccess(
            my_izx + (iy + 3 - m_stencilStart[1]) * m_vSize0);
        const ElementType *ptrSrc0 = &b(s[0] - code[0] * nX, iy - code[1] * nY);
        const ElementType *ptrSrc1 =
            &b(s[0] - code[0] * nX, iy + 1 - code[1] * nY);
        const ElementType *ptrSrc2 =
            &b(s[0] - code[0] * nX, iy + 2 - code[1] * nY);
        const ElementType *ptrSrc3 =
            &b(s[0] - code[0] * nX, iy + 3 - code[1] * nY);
        memcpy(ptrDest0, ptrSrc0, bytes);
        memcpy(ptrDest1, ptrSrc1, bytes);
        memcpy(ptrDest2, ptrSrc2, bytes);
        memcpy(ptrDest3, ptrSrc3, bytes);
      }
#pragma GCC ivdep
      for (int iy = e[1] - mod; iy < e[1]; iy++) {
        ElementType *__restrict__ ptrDest = &m_cacheBlock->LinAccess(
            my_izx + (iy - m_stencilStart[1]) * m_vSize0);
        const ElementType *ptrSrc = &b(s[0] - code[0] * nX, iy - code[1] * nY);
        memcpy(ptrDest, ptrSrc, bytes);
      }
    }
  }
  ElementType AverageDown(const ElementType &e0, const ElementType &e1,
                          const ElementType &e2, const ElementType &e3) {
    return 0.25 * ((e0 + e3) + (e1 + e2));
  }
  void LI(ElementType &a, ElementType b, ElementType c) {
    auto kappa = ((4.0 / 15.0) * a + (6.0 / 15.0) * c) + (-10.0 / 15.0) * b;
    auto lambda = (b - c) - kappa;
    a = (4.0 * kappa + 2.0 * lambda) + c;
  }
  void LE(ElementType &a, ElementType b, ElementType c) {
    auto kappa = ((4.0 / 15.0) * a + (6.0 / 15.0) * c) + (-10.0 / 15.0) * b;
    auto lambda = (b - c) - kappa;
    a = (9.0 * kappa + 3.0 * lambda) + c;
  }
  virtual void TestInterp(ElementType *C[3][3], ElementType &R, int x, int y) {
    const double dx = 0.25 * (2 * x - 1);
    const double dy = 0.25 * (2 * y - 1);
    ElementType dudx = 0.5 * ((*C[2][1]) - (*C[0][1]));
    ElementType dudy = 0.5 * ((*C[1][2]) - (*C[1][0]));
    ElementType dudxdy =
        0.25 * (((*C[0][0]) + (*C[2][2])) - ((*C[2][0]) + (*C[0][2])));
    ElementType dudx2 = ((*C[0][1]) + (*C[2][1])) - 2.0 * (*C[1][1]);
    ElementType dudy2 = ((*C[1][0]) + (*C[1][2])) - 2.0 * (*C[1][1]);
    R = (*C[1][1] + (dx * dudx + dy * dudy)) +
        (((0.5 * dx * dx) * dudx2 + (0.5 * dy * dy) * dudy2) +
         (dx * dy) * dudxdy);
  }
  void FineToCoarseExchange(const BlockInfo &info, const int *const code,
                            const int *const s, const int *const e) {
    const int bytes = (abs(code[0]) * (e[0] - s[0]) +
                       (1 - abs(code[0])) * ((e[0] - s[0]) / 2)) *
                      sizeof(ElementType);
    if (!bytes)
      return;
    const int nX = _BS_;
    const int nY = _BS_;
    const int nZ = 1;
    const int m_vSize0 = m_cacheBlock->getSize(0);
    const int m_nElemsPerSlice = m_cacheBlock->getNumberOfElementsPerSlice();
    const int yStep = (code[1] == 0) ? 2 : 1;
    const int zStep = (code[2] == 0) ? 2 : 1;
    const int mod = ((e[1] - s[1]) / yStep) % 4;
    int Bstep = 1;
    if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 2))
      Bstep = 3;
    else if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 3))
      Bstep = 4;
    for (int B = 0; B <= 3; B += Bstep) {
      const int aux = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
      BlockType *b_ptr =
          m_refGrid->avail1(2 * info.index[0] + std::max(code[0], 0) + code[0] +
                                (B % 2) * std::max(0, 1 - abs(code[0])),
                            2 * info.index[1] + std::max(code[1], 0) + code[1] +
                                aux * std::max(0, 1 - abs(code[1])),
                            info.level + 1);
      if (b_ptr == nullptr)
        continue;
      BlockType &b = *b_ptr;
      const int my_ix = abs(code[0]) * (s[0] - m_stencilStart[0]) +
                        (1 - abs(code[0])) * (s[0] - m_stencilStart[0] +
                                              (B % 2) * (e[0] - s[0]) / 2);
      const int XX = s[0] - code[0] * nX + std::min(0, code[0]) * (e[0] - s[0]);
#pragma GCC ivdep
      for (int iz = s[2]; iz < e[2]; iz += zStep) {
        const int ZZ = (abs(code[2]) == 1)
                           ? 2 * (iz - code[2] * nZ) + std::min(0, code[2]) * nZ
                           : iz;
        const int my_izx =
            (abs(code[2]) * (iz - m_stencilStart[2]) +
             (1 - abs(code[2])) *
                 (iz / 2 - m_stencilStart[2] + (B / 2) * (e[2] - s[2]) / 2)) *
                m_nElemsPerSlice +
            my_ix;
#pragma GCC ivdep
        for (int iy = s[1]; iy < e[1] - mod; iy += 4 * yStep) {
          ElementType *__restrict__ ptrDest0 = &m_cacheBlock->LinAccess(
              my_izx +
              (abs(code[1]) * (iy + 0 * yStep - m_stencilStart[1]) +
               (1 - abs(code[1])) * ((iy + 0 * yStep) / 2 - m_stencilStart[1] +
                                     aux * (e[1] - s[1]) / 2)) *
                  m_vSize0);
          ElementType *__restrict__ ptrDest1 = &m_cacheBlock->LinAccess(
              my_izx +
              (abs(code[1]) * (iy + 1 * yStep - m_stencilStart[1]) +
               (1 - abs(code[1])) * ((iy + 1 * yStep) / 2 - m_stencilStart[1] +
                                     aux * (e[1] - s[1]) / 2)) *
                  m_vSize0);
          ElementType *__restrict__ ptrDest2 = &m_cacheBlock->LinAccess(
              my_izx +
              (abs(code[1]) * (iy + 2 * yStep - m_stencilStart[1]) +
               (1 - abs(code[1])) * ((iy + 2 * yStep) / 2 - m_stencilStart[1] +
                                     aux * (e[1] - s[1]) / 2)) *
                  m_vSize0);
          ElementType *__restrict__ ptrDest3 = &m_cacheBlock->LinAccess(
              my_izx +
              (abs(code[1]) * (iy + 3 * yStep - m_stencilStart[1]) +
               (1 - abs(code[1])) * ((iy + 3 * yStep) / 2 - m_stencilStart[1] +
                                     aux * (e[1] - s[1]) / 2)) *
                  m_vSize0);
          const int YY0 = (abs(code[1]) == 1)
                              ? 2 * (iy + 0 * yStep - code[1] * nY) +
                                    std::min(0, code[1]) * nY
                              : iy + 0 * yStep;
          const int YY1 = (abs(code[1]) == 1)
                              ? 2 * (iy + 1 * yStep - code[1] * nY) +
                                    std::min(0, code[1]) * nY
                              : iy + 1 * yStep;
          const int YY2 = (abs(code[1]) == 1)
                              ? 2 * (iy + 2 * yStep - code[1] * nY) +
                                    std::min(0, code[1]) * nY
                              : iy + 2 * yStep;
          const int YY3 = (abs(code[1]) == 1)
                              ? 2 * (iy + 3 * yStep - code[1] * nY) +
                                    std::min(0, code[1]) * nY
                              : iy + 3 * yStep;
          const ElementType *ptrSrc_00 = &b(XX, YY0);
          const ElementType *ptrSrc_10 = &b(XX, YY0 + 1);
          const ElementType *ptrSrc_01 = &b(XX, YY1);
          const ElementType *ptrSrc_11 = &b(XX, YY1 + 1);
          const ElementType *ptrSrc_02 = &b(XX, YY2);
          const ElementType *ptrSrc_12 = &b(XX, YY2 + 1);
          const ElementType *ptrSrc_03 = &b(XX, YY3);
          const ElementType *ptrSrc_13 = &b(XX, YY3 + 1);
#pragma GCC ivdep
          for (int ee = 0; ee < (abs(code[0]) * (e[0] - s[0]) +
                                 (1 - abs(code[0])) * ((e[0] - s[0]) / 2));
               ee++) {
            ptrDest0[ee] = AverageDown(
                *(ptrSrc_00 + 2 * ee), *(ptrSrc_10 + 2 * ee),
                *(ptrSrc_00 + 2 * ee + 1), *(ptrSrc_10 + 2 * ee + 1));
            ptrDest1[ee] = AverageDown(
                *(ptrSrc_01 + 2 * ee), *(ptrSrc_11 + 2 * ee),
                *(ptrSrc_01 + 2 * ee + 1), *(ptrSrc_11 + 2 * ee + 1));
            ptrDest2[ee] = AverageDown(
                *(ptrSrc_02 + 2 * ee), *(ptrSrc_12 + 2 * ee),
                *(ptrSrc_02 + 2 * ee + 1), *(ptrSrc_12 + 2 * ee + 1));
            ptrDest3[ee] = AverageDown(
                *(ptrSrc_03 + 2 * ee), *(ptrSrc_13 + 2 * ee),
                *(ptrSrc_03 + 2 * ee + 1), *(ptrSrc_13 + 2 * ee + 1));
          }
        }
#pragma GCC ivdep
        for (int iy = e[1] - mod; iy < e[1]; iy += yStep) {
          ElementType *ptrDest = (ElementType *)&m_cacheBlock->LinAccess(
              my_izx + (abs(code[1]) * (iy - m_stencilStart[1]) +
                        (1 - abs(code[1])) * (iy / 2 - m_stencilStart[1] +
                                              aux * (e[1] - s[1]) / 2)) *
                           m_vSize0);
          const int YY = (abs(code[1]) == 1) ? 2 * (iy - code[1] * nY) +
                                                   std::min(0, code[1]) * nY
                                             : iy;
          const ElementType *ptrSrc_0 = &b(XX, YY);
          const ElementType *ptrSrc_1 = &b(XX, YY + 1);
#pragma GCC ivdep
          for (int ee = 0; ee < (abs(code[0]) * (e[0] - s[0]) +
                                 (1 - abs(code[0])) * ((e[0] - s[0]) / 2));
               ee++) {
            ptrDest[ee] =
                AverageDown(*(ptrSrc_0 + 2 * ee), *(ptrSrc_1 + 2 * ee),
                            *(ptrSrc_0 + 2 * ee + 1), *(ptrSrc_1 + 2 * ee + 1));
          }
        }
      }
    }
  }
  void CoarseFineExchange(const BlockInfo &info, const int *const code) {
    const int infoNei_index[3] = {(info.index[0] + code[0] + NX) % NX,
                                  (info.index[1] + code[1] + NY) % NY,
                                  (info.index[2] + code[2] + NZ) % NZ};
    const int infoNei_index_true[3] = {(info.index[0] + code[0]),
                                       (info.index[1] + code[1]),
                                       (info.index[2] + code[2])};
    BlockType *b_ptr = m_refGrid->avail1(
        (infoNei_index[0]) / 2, (infoNei_index[1]) / 2, info.level - 1);
    if (b_ptr == nullptr)
      return;
    const BlockType &b = *b_ptr;
    const int s[3] = {
        code[0] < 1 ? (code[0] < 0 ? offset[0] : 0) : CoarseBlockSize[0],
        code[1] < 1 ? (code[1] < 0 ? offset[1] : 0) : CoarseBlockSize[1],
        code[2] < 1 ? (code[2] < 0 ? offset[2] : 0) : CoarseBlockSize[2]};
    const int e[3] = {code[0] < 1 ? (code[0] < 0 ? 0 : CoarseBlockSize[0])
                                  : CoarseBlockSize[0] + (m_stencilEnd[0]) / 2 +
                                        m_InterpStencilEnd[0] - 1,
                      code[1] < 1 ? (code[1] < 0 ? 0 : CoarseBlockSize[1])
                                  : CoarseBlockSize[1] + (m_stencilEnd[1]) / 2 +
                                        m_InterpStencilEnd[1] - 1,
                      code[2] < 1 ? (code[2] < 0 ? 0 : CoarseBlockSize[2])
                                  : CoarseBlockSize[2] + (m_stencilEnd[2]) / 2 +
                                        m_InterpStencilEnd[2] - 1};
    const int bytes = (e[0] - s[0]) * sizeof(ElementType);
    if (!bytes)
      return;
    const int base[3] = {(info.index[0] + code[0]) % 2,
                         (info.index[1] + code[1]) % 2,
                         (info.index[2] + code[2]) % 2};
    int CoarseEdge[3];
    CoarseEdge[0] = (code[0] == 0) ? 0
                    : (((info.index[0] % 2 == 0) &&
                        (infoNei_index_true[0] > info.index[0])) ||
                       ((info.index[0] % 2 == 1) &&
                        (infoNei_index_true[0] < info.index[0])))
                        ? 1
                        : 0;
    CoarseEdge[1] = (code[1] == 0) ? 0
                    : (((info.index[1] % 2 == 0) &&
                        (infoNei_index_true[1] > info.index[1])) ||
                       ((info.index[1] % 2 == 1) &&
                        (infoNei_index_true[1] < info.index[1])))
                        ? 1
                        : 0;
    CoarseEdge[2] = (code[2] == 0) ? 0
                    : (((info.index[2] % 2 == 0) &&
                        (infoNei_index_true[2] > info.index[2])) ||
                       ((info.index[2] % 2 == 1) &&
                        (infoNei_index_true[2] < info.index[2])))
                        ? 1
                        : 0;
    const int start[3] = {
        std::max(code[0], 0) * _BS_ / 2 +
            (1 - abs(code[0])) * base[0] * _BS_ / 2 - code[0] * _BS_ +
            CoarseEdge[0] * code[0] * _BS_ / 2,
        std::max(code[1], 0) * _BS_ / 2 +
            (1 - abs(code[1])) * base[1] * _BS_ / 2 - code[1] * _BS_ +
            CoarseEdge[1] * code[1] * _BS_ / 2,
        std::max(code[2], 0) / 2 + (1 - abs(code[2])) * base[2] / 2 - code[2] +
            CoarseEdge[2] * code[2] / 2};

    const int m_vSize0 = m_CoarsenedBlock->getSize(0);
    const int m_nElemsPerSlice =
        m_CoarsenedBlock->getNumberOfElementsPerSlice();
    const int my_ix = s[0] - offset[0];
    const int mod = (e[1] - s[1]) % 4;
#pragma GCC ivdep
    for (int iz = s[2]; iz < e[2]; iz++) {
      const int my_izx = (iz - offset[2]) * m_nElemsPerSlice + my_ix;
#pragma GCC ivdep
      for (int iy = s[1]; iy < e[1] - mod; iy += 4) {
        ElementType *__restrict__ ptrDest0 = &m_CoarsenedBlock->LinAccess(
            my_izx + (iy + 0 - offset[1]) * m_vSize0);
        ElementType *__restrict__ ptrDest1 = &m_CoarsenedBlock->LinAccess(
            my_izx + (iy + 1 - offset[1]) * m_vSize0);
        ElementType *__restrict__ ptrDest2 = &m_CoarsenedBlock->LinAccess(
            my_izx + (iy + 2 - offset[1]) * m_vSize0);
        ElementType *__restrict__ ptrDest3 = &m_CoarsenedBlock->LinAccess(
            my_izx + (iy + 3 - offset[1]) * m_vSize0);
        const ElementType *ptrSrc0 = &b(s[0] + start[0], iy + 0 + start[1]);
        const ElementType *ptrSrc1 = &b(s[0] + start[0], iy + 1 + start[1]);
        const ElementType *ptrSrc2 = &b(s[0] + start[0], iy + 2 + start[1]);
        const ElementType *ptrSrc3 = &b(s[0] + start[0], iy + 3 + start[1]);
        memcpy(ptrDest0, ptrSrc0, bytes);
        memcpy(ptrDest1, ptrSrc1, bytes);
        memcpy(ptrDest2, ptrSrc2, bytes);
        memcpy(ptrDest3, ptrSrc3, bytes);
      }
#pragma GCC ivdep
      for (int iy = e[1] - mod; iy < e[1]; iy++) {
        ElementType *ptrDest =
            &m_CoarsenedBlock->LinAccess(my_izx + (iy - offset[1]) * m_vSize0);
        const ElementType *ptrSrc = &b(s[0] + start[0], iy + start[1]);
        memcpy(ptrDest, ptrSrc, bytes);
      }
    }
  }
  void FillCoarseVersion(const int *const code) {
    const int icode = (code[0] + 1) + 3 * (code[1] + 1) + 9 * (code[2] + 1);
    if (myblocks[icode] == nullptr)
      return;
    const BlockType &b = *myblocks[icode];
    const int nX = _BS_;
    const int nY = _BS_;
    const int nZ = 1;
    const int eC[3] = {(m_stencilEnd[0]) / 2 + m_InterpStencilEnd[0],
                       (m_stencilEnd[1]) / 2 + m_InterpStencilEnd[1],
                       (m_stencilEnd[2]) / 2 + m_InterpStencilEnd[2]};
    const int s[3] = {
        code[0] < 1 ? (code[0] < 0 ? offset[0] : 0) : CoarseBlockSize[0],
        code[1] < 1 ? (code[1] < 0 ? offset[1] : 0) : CoarseBlockSize[1],
        code[2] < 1 ? (code[2] < 0 ? offset[2] : 0) : CoarseBlockSize[2]};
    const int e[3] = {code[0] < 1 ? (code[0] < 0 ? 0 : CoarseBlockSize[0])
                                  : CoarseBlockSize[0] + eC[0] - 1,
                      code[1] < 1 ? (code[1] < 0 ? 0 : CoarseBlockSize[1])
                                  : CoarseBlockSize[1] + eC[1] - 1,
                      code[2] < 1 ? (code[2] < 0 ? 0 : CoarseBlockSize[2])
                                  : CoarseBlockSize[2] + eC[2] - 1};
    const int bytes = (e[0] - s[0]) * sizeof(ElementType);
    if (!bytes)
      return;
    const int start[3] = {
        s[0] + std::max(code[0], 0) * CoarseBlockSize[0] - code[0] * nX +
            std::min(0, code[0]) * (e[0] - s[0]),
        s[1] + std::max(code[1], 0) * CoarseBlockSize[1] - code[1] * nY +
            std::min(0, code[1]) * (e[1] - s[1]),
        s[2] + std::max(code[2], 0) * CoarseBlockSize[2] - code[2] * nZ +
            std::min(0, code[2]) * (e[2] - s[2])};
    const int m_vSize0 = m_CoarsenedBlock->getSize(0);
    const int m_nElemsPerSlice =
        m_CoarsenedBlock->getNumberOfElementsPerSlice();
    const int my_ix = s[0] - offset[0];
    const int XX = start[0];
#pragma GCC ivdep
    for (int iz = s[2]; iz < e[2]; iz++) {
      const int ZZ = 2 * (iz - s[2]) + start[2];
      const int my_izx = (iz - offset[2]) * m_nElemsPerSlice + my_ix;
#pragma GCC ivdep
      for (int iy = s[1]; iy < e[1]; iy++) {
        if (code[1] == 0 && code[2] == 0 && iy > -m_InterpStencilStart[1] &&
            iy < nY / 2 - m_InterpStencilEnd[1] &&
            iz > -m_InterpStencilStart[2] &&
            iz < nZ / 2 - m_InterpStencilEnd[2])
          continue;
        ElementType *__restrict__ ptrDest1 =
            &m_CoarsenedBlock->LinAccess(my_izx + (iy - offset[1]) * m_vSize0);
        const int YY = 2 * (iy - s[1]) + start[1];
        const ElementType *ptrSrc_0 = (const ElementType *)&b(XX, YY);
        const ElementType *ptrSrc_1 = (const ElementType *)&b(XX, YY + 1);
#pragma GCC ivdep
        for (int ee = 0; ee < e[0] - s[0]; ee++) {
          ptrDest1[ee] =
              AverageDown(*(ptrSrc_0 + 2 * ee), *(ptrSrc_1 + 2 * ee),
                          *(ptrSrc_0 + 2 * ee + 1), *(ptrSrc_1 + 2 * ee + 1));
        }
      }
    }
  }
  void CoarseFineInterpolation(const BlockInfo &info) {
    const int nX = _BS_;
    const int nY = _BS_;
    const int nZ = 1;
    const bool xperiodic = is_xperiodic();
    const bool yperiodic = is_yperiodic();
    const bool zperiodic = is_zperiodic();
    const std::array<int, 3> blocksPerDim = m_refGrid->getMaxBlocks();
    const int aux = 1 << info.level;
    const bool xskin =
        info.index[0] == 0 || info.index[0] == blocksPerDim[0] * aux - 1;
    const bool yskin =
        info.index[1] == 0 || info.index[1] == blocksPerDim[1] * aux - 1;
    const bool zskin =
        info.index[2] == 0 || info.index[2] == blocksPerDim[2] * aux - 1;
    const int xskip = info.index[0] == 0 ? -1 : 1;
    const int yskip = info.index[1] == 0 ? -1 : 1;
    const int zskip = info.index[2] == 0 ? -1 : 1;
    for (int ii = 0; ii < coarsened_nei_codes_size; ++ii) {
      const int icode = coarsened_nei_codes[ii];
      if (icode == 1 * 1 + 3 * 1 + 9 * 1)
        continue;
      const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                           (icode / 9) % 3 - 1};
      if (code[2] != 0)
        continue;
      if (!xperiodic && code[0] == xskip && xskin)
        continue;
      if (!yperiodic && code[1] == yskip && yskin)
        continue;
      if (!zperiodic && code[2] == zskip && zskin)
        continue;
      if (!istensorial && !use_averages &&
          abs(code[0]) + abs(code[1]) + abs(code[2]) > 1)
        continue;
      const int s[3] = {
          code[0] < 1 ? (code[0] < 0 ? m_stencilStart[0] : 0) : nX,
          code[1] < 1 ? (code[1] < 0 ? m_stencilStart[1] : 0) : nY,
          code[2] < 1 ? (code[2] < 0 ? m_stencilStart[2] : 0) : nZ};
      const int e[3] = {
          code[0] < 1 ? (code[0] < 0 ? 0 : nX) : nX + m_stencilEnd[0] - 1,
          code[1] < 1 ? (code[1] < 0 ? 0 : nY) : nY + m_stencilEnd[1] - 1,
          code[2] < 1 ? (code[2] < 0 ? 0 : nZ) : nZ + m_stencilEnd[2] - 1};
      const int sC[3] = {
          code[0] < 1 ? (code[0] < 0 ? ((m_stencilStart[0] - 1) / 2) : 0)
                      : CoarseBlockSize[0],
          code[1] < 1 ? (code[1] < 0 ? ((m_stencilStart[1] - 1) / 2) : 0)
                      : CoarseBlockSize[1],
          code[2] < 1 ? (code[2] < 0 ? ((m_stencilStart[2] - 1) / 2) : 0)
                      : CoarseBlockSize[2]};
      const int bytes = (e[0] - s[0]) * sizeof(ElementType);
      if (!bytes)
        continue;
      if (use_averages) {
#pragma GCC ivdep
        for (int iy = s[1]; iy < e[1]; iy += 1) {
          const int YY =
              (iy - s[1] - std::min(0, code[1]) * ((e[1] - s[1]) % 2)) / 2 +
              sC[1];
#pragma GCC ivdep
          for (int ix = s[0]; ix < e[0]; ix += 1) {
            const int XX =
                (ix - s[0] - std::min(0, code[0]) * ((e[0] - s[0]) % 2)) / 2 +
                sC[0];
            ElementType *Test[3][3];
            for (int i = 0; i < 3; i++)
              for (int j = 0; j < 3; j++)
                Test[i][j] = &m_CoarsenedBlock->Access(
                    XX - 1 + i - offset[0], YY - 1 + j - offset[1], 0);
            TestInterp(
                Test,
                m_cacheBlock->Access(ix - m_stencilStart[0],
                                     iy - m_stencilStart[1], 0),
                abs(ix - s[0] - std::min(0, code[0]) * ((e[0] - s[0]) % 2)) % 2,
                abs(iy - s[1] - std::min(0, code[1]) * ((e[1] - s[1]) % 2)) %
                    2);
          }
        }
      }
      if (m_refGrid->FiniteDifferences && abs(code[0]) + abs(code[1]) == 1) {
#pragma GCC ivdep
        for (int iy = s[1]; iy < e[1]; iy += 2) {
          const int YY =
              (iy - s[1] - std::min(0, code[1]) * ((e[1] - s[1]) % 2)) / 2 +
              sC[1] - offset[1];
          const int y =
              abs(iy - s[1] - std::min(0, code[1]) * ((e[1] - s[1]) % 2)) % 2;
          const int iyp = (abs(iy) % 2 == 1) ? -1 : 1;
          const double dy = 0.25 * (2 * y - 1);
#pragma GCC ivdep
          for (int ix = s[0]; ix < e[0]; ix += 2) {
            const int XX =
                (ix - s[0] - std::min(0, code[0]) * ((e[0] - s[0]) % 2)) / 2 +
                sC[0] - offset[0];
            const int x =
                abs(ix - s[0] - std::min(0, code[0]) * ((e[0] - s[0]) % 2)) % 2;
            const int ixp = (abs(ix) % 2 == 1) ? -1 : 1;
            const double dx = 0.25 * (2 * x - 1);
            if (ix < -2 || iy < -2 || ix > nX + 1 || iy > nY + 1)
              continue;
            if (code[0] != 0) {
              ElementType dudy, dudy2;
              if (YY + offset[1] == 0) {
                dudy = (-0.5 * m_CoarsenedBlock->Access(XX, YY + 2, 0) -
                        1.5 * m_CoarsenedBlock->Access(XX, YY, 0)) +
                       2.0 * m_CoarsenedBlock->Access(XX, YY + 1, 0);
                dudy2 = (m_CoarsenedBlock->Access(XX, YY + 2, 0) +
                         m_CoarsenedBlock->Access(XX, YY, 0)) -
                        2.0 * m_CoarsenedBlock->Access(XX, YY + 1, 0);
              } else if (YY + offset[1] == CoarseBlockSize[1] - 1) {
                dudy = (0.5 * m_CoarsenedBlock->Access(XX, YY - 2, 0) +
                        1.5 * m_CoarsenedBlock->Access(XX, YY, 0)) -
                       2.0 * m_CoarsenedBlock->Access(XX, YY - 1, 0);
                dudy2 = (m_CoarsenedBlock->Access(XX, YY - 2, 0) +
                         m_CoarsenedBlock->Access(XX, YY, 0)) -
                        2.0 * m_CoarsenedBlock->Access(XX, YY - 1, 0);
              } else {
                dudy = 0.5 * (m_CoarsenedBlock->Access(XX, YY + 1, 0) -
                              m_CoarsenedBlock->Access(XX, YY - 1, 0));
                dudy2 = (m_CoarsenedBlock->Access(XX, YY + 1, 0) +
                         m_CoarsenedBlock->Access(XX, YY - 1, 0)) -
                        2.0 * m_CoarsenedBlock->Access(XX, YY, 0);
              }
              m_cacheBlock->Access(ix - m_stencilStart[0],
                                   iy - m_stencilStart[1], 0) =
                  m_CoarsenedBlock->Access(XX, YY, 0) + dy * dudy +
                  (0.5 * dy * dy) * dudy2;
              if (iy + iyp >= s[1] && iy + iyp < e[1])
                m_cacheBlock->Access(ix - m_stencilStart[0],
                                     iy - m_stencilStart[1] + iyp, 0) =
                    m_CoarsenedBlock->Access(XX, YY, 0) - dy * dudy +
                    (0.5 * dy * dy) * dudy2;
              if (ix + ixp >= s[0] && ix + ixp < e[0])
                m_cacheBlock->Access(ix - m_stencilStart[0] + ixp,
                                     iy - m_stencilStart[1], 0) =
                    m_CoarsenedBlock->Access(XX, YY, 0) + dy * dudy +
                    (0.5 * dy * dy) * dudy2;
              if (ix + ixp >= s[0] && ix + ixp < e[0] && iy + iyp >= s[1] &&
                  iy + iyp < e[1])
                m_cacheBlock->Access(ix - m_stencilStart[0] + ixp,
                                     iy - m_stencilStart[1] + iyp, 0) =
                    m_CoarsenedBlock->Access(XX, YY, 0) - dy * dudy +
                    (0.5 * dy * dy) * dudy2;
            } else {
              ElementType dudx, dudx2;
              if (XX + offset[0] == 0) {
                dudx = (-0.5 * m_CoarsenedBlock->Access(XX + 2, YY, 0) -
                        1.5 * m_CoarsenedBlock->Access(XX, YY, 0)) +
                       2.0 * m_CoarsenedBlock->Access(XX + 1, YY, 0);
                dudx2 = (m_CoarsenedBlock->Access(XX + 2, YY, 0) +
                         m_CoarsenedBlock->Access(XX, YY, 0)) -
                        2.0 * m_CoarsenedBlock->Access(XX + 1, YY, 0);
              } else if (XX + offset[0] == CoarseBlockSize[0] - 1) {
                dudx = (0.5 * m_CoarsenedBlock->Access(XX - 2, YY, 0) +
                        1.5 * m_CoarsenedBlock->Access(XX, YY, 0)) -
                       2.0 * m_CoarsenedBlock->Access(XX - 1, YY, 0);
                dudx2 = (m_CoarsenedBlock->Access(XX - 2, YY, 0) +
                         m_CoarsenedBlock->Access(XX, YY, 0)) -
                        2.0 * m_CoarsenedBlock->Access(XX - 1, YY, 0);
              } else {
                dudx = 0.5 * (m_CoarsenedBlock->Access(XX + 1, YY, 0) -
                              m_CoarsenedBlock->Access(XX - 1, YY, 0));
                dudx2 = (m_CoarsenedBlock->Access(XX + 1, YY, 0) +
                         m_CoarsenedBlock->Access(XX - 1, YY, 0)) -
                        2.0 * m_CoarsenedBlock->Access(XX, YY, 0);
              }
              m_cacheBlock->Access(ix - m_stencilStart[0],
                                   iy - m_stencilStart[1], 0) =
                  m_CoarsenedBlock->Access(XX, YY, 0) + dx * dudx +
                  (0.5 * dx * dx) * dudx2;
              if (iy + iyp >= s[1] && iy + iyp < e[1])
                m_cacheBlock->Access(ix - m_stencilStart[0],
                                     iy - m_stencilStart[1] + iyp, 0) =
                    m_CoarsenedBlock->Access(XX, YY, 0) + dx * dudx +
                    (0.5 * dx * dx) * dudx2;
              if (ix + ixp >= s[0] && ix + ixp < e[0])
                m_cacheBlock->Access(ix - m_stencilStart[0] + ixp,
                                     iy - m_stencilStart[1], 0) =
                    m_CoarsenedBlock->Access(XX, YY, 0) - dx * dudx +
                    (0.5 * dx * dx) * dudx2;
              if (ix + ixp >= s[0] && ix + ixp < e[0] && iy + iyp >= s[1] &&
                  iy + iyp < e[1])
                m_cacheBlock->Access(ix - m_stencilStart[0] + ixp,
                                     iy - m_stencilStart[1] + iyp, 0) =
                    m_CoarsenedBlock->Access(XX, YY, 0) - dx * dudx +
                    (0.5 * dx * dx) * dudx2;
            }
          }
        }
        for (int iy = s[1]; iy < e[1]; iy += 1) {
#pragma GCC ivdep
          for (int ix = s[0]; ix < e[0]; ix += 1) {
            if (ix < -2 || iy < -2 || ix > nX + 1 || iy > nY + 1)
              continue;
            const int x =
                abs(ix - s[0] - std::min(0, code[0]) * ((e[0] - s[0]) % 2)) % 2;
            const int y =
                abs(iy - s[1] - std::min(0, code[1]) * ((e[1] - s[1]) % 2)) % 2;
            auto &a = m_cacheBlock->Access(ix - m_stencilStart[0],
                                           iy - m_stencilStart[1], 0);
            if (code[0] == 0 && code[1] == 1) {
              if (y == 0) {
                auto &b = m_cacheBlock->Access(ix - m_stencilStart[0],
                                               iy - m_stencilStart[1] - 1, 0);
                auto &c = m_cacheBlock->Access(ix - m_stencilStart[0],
                                               iy - m_stencilStart[1] - 2, 0);
                LI(a, b, c);
              } else if (y == 1) {
                auto &b = m_cacheBlock->Access(ix - m_stencilStart[0],
                                               iy - m_stencilStart[1] - 2, 0);
                auto &c = m_cacheBlock->Access(ix - m_stencilStart[0],
                                               iy - m_stencilStart[1] - 3, 0);
                LE(a, b, c);
              }
            } else if (code[0] == 0 && code[1] == -1) {
              if (y == 1) {
                auto &b = m_cacheBlock->Access(ix - m_stencilStart[0],
                                               iy - m_stencilStart[1] + 1, 0);
                auto &c = m_cacheBlock->Access(ix - m_stencilStart[0],
                                               iy - m_stencilStart[1] + 2, 0);
                LI(a, b, c);
              } else if (y == 0) {
                auto &b = m_cacheBlock->Access(ix - m_stencilStart[0],
                                               iy - m_stencilStart[1] + 2, 0);
                auto &c = m_cacheBlock->Access(ix - m_stencilStart[0],
                                               iy - m_stencilStart[1] + 3, 0);
                LE(a, b, c);
              }
            } else if (code[1] == 0 && code[0] == 1) {
              if (x == 0) {
                auto &b = m_cacheBlock->Access(ix - m_stencilStart[0] - 1,
                                               iy - m_stencilStart[1], 0);
                auto &c = m_cacheBlock->Access(ix - m_stencilStart[0] - 2,
                                               iy - m_stencilStart[1], 0);
                LI(a, b, c);
              } else if (x == 1) {
                auto &b = m_cacheBlock->Access(ix - m_stencilStart[0] - 2,
                                               iy - m_stencilStart[1], 0);
                auto &c = m_cacheBlock->Access(ix - m_stencilStart[0] - 3,
                                               iy - m_stencilStart[1], 0);
                LE(a, b, c);
              }
            } else if (code[1] == 0 && code[0] == -1) {
              if (x == 1) {
                auto &b = m_cacheBlock->Access(ix - m_stencilStart[0] + 1,
                                               iy - m_stencilStart[1], 0);
                auto &c = m_cacheBlock->Access(ix - m_stencilStart[0] + 2,
                                               iy - m_stencilStart[1], 0);
                LI(a, b, c);
              } else if (x == 0) {
                auto &b = m_cacheBlock->Access(ix - m_stencilStart[0] + 2,
                                               iy - m_stencilStart[1], 0);
                auto &c = m_cacheBlock->Access(ix - m_stencilStart[0] + 3,
                                               iy - m_stencilStart[1], 0);
                LE(a, b, c);
              }
            }
          }
        }
      }
    }
  }
  virtual void _apply_bc(const BlockInfo &info, const Real t = 0,
                         bool coarse = false) {}
  template <typename T> void _release(T *&t) {
    if (t != NULL) {
      delete t;
    }
    t = NULL;
  }
  BlockLab(const BlockLab &) = delete;
  BlockLab &operator=(const BlockLab &) = delete;
};
template <typename MyBlockLab> struct BlockLabMPI : public MyBlockLab {
  using GridType = typename MyBlockLab::GridType;
  using BlockType = typename GridType::BlockType;
  typedef SynchronizerMPI_AMR<GridType> SynchronizerMPIType;
  SynchronizerMPIType *refSynchronizerMPI;
  virtual void prepare(GridType &grid, const StencilInfo &stencil,
                       const int[3] = default_start,
                       const int[3] = default_end) override {
    auto itSynchronizerMPI = grid.SynchronizerMPIs.find(stencil);
    refSynchronizerMPI = itSynchronizerMPI->second;
    MyBlockLab::prepare(grid, stencil);
  }
  virtual void load(const BlockInfo &info, const Real t = 0,
                    const bool applybc = true) override {
    MyBlockLab::load(info, t, applybc);
    Real *dst = (Real *)&MyBlockLab ::m_cacheBlock->LinAccess(0);
    Real *dst1 = (Real *)&MyBlockLab ::m_CoarsenedBlock->LinAccess(0);
    refSynchronizerMPI->fetch(info, MyBlockLab::m_cacheBlock->getSize(),
                              MyBlockLab::m_CoarsenedBlock->getSize(), dst,
                              dst1);
    if (MyBlockLab::m_refGrid->get_world_size() > 1)
      MyBlockLab::post_load(info, t, applybc);
  }
};
template <typename TGrid> struct LoadBalancer {
  typedef typename TGrid::Block BlockType;
  typedef typename TGrid::Block::ElementType ElementType;
  bool movedBlocks;
  TGrid *grid;
  MPI_Datatype MPI_BLOCK;
  struct MPI_Block {
    long long mn[2];
    Real data[sizeof(BlockType) / sizeof(Real)];
    MPI_Block(const BlockInfo &info, const bool Fillptr = true) {
      prepare(info, Fillptr);
    }
    void prepare(const BlockInfo &info, const bool Fillptr = true) {
      mn[0] = info.level;
      mn[1] = info.Z;
      if (Fillptr) {
        Real *aux = &((BlockType *)info.ptrBlock)->data[0][0].member(0);
        std::memcpy(&data[0], aux, sizeof(BlockType));
      }
    }
    MPI_Block() {}
  };
  void AddBlock(const int level, const long long Z, Real *data) {
    grid->_alloc(level, Z);
    BlockInfo &info = grid->getBlockInfoAll(level, Z);
    BlockType *b1 = (BlockType *)info.ptrBlock;
    assert(b1 != NULL);
    Real *a1 = &b1->data[0][0].member(0);
    std::memcpy(a1, data, sizeof(BlockType));
    int p[2];
    BlockInfo::inverse(Z, level, p[0], p[1]);
    if (level < grid->getlevelMax() - 1)
      for (int j1 = 0; j1 < 2; j1++)
        for (int i1 = 0; i1 < 2; i1++) {
          const long long nc =
              grid->getZforward(level + 1, 2 * p[0] + i1, 2 * p[1] + j1);
          grid->Tree(level + 1, nc).setCheckCoarser();
        }
    if (level > 0) {
      const long long nf = grid->getZforward(level - 1, p[0] / 2, p[1] / 2);
      grid->Tree(level - 1, nf).setCheckFiner();
    }
  }
  LoadBalancer(TGrid &a_grid) {
    grid = &a_grid;
    movedBlocks = false;
    int array_of_blocklengths[2] = {2, sizeof(BlockType) / sizeof(Real)};
    MPI_Aint array_of_displacements[2] = {0, 2 * sizeof(long long)};
    MPI_Datatype array_of_types[2];
    array_of_types[0] = MPI_LONG_LONG;
    if (sizeof(Real) == sizeof(float))
      array_of_types[1] = MPI_FLOAT;
    else if (sizeof(Real) == sizeof(double))
      array_of_types[1] = MPI_DOUBLE;
    else if (sizeof(Real) == sizeof(long double))
      array_of_types[1] = MPI_LONG_DOUBLE;
    MPI_Type_create_struct(2, array_of_blocklengths, array_of_displacements,
                           array_of_types, &MPI_BLOCK);
    MPI_Type_commit(&MPI_BLOCK);
  }
  ~LoadBalancer() { MPI_Type_free(&MPI_BLOCK); }
  void PrepareCompression() {
    const int size = grid->get_world_size();
    const int rank = grid->rank();
    std::vector<BlockInfo> &I = grid->m_vInfo;
    std::vector<std::vector<MPI_Block>> send_blocks(size);
    std::vector<std::vector<MPI_Block>> recv_blocks(size);
    for (auto &b : I) {
      const long long nBlock = grid->getZforward(b.level, 2 * (b.index[0] / 2),
                                                 2 * (b.index[1] / 2));
      const BlockInfo &base = grid->getBlockInfoAll(b.level, nBlock);
      if (!grid->Tree(base).Exists() || base.state != Compress)
        continue;
      const BlockInfo &bCopy = grid->getBlockInfoAll(b.level, b.Z);
      const int baserank = grid->Tree(b.level, nBlock).rank();
      const int brank = grid->Tree(b.level, b.Z).rank();
      if (b.Z != nBlock) {
        if (baserank != rank && brank == rank) {
          send_blocks[baserank].push_back({bCopy});
          grid->Tree(b.level, b.Z).setrank(baserank);
        }
      } else {
        for (int j = 0; j < 2; j++)
          for (int i = 0; i < 2; i++) {
            const long long n =
                grid->getZforward(b.level, b.index[0] + i, b.index[1] + j);
            if (n == nBlock)
              continue;
            BlockInfo &temp = grid->getBlockInfoAll(b.level, n);
            const int temprank = grid->Tree(b.level, n).rank();
            if (temprank != rank) {
              recv_blocks[temprank].push_back({temp, false});
              grid->Tree(b.level, n).setrank(baserank);
            }
          }
      }
    }
    std::vector<MPI_Request> requests;
    for (int r = 0; r < size; r++)
      if (r != rank) {
        if (recv_blocks[r].size() != 0) {
          MPI_Request req{};
          requests.push_back(req);
          MPI_Irecv(&recv_blocks[r][0], recv_blocks[r].size(), MPI_BLOCK, r,
                    2468, MPI_COMM_WORLD, &requests.back());
        }
        if (send_blocks[r].size() != 0) {
          MPI_Request req{};
          requests.push_back(req);
          MPI_Isend(&send_blocks[r][0], send_blocks[r].size(), MPI_BLOCK, r,
                    2468, MPI_COMM_WORLD, &requests.back());
        }
      }
    for (int r = 0; r < size; r++)
      for (int i = 0; i < (int)send_blocks[r].size(); i++) {
        grid->_dealloc(send_blocks[r][i].mn[0], send_blocks[r][i].mn[1]);
        grid->Tree(send_blocks[r][i].mn[0], send_blocks[r][i].mn[1])
            .setCheckCoarser();
      }
    if (requests.size() != 0) {
      movedBlocks = true;
      MPI_Waitall(requests.size(), &requests[0], MPI_STATUSES_IGNORE);
    }
    for (int r = 0; r < size; r++)
      for (int i = 0; i < (int)recv_blocks[r].size(); i++) {
        const int level = (int)recv_blocks[r][i].mn[0];
        const long long Z = recv_blocks[r][i].mn[1];
        grid->_alloc(level, Z);
        BlockInfo &info = grid->getBlockInfoAll(level, Z);
        BlockType *b1 = (BlockType *)info.ptrBlock;
        assert(b1 != NULL);
        Real *a1 = &b1->data[0][0].member(0);
        std::memcpy(a1, recv_blocks[r][i].data, sizeof(BlockType));
      }
  }
  void Balance_Diffusion(std::vector<long long> &block_distribution) {
    const int size = grid->get_world_size();
    const int rank = grid->rank();
    movedBlocks = false;
    {
      long long max_b = block_distribution[0];
      long long min_b = block_distribution[0];
      for (auto &b : block_distribution) {
        max_b = std::max(max_b, b);
        min_b = std::min(min_b, b);
      }
      const double ratio = static_cast<double>(max_b) / min_b;
      if (ratio > 1.01 || min_b == 0) {
        Balance_Global(block_distribution);
        return;
      }
    }
    const int right = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
    const int left = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    const int my_blocks = grid->m_vInfo.size();
    int right_blocks, left_blocks;
    MPI_Request reqs[4];
    MPI_Irecv(&left_blocks, 1, MPI_INT, left, 123, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(&right_blocks, 1, MPI_INT, right, 456, MPI_COMM_WORLD, &reqs[1]);
    MPI_Isend(&my_blocks, 1, MPI_INT, left, 456, MPI_COMM_WORLD, &reqs[2]);
    MPI_Isend(&my_blocks, 1, MPI_INT, right, 123, MPI_COMM_WORLD, &reqs[3]);
    MPI_Waitall(4, &reqs[0], MPI_STATUSES_IGNORE);
    const int nu = 4;
    const int flux_left = (rank == 0) ? 0 : (my_blocks - left_blocks) / nu;
    const int flux_right =
        (rank == size - 1) ? 0 : (my_blocks - right_blocks) / nu;
    std::vector<BlockInfo> SortedInfos = grid->m_vInfo;
    if (flux_right != 0 || flux_left != 0)
      std::sort(SortedInfos.begin(), SortedInfos.end());
    std::vector<MPI_Block> send_left;
    std::vector<MPI_Block> recv_left;
    std::vector<MPI_Block> send_right;
    std::vector<MPI_Block> recv_right;
    std::vector<MPI_Request> request;
    if (flux_left > 0) {
      send_left.resize(flux_left);
#pragma omp parallel for schedule(runtime)
      for (int i = 0; i < flux_left; i++)
        send_left[i].prepare(SortedInfos[i]);
      MPI_Request req{};
      request.push_back(req);
      MPI_Isend(&send_left[0], send_left.size(), MPI_BLOCK, left, 7890,
                MPI_COMM_WORLD, &request.back());
    } else if (flux_left < 0) {
      recv_left.resize(abs(flux_left));
      MPI_Request req{};
      request.push_back(req);
      MPI_Irecv(&recv_left[0], recv_left.size(), MPI_BLOCK, left, 4560,
                MPI_COMM_WORLD, &request.back());
    }
    if (flux_right > 0) {
      send_right.resize(flux_right);
#pragma omp parallel for schedule(runtime)
      for (int i = 0; i < flux_right; i++)
        send_right[i].prepare(SortedInfos[my_blocks - i - 1]);
      MPI_Request req{};
      request.push_back(req);
      MPI_Isend(&send_right[0], send_right.size(), MPI_BLOCK, right, 4560,
                MPI_COMM_WORLD, &request.back());
    } else if (flux_right < 0) {
      recv_right.resize(abs(flux_right));
      MPI_Request req{};
      request.push_back(req);
      MPI_Irecv(&recv_right[0], recv_right.size(), MPI_BLOCK, right, 7890,
                MPI_COMM_WORLD, &request.back());
    }
    for (int i = 0; i < flux_right; i++) {
      BlockInfo &info = SortedInfos[my_blocks - i - 1];
      grid->_dealloc(info.level, info.Z);
      grid->Tree(info.level, info.Z).setrank(right);
    }
    for (int i = 0; i < flux_left; i++) {
      BlockInfo &info = SortedInfos[i];
      grid->_dealloc(info.level, info.Z);
      grid->Tree(info.level, info.Z).setrank(left);
    }
    if (request.size() != 0) {
      movedBlocks = true;
      MPI_Waitall(request.size(), &request[0], MPI_STATUSES_IGNORE);
    }
    int temp = movedBlocks ? 1 : 0;
    MPI_Request request_reduction;
    MPI_Iallreduce(MPI_IN_PLACE, &temp, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD,
                   &request_reduction);
    for (int i = 0; i < -flux_left; i++)
      AddBlock(recv_left[i].mn[0], recv_left[i].mn[1], recv_left[i].data);
    for (int i = 0; i < -flux_right; i++)
      AddBlock(recv_right[i].mn[0], recv_right[i].mn[1], recv_right[i].data);
    MPI_Wait(&request_reduction, MPI_STATUS_IGNORE);
    movedBlocks = (temp >= 1);
    grid->FillPos();
  }
  void Balance_Global(std::vector<long long> &all_b) {
    const int size = grid->get_world_size();
    const int rank = grid->rank();
    std::vector<BlockInfo> SortedInfos = grid->m_vInfo;
    std::sort(SortedInfos.begin(), SortedInfos.end());
    long long total_load = 0;
    for (int r = 0; r < size; r++)
      total_load += all_b[r];
    long long my_load = total_load / size;
    if (rank < (total_load % size))
      my_load += 1;
    std::vector<long long> index_start(size);
    index_start[0] = 0;
    for (int r = 1; r < size; r++)
      index_start[r] = index_start[r - 1] + all_b[r - 1];
    long long ideal_index = (total_load / size) * rank;
    ideal_index += (rank < (total_load % size)) ? rank : (total_load % size);
    std::vector<std::vector<MPI_Block>> send_blocks(size);
    std::vector<std::vector<MPI_Block>> recv_blocks(size);
    for (int r = 0; r < size; r++)
      if (rank != r) {
        {
          const long long a1 = ideal_index;
          const long long a2 = ideal_index + my_load - 1;
          const long long b1 = index_start[r];
          const long long b2 = index_start[r] + all_b[r] - 1;
          const long long c1 = std::max(a1, b1);
          const long long c2 = std::min(a2, b2);
          if (c2 - c1 + 1 > 0)
            recv_blocks[r].resize(c2 - c1 + 1);
        }
        {
          long long other_ideal_index = (total_load / size) * r;
          other_ideal_index +=
              (r < (total_load % size)) ? r : (total_load % size);
          long long other_load = total_load / size;
          if (r < (total_load % size))
            other_load += 1;
          const long long a1 = other_ideal_index;
          const long long a2 = other_ideal_index + other_load - 1;
          const long long b1 = index_start[rank];
          const long long b2 = index_start[rank] + all_b[rank] - 1;
          const long long c1 = std::max(a1, b1);
          const long long c2 = std::min(a2, b2);
          if (c2 - c1 + 1 > 0)
            send_blocks[r].resize(c2 - c1 + 1);
        }
      }
    int tag = 12345;
    std::vector<MPI_Request> requests;
    for (int r = 0; r < size; r++)
      if (recv_blocks[r].size() != 0) {
        MPI_Request req{};
        requests.push_back(req);
        MPI_Irecv(recv_blocks[r].data(), recv_blocks[r].size(), MPI_BLOCK, r,
                  tag, MPI_COMM_WORLD, &requests.back());
      }
    long long counter_S = 0;
    long long counter_E = 0;
    for (int r = 0; r < rank; r++)
      if (send_blocks[r].size() != 0) {
        for (size_t i = 0; i < send_blocks[r].size(); i++)
          send_blocks[r][i].prepare(SortedInfos[counter_S + i]);
        counter_S += send_blocks[r].size();
        MPI_Request req{};
        requests.push_back(req);
        MPI_Isend(send_blocks[r].data(), send_blocks[r].size(), MPI_BLOCK, r,
                  tag, MPI_COMM_WORLD, &requests.back());
      }
    for (int r = size - 1; r > rank; r--)
      if (send_blocks[r].size() != 0) {
        for (size_t i = 0; i < send_blocks[r].size(); i++)
          send_blocks[r][i].prepare(
              SortedInfos[SortedInfos.size() - 1 - (counter_E + i)]);
        counter_E += send_blocks[r].size();
        MPI_Request req{};
        requests.push_back(req);
        MPI_Isend(send_blocks[r].data(), send_blocks[r].size(), MPI_BLOCK, r,
                  tag, MPI_COMM_WORLD, &requests.back());
      }
    movedBlocks = true;
    std::vector<long long> deallocIDs;
    counter_S = 0;
    counter_E = 0;
    for (int r = 0; r < size; r++)
      if (send_blocks[r].size() != 0) {
        if (r < rank) {
          for (size_t i = 0; i < send_blocks[r].size(); i++) {
            BlockInfo &info = SortedInfos[counter_S + i];
            deallocIDs.push_back(info.blockID_2);
            grid->Tree(info.level, info.Z).setrank(r);
          }
          counter_S += send_blocks[r].size();
        } else {
          for (size_t i = 0; i < send_blocks[r].size(); i++) {
            BlockInfo &info =
                SortedInfos[SortedInfos.size() - 1 - (counter_E + i)];
            deallocIDs.push_back(info.blockID_2);
            grid->Tree(info.level, info.Z).setrank(r);
          }
          counter_E += send_blocks[r].size();
        }
      }
    grid->dealloc_many(deallocIDs);
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
#pragma omp parallel
    {
      for (int r = 0; r < size; r++)
        if (recv_blocks[r].size() != 0) {
#pragma omp for
          for (size_t i = 0; i < recv_blocks[r].size(); i++)
            AddBlock(recv_blocks[r][i].mn[0], recv_blocks[r][i].mn[1],
                     recv_blocks[r][i].data);
        }
    }
    grid->FillPos();
  }
};
template <typename TLab> struct MeshAdaptation {
  typedef typename TLab::GridType TGrid;
  typedef typename TGrid::Block BlockType;
  typedef typename TGrid::BlockType::ElementType ElementType;
  typedef SynchronizerMPI_AMR<TGrid> SynchronizerMPIType;
  StencilInfo stencil;
  bool CallValidStates;
  bool boundary_needed;
  LoadBalancer<TGrid> *Balancer;
  TGrid *grid;
  double time;
  bool basic_refinement;
  double tolerance_for_refinement;
  double tolerance_for_compression;
  std::vector<long long> dealloc_IDs;
  MeshAdaptation(TGrid &g, double Rtol, double Ctol) {
    grid = &g;
    tolerance_for_refinement = Rtol;
    tolerance_for_compression = Ctol;
    boundary_needed = false;
    constexpr int Gx = 1;
    constexpr int Gy = 1;
    constexpr int Gz = DIMENSION == 3 ? 1 : 0;
    stencil.sx = -Gx;
    stencil.sy = -Gy;
    stencil.sz = -Gz;
    stencil.ex = Gx + 1;
    stencil.ey = Gy + 1;
    stencil.ez = Gz + 1;
    stencil.tensorial = true;
    for (int i = 0; i < ElementType::DIM; i++)
      stencil.selcomponents.push_back(i);
    Balancer = new LoadBalancer<TGrid>(*grid);
  }
  void Tag(double t = 0) {
    time = t;
    boundary_needed = true;
    SynchronizerMPI_AMR<TGrid> *Synch = grid->sync(stencil);
    CallValidStates = false;
    bool Reduction = false;
    MPI_Request Reduction_req;
    int tmp;
    std::vector<BlockInfo *> &inner = Synch->avail_inner();
    TagBlocksVector(inner, Reduction, Reduction_req, tmp);
    std::vector<BlockInfo *> &halo = Synch->avail_halo();
    TagBlocksVector(halo, Reduction, Reduction_req, tmp);
    if (!Reduction) {
      tmp = CallValidStates ? 1 : 0;
      Reduction = true;
      MPI_Iallreduce(MPI_IN_PLACE, &tmp, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD,
                     &Reduction_req);
    }
    MPI_Wait(&Reduction_req, MPI_STATUS_IGNORE);
    CallValidStates = (tmp > 0);
    grid->boundary = halo;
    if (CallValidStates)
      ValidStates();
  }
  void Adapt(bool basic) {
    basic_refinement = basic;
    SynchronizerMPI_AMR<TGrid> *Synch = nullptr;
    if (basic == false) {
      Synch = grid->sync(stencil);
      grid->boundary = Synch->avail_halo();
      if (boundary_needed)
        grid->UpdateBoundary();
    }
    int r = 0;
    int c = 0;
    std::vector<int> m_com;
    std::vector<int> m_ref;
    std::vector<long long> n_com;
    std::vector<long long> n_ref;
    std::vector<BlockInfo> &I = grid->m_vInfo;
    long long blocks_after = I.size();
    for (auto &info : I) {
      if (info.state == Refine) {
        m_ref.push_back(info.level);
        n_ref.push_back(info.Z);
        blocks_after += (1 << DIMENSION) - 1;
        r++;
      } else if (info.state == Compress && info.index[0] % 2 == 0 &&
                 info.index[1] % 2 == 0 && info.index[2] % 2 == 0) {
        m_com.push_back(info.level);
        n_com.push_back(info.Z);
        c++;
      } else if (info.state == Compress) {
        blocks_after--;
      }
    }
    MPI_Request requests[2];
    int temp[2] = {r, c};
    int result[2];
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::vector<long long> block_distribution(size);
    MPI_Iallreduce(&temp, &result, 2, MPI_INT, MPI_SUM, MPI_COMM_WORLD,
                   &requests[0]);
    MPI_Iallgather(&blocks_after, 1, MPI_LONG_LONG, block_distribution.data(),
                   1, MPI_LONG_LONG, MPI_COMM_WORLD, &requests[1]);
    dealloc_IDs.clear();
    {
      TLab lab;
      if (Synch != nullptr)
        lab.prepare(*grid, Synch->stencil);
      for (size_t i = 0; i < m_ref.size(); i++) {
        refine_1(m_ref[i], n_ref[i], lab);
      }
      for (size_t i = 0; i < m_ref.size(); i++) {
        refine_2(m_ref[i], n_ref[i]);
      }
    }
    grid->dealloc_many(dealloc_IDs);
    Balancer->PrepareCompression();
    dealloc_IDs.clear();
    for (size_t i = 0; i < m_com.size(); i++) {
      compress(m_com[i], n_com[i]);
    }
    grid->dealloc_many(dealloc_IDs);
    MPI_Waitall(2, requests, MPI_STATUS_IGNORE);
    Balancer->Balance_Diffusion(block_distribution);
    if (result[0] > 0 || result[1] > 0 || Balancer->movedBlocks) {
      grid->UpdateFluxCorrection = true;
      grid->UpdateGroups = true;
      grid->UpdateBlockInfoAll_States(false);
      auto it = grid->SynchronizerMPIs.begin();
      while (it != grid->SynchronizerMPIs.end()) {
        (*it->second)._Setup();
        it++;
      }
    }
  }
  void TagLike(const std::vector<BlockInfo> &I1) {
    std::vector<BlockInfo> &I2 = grid->m_vInfo;
    for (size_t i1 = 0; i1 < I2.size(); i1++) {
      BlockInfo &ary0 = I2[i1];
      BlockInfo &info = grid->getBlockInfoAll(ary0.level, ary0.Z);
      for (int i = 2 * (info.index[0] / 2); i <= 2 * (info.index[0] / 2) + 1;
           i++)
        for (int j = 2 * (info.index[1] / 2); j <= 2 * (info.index[1] / 2) + 1;
             j++) {
          const long long n = grid->getZforward(info.level, i, j);
          BlockInfo &infoNei = grid->getBlockInfoAll(info.level, n);
          infoNei.state = Leave;
        }
      info.state = Leave;
      ary0.state = Leave;
    }
#pragma omp parallel for
    for (size_t i = 0; i < I1.size(); i++) {
      const BlockInfo &info1 = I1[i];
      BlockInfo &info2 = I2[i];
      BlockInfo &info3 = grid->getBlockInfoAll(info2.level, info2.Z);
      info2.state = info1.state;
      info3.state = info1.state;
      if (info2.state == Compress) {
        const int i2 = 2 * (info2.index[0] / 2);
        const int j2 = 2 * (info2.index[1] / 2);
        const long long n = grid->getZforward(info2.level, i2, j2);
        BlockInfo &infoNei = grid->getBlockInfoAll(info2.level, n);
        infoNei.state = Compress;
      }
    }
  }
  void TagBlocksVector(std::vector<BlockInfo *> &I, bool &Reduction,
                       MPI_Request &Reduction_req, int &tmp) {
    const int levelMax = grid->getlevelMax();
#pragma omp parallel
    {
#pragma omp for schedule(dynamic, 1)
      for (size_t i = 0; i < I.size(); i++) {
        BlockInfo &info = grid->getBlockInfoAll(I[i]->level, I[i]->Z);
        I[i]->state = TagLoadedBlock(info);
        const bool maxLevel =
            (I[i]->state == Refine) && (I[i]->level == levelMax - 1);
        const bool minLevel = (I[i]->state == Compress) && (I[i]->level == 0);
        if (maxLevel || minLevel)
          I[i]->state = Leave;
        info.state = I[i]->state;
        if (info.state != Leave) {
#pragma omp critical
          {
            CallValidStates = true;
            if (!Reduction) {
              tmp = 1;
              Reduction = true;
              MPI_Iallreduce(MPI_IN_PLACE, &tmp, 1, MPI_INT, MPI_SUM,
                             MPI_COMM_WORLD, &Reduction_req);
            }
          }
        }
      }
    }
  }
  void refine_1(const int level, const long long Z, TLab &lab) {
    BlockInfo &parent = grid->getBlockInfoAll(level, Z);
    parent.state = Leave;
    if (basic_refinement == false)
      lab.load(parent, time, true);
    const int p[3] = {parent.index[0], parent.index[1], parent.index[2]};
    assert(parent.ptrBlock != NULL);
    assert(level <= grid->getlevelMax() - 1);
    BlockType *Blocks[4];
    for (int j = 0; j < 2; j++)
      for (int i = 0; i < 2; i++) {
        const long long nc =
            grid->getZforward(level + 1, 2 * p[0] + i, 2 * p[1] + j);
        BlockInfo &Child = grid->getBlockInfoAll(level + 1, nc);
        Child.state = Leave;
        grid->_alloc(level + 1, nc);
        grid->Tree(level + 1, nc).setCheckCoarser();
        Blocks[j * 2 + i] = (BlockType *)Child.ptrBlock;
      }
    if (basic_refinement == false)
      RefineBlocks(Blocks, lab);
  }
  void refine_2(const int level, const long long Z) {
#pragma omp critical
    { dealloc_IDs.push_back(grid->getBlockInfoAll(level, Z).blockID_2); }
    BlockInfo &parent = grid->getBlockInfoAll(level, Z);
    grid->Tree(parent).setCheckFiner();
    parent.state = Leave;
    int p[3] = {parent.index[0], parent.index[1], parent.index[2]};
    for (int j = 0; j < 2; j++)
      for (int i = 0; i < 2; i++) {
        const long long nc =
            grid->getZforward(level + 1, 2 * p[0] + i, 2 * p[1] + j);
        BlockInfo &Child = grid->getBlockInfoAll(level + 1, nc);
        grid->Tree(Child).setrank(grid->rank());
        if (level + 2 < grid->getlevelMax())
          for (int i0 = 0; i0 < 2; i0++)
            for (int i1 = 0; i1 < 2; i1++)
              grid->Tree(level + 2, Child.Zchild[i0][i1][1]).setCheckCoarser();
      }
  }
  void compress(const int level, const long long Z) {
    assert(level > 0);
    BlockInfo &info = grid->getBlockInfoAll(level, Z);
    assert(info.state == Compress);
    BlockType *Blocks[4];
    for (int J = 0; J < 2; J++)
      for (int I = 0; I < 2; I++) {
        const int blk = J * 2 + I;
        const long long n =
            grid->getZforward(level, info.index[0] + I, info.index[1] + J);
        Blocks[blk] = (BlockType *)(grid->getBlockInfoAll(level, n)).ptrBlock;
      }
    const int nx = _BS_;
    const int ny = _BS_;
    const int offsetX[2] = {0, nx / 2};
    const int offsetY[2] = {0, ny / 2};
    if (basic_refinement == false)
      for (int J = 0; J < 2; J++)
        for (int I = 0; I < 2; I++) {
          BlockType &b = *Blocks[J * 2 + I];
          for (int j = 0; j < ny; j += 2)
            for (int i = 0; i < nx; i += 2) {
              ElementType average = 0.25 * ((b(i, j) + b(i + 1, j + 1)) +
                                            (b(i + 1, j) + b(i, j + 1)));
              (*Blocks[0])(i / 2 + offsetX[I], j / 2 + offsetY[J]) = average;
            }
        }
    const long long np =
        grid->getZforward(level - 1, info.index[0] / 2, info.index[1] / 2);
    BlockInfo &parent = grid->getBlockInfoAll(level - 1, np);
    grid->Tree(parent.level, parent.Z).setrank(grid->rank());
    parent.ptrBlock = info.ptrBlock;
    parent.state = Leave;
    if (level - 2 >= 0)
      grid->Tree(level - 2, parent.Zparent).setCheckFiner();
    for (int J = 0; J < 2; J++)
      for (int I = 0; I < 2; I++) {
        const long long n =
            grid->getZforward(level, info.index[0] + I, info.index[1] + J);
        if (I + J == 0) {
          grid->FindBlockInfo(level, n, level - 1, np);
        } else {
#pragma omp critical
          { dealloc_IDs.push_back(grid->getBlockInfoAll(level, n).blockID_2); }
        }
        grid->Tree(level, n).setCheckCoarser();
        grid->getBlockInfoAll(level, n).state = Leave;
      }
  }
  void ValidStates() {
    const std::array<int, 3> blocksPerDim = grid->getMaxBlocks();
    const int levelMin = 0;
    const int levelMax = grid->getlevelMax();
    const bool xperiodic = grid->xperiodic;
    const bool yperiodic = grid->yperiodic;
    const bool zperiodic = grid->zperiodic;
    std::vector<BlockInfo> &I = grid->m_vInfo;
#pragma omp parallel for
    for (size_t j = 0; j < I.size(); j++) {
      BlockInfo &info = I[j];
      if ((info.state == Refine && info.level == levelMax - 1) ||
          (info.state == Compress && info.level == levelMin)) {
        info.state = Leave;
        (grid->getBlockInfoAll(info.level, info.Z)).state = Leave;
      }
      if (info.state != Leave) {
        info.changed2 = true;
        (grid->getBlockInfoAll(info.level, info.Z)).changed2 = info.changed2;
      }
    }
    bool clean_boundary = true;
    for (int m = levelMax - 1; m >= levelMin; m--) {
      for (size_t j = 0; j < I.size(); j++) {
        BlockInfo &info = I[j];
        if (info.level == m && info.state != Refine &&
            info.level != levelMax - 1) {
          const int TwoPower = 1 << info.level;
          const bool xskin = info.index[0] == 0 ||
                             info.index[0] == blocksPerDim[0] * TwoPower - 1;
          const bool yskin = info.index[1] == 0 ||
                             info.index[1] == blocksPerDim[1] * TwoPower - 1;
          const bool zskin = info.index[2] == 0 ||
                             info.index[2] == blocksPerDim[2] * TwoPower - 1;
          const int xskip = info.index[0] == 0 ? -1 : 1;
          const int yskip = info.index[1] == 0 ? -1 : 1;
          const int zskip = info.index[2] == 0 ? -1 : 1;
          for (int icode = 0; icode < 27; icode++) {
            if (info.state == Refine)
              break;
            if (icode == 1 * 1 + 3 * 1 + 9 * 1)
              continue;
            const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                                 (icode / 9) % 3 - 1};
            if (!xperiodic && code[0] == xskip && xskin)
              continue;
            if (!yperiodic && code[1] == yskip && yskin)
              continue;
            if (!zperiodic && code[2] == zskip && zskin)
              continue;
            if (code[2] != 0)
              continue;
            if (grid->Tree(info.level, info.Znei_(code[0], code[1], code[2]))
                    .CheckFiner()) {
              if (info.state == Compress) {
                info.state = Leave;
                (grid->getBlockInfoAll(info.level, info.Z)).state = Leave;
              }
              const int tmp = abs(code[0]) + abs(code[1]) + abs(code[2]);
              int Bstep = 1;
              if (tmp == 2)
                Bstep = 3;
              else if (tmp == 3)
                Bstep = 4;
              for (int B = 0; B <= 1; B += Bstep) {
                const int aux = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
                const int iNei = 2 * info.index[0] + std::max(code[0], 0) +
                                 code[0] +
                                 (B % 2) * std::max(0, 1 - abs(code[0]));
                const int jNei = 2 * info.index[1] + std::max(code[1], 0) +
                                 code[1] + aux * std::max(0, 1 - abs(code[1]));
                const long long zzz = grid->getZforward(m + 1, iNei, jNei);
                BlockInfo &FinerNei = grid->getBlockInfoAll(m + 1, zzz);
                State NeiState = FinerNei.state;
                if (NeiState == Refine) {
                  info.state = Refine;
                  (grid->getBlockInfoAll(info.level, info.Z)).state = Refine;
                  info.changed2 = true;
                  (grid->getBlockInfoAll(info.level, info.Z)).changed2 = true;
                  break;
                }
              }
            }
          }
        }
      }
      grid->UpdateBoundary(clean_boundary);
      clean_boundary = false;
      if (m == levelMin)
        break;
      for (size_t j = 0; j < I.size(); j++) {
        BlockInfo &info = I[j];
        if (info.level == m && info.state == Compress) {
          const int aux = 1 << info.level;
          const bool xskin =
              info.index[0] == 0 || info.index[0] == blocksPerDim[0] * aux - 1;
          const bool yskin =
              info.index[1] == 0 || info.index[1] == blocksPerDim[1] * aux - 1;
          const bool zskin =
              info.index[2] == 0 || info.index[2] == blocksPerDim[2] * aux - 1;
          const int xskip = info.index[0] == 0 ? -1 : 1;
          const int yskip = info.index[1] == 0 ? -1 : 1;
          const int zskip = info.index[2] == 0 ? -1 : 1;
          for (int icode = 0; icode < 27; icode++) {
            if (icode == 1 * 1 + 3 * 1 + 9 * 1)
              continue;
            const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                                 (icode / 9) % 3 - 1};
            if (!xperiodic && code[0] == xskip && xskin)
              continue;
            if (!yperiodic && code[1] == yskip && yskin)
              continue;
            if (!zperiodic && code[2] == zskip && zskin)
              continue;
            if (code[2] != 0)
              continue;
            BlockInfo &infoNei = grid->getBlockInfoAll(
                info.level, info.Znei_(code[0], code[1], code[2]));
            if (grid->Tree(infoNei).Exists() && infoNei.state == Refine) {
              info.state = Leave;
              (grid->getBlockInfoAll(info.level, info.Z)).state = Leave;
              break;
            }
          }
        }
      }
    }
    for (size_t jjj = 0; jjj < I.size(); jjj++) {
      BlockInfo &info = I[jjj];
      const int m = info.level;
      bool found = false;
      for (int i = 2 * (info.index[0] / 2); i <= 2 * (info.index[0] / 2) + 1;
           i++)
        for (int j = 2 * (info.index[1] / 2); j <= 2 * (info.index[1] / 2) + 1;
             j++)
          for (int k = 2 * (info.index[2] / 2);
               k <= 2 * (info.index[2] / 2) + 1; k++) {
            const long long n = grid->getZforward(m, i, j);
            BlockInfo &infoNei = grid->getBlockInfoAll(m, n);
            if (grid->Tree(infoNei).Exists() == false ||
                infoNei.state != Compress) {
              found = true;
              if (info.state == Compress) {
                info.state = Leave;
                (grid->getBlockInfoAll(info.level, info.Z)).state = Leave;
              }
              break;
            }
          }
      if (found)
        for (int i = 2 * (info.index[0] / 2); i <= 2 * (info.index[0] / 2) + 1;
             i++)
          for (int j = 2 * (info.index[1] / 2);
               j <= 2 * (info.index[1] / 2) + 1; j++)
            for (int k = 2 * (info.index[2] / 2);
                 k <= 2 * (info.index[2] / 2) + 1; k++) {
              const long long n = grid->getZforward(m, i, j);
              BlockInfo &infoNei = grid->getBlockInfoAll(m, n);
              if (grid->Tree(infoNei).Exists() && infoNei.state == Compress)
                infoNei.state = Leave;
            }
    }
  }
  virtual void RefineBlocks(BlockType *B[8], TLab &Lab) {
    const int nx = _BS_;
    const int ny = _BS_;
    int offsetX[2] = {0, nx / 2};
    int offsetY[2] = {0, ny / 2};
    for (int J = 0; J < 2; J++)
      for (int I = 0; I < 2; I++) {
        BlockType &b = *B[J * 2 + I];
        for (size_t y = 0; y < _BS_; y++)
          for (size_t x = 0; x < _BS_; x++)
            b.data[y][x].clear();
        for (int j = 0; j < ny; j += 2)
          for (int i = 0; i < nx; i += 2) {
            ElementType dudx =
                0.5 * (Lab(i / 2 + offsetX[I] + 1, j / 2 + offsetY[J]) -
                       Lab(i / 2 + offsetX[I] - 1, j / 2 + offsetY[J]));
            ElementType dudy =
                0.5 * (Lab(i / 2 + offsetX[I], j / 2 + offsetY[J] + 1) -
                       Lab(i / 2 + offsetX[I], j / 2 + offsetY[J] - 1));
            ElementType dudx2 =
                (Lab(i / 2 + offsetX[I] + 1, j / 2 + offsetY[J]) +
                 Lab(i / 2 + offsetX[I] - 1, j / 2 + offsetY[J])) -
                2.0 * Lab(i / 2 + offsetX[I], j / 2 + offsetY[J]);
            ElementType dudy2 =
                (Lab(i / 2 + offsetX[I], j / 2 + offsetY[J] + 1) +
                 Lab(i / 2 + offsetX[I], j / 2 + offsetY[J] - 1)) -
                2.0 * Lab(i / 2 + offsetX[I], j / 2 + offsetY[J]);
            ElementType dudxdy =
                0.25 * ((Lab(i / 2 + offsetX[I] + 1, j / 2 + offsetY[J] + 1) +
                         Lab(i / 2 + offsetX[I] - 1, j / 2 + offsetY[J] - 1)) -
                        (Lab(i / 2 + offsetX[I] + 1, j / 2 + offsetY[J] - 1) +
                         Lab(i / 2 + offsetX[I] - 1, j / 2 + offsetY[J] + 1)));
            b(i, j) = (Lab(i / 2 + offsetX[I], j / 2 + offsetY[J]) +
                       (-0.25 * dudx - 0.25 * dudy)) +
                      ((0.03125 * dudx2 + 0.03125 * dudy2) + 0.0625 * dudxdy);
            b(i + 1, j) =
                (Lab(i / 2 + offsetX[I], j / 2 + offsetY[J]) +
                 (+0.25 * dudx - 0.25 * dudy)) +
                ((0.03125 * dudx2 + 0.03125 * dudy2) - 0.0625 * dudxdy);
            b(i, j + 1) =
                (Lab(i / 2 + offsetX[I], j / 2 + offsetY[J]) +
                 (-0.25 * dudx + 0.25 * dudy)) +
                ((0.03125 * dudx2 + 0.03125 * dudy2) - 0.0625 * dudxdy);
            b(i + 1, j + 1) =
                (Lab(i / 2 + offsetX[I], j / 2 + offsetY[J]) +
                 (+0.25 * dudx + 0.25 * dudy)) +
                ((0.03125 * dudx2 + 0.03125 * dudy2) + 0.0625 * dudxdy);
          }
      }
  }
  virtual State TagLoadedBlock(BlockInfo &info) {
    const int nx = _BS_;
    const int ny = _BS_;
    BlockType &b = *(BlockType *)info.ptrBlock;
    double Linf = 0.0;
    for (int j = 0; j < ny; j++)
      for (int i = 0; i < nx; i++) {
        Linf = std::max(Linf, std::fabs(b(i, j).magnitude()));
      }
    if (Linf > tolerance_for_refinement)
      return Refine;
    else if (Linf < tolerance_for_compression)
      return Compress;
    return Leave;
  }
};
template <typename Lab, typename Kernel, typename TGrid,
          typename TGrid_corr = TGrid>
void compute(Kernel &&kernel, TGrid *g, TGrid_corr *g_corr = nullptr) {
  if (g_corr != nullptr)
    g_corr->Corrector.prepare(*g_corr);
  SynchronizerMPI_AMR<TGrid> &Synch = *(g->sync(kernel.stencil));
  std::vector<BlockInfo *> *inner = &Synch.avail_inner();
  std::vector<BlockInfo *> *halo_next;
  bool done = false;
#pragma omp parallel
  {
    Lab lab;
    lab.prepare(*g, kernel.stencil);
#pragma omp for nowait
    for (const auto &I : *inner) {
      lab.load(*I, 0);
      kernel(lab, *I);
    }
    while (done == false) {
#pragma omp master
      halo_next = &Synch.avail_next();
#pragma omp barrier
#pragma omp for nowait
      for (const auto &I : *halo_next) {
        lab.load(*I, 0);
        kernel(lab, *I);
      }
#pragma omp single
      {
        if (halo_next->size() == 0)
          done = true;
      }
    }
  }
  Synch.avail_halo();
  if (g_corr != nullptr)
    g_corr->Corrector.FillBlockCases();
}
template <typename Kernel, typename TGrid, typename LabMPI, typename TGrid2,
          typename LabMPI2, typename TGrid_corr = TGrid>
static void compute(const Kernel &kernel, TGrid &grid, TGrid2 &grid2,
                    const bool applyFluxCorrection = false,
                    TGrid_corr *corrected_grid = nullptr) {
  if (applyFluxCorrection)
    corrected_grid->Corrector.prepare(*corrected_grid);
  SynchronizerMPI_AMR<TGrid> &Synch = *grid.sync(kernel.stencil);
  Kernel kernel2 = kernel;
  kernel2.stencil.sx = kernel2.stencil2.sx;
  kernel2.stencil.sy = kernel2.stencil2.sy;
  kernel2.stencil.sz = kernel2.stencil2.sz;
  kernel2.stencil.ex = kernel2.stencil2.ex;
  kernel2.stencil.ey = kernel2.stencil2.ey;
  kernel2.stencil.ez = kernel2.stencil2.ez;
  kernel2.stencil.tensorial = kernel2.stencil2.tensorial;
  kernel2.stencil.selcomponents.clear();
  kernel2.stencil.selcomponents = kernel2.stencil2.selcomponents;
  SynchronizerMPI_AMR<TGrid2> &Synch2 = *grid2.sync(kernel2.stencil);
  const StencilInfo &stencil = Synch.stencil;
  const StencilInfo &stencil2 = Synch2.stencil;
  std::vector<BlockInfo> &blk = grid.m_vInfo;
  std::vector<bool> ready(blk.size(), false);
  std::vector<BlockInfo *> &avail0 = Synch.avail_inner();
  std::vector<BlockInfo *> &avail02 = Synch2.avail_inner();
  const int Ninner = avail0.size();
  std::vector<BlockInfo *> avail1;
  std::vector<BlockInfo *> avail12;
#pragma omp parallel
  {
    LabMPI lab;
    LabMPI2 lab2;
    lab.prepare(grid, stencil);
    lab2.prepare(grid2, stencil2);
#pragma omp for
    for (int i = 0; i < Ninner; i++) {
      const BlockInfo &I = *avail0[i];
      const BlockInfo &I2 = *avail02[i];
      lab.load(I, 0);
      lab2.load(I2, 0);
      kernel(lab, lab2, I, I2);
      ready[I.blockID] = true;
    }
#pragma omp master
    {
      avail1 = Synch.avail_halo();
      avail12 = Synch2.avail_halo();
    }
#pragma omp barrier
    const int Nhalo = avail1.size();
#pragma omp for
    for (int i = 0; i < Nhalo; i++) {
      const BlockInfo &I = *avail1[i];
      const BlockInfo &I2 = *avail12[i];
      lab.load(I, 0);
      lab2.load(I2, 0);
      kernel(lab, lab2, I, I2);
    }
  }
  if (applyFluxCorrection)
    corrected_grid->Corrector.FillBlockCases();
}
struct ScalarElement {
  Real s = 0;
  void clear() { s = 0; }
  ScalarElement &operator*=(const Real a) {
    this->s *= a;
    return *this;
  }
  ScalarElement &operator+=(const ScalarElement &rhs) {
    this->s += rhs.s;
    return *this;
  }
  ScalarElement &operator-=(const ScalarElement &rhs) {
    this->s -= rhs.s;
    return *this;
  }
  friend ScalarElement operator*(const Real a, ScalarElement el) {
    return (el *= a);
  }
  friend ScalarElement operator+(ScalarElement lhs, const ScalarElement &rhs) {
    return (lhs += rhs);
  }
  friend ScalarElement operator-(ScalarElement lhs, const ScalarElement &rhs) {
    return (lhs -= rhs);
  }
  Real magnitude() { return s; }
  Real &member(int i) { return s; }
  static constexpr int DIM = 1;
};
struct VectorElement {
  static constexpr int DIM = 2;
  Real u[2];
  VectorElement() { u[0] = u[1] = 0; }
  void clear() { u[0] = u[1] = 0; }
  VectorElement &operator=(const VectorElement &c) = default;
  VectorElement &operator*=(const Real a) {
    u[0] *= a;
    u[1] *= a;
    return *this;
  }
  VectorElement &operator+=(const VectorElement &rhs) {
    u[0] += rhs.u[0];
    u[1] += rhs.u[1];
    return *this;
  }
  VectorElement &operator-=(const VectorElement &rhs) {
    u[0] -= rhs.u[0];
    u[1] -= rhs.u[1];
    return *this;
  }
  friend VectorElement operator*(const Real a, VectorElement el) {
    return (el *= a);
  }
  friend VectorElement operator+(VectorElement lhs, const VectorElement &rhs) {
    return (lhs += rhs);
  }
  friend VectorElement operator-(VectorElement lhs, const VectorElement &rhs) {
    return (lhs -= rhs);
  }
  Real magnitude() { return sqrt(u[0] * u[0] + u[1] * u[1]); }
  Real &member(int i) { return u[i]; }
};
template <typename T> struct GridBlock {
  using ElementType = T;
  T data[_BS_][_BS_];
  const T &operator()(int ix, int iy = 0) const { return data[iy][ix]; }
  T &operator()(int ix, int iy = 0) { return data[iy][ix]; }
  GridBlock(const GridBlock &) = delete;
  GridBlock &operator=(const GridBlock &) = delete;
};
enum BCflag { freespace, periodic, wall };
BCflag string2BCflag(const std::string &strFlag) {
  if (strFlag == "periodic") {
    return periodic;
  } else if (strFlag == "freespace") {
    return freespace;
  } else if (strFlag == "wall") {
    return wall;
  } else {
    fprintf(stderr, "BC not recognized %s\n", strFlag.c_str());
    fflush(0);
    abort();
    return periodic;
  }
}
static BCflag cubismBCX;
static BCflag cubismBCY;
template <typename TGrid, typename ElementType>
struct BlockLabDirichlet : public BlockLab<TGrid, ElementType> {
  static constexpr int sizeX = _BS_;
  static constexpr int sizeY = _BS_;
  static constexpr int sizeZ = 1;
  virtual bool is_xperiodic() override { return cubismBCX == periodic; }
  virtual bool is_yperiodic() override { return cubismBCY == periodic; }
  virtual bool is_zperiodic() override { return false; }
  template <int dir, int side>
  void applyBCface(bool wall, bool coarse = false) {
    const int A = 1 - dir;
    if (!coarse) {
      auto *const cb = this->m_cacheBlock;
      int s[3] = {0, 0, 0}, e[3] = {0, 0, 0};
      const int *const stenBeg = this->m_stencilStart;
      const int *const stenEnd = this->m_stencilEnd;
      s[0] = dir == 0 ? (side == 0 ? stenBeg[0] : sizeX) : stenBeg[0];
      s[1] = dir == 1 ? (side == 0 ? stenBeg[1] : sizeY) : stenBeg[1];
      e[0] = dir == 0 ? (side == 0 ? 0 : sizeX + stenEnd[0] - 1)
                      : sizeX + stenEnd[0] - 1;
      e[1] = dir == 1 ? (side == 0 ? 0 : sizeY + stenEnd[1] - 1)
                      : sizeY + stenEnd[1] - 1;
      if (!wall)
        for (int iy = s[1]; iy < e[1]; iy++)
          for (int ix = s[0]; ix < e[0]; ix++) {
            const int x =
                (dir == 0 ? (side == 0 ? 0 : sizeX - 1) : ix) - stenBeg[0];
            const int y =
                (dir == 1 ? (side == 0 ? 0 : sizeY - 1) : iy) - stenBeg[1];
            cb->Access(ix - stenBeg[0], iy - stenBeg[1], 0).member(1 - A) =
                (-1.0) * cb->Access(x, y, 0).member(1 - A);
            cb->Access(ix - stenBeg[0], iy - stenBeg[1], 0).member(A) =
                cb->Access(x, y, 0).member(A);
          }
      else
        for (int iy = s[1]; iy < e[1]; iy++)
          for (int ix = s[0]; ix < e[0]; ix++) {
            const int x =
                (dir == 0 ? (side == 0 ? 0 : sizeX - 1) : ix) - stenBeg[0];
            const int y =
                (dir == 1 ? (side == 0 ? 0 : sizeY - 1) : iy) - stenBeg[1];
            cb->Access(ix - stenBeg[0], iy - stenBeg[1], 0) =
                (-1.0) * cb->Access(x, y, 0);
          }
    } else {
      auto *const cb = this->m_CoarsenedBlock;
      const int eI[3] = {
          (this->m_stencilEnd[0]) / 2 + 1 + this->m_InterpStencilEnd[0] - 1,
          (this->m_stencilEnd[1]) / 2 + 1 + this->m_InterpStencilEnd[1] - 1,
          (this->m_stencilEnd[2]) / 2 + 1 + this->m_InterpStencilEnd[2] - 1};
      const int sI[3] = {
          (this->m_stencilStart[0] - 1) / 2 + this->m_InterpStencilStart[0],
          (this->m_stencilStart[1] - 1) / 2 + this->m_InterpStencilStart[1],
          (this->m_stencilStart[2] - 1) / 2 + this->m_InterpStencilStart[2]};
      const int *const stenBeg = sI;
      const int *const stenEnd = eI;
      int s[3] = {0, 0, 0}, e[3] = {0, 0, 0};
      s[0] = dir == 0 ? (side == 0 ? stenBeg[0] : sizeX / 2) : stenBeg[0];
      s[1] = dir == 1 ? (side == 0 ? stenBeg[1] : sizeY / 2) : stenBeg[1];
      e[0] = dir == 0 ? (side == 0 ? 0 : sizeX / 2 + stenEnd[0] - 1)
                      : sizeX / 2 + stenEnd[0] - 1;
      e[1] = dir == 1 ? (side == 0 ? 0 : sizeY / 2 + stenEnd[1] - 1)
                      : sizeY / 2 + stenEnd[1] - 1;
      if (!wall)
        for (int iy = s[1]; iy < e[1]; iy++)
          for (int ix = s[0]; ix < e[0]; ix++) {
            const int x =
                (dir == 0 ? (side == 0 ? 0 : sizeX / 2 - 1) : ix) - stenBeg[0];
            const int y =
                (dir == 1 ? (side == 0 ? 0 : sizeY / 2 - 1) : iy) - stenBeg[1];
            cb->Access(ix - stenBeg[0], iy - stenBeg[1], 0).member(1 - A) =
                (-1.0) * cb->Access(x, y, 0).member(1 - A);
            cb->Access(ix - stenBeg[0], iy - stenBeg[1], 0).member(A) =
                cb->Access(x, y, 0).member(A);
          }
      else
        for (int iy = s[1]; iy < e[1]; iy++)
          for (int ix = s[0]; ix < e[0]; ix++) {
            const int x =
                (dir == 0 ? (side == 0 ? 0 : sizeX / 2 - 1) : ix) - stenBeg[0];
            const int y =
                (dir == 1 ? (side == 0 ? 0 : sizeY / 2 - 1) : iy) - stenBeg[1];
            cb->Access(ix - stenBeg[0], iy - stenBeg[1], 0) =
                (-1.0) * cb->Access(x, y, 0);
          }
    }
  }
  void _apply_bc(const BlockInfo &info, const Real t = 0,
                 const bool coarse = false) override {
    const BCflag BCX = cubismBCX;
    const BCflag BCY = cubismBCY;
    if (!coarse) {
      if (is_xperiodic() == false) {
        if (info.index[0] == 0)
          this->template applyBCface<0, 0>(BCX == wall);
        if (info.index[0] == this->NX - 1)
          this->template applyBCface<0, 1>(BCX == wall);
      }
      if (is_yperiodic() == false) {
        if (info.index[1] == 0)
          this->template applyBCface<1, 0>(BCY == wall);
        if (info.index[1] == this->NY - 1)
          this->template applyBCface<1, 1>(BCY == wall);
      }
    } else {
      if (is_xperiodic() == false) {
        if (info.index[0] == 0)
          this->template applyBCface<0, 0>(BCX == wall, coarse);
        if (info.index[0] == this->NX - 1)
          this->template applyBCface<0, 1>(BCX == wall, coarse);
      }
      if (is_yperiodic() == false) {
        if (info.index[1] == 0)
          this->template applyBCface<1, 0>(BCY == wall, coarse);
        if (info.index[1] == this->NY - 1)
          this->template applyBCface<1, 1>(BCY == wall, coarse);
      }
    }
  }
  BlockLabDirichlet() : BlockLab<TGrid, ElementType>() {}
  BlockLabDirichlet(const BlockLabDirichlet &) = delete;
  BlockLabDirichlet &operator=(const BlockLabDirichlet &) = delete;
};
template <typename TGrid, typename ElementType>
struct BlockLabNeumann : public BlockLab<TGrid, ElementType> {
  static constexpr int sizeX = _BS_;
  static constexpr int sizeY = _BS_;
  static constexpr int sizeZ = 1;
  template <int dir, int side> void Neumann2D(const bool coarse = false) {
    int stenBeg[2];
    int stenEnd[2];
    int bsize[2];
    if (!coarse) {
      stenEnd[0] = this->m_stencilEnd[0];
      stenEnd[1] = this->m_stencilEnd[1];
      stenBeg[0] = this->m_stencilStart[0];
      stenBeg[1] = this->m_stencilStart[1];
      bsize[0] = sizeX;
      bsize[1] = sizeY;
    } else {
      stenEnd[0] =
          (this->m_stencilEnd[0]) / 2 + 1 + this->m_InterpStencilEnd[0] - 1;
      stenEnd[1] =
          (this->m_stencilEnd[1]) / 2 + 1 + this->m_InterpStencilEnd[1] - 1;
      stenBeg[0] =
          (this->m_stencilStart[0] - 1) / 2 + this->m_InterpStencilStart[0];
      stenBeg[1] =
          (this->m_stencilStart[1] - 1) / 2 + this->m_InterpStencilStart[1];
      bsize[0] = sizeX / 2;
      bsize[1] = sizeY / 2;
    }
    auto *const cb = coarse ? this->m_CoarsenedBlock : this->m_cacheBlock;
    int s[2];
    int e[2];
    s[0] = dir == 0 ? (side == 0 ? stenBeg[0] : bsize[0]) : stenBeg[0];
    s[1] = dir == 1 ? (side == 0 ? stenBeg[1] : bsize[1]) : stenBeg[1];
    e[0] = dir == 0 ? (side == 0 ? 0 : bsize[0] + stenEnd[0] - 1)
                    : bsize[0] + stenEnd[0] - 1;
    e[1] = dir == 1 ? (side == 0 ? 0 : bsize[1] + stenEnd[1] - 1)
                    : bsize[1] + stenEnd[1] - 1;
    for (int iy = s[1]; iy < e[1]; iy++)
      for (int ix = s[0]; ix < e[0]; ix++)
        cb->Access(ix - stenBeg[0], iy - stenBeg[1], 0) = cb->Access(
            (dir == 0 ? (side == 0 ? 0 : bsize[0] - 1) : ix) - stenBeg[0],
            (dir == 1 ? (side == 0 ? 0 : bsize[1] - 1) : iy) - stenBeg[1], 0);
  }
  typedef typename TGrid::BlockType::ElementType ElementTypeBlock;
  bool is_xperiodic() override { return cubismBCX == periodic; }
  bool is_yperiodic() override { return cubismBCY == periodic; }
  bool is_zperiodic() override { return false; }
  BlockLabNeumann() = default;
  BlockLabNeumann(const BlockLabNeumann &) = delete;
  BlockLabNeumann &operator=(const BlockLabNeumann &) = delete;
  virtual void _apply_bc(const BlockInfo &info, const Real t = 0,
                         const bool coarse = false) override {
    if (is_xperiodic() == false) {
      if (info.index[0] == 0)
        Neumann2D<0, 0>(coarse);
      if (info.index[0] == this->NX - 1)
        Neumann2D<0, 1>(coarse);
    }
    if (is_yperiodic() == false) {
      if (info.index[1] == 0)
        Neumann2D<1, 0>(coarse);
      if (info.index[1] == this->NY - 1)
        Neumann2D<1, 1>(coarse);
    }
  }
};
typedef GridBlock<ScalarElement> ScalarBlock;
typedef GridBlock<VectorElement> VectorBlock;
typedef GridMPI<Grid<ScalarBlock, ScalarElement>, ScalarElement> ScalarGrid;
typedef GridMPI<Grid<VectorBlock, VectorElement>, VectorElement> VectorGrid;
typedef BlockLabMPI<BlockLabDirichlet<VectorGrid, VectorElement>> VectorLab;
typedef BlockLabMPI<BlockLabNeumann<ScalarGrid, ScalarElement>> ScalarLab;
typedef MeshAdaptation<ScalarLab> ScalarAMR;
typedef MeshAdaptation<VectorLab> VectorAMR;
struct FishSkin {
  size_t Npoints;
  Real *xSurf;
  Real *ySurf;
  Real *normXSurf;
  Real *normYSurf;
  Real *midX;
  Real *midY;
  FishSkin(const size_t N)
      : Npoints(N), xSurf(new Real[Npoints]), ySurf(new Real[Npoints]),
        normXSurf(new Real[Npoints - 1]), normYSurf(new Real[Npoints - 1]),
        midX(new Real[Npoints - 1]), midY(new Real[Npoints - 1]) {}
};
struct Shape;
static struct {
  int rank;
  int size;
  int bpdx;
  int bpdy;
  int levelMax;
  int levelStart;
  Real Rtol;
  Real Ctol;
  int AdaptSteps{20};
  bool bAdaptChiGradient;
  Real extent;
  Real extents[2];
  Real dt;
  Real CFL;
  int nsteps;
  Real endTime;
  Real lambda;
  Real dlm;
  Real nu;
  Real PoissonTol;
  Real PoissonTolRel;
  int maxPoissonRestarts;
  int maxPoissonIterations;
  int bMeanConstraint;
  int dumpFreq;
  Real dumpTime;
  ScalarGrid *chi = nullptr;
  VectorGrid *vel = nullptr;
  VectorGrid *vOld = nullptr;
  ScalarGrid *pres = nullptr;
  VectorGrid *tmpV = nullptr;
  ScalarGrid *tmp = nullptr;
  ScalarGrid *pold = nullptr;
  ScalarGrid *Cs = nullptr;
  std::vector<Shape *> shapes;
  Real time = 0;
  int step = 0;
  Real uinfx = 0;
  Real uinfy = 0;
  Real nextDumpTime = 0;
  std::vector<int> bCollisionID;
  Real minH;
  ScalarAMR *tmp_amr = nullptr;
  ScalarAMR *chi_amr = nullptr;
  ScalarAMR *pres_amr = nullptr;
  ScalarAMR *pold_amr = nullptr;
  VectorAMR *vel_amr = nullptr;
  VectorAMR *vOld_amr = nullptr;
  VectorAMR *tmpV_amr = nullptr;
  ScalarAMR *Cs_amr = nullptr;
} sim;
using CHI_MAT = Real[_BS_][_BS_];
using UDEFMAT = Real[_BS_][_BS_][2];
struct surface_data {
  const int ix, iy;
  const Real dchidx, dchidy, delta;
  surface_data(const int _ix, const int _iy, const Real Xdx, const Real Xdy,
               const Real D)
      : ix(_ix), iy(_iy), dchidx(Xdx), dchidy(Xdy), delta(D) {}
};
struct ObstacleBlock {
  Real chi[_BS_][_BS_];
  Real dist[_BS_][_BS_];
  Real udef[_BS_][_BS_][2];
  size_t n_surfPoints = 0;
  bool filled = false;
  std::vector<surface_data *> surface;
  Real *x_s = nullptr;
  Real *y_s = nullptr;
  Real *p_s = nullptr;
  Real *u_s = nullptr;
  Real *v_s = nullptr;
  Real *nx_s = nullptr;
  Real *ny_s = nullptr;
  Real *omega_s = nullptr;
  Real *uDef_s = nullptr;
  Real *vDef_s = nullptr;
  Real *fX_s = nullptr;
  Real *fY_s = nullptr;
  Real *fXv_s = nullptr;
  Real *fYv_s = nullptr;
  Real perimeter = 0, forcex = 0, forcey = 0, forcex_P = 0, forcey_P = 0;
  Real forcex_V = 0, forcey_V = 0, torque = 0, torque_P = 0, torque_V = 0;
  Real drag = 0, thrust = 0, lift = 0, Pout = 0, PoutNew = 0, PoutBnd = 0,
       defPower = 0, defPowerBnd = 0;
  Real circulation = 0;
  Real COM_x = 0;
  Real COM_y = 0;
  Real Mass = 0;
  ObstacleBlock() {
    clear_surface();
    std::fill(dist[0], dist[0] + _BS_ * _BS_, -1);
    std::fill(chi[0], chi[0] + _BS_ * _BS_, 0);
    memset(udef, 0, sizeof(Real) * _BS_ * _BS_ * 2);
    surface.reserve(4 * _BS_);
  }
  void clear_surface() {
    filled = false;
    n_surfPoints = 0;
    perimeter = forcex = forcey = forcex_P = forcey_P = 0;
    forcex_V = forcey_V = torque = torque_P = torque_V = drag = thrust = lift =
        0;
    Pout = PoutBnd = defPower = defPowerBnd = circulation = 0;
    for (auto &trash : surface) {
      if (trash == nullptr)
        continue;
      delete trash;
      trash = nullptr;
    }
    surface.clear();
    free(x_s);
    free(y_s);
    free(p_s);
    free(u_s);
    free(v_s);
    free(nx_s);
    free(ny_s);
    free(omega_s);
    free(uDef_s);
    free(vDef_s);
    free(fX_s);
    free(fY_s);
    free(fXv_s);
    free(fYv_s);
  }
};
struct KernelVorticity {
  KernelVorticity() {}
  const std::vector<BlockInfo> &tmpInfo = sim.tmp->m_vInfo;
  const StencilInfo stencil{-1, -1, 0, 2, 2, 1, false, {0, 1}};
  void operator()(VectorLab &lab, const BlockInfo &info) const {
    const Real i2h = 0.5 / info.h;
    auto &__restrict__ TMP = *(ScalarBlock *)tmpInfo[info.blockID].ptrBlock;
    for (int y = 0; y < _BS_; ++y)
      for (int x = 0; x < _BS_; ++x)
        TMP(x, y).s = i2h * ((lab(x, y - 1).u[0] - lab(x, y + 1).u[0]) +
                             (lab(x + 1, y).u[1] - lab(x - 1, y).u[1]));
  }
};
static void dump(Real time, ScalarGrid *grid, char *path) {
  long i, j, k, l, x, y, ncell, ncell_total, offset;
  char xyz_path[FILENAME_MAX], attr_path[FILENAME_MAX], xdmf_path[FILENAME_MAX],
      *xyz_base, *attr_base;
  MPI_File mpi_file;
  FILE *xmf;
  float *xyz, *attr;
  snprintf(xyz_path, sizeof xyz_path, "%s.xyz.raw", path);
  snprintf(attr_path, sizeof attr_path, "%s.attr.raw", path);
  snprintf(xdmf_path, sizeof xdmf_path, "%s.xdmf2", path);
  xyz_base = xyz_path;
  attr_base = attr_path;
  for (j = 0; xyz_path[j] != '\0'; j++) {
    if (xyz_path[j] == '/' && xyz_path[j + 1] != '\0') {
      xyz_base = &xyz_path[j + 1];
      attr_base = &attr_path[j + 1];
    }
  }
  ncell = grid->m_vInfo.size() * _BS_ * _BS_;
  MPI_Exscan(&ncell, &offset, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  if (sim.rank == 0)
    offset = 0;
  if (sim.rank == sim.size - 1) {
    ncell_total = ncell + offset;
    xmf = fopen(xdmf_path, "w");
    fprintf(xmf,
            "<Xdmf\n"
            "    Version=\"2.0\">\n"
            "  <Domain>\n"
            "    <Grid>\n"
            "      <Time Value=\"%.16e\"/>\n"
            "      <Topology\n"
            "	  Dimensions=\"%ld\"\n"
            "	  TopologyType=\"Quadrilateral\"/>\n"
            "      <Geometry\n"
            "	  GeometryType=\"XY\">\n"
            "	<DataItem\n"
            "	    Dimensions=\"%ld 2\"\n"
            "	    Format=\"Binary\">\n"
            "	  %s\n"
            "	</DataItem>\n"
            "      </Geometry>\n"
            "      <Attribute\n"
            "	  Name=\"vort\"\n"
            "	  Center=\"Cell\">\n"
            "	<DataItem\n"
            "	    Dimensions=\"%ld\"\n"
            "	    Format=\"Binary\">\n"
            "	  %s\n"
            "	</DataItem>\n"
            "      </Attribute>\n"
            "    </Grid>\n"
            "  </Domain>\n"
            "</Xdmf>\n",
            time, ncell_total, 4 * ncell_total, xyz_base, ncell_total,
            attr_base);
    fclose(xmf);
  }
  xyz = (float *)malloc(8 * ncell * sizeof *xyz);
  attr = (float *)malloc(ncell * sizeof *xyz);
  k = 0;
  l = 0;
  for (i = 0; i < grid->m_vInfo.size(); i++) {
    const BlockInfo &info = grid->m_vInfo[i];
    ScalarBlock &b = *(ScalarBlock *)info.ptrBlock;
    for (y = 0; y < _BS_; y++)
      for (x = 0; x < _BS_; x++) {
        double u, v;
        u = info.origin[0] + info.h * x;
        v = info.origin[1] + info.h * y;
        xyz[k++] = u;
        xyz[k++] = v;
        xyz[k++] = u;
        xyz[k++] = v + info.h;
        xyz[k++] = u + info.h;
        xyz[k++] = v + info.h;
        xyz[k++] = u + info.h;
        xyz[k++] = v;
        attr[l++] = b.data[y][x].s;
      }
  }
  MPI_File_open(MPI_COMM_WORLD, xyz_path, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &mpi_file);
  MPI_File_write_at_all(mpi_file, 8 * offset * sizeof *xyz, xyz,
                        8 * ncell * sizeof *xyz, MPI_BYTE, MPI_STATUS_IGNORE);
  MPI_File_close(&mpi_file);
  free(xyz);
  MPI_File_open(MPI_COMM_WORLD, attr_path, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &mpi_file);
  MPI_File_write_at_all(mpi_file, offset * sizeof *attr, attr,
                        ncell * sizeof *attr, MPI_BYTE, MPI_STATUS_IGNORE);
  MPI_File_close(&mpi_file);
  free(attr);
}
struct Integrals {
  const Real x, y, m, j, u, v, a;
  Integrals(Real _x, Real _y, Real _m, Real _j, Real _u, Real _v, Real _a)
      : x(_x), y(_y), m(_m), j(_j), u(_u), v(_v), a(_a) {}
  Integrals(const Integrals &c)
      : x(c.x), y(c.y), m(c.m), j(c.j), u(c.u), v(c.v), a(c.a) {}
};
struct IF2D_Interpolation1D {
  static void naturalCubicSpline(const Real *x, const Real *y, const unsigned n,
                                 const Real *xx, Real *yy, const unsigned nn) {
    return naturalCubicSpline(x, y, n, xx, yy, nn, 0);
  }
  static void naturalCubicSpline(const Real *x, const Real *y, const unsigned n,
                                 const Real *xx, Real *yy, const unsigned nn,
                                 const Real offset) {
    std::vector<Real> y2(n), u(n - 1);
    y2[0] = 0;
    u[0] = 0;
    for (unsigned i = 1; i < n - 1; i++) {
      const Real sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
      const Real p = sig * y2[i - 1] + 2;
      y2[i] = (sig - 1) / p;
      u[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) -
             (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
      u[i] = (6 * u[i] / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p;
    }
    const Real qn = 0;
    const Real un = 0;
    y2[n - 1] = (un - qn * u[n - 2]) / (qn * y2[n - 2] + 1);
    for (unsigned k = n - 2; k > 0; k--)
      y2[k] = y2[k] * y2[k + 1] + u[k];
    for (unsigned j = 0; j < nn; j++) {
      unsigned int klo = 0;
      unsigned int khi = n - 1;
      unsigned int k = 0;
      while (khi - klo > 1) {
        k = (khi + klo) >> 1;
        if (x[k] > (xx[j] + offset))
          khi = k;
        else
          klo = k;
      }
      const Real h = x[khi] - x[klo];
      if (h <= 0.0) {
        std::cout << "Interpolation points must be distinct!" << std::endl;
        abort();
      }
      const Real a = (x[khi] - (xx[j] + offset)) / h;
      const Real b = ((xx[j] + offset) - x[klo]) / h;
      yy[j] =
          a * y[klo] + b * y[khi] +
          ((a * a * a - a) * y2[klo] + (b * b * b - b) * y2[khi]) * (h * h) / 6;
    }
  }
  static void cubicInterpolation(const Real x0, const Real x1, const Real x,
                                 const Real y0, const Real y1, const Real dy0,
                                 const Real dy1, Real &y, Real &dy) {
    const Real xrel = (x - x0);
    const Real deltax = (x1 - x0);
    const Real a = (dy0 + dy1) / (deltax * deltax) -
                   2 * (y1 - y0) / (deltax * deltax * deltax);
    const Real b =
        (-2 * dy0 - dy1) / deltax + 3 * (y1 - y0) / (deltax * deltax);
    const Real c = dy0;
    const Real d = y0;
    y = a * xrel * xrel * xrel + b * xrel * xrel + c * xrel + d;
    dy = 3 * a * xrel * xrel + 2 * b * xrel + c;
  }
  static void cubicInterpolation(const Real x0, const Real x1, const Real x,
                                 const Real y0, const Real y1, Real &y,
                                 Real &dy) {
    return cubicInterpolation(x0, x1, x, y0, y1, 0, 0, y, dy);
  }
  static void linearInterpolation(const Real x0, const Real x1, const Real x,
                                  const Real y0, const Real y1, Real &y,
                                  Real &dy) {
    y = (y1 - y0) / (x1 - x0) * (x - x0) + y0;
    dy = (y1 - y0) / (x1 - x0);
  }
};
namespace Schedulers {
template <int Npoints> struct ParameterScheduler {
  static constexpr int npoints = Npoints;
  std::array<Real, Npoints> parameters_t0;
  std::array<Real, Npoints> parameters_t1;
  std::array<Real, Npoints> dparameters_t0;
  Real t0, t1;
  ParameterScheduler() {
    t0 = -1;
    t1 = 0;
    parameters_t0 = std::array<Real, Npoints>();
    parameters_t1 = std::array<Real, Npoints>();
    dparameters_t0 = std::array<Real, Npoints>();
  }
  void transition(const Real t, const Real tstart, const Real tend,
                  const std::array<Real, Npoints> parameters_tend,
                  const bool UseCurrentDerivative = false) {
    if (t < tstart or t > tend)
      return;
    std::array<Real, Npoints> parameters;
    std::array<Real, Npoints> dparameters;
    gimmeValues(tstart, parameters, dparameters);
    t0 = tstart;
    t1 = tend;
    parameters_t0 = parameters;
    parameters_t1 = parameters_tend;
    dparameters_t0 =
        UseCurrentDerivative ? dparameters : std::array<Real, Npoints>();
  }
  void transition(const Real t, const Real tstart, const Real tend,
                  const std::array<Real, Npoints> parameters_tstart,
                  const std::array<Real, Npoints> parameters_tend) {
    if (t < tstart or t > tend)
      return;
    if (tstart < t0)
      return;
    t0 = tstart;
    t1 = tend;
    parameters_t0 = parameters_tstart;
    parameters_t1 = parameters_tend;
  }
  void gimmeValues(const Real t, std::array<Real, Npoints> &parameters,
                   std::array<Real, Npoints> &dparameters) {
    if (t < t0 or t0 < 0) {
      parameters = parameters_t0;
      dparameters = std::array<Real, Npoints>();
    } else if (t > t1) {
      parameters = parameters_t1;
      dparameters = std::array<Real, Npoints>();
    } else {
      for (int i = 0; i < Npoints; ++i)
        IF2D_Interpolation1D::cubicInterpolation(
            t0, t1, t, parameters_t0[i], parameters_t1[i], dparameters_t0[i],
            0.0, parameters[i], dparameters[i]);
    }
  }
  void gimmeValuesLinear(const Real t, std::array<Real, Npoints> &parameters,
                         std::array<Real, Npoints> &dparameters) {
    if (t < t0 or t0 < 0) {
      parameters = parameters_t0;
      dparameters = std::array<Real, Npoints>();
    } else if (t > t1) {
      parameters = parameters_t1;
      dparameters = std::array<Real, Npoints>();
    } else {
      for (int i = 0; i < Npoints; ++i)
        IF2D_Interpolation1D::linearInterpolation(
            t0, t1, t, parameters_t0[i], parameters_t1[i], parameters[i],
            dparameters[i]);
    }
  }
  void gimmeValues(const Real t, std::array<Real, Npoints> &parameters) {
    std::array<Real, Npoints> dparameters_whocares;
    return gimmeValues(t, parameters, dparameters_whocares);
  }
};
struct ParameterSchedulerScalar : ParameterScheduler<1> {
  void transition(const Real t, const Real tstart, const Real tend,
                  const Real parameter_tend, const bool keepSlope = false) {
    const std::array<Real, 1> myParameter = {parameter_tend};
    return ParameterScheduler<1>::transition(t, tstart, tend, myParameter,
                                             keepSlope);
  }
  void transition(const Real t, const Real tstart, const Real tend,
                  const Real parameter_tstart, const Real parameter_tend) {
    const std::array<Real, 1> myParameterStart = {parameter_tstart};
    const std::array<Real, 1> myParameterEnd = {parameter_tend};
    return ParameterScheduler<1>::transition(t, tstart, tend, myParameterStart,
                                             myParameterEnd);
  }
  void gimmeValues(const Real t, Real &parameter, Real &dparameter) {
    std::array<Real, 1> myParameter, mydParameter;
    ParameterScheduler<1>::gimmeValues(t, myParameter, mydParameter);
    parameter = myParameter[0];
    dparameter = mydParameter[0];
  }
  void gimmeValues(const Real t, Real &parameter) {
    std::array<Real, 1> myParameter;
    ParameterScheduler<1>::gimmeValues(t, myParameter);
    parameter = myParameter[0];
  }
};
template <int Npoints>
struct ParameterSchedulerVector : ParameterScheduler<Npoints> {
  void gimmeValues(const Real t, const std::array<Real, Npoints> &positions,
                   const int Nfine, const Real *const positions_fine,
                   Real *const parameters_fine, Real *const dparameters_fine) {
    Real *parameters_t0_fine = new Real[Nfine];
    Real *parameters_t1_fine = new Real[Nfine];
    Real *dparameters_t0_fine = new Real[Nfine];
    IF2D_Interpolation1D::naturalCubicSpline(
        positions.data(), this->parameters_t0.data(), Npoints, positions_fine,
        parameters_t0_fine, Nfine);
    IF2D_Interpolation1D::naturalCubicSpline(
        positions.data(), this->parameters_t1.data(), Npoints, positions_fine,
        parameters_t1_fine, Nfine);
    IF2D_Interpolation1D::naturalCubicSpline(
        positions.data(), this->dparameters_t0.data(), Npoints, positions_fine,
        dparameters_t0_fine, Nfine);
    if (t < this->t0 or this->t0 < 0) {
      memcpy(parameters_fine, parameters_t0_fine, Nfine * sizeof(Real));
      memset(dparameters_fine, 0, Nfine * sizeof(Real));
    } else if (t > this->t1) {
      memcpy(parameters_fine, parameters_t1_fine, Nfine * sizeof(Real));
      memset(dparameters_fine, 0, Nfine * sizeof(Real));
    } else {
      for (int i = 0; i < Nfine; ++i)
        IF2D_Interpolation1D::cubicInterpolation(
            this->t0, this->t1, t, parameters_t0_fine[i], parameters_t1_fine[i],
            dparameters_t0_fine[i], 0, parameters_fine[i], dparameters_fine[i]);
    }
    delete[] parameters_t0_fine;
    delete[] parameters_t1_fine;
    delete[] dparameters_t0_fine;
  }
  void gimmeValues(const Real t, std::array<Real, Npoints> &parameters) {
    ParameterScheduler<Npoints>::gimmeValues(t, parameters);
  }
  void gimmeValues(const Real t, std::array<Real, Npoints> &parameters,
                   std::array<Real, Npoints> &dparameters) {
    ParameterScheduler<Npoints>::gimmeValues(t, parameters, dparameters);
  }
};
template <int Npoints>
struct ParameterSchedulerLearnWave : ParameterScheduler<Npoints> {
  template <typename T>
  void gimmeValues(const Real t, const Real Twave, const Real Length,
                   const std::array<Real, Npoints> &positions, const int Nfine,
                   const T *const positions_fine, T *const parameters_fine,
                   Real *const dparameters_fine) {
    const Real _1oL = 1. / Length;
    const Real _1oT = 1. / Twave;
    for (int i = 0; i < Nfine; ++i) {
      const Real c = positions_fine[i] * _1oL - (t - this->t0) * _1oT;
      bool bCheck = true;
      if (c < positions[0]) {
        IF2D_Interpolation1D::cubicInterpolation(
            c, positions[0], c, this->parameters_t0[0], this->parameters_t0[0],
            parameters_fine[i], dparameters_fine[i]);
        bCheck = false;
      } else if (c > positions[Npoints - 1]) {
        IF2D_Interpolation1D::cubicInterpolation(
            positions[Npoints - 1], c, c, this->parameters_t0[Npoints - 1],
            this->parameters_t0[Npoints - 1], parameters_fine[i],
            dparameters_fine[i]);
        bCheck = false;
      } else {
        for (int j = 1; j < Npoints; ++j) {
          if ((c >= positions[j - 1]) && (c <= positions[j])) {
            IF2D_Interpolation1D::cubicInterpolation(
                positions[j - 1], positions[j], c, this->parameters_t0[j - 1],
                this->parameters_t0[j], parameters_fine[i],
                dparameters_fine[i]);
            dparameters_fine[i] = -dparameters_fine[i] * _1oT;
            bCheck = false;
          }
        }
      }
      if (bCheck) {
        std::cout << "Ciaone2!" << std::endl;
        abort();
      }
    }
  }
  void Turn(const Real b, const Real t_turn) {
    this->t0 = t_turn;
    for (int i = Npoints - 1; i > 1; --i)
      this->parameters_t0[i] = this->parameters_t0[i - 2];
    this->parameters_t0[1] = b;
    this->parameters_t0[0] = 0;
  }
};
} // namespace Schedulers
struct Shape {
  Shape(CommandlineParser &p, Real C[2])
      : center{C[0], C[1]}, centerOfMass{C[0], C[1]},
        orientation(p("-angle").asDouble(0) * M_PI / 180),
        bForced(p("-bForced").asBool(false)),
        bForcedx(p("-bForcedx").asBool(bForced)),
        bForcedy(p("-bForcedy").asBool(bForced)),
        bBlockang(p("-bBlockAng").asBool(bForcedx || bForcedy)),
        forcedu(-p("-xvel").asDouble(0)), forcedv(-p("-yvel").asDouble(0)),
        forcedomega(-p("-angvel").asDouble(0)), length(p("-L").asDouble(0.1)),
        Tperiod(p("-T").asDouble(1)), phaseShift(p("-phi").asDouble(0)),
        rK(new Real[Nm]), vK(new Real[Nm]), rC(new Real[Nm]), vC(new Real[Nm]),
        rB(new Real[Nm]), vB(new Real[Nm]), rS(new Real[Nm]), rX(new Real[Nm]),
        rY(new Real[Nm]), vX(new Real[Nm]), vY(new Real[Nm]),
        norX(new Real[Nm]), norY(new Real[Nm]), vNorX(new Real[Nm]),
        vNorY(new Real[Nm]), width(new Real[Nm]) {}
  std::vector<ObstacleBlock *> obstacleBlocks;
  Real center[2];
  Real centerOfMass[2];
  Real orientation;
  Real d_gm[2] = {0, 0};
  const bool bForced;
  const bool bForcedx;
  const bool bForcedy;
  const bool bBlockang;
  const Real forcedu;
  const Real forcedv;
  const Real forcedomega;
  Real M = 0;
  Real J = 0;
  Real u = forcedu;
  Real v = forcedv;
  Real omega = forcedomega;
  Real fluidAngMom = 0;
  Real fluidMomX = 0;
  Real fluidMomY = 0;
  Real penalDX = 0;
  Real penalDY = 0;
  Real penalM = 0;
  Real penalJ = 0;
  Real appliedForceX = 0;
  Real appliedForceY = 0;
  Real appliedTorque = 0;
  Real perimeter = 0, forcex = 0, forcey = 0, forcex_P = 0, forcey_P = 0;
  Real forcex_V = 0, forcey_V = 0, torque = 0, torque_P = 0, torque_V = 0;
  Real drag = 0, thrust = 0, lift = 0, circulation = 0, Pout = 0, PoutNew = 0,
       PoutBnd = 0, defPower = 0;
  Real defPowerBnd = 0, Pthrust = 0, Pdrag = 0, EffPDef = 0, EffPDefBnd = 0;
  const Real Tperiod, phaseShift;
  Real area_internal = 0, J_internal = 0;
  Real CoM_internal[2] = {0, 0}, vCoM_internal[2] = {0, 0};
  Real theta_internal = 0, angvel_internal = 0;
  Real length, h;
  Real fracRefined = 0.1, fracMid = 1 - 2 * fracRefined;
  Real dSmid_tgt = sim.minH / std::sqrt(2);
  Real dSrefine_tgt = 0.125 * sim.minH;
  int Nmid = (int)std::ceil(length * fracMid / dSmid_tgt / 8) * 8;
  Real dSmid = length * fracMid / Nmid;
  int Nend =
      (int)std::ceil(fracRefined * length * 2 / (dSmid + dSrefine_tgt) / 4) * 4;
  Real dSref = fracRefined * length * 2 / Nend - dSmid;
  int Nm = Nmid + 2 * Nend + 1;
  Real *rS;
  Real *rX;
  Real *rY;
  Real *vX;
  Real *vY;
  Real *norX;
  Real *norY;
  Real *vNorX;
  Real *vNorY;
  Real *width;
  Real linMom[2], area, angMom;
  FishSkin upperSkin = FishSkin(Nm);
  FishSkin lowerSkin = FishSkin(Nm);
  Real amplitudeFactor;
  Real curv_PID_fac = 0;
  Real curv_PID_dif = 0;
  Real avgDeltaY = 0;
  Real avgDangle = 0;
  Real avgAngVel = 0;
  Real lastTact = 0;
  Real lastCurv = 0;
  Real oldrCurv = 0;
  Real periodPIDval = Tperiod;
  Real periodPIDdif = 0;
  bool TperiodPID = false;
  Real time0 = 0;
  Real timeshift = 0;
  Real lastTime = 0;
  Real lastAvel = 0;
  Schedulers::ParameterSchedulerVector<6> curvatureScheduler;
  Schedulers::ParameterSchedulerLearnWave<7> rlBendingScheduler;
  Schedulers::ParameterSchedulerScalar periodScheduler;
  Real current_period = Tperiod;
  Real next_period = Tperiod;
  Real transition_start = 0.0;
  Real transition_duration = 0.1 * Tperiod;
  Real *rK;
  Real *vK;
  Real *rC;
  Real *vC;
  Real *rB;
  Real *vB;
};
struct ComputeSurfaceNormals {
  ComputeSurfaceNormals(){};
  StencilInfo stencil{-1, -1, 0, 2, 2, 1, false, {0}};
  StencilInfo stencil2{-1, -1, 0, 2, 2, 1, false, {0}};
  void operator()(ScalarLab &labChi, ScalarLab &labSDF,
                  const BlockInfo &infoChi, const BlockInfo &infoSDF) const {
    for (const auto &shape : sim.shapes) {
      const std::vector<ObstacleBlock *> &OBLOCK = shape->obstacleBlocks;
      if (OBLOCK[infoChi.blockID] == nullptr)
        continue;
      const Real h = infoChi.h;
      ObstacleBlock &o = *OBLOCK[infoChi.blockID];
      const Real i2h = 0.5 / h;
      const Real fac = 0.5 * h;
      for (int iy = 0; iy < _BS_; iy++)
        for (int ix = 0; ix < _BS_; ix++) {
          const Real gradHX = labChi(ix + 1, iy).s - labChi(ix - 1, iy).s;
          const Real gradHY = labChi(ix, iy + 1).s - labChi(ix, iy - 1).s;
          if (gradHX * gradHX + gradHY * gradHY < 1e-12)
            continue;
          const Real gradUX =
              i2h * (labSDF(ix + 1, iy).s - labSDF(ix - 1, iy).s);
          const Real gradUY =
              i2h * (labSDF(ix, iy + 1).s - labSDF(ix, iy - 1).s);
          const Real gradUSq = (gradUX * gradUX + gradUY * gradUY) + EPS;
          const Real D = fac * (gradHX * gradUX + gradHY * gradUY) / gradUSq;
          if (std::fabs(D) > EPS) {
            o.n_surfPoints++;
            const Real dchidx = -D * gradUX, dchidy = -D * gradUY;
            o.surface.push_back(new surface_data(ix, iy, dchidx, dchidy, D));
          }
        }
      o.filled = true;
      o.x_s = (Real *)calloc(o.n_surfPoints, sizeof(Real));
      o.y_s = (Real *)calloc(o.n_surfPoints, sizeof(Real));
      o.p_s = (Real *)calloc(o.n_surfPoints, sizeof(Real));
      o.u_s = (Real *)calloc(o.n_surfPoints, sizeof(Real));
      o.v_s = (Real *)calloc(o.n_surfPoints, sizeof(Real));
      o.nx_s = (Real *)calloc(o.n_surfPoints, sizeof(Real));
      o.ny_s = (Real *)calloc(o.n_surfPoints, sizeof(Real));
      o.omega_s = (Real *)calloc(o.n_surfPoints, sizeof(Real));
      o.uDef_s = (Real *)calloc(o.n_surfPoints, sizeof(Real));
      o.vDef_s = (Real *)calloc(o.n_surfPoints, sizeof(Real));
      o.fX_s = (Real *)calloc(o.n_surfPoints, sizeof(Real));
      o.fY_s = (Real *)calloc(o.n_surfPoints, sizeof(Real));
      o.fXv_s = (Real *)calloc(o.n_surfPoints, sizeof(Real));
      o.fYv_s = (Real *)calloc(o.n_surfPoints, sizeof(Real));
    }
  }
};
struct AreaSegment {
  const Real safe_distance;
  const std::pair<int, int> s_range;
  Real w[2], c[2];
  Real normalI[2] = {(Real)1, (Real)0};
  Real normalJ[2] = {(Real)0, (Real)1};
  Real objBoxLabFr[2][2] = {{0, 0}, {0, 0}};
  Real objBoxObjFr[2][2] = {{0, 0}, {0, 0}};
  AreaSegment(std::pair<int, int> sr, const Real bb[2][2], const Real safe)
      : safe_distance(safe), s_range(sr), w{(bb[0][1] - bb[0][0]) / 2 + safe,
                                            (bb[1][1] - bb[1][0]) / 2 + safe},
        c{(bb[0][1] + bb[0][0]) / 2, (bb[1][1] + bb[1][0]) / 2} {
    assert(w[0] > 0);
    assert(w[1] > 0);
  }
  void changeToComputationalFrame(const Real pos[2], const Real angle) {
    Real Rmatrix2D[2][2] = {{std::cos(angle), -std::sin(angle)},
                            {std::sin(angle), std::cos(angle)}};
    Real p[2] = {c[0], c[1]};
    Real nx[2] = {normalI[0], normalI[1]};
    Real ny[2] = {normalJ[0], normalJ[1]};
    for (int i = 0; i < 2; ++i) {
      c[i] = Rmatrix2D[i][0] * p[0] + Rmatrix2D[i][1] * p[1];
      normalI[i] = Rmatrix2D[i][0] * nx[0] + Rmatrix2D[i][1] * nx[1];
      normalJ[i] = Rmatrix2D[i][0] * ny[0] + Rmatrix2D[i][1] * ny[1];
    }
    c[0] += pos[0];
    c[1] += pos[1];
    Real magI = std::sqrt(normalI[0] * normalI[0] + normalI[1] * normalI[1]);
    Real magJ = std::sqrt(normalJ[0] * normalJ[0] + normalJ[1] * normalJ[1]);
    assert(magI > std::numeric_limits<Real>::epsilon());
    assert(magJ > std::numeric_limits<Real>::epsilon());
    Real invMagI = 1 / magI, invMagJ = 1 / magJ;
    for (int i = 0; i < 2; ++i) {
      normalI[i] = std::fabs(normalI[i]) * invMagI;
      normalJ[i] = std::fabs(normalJ[i]) * invMagJ;
    }
    assert(normalI[0] >= 0 && normalI[1] >= 0);
    assert(normalJ[0] >= 0 && normalJ[1] >= 0);
    Real widthXvec[] = {w[0] * normalI[0], w[0] * normalI[1]};
    Real widthYvec[] = {w[1] * normalJ[0], w[1] * normalJ[1]};
    for (int i = 0; i < 2; ++i) {
      objBoxLabFr[i][0] = c[i] - widthXvec[i] - widthYvec[i];
      objBoxLabFr[i][1] = c[i] + widthXvec[i] + widthYvec[i];
      objBoxObjFr[i][0] = c[i] - w[i];
      objBoxObjFr[i][1] = c[i] + w[i];
    }
  }
  bool isIntersectingWithAABB(const Real start[2], const Real end[2]) const {
    Real AABB_w[2] = {(end[0] - start[0]) / 2 + safe_distance,
                      (end[1] - start[1]) / 2 + safe_distance};
    Real AABB_c[2] = {(end[0] + start[0]) / 2, (end[1] + start[1]) / 2};
    Real AABB_box[2][2] = {{AABB_c[0] - AABB_w[0], AABB_c[0] + AABB_w[0]},
                           {AABB_c[1] - AABB_w[1], AABB_c[1] + AABB_w[1]}};
    assert(AABB_w[0] > 0 && AABB_w[1] > 0);
    Real intersectionLabFrame[2][2] = {
        {std::max(objBoxLabFr[0][0], AABB_box[0][0]),
         std::min(objBoxLabFr[0][1], AABB_box[0][1])},
        {std::max(objBoxLabFr[1][0], AABB_box[1][0]),
         std::min(objBoxLabFr[1][1], AABB_box[1][1])}};
    if (intersectionLabFrame[0][1] - intersectionLabFrame[0][0] < 0 ||
        intersectionLabFrame[1][1] - intersectionLabFrame[1][0] < 0)
      return false;
    Real widthXbox[2] = {AABB_w[0] * normalI[0], AABB_w[0] * normalJ[0]};
    Real widthYbox[2] = {AABB_w[1] * normalI[1], AABB_w[1] * normalJ[1]};
    Real boxBox[2][2] = {{AABB_c[0] - widthXbox[0] - widthYbox[0],
                          AABB_c[0] + widthXbox[0] + widthYbox[0]},
                         {AABB_c[1] - widthXbox[1] - widthYbox[1],
                          AABB_c[1] + widthXbox[1] + widthYbox[1]}};
    Real intersectionFishFrame[2][2] = {
        {std::max(boxBox[0][0], objBoxObjFr[0][0]),
         std::min(boxBox[0][1], objBoxObjFr[0][1])},
        {std::max(boxBox[1][0], objBoxObjFr[1][0]),
         std::min(boxBox[1][1], objBoxObjFr[1][1])}};
    if (intersectionFishFrame[0][1] - intersectionFishFrame[0][0] < 0 ||
        intersectionFishFrame[1][1] - intersectionFishFrame[1][0] < 0)
      return false;
    return true;
  }
};
struct PutChiOnGrid {
  PutChiOnGrid(){};
  StencilInfo stencil{-1, -1, 0, 2, 2, 1, false, {0}};
  std::vector<BlockInfo> &chiInfo = sim.chi->m_vInfo;
  void operator()(ScalarLab &lab, const BlockInfo &info) const {
    for (auto &shape : sim.shapes) {
      std::vector<ObstacleBlock *> &OBLOCK = shape->obstacleBlocks;
      if (OBLOCK[info.blockID] == nullptr)
        continue;
      Real h = info.h;
      Real h2 = h * h;
      ObstacleBlock &o = *OBLOCK[info.blockID];
      CHI_MAT &__restrict__ X = o.chi;
      const CHI_MAT &__restrict__ sdf = o.dist;
      o.COM_x = 0;
      o.COM_y = 0;
      o.Mass = 0;
      auto &__restrict__ CHI = *(ScalarBlock *)chiInfo[info.blockID].ptrBlock;
      for (int iy = 0; iy < _BS_; iy++)
        for (int ix = 0; ix < _BS_; ix++) {
          if (sdf[iy][ix] > +h || sdf[iy][ix] < -h) {
            X[iy][ix] = sdf[iy][ix] > 0 ? 1 : 0;
          } else {
            Real distPx = lab(ix + 1, iy).s;
            Real distMx = lab(ix - 1, iy).s;
            Real distPy = lab(ix, iy + 1).s;
            Real distMy = lab(ix, iy - 1).s;
            Real IplusX = std::max((Real)0.0, distPx);
            Real IminuX = std::max((Real)0.0, distMx);
            Real IplusY = std::max((Real)0.0, distPy);
            Real IminuY = std::max((Real)0.0, distMy);
            Real gradIX = IplusX - IminuX;
            Real gradIY = IplusY - IminuY;
            Real gradUX = distPx - distMx;
            Real gradUY = distPy - distMy;
            Real gradUSq = (gradUX * gradUX + gradUY * gradUY) + EPS;
            X[iy][ix] = (gradIX * gradUX + gradIY * gradUY) / gradUSq;
          }
          CHI(ix, iy).s = std::max(CHI(ix, iy).s, X[iy][ix]);
          if (X[iy][ix] > 0) {
            Real p[2];
            info.pos(p, ix, iy);
            o.COM_x += X[iy][ix] * h2 * (p[0] - shape->centerOfMass[0]);
            o.COM_y += X[iy][ix] * h2 * (p[1] - shape->centerOfMass[1]);
            o.Mass += X[iy][ix] * h2;
          }
        }
    }
  }
};
static void if2d_solve(unsigned Nm, Real *rS, Real *curv, Real *curv_dt,
                       Real *rX, Real *rY, Real *vX, Real *vY, Real *norX,
                       Real *norY, Real *vNorX, Real *vNorY) {
  rX[0] = 0.0;
  rY[0] = 0.0;
  norX[0] = 0.0;
  norY[0] = 1.0;
  Real ksiX = 1.0;
  Real ksiY = 0.0;
  vX[0] = 0.0;
  vY[0] = 0.0;
  vNorX[0] = 0.0;
  vNorY[0] = 0.0;
  Real vKsiX = 0.0;
  Real vKsiY = 0.0;
  for (unsigned i = 1; i < Nm; i++) {
    const Real dksiX = curv[i - 1] * norX[i - 1];
    const Real dksiY = curv[i - 1] * norY[i - 1];
    const Real dnuX = -curv[i - 1] * ksiX;
    const Real dnuY = -curv[i - 1] * ksiY;
    const Real dvKsiX =
        curv_dt[i - 1] * norX[i - 1] + curv[i - 1] * vNorX[i - 1];
    const Real dvKsiY =
        curv_dt[i - 1] * norY[i - 1] + curv[i - 1] * vNorY[i - 1];
    const Real dvNuX = -curv_dt[i - 1] * ksiX - curv[i - 1] * vKsiX;
    const Real dvNuY = -curv_dt[i - 1] * ksiY - curv[i - 1] * vKsiY;
    const Real ds = rS[i] - rS[i - 1];
    rX[i] = rX[i - 1] + ds * ksiX;
    rY[i] = rY[i - 1] + ds * ksiY;
    norX[i] = norX[i - 1] + ds * dnuX;
    norY[i] = norY[i - 1] + ds * dnuY;
    ksiX += ds * dksiX;
    ksiY += ds * dksiY;
    vX[i] = vX[i - 1] + ds * vKsiX;
    vY[i] = vY[i - 1] + ds * vKsiY;
    vNorX[i] = vNorX[i - 1] + ds * dvNuX;
    vNorY[i] = vNorY[i - 1] + ds * dvNuY;
    vKsiX += ds * dvKsiX;
    vKsiY += ds * dvKsiY;
    const Real d1 = ksiX * ksiX + ksiY * ksiY;
    const Real d2 = norX[i] * norX[i] + norY[i] * norY[i];
    if (d1 > std::numeric_limits<Real>::epsilon()) {
      const Real normfac = 1 / std::sqrt(d1);
      ksiX *= normfac;
      ksiY *= normfac;
    }
    if (d2 > std::numeric_limits<Real>::epsilon()) {
      const Real normfac = 1 / std::sqrt(d2);
      norX[i] *= normfac;
      norY[i] *= normfac;
    }
  }
}
struct PutFishOnBlocks {
  Real position[2];
  Real Rmatrix2D[2][2];
  void changeVelocityToComputationalFrame(Real x[2]) const {
    const Real p[2] = {x[0], x[1]};
    x[0] = Rmatrix2D[0][0] * p[0] + Rmatrix2D[0][1] * p[1];
    x[1] = Rmatrix2D[1][0] * p[0] + Rmatrix2D[1][1] * p[1];
  }
  void changeToComputationalFrame(Real x[2]) const {
    const Real p[2] = {x[0], x[1]};
    x[0] = Rmatrix2D[0][0] * p[0] + Rmatrix2D[0][1] * p[1];
    x[1] = Rmatrix2D[1][0] * p[0] + Rmatrix2D[1][1] * p[1];
    x[0] += position[0];
    x[1] += position[1];
  }
  void changeFromComputationalFrame(Real x[2]) const {
    const Real p[2] = {x[0] - position[0], x[1] - position[1]};
    x[0] = Rmatrix2D[0][0] * p[0] + Rmatrix2D[1][0] * p[1];
    x[1] = Rmatrix2D[0][1] * p[0] + Rmatrix2D[1][1] * p[1];
  }
};
static void ongrid(Real dt) {
  const std::vector<BlockInfo> &velInfo = sim.vel->m_vInfo;
  const std::vector<BlockInfo> &tmpInfo = sim.tmp->m_vInfo;
  const std::vector<BlockInfo> &chiInfo = sim.chi->m_vInfo;
  int nSum[2] = {0, 0};
  Real uSum[2] = {0, 0};
  if (nSum[0] > 0) {
    sim.uinfx = uSum[0] / nSum[0];
  }
  if (nSum[1] > 0) {
    sim.uinfy = uSum[1] / nSum[1];
  }
  for (const auto &shape : sim.shapes) {
    shape->centerOfMass[0] += dt * (shape->u + sim.uinfx);
    shape->centerOfMass[1] += dt * (shape->v + sim.uinfy);
    shape->orientation += dt * shape->omega;
    shape->orientation = shape->orientation > M_PI
                             ? shape->orientation - 2 * M_PI
                             : shape->orientation;
    shape->orientation = shape->orientation < -M_PI
                             ? shape->orientation + 2 * M_PI
                             : shape->orientation;
    Real cosang = std::cos(shape->orientation);
    Real sinang = std::sin(shape->orientation);
    shape->center[0] = shape->centerOfMass[0] + cosang * shape->d_gm[0] -
                       sinang * shape->d_gm[1];
    shape->center[1] = shape->centerOfMass[1] + sinang * shape->d_gm[0] +
                       cosang * shape->d_gm[1];
    shape->theta_internal -= dt * shape->angvel_internal;
    if (shape->center[0] < 0 || shape->center[0] > sim.extents[0] ||
        shape->center[1] < 0 || shape->center[1] > sim.extents[1]) {
      printf("[CUP2D] ABORT: Body out of domain\n");
      fflush(0);
      abort();
    }
  }
  const size_t Nblocks = velInfo.size();
#pragma omp parallel for
  for (size_t i = 0; i < Nblocks; i++)
    for (int x = 0; x < _BS_; x++)
      for (int y = 0; y < _BS_; y++) {
        ((ScalarBlock *)chiInfo[i].ptrBlock)->data[x][y].clear();
        ((ScalarBlock *)tmpInfo[i].ptrBlock)->data[x][y].s = -1;
      }
  for (const auto &shape : sim.shapes) {
    for (auto &entry : shape->obstacleBlocks)
      delete entry;
    shape->obstacleBlocks.clear();
    shape->periodScheduler.transition(
        sim.time, shape->transition_start,
        shape->transition_start + shape->transition_duration,
        shape->current_period, shape->next_period);
    shape->periodScheduler.gimmeValues(sim.time, shape->periodPIDval,
                                       shape->periodPIDdif);
    if (shape->transition_start < sim.time &&
        sim.time < shape->transition_start + shape->transition_duration) {
      shape->timeshift =
          (sim.time - shape->time0) / shape->periodPIDval + shape->timeshift;
      shape->time0 = sim.time;
    }
    const std::array<Real, 6> curvaturePoints = {(Real)0,
                                                 (Real).15 * shape->length,
                                                 (Real).4 * shape->length,
                                                 (Real).65 * shape->length,
                                                 (Real).9 * shape->length,
                                                 shape->length};
    const std::array<Real, 6> curvatureValues = {
        (Real)0.82014 / shape->length, (Real)1.46515 / shape->length,
        (Real)2.57136 / shape->length, (Real)3.75425 / shape->length,
        (Real)5.09147 / shape->length, (Real)5.70449 / shape->length};
    const std::array<Real, 7> bendPoints = {(Real)-.5, (Real)-.25, (Real)0,
                                            (Real).25, (Real).5,   (Real).75,
                                            (Real)1};
    const std::array<Real, 6> curvatureZeros = {
        0.01 * curvatureValues[0], 0.01 * curvatureValues[1],
        0.01 * curvatureValues[2], 0.01 * curvatureValues[3],
        0.01 * curvatureValues[4], 0.01 * curvatureValues[5],
    };
    shape->curvatureScheduler.transition(0, 0, shape->Tperiod, curvatureZeros,
                                         curvatureValues);
    shape->curvatureScheduler.gimmeValues(sim.time, curvaturePoints, shape->Nm,
                                          shape->rS, shape->rC, shape->vC);
    shape->rlBendingScheduler.gimmeValues(sim.time, shape->periodPIDval,
                                          shape->length, bendPoints, shape->Nm,
                                          shape->rS, shape->rB, shape->vB);
    const Real diffT = 1 - (sim.time - shape->time0) * shape->periodPIDdif /
                               shape->periodPIDval;
    const Real darg = 2 * M_PI / shape->periodPIDval * diffT;
    const Real arg0 = 2 * M_PI *
                          ((sim.time - shape->time0) / shape->periodPIDval +
                           shape->timeshift) +
                      M_PI * shape->phaseShift;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < shape->Nm; ++i) {
      const Real arg = arg0 - 2 * M_PI * shape->rS[i] / shape->length;
      shape->rK[i] = shape->amplitudeFactor * shape->rC[i] *
                     (std::sin(arg) + shape->rB[i] + shape->curv_PID_fac);
      shape->vK[i] =
          shape->amplitudeFactor *
          (shape->vC[i] * (std::sin(arg) + shape->rB[i] + shape->curv_PID_fac) +
           shape->rC[i] *
               (std::cos(arg) * darg + shape->vB[i] + shape->curv_PID_dif));
    }
    if2d_solve(shape->Nm, shape->rS, shape->rK, shape->vK, shape->rX, shape->rY,
               shape->vX, shape->vY, shape->norX, shape->norY, shape->vNorX,
               shape->vNorY);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < shape->lowerSkin.Npoints; ++i) {
      Real norm[2] = {shape->norX[i], shape->norY[i]};
      Real const norm_mod1 = std::sqrt(norm[0] * norm[0] + norm[1] * norm[1]);
      norm[0] /= norm_mod1;
      norm[1] /= norm_mod1;
      shape->lowerSkin.xSurf[i] = shape->rX[i] - shape->width[i] * norm[0];
      shape->lowerSkin.ySurf[i] = shape->rY[i] - shape->width[i] * norm[1];
      shape->upperSkin.xSurf[i] = shape->rX[i] + shape->width[i] * norm[0];
      shape->upperSkin.ySurf[i] = shape->rY[i] + shape->width[i] * norm[1];
    }
    Real _area = 0, _cmx = 0, _cmy = 0, _lmx = 0, _lmy = 0;
#pragma omp parallel for schedule(static)                                      \
    reduction(+ : _area, _cmx, _cmy, _lmx, _lmy)
    for (int i = 0; i < shape->Nm; ++i) {
      const Real ds =
          (i == 0) ? shape->rS[1] - shape->rS[0]
                   : ((i == shape->Nm - 1)
                          ? shape->rS[shape->Nm - 1] - shape->rS[shape->Nm - 2]
                          : shape->rS[i + 1] - shape->rS[i - 1]);
      const Real fac1 = 2 * shape->width[i];
      const Real fac2 =
          2 * std::pow(shape->width[i], 3) *
          (dds(i, shape->Nm, shape->norX, shape->rS) * shape->norY[i] -
           dds(i, shape->Nm, shape->norY, shape->rS) * shape->norX[i]) /
          3;
      _area += fac1 * ds / 2;
      _cmx += (shape->rX[i] * fac1 + shape->norX[i] * fac2) * ds / 2;
      _cmy += (shape->rY[i] * fac1 + shape->norY[i] * fac2) * ds / 2;
      _lmx += (shape->vX[i] * fac1 + shape->vNorX[i] * fac2) * ds / 2;
      _lmy += (shape->vY[i] * fac1 + shape->vNorY[i] * fac2) * ds / 2;
    }
    shape->area = _area;
    shape->CoM_internal[0] = _cmx;
    shape->CoM_internal[1] = _cmy;
    shape->linMom[0] = _lmx;
    shape->linMom[1] = _lmy;
    shape->CoM_internal[0] /= shape->area;
    shape->CoM_internal[1] /= shape->area;
    shape->vCoM_internal[0] = shape->linMom[0] / shape->area;
    shape->vCoM_internal[1] = shape->linMom[1] / shape->area;
    shape->area_internal = shape->area;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < shape->Nm; ++i) {
      shape->rX[i] -= shape->CoM_internal[0];
      shape->rY[i] -= shape->CoM_internal[1];
      shape->vX[i] -= shape->vCoM_internal[0];
      shape->vY[i] -= shape->vCoM_internal[1];
    }
    Real _J = 0, _am = 0;
#pragma omp parallel for reduction(+ : _J, _am) schedule(static)
    for (int i = 0; i < shape->Nm; ++i) {
      const Real ds =
          (i == 0) ? shape->rS[1] - shape->rS[0]
                   : ((i == shape->Nm - 1)
                          ? shape->rS[shape->Nm - 1] - shape->rS[shape->Nm - 2]
                          : shape->rS[i + 1] - shape->rS[i - 1]);
      const Real fac1 = 2 * shape->width[i];
      const Real fac2 =
          2 * std::pow(shape->width[i], 3) *
          (dds(i, shape->Nm, shape->norX, shape->rS) * shape->norY[i] -
           dds(i, shape->Nm, shape->norY, shape->rS) * shape->norX[i]) /
          3;
      const Real fac3 = 2 * std::pow(shape->width[i], 3) / 3;
      const Real tmp_M =
          (shape->rX[i] * shape->vY[i] - shape->rY[i] * shape->vX[i]) * fac1 +
          (shape->rX[i] * shape->vNorY[i] - shape->rY[i] * shape->vNorX[i] +
           shape->vY[i] * shape->norX[i] - shape->vX[i] * shape->norY[i]) *
              fac2 +
          (shape->norX[i] * shape->vNorY[i] -
           shape->norY[i] * shape->vNorX[i]) *
              fac3;
      const Real tmp_J =
          (shape->rX[i] * shape->rX[i] + shape->rY[i] * shape->rY[i]) * fac1 +
          2 * (shape->rX[i] * shape->norX[i] + shape->rY[i] * shape->norY[i]) *
              fac2 +
          fac3;
      _am += tmp_M * ds / 2;
      _J += tmp_J * ds / 2;
    }
    shape->J = _J;
    shape->angMom = _am;
    shape->angvel_internal = shape->angMom / shape->J;
    shape->J_internal = shape->J;
    const Real Rmatrix2D[2][2] = {
        {std::cos(shape->theta_internal), -std::sin(shape->theta_internal)},
        {std::sin(shape->theta_internal), std::cos(shape->theta_internal)}};
#pragma omp parallel for schedule(static)
    for (int i = 0; i < shape->Nm; ++i) {
      shape->vX[i] += shape->angvel_internal * shape->rY[i];
      shape->vY[i] -= shape->angvel_internal * shape->rX[i];
      rotate2D(Rmatrix2D, shape->rX[i], shape->rY[i]);
      rotate2D(Rmatrix2D, shape->vX[i], shape->vY[i]);
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < shape->Nm - 1; i++) {
      const auto ds = shape->rS[i + 1] - shape->rS[i];
      const auto tX = shape->rX[i + 1] - shape->rX[i];
      const auto tY = shape->rY[i + 1] - shape->rY[i];
      const auto tVX = shape->vX[i + 1] - shape->vX[i];
      const auto tVY = shape->vY[i + 1] - shape->vY[i];
      shape->norX[i] = -tY / ds;
      shape->norY[i] = tX / ds;
      shape->vNorX[i] = -tVY / ds;
      shape->vNorY[i] = tVX / ds;
    }
    shape->norX[shape->Nm - 1] = shape->norX[shape->Nm - 2];
    shape->norY[shape->Nm - 1] = shape->norY[shape->Nm - 2];
    shape->vNorX[shape->Nm - 1] = shape->vNorX[shape->Nm - 2];
    shape->vNorY[shape->Nm - 1] = shape->vNorY[shape->Nm - 2];
    {
      const Real Rmatrix2D[2][2] = {
          {std::cos(shape->theta_internal), -std::sin(shape->theta_internal)},
          {std::sin(shape->theta_internal), std::cos(shape->theta_internal)}};
#pragma omp parallel for schedule(static)
      for (size_t i = 0; i < shape->upperSkin.Npoints; ++i) {
        shape->upperSkin.xSurf[i] -= shape->CoM_internal[0];
        shape->upperSkin.ySurf[i] -= shape->CoM_internal[1];
        rotate2D(Rmatrix2D, shape->upperSkin.xSurf[i],
                 shape->upperSkin.ySurf[i]);
        shape->lowerSkin.xSurf[i] -= shape->CoM_internal[0];
        shape->lowerSkin.ySurf[i] -= shape->CoM_internal[1];
        rotate2D(Rmatrix2D, shape->lowerSkin.xSurf[i],
                 shape->lowerSkin.ySurf[i]);
      }
    }
    const int Nsegments = (shape->Nm - 1) / 8;
    const int Nm = shape->Nm;
    assert((Nm - 1) % Nsegments == 0);
    std::vector<AreaSegment *> vSegments(Nsegments, nullptr);
    Real h = std::numeric_limits<Real>::infinity();
    for (size_t i = 0; i < sim.vel->m_vInfo.size(); i++)
      h = std::min(sim.vel->m_vInfo[i].h, h);
    MPI_Allreduce(MPI_IN_PLACE, &h, 1, MPI_Real, MPI_MIN, MPI_COMM_WORLD);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < Nsegments; ++i) {
      const int next_idx = (i + 1) * (Nm - 1) / Nsegments;
      const int idx = i * (Nm - 1) / Nsegments;
      Real bbox[2][2] = {{1e9, -1e9}, {1e9, -1e9}};
      for (int ss = idx; ss <= next_idx; ++ss) {
        const Real xBnd[2] = {
            shape->rX[ss] - shape->norX[ss] * shape->width[ss],
            shape->rX[ss] + shape->norX[ss] * shape->width[ss]};
        const Real yBnd[2] = {
            shape->rY[ss] - shape->norY[ss] * shape->width[ss],
            shape->rY[ss] + shape->norY[ss] * shape->width[ss]};
        const Real maxX = std::max(xBnd[0], xBnd[1]),
                   minX = std::min(xBnd[0], xBnd[1]);
        const Real maxY = std::max(yBnd[0], yBnd[1]),
                   minY = std::min(yBnd[0], yBnd[1]);
        bbox[0][0] = std::min(bbox[0][0], minX);
        bbox[0][1] = std::max(bbox[0][1], maxX);
        bbox[1][0] = std::min(bbox[1][0], minY);
        bbox[1][1] = std::max(bbox[1][1], maxY);
      }
      const Real DD = 4 * h;
      AreaSegment *const tAS =
          new AreaSegment(std::make_pair(idx, next_idx), bbox, DD);
      tAS->changeToComputationalFrame(shape->center, shape->orientation);
      vSegments[i] = tAS;
    }
    const auto N = tmpInfo.size();
    std::vector<std::vector<AreaSegment *> *> segmentsPerBlock(N, nullptr);
    shape->obstacleBlocks = std::vector<ObstacleBlock *>(N, nullptr);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < tmpInfo.size(); ++i) {
      const BlockInfo &info = tmpInfo[i];
      Real pStart[2], pEnd[2];
      info.pos(pStart, 0, 0);
      info.pos(pEnd, _BS_ - 1, _BS_ - 1);
      for (size_t s = 0; s < vSegments.size(); ++s)
        if (vSegments[s]->isIntersectingWithAABB(pStart, pEnd)) {
          if (segmentsPerBlock[info.blockID] == nullptr)
            segmentsPerBlock[info.blockID] = new std::vector<AreaSegment *>(0);
          segmentsPerBlock[info.blockID]->push_back(vSegments[s]);
        }
      if (segmentsPerBlock[info.blockID] not_eq nullptr) {
        ObstacleBlock *const block = new ObstacleBlock();
        assert(block not_eq nullptr);
        shape->obstacleBlocks[info.blockID] = block;
        block->clear_surface();
        std::fill(block->dist[0], block->dist[0] + _BS_ * _BS_, -1);
        std::fill(block->chi[0], block->chi[0] + _BS_ * _BS_, 0);
        memset(block->udef, 0, sizeof(Real) * _BS_ * _BS_ * 2);
      }
    }
    assert(not segmentsPerBlock.empty());
#pragma omp parallel
    {
      PutFishOnBlocks putfish;
      putfish.position[0] = shape->center[0];
      putfish.position[1] = shape->center[1];

      putfish.Rmatrix2D[0][0] = std::cos(shape->orientation);
      putfish.Rmatrix2D[0][1] = -std::sin(shape->orientation);
      putfish.Rmatrix2D[1][0] = std::sin(shape->orientation);
      putfish.Rmatrix2D[1][1] = std::cos(shape->orientation);

#pragma omp for schedule(dynamic)
      for (size_t i = 0; i < tmpInfo.size(); i++) {
        const auto pos = segmentsPerBlock[tmpInfo[i].blockID];
        if (pos not_eq nullptr) {
          ObstacleBlock *const block =
              shape->obstacleBlocks[tmpInfo[i].blockID];
          assert(block not_eq nullptr);
          const BlockInfo &info = tmpInfo[i];
          ScalarBlock &b = *(ScalarBlock *)tmpInfo[i].ptrBlock;
          ObstacleBlock *const o = block;
          const std::vector<AreaSegment *> &v = *pos;

          // putfish(info, b, o, v);
          Real org[2];
          info.pos(org, 0, 0);
          const Real h = info.h, invh = 1.0 / info.h;
          const Real *const rX = shape->rX, *const norX = shape->norX;
          const Real *const rY = shape->rY, *const norY = shape->norY;
          const Real *const vX = shape->vX, *const vNorX = shape->vNorX;
          const Real *const vY = shape->vY, *const vNorY = shape->vNorY;
          const Real *const width = shape->width;
          static constexpr int BS[2] = {_BS_, _BS_};
          std::fill(o->dist[0], o->dist[0] + BS[1] * BS[0], -1);
          std::fill(o->chi[0], o->chi[0] + BS[1] * BS[0], 0);
          for (int i = 0; i < (int)v.size(); ++i) {
            const int firstSegm = std::max(v[i]->s_range.first, 1);
            const int lastSegm = std::min(v[i]->s_range.second, shape->Nm - 2);
            for (int ss = firstSegm; ss <= lastSegm; ++ss) {
              assert(width[ss] > 0);
              for (int signp = -1; signp <= 1; signp += 2) {
                Real myP[2] = {
                    rX[ss + 0] + width[ss + 0] * signp * norX[ss + 0],
                    rY[ss + 0] + width[ss + 0] * signp * norY[ss + 0]};
                putfish.changeToComputationalFrame(myP);
                const int iap[2] = {(int)std::floor((myP[0] - org[0]) * invh),
                                    (int)std::floor((myP[1] - org[1]) * invh)};
                if (iap[0] + 3 <= 0 || iap[0] - 1 >= BS[0])
                  continue;
                if (iap[1] + 3 <= 0 || iap[1] - 1 >= BS[1])
                  continue;
                Real pP[2] = {rX[ss + 1] + width[ss + 1] * signp * norX[ss + 1],
                              rY[ss + 1] +
                                  width[ss + 1] * signp * norY[ss + 1]};
                putfish.changeToComputationalFrame(pP);
                Real pM[2] = {rX[ss - 1] + width[ss - 1] * signp * norX[ss - 1],
                              rY[ss - 1] +
                                  width[ss - 1] * signp * norY[ss - 1]};
                putfish.changeToComputationalFrame(pM);
                Real udef[2] = {
                    vX[ss + 0] + width[ss + 0] * signp * vNorX[ss + 0],
                    vY[ss + 0] + width[ss + 0] * signp * vNorY[ss + 0]};
                putfish.changeVelocityToComputationalFrame(udef);
                for (int sy = std::max(0, iap[1] - 2);
                     sy < std::min(iap[1] + 4, BS[1]); ++sy)
                  for (int sx = std::max(0, iap[0] - 2);
                       sx < std::min(iap[0] + 4, BS[0]); ++sx) {
                    Real p[2];
                    info.pos(p, sx, sy);
                    const Real dist0 = dist(p, myP);
                    const Real distP = dist(p, pP);
                    const Real distM = dist(p, pM);
                    if (std::fabs(o->dist[sy][sx]) <
                        std::min({dist0, distP, distM}))
                      continue;
                    putfish.changeFromComputationalFrame(p);
#ifndef NDEBUG
                    Real p0[2] = {rX[ss] + width[ss] * signp * norX[ss],
                                  rY[ss] + width[ss] * signp * norY[ss]};
                    Real distC = dist(p, p0);
                    assert(std::fabs(distC - dist0) < EPS);
#endif
                    int close_s = ss, secnd_s = ss + (distP < distM ? 1 : -1);
                    Real dist1 = dist0, dist2 = distP < distM ? distP : distM;
                    if (distP < dist0 || distM < dist0) {
                      dist1 = dist2;
                      dist2 = dist0;
                      close_s = secnd_s;
                      secnd_s = ss;
                    }
                    Real dSsq = std::pow(rX[close_s] - rX[secnd_s], 2) +
                                std::pow(rY[close_s] - rY[secnd_s], 2);
                    assert(dSsq > 2.2e-16);
                    Real cnt2ML = std::pow(width[close_s], 2);
                    Real nxt2ML = std::pow(width[secnd_s], 2);
                    Real safeW =
                        std::max(width[close_s], width[secnd_s]) + 2 * h;
                    Real xMidl[2] = {rX[close_s], rY[close_s]};
                    Real grd2ML = dist(p, xMidl);
                    Real diffH = std::fabs(width[close_s] - width[secnd_s]);
                    Real sign2d = 0;
                    if (dSsq > diffH * diffH || grd2ML > safeW * safeW) {
                      sign2d = grd2ML > cnt2ML ? -1 : 1;
                    } else {
                      Real corr = 2 * std::sqrt(cnt2ML * nxt2ML);
                      Real Rsq = (cnt2ML + nxt2ML - corr + dSsq) *
                                 (cnt2ML + nxt2ML + corr + dSsq) / 4 / dSsq;
                      Real maxAx = std::max(cnt2ML, nxt2ML);
                      int idAx1 = cnt2ML > nxt2ML ? close_s : secnd_s;
                      int idAx2 = idAx1 == close_s ? secnd_s : close_s;
                      Real d = std::sqrt((Rsq - maxAx) / dSsq);
                      Real xCentr[2] = {rX[idAx1] + (rX[idAx1] - rX[idAx2]) * d,
                                        rY[idAx1] +
                                            (rY[idAx1] - rY[idAx2]) * d};
                      Real grd2Core = dist(p, xCentr);
                      sign2d = grd2Core > Rsq ? -1 : 1;
                    }
                    if (std::fabs(o->dist[sy][sx]) > dist1) {
                      Real W =
                          1 - std::min((Real)1, std::sqrt(dist1) * (invh / 3));
                      assert(W >= 0);
                      o->udef[sy][sx][0] = W * udef[0];
                      o->udef[sy][sx][1] = W * udef[1];
                      o->dist[sy][sx] = sign2d * dist1;
                      o->chi[sy][sx] = W;
                    }
                  }
              }
            }
          }
          info.pos(org, 0, 0);
          for (int i = 0; i < (int)v.size(); ++i) {
            const int firstSegm = std::max(v[i]->s_range.first, 1);
            const int lastSegm = std::min(v[i]->s_range.second, shape->Nm - 2);
            for (int ss = firstSegm; ss <= lastSegm; ++ss) {
              const Real myWidth = shape->width[ss];
              assert(myWidth > 0);
              const int Nw = std::floor(myWidth / h);
              for (int iw = -Nw + 1; iw < Nw; ++iw) {
                const Real offsetW = iw * h;
                Real xp[2] = {shape->rX[ss] + offsetW * shape->norX[ss],
                              shape->rY[ss] + offsetW * shape->norY[ss]};
                putfish.changeToComputationalFrame(xp);
                xp[0] = (xp[0] - org[0]) * invh;
                xp[1] = (xp[1] - org[1]) * invh;
                const Real ap[2] = {std::floor(xp[0]), std::floor(xp[1])};
                const int iap[2] = {(int)ap[0], (int)ap[1]};
                if (iap[0] + 2 <= 0 || iap[0] >= BS[0])
                  continue;
                if (iap[1] + 2 <= 0 || iap[1] >= BS[1])
                  continue;
                Real udef[2] = {shape->vX[ss] + offsetW * shape->vNorX[ss],
                                shape->vY[ss] + offsetW * shape->vNorY[ss]};
                putfish.changeVelocityToComputationalFrame(udef);
                Real wghts[2][2];
                for (int c = 0; c < 2; ++c) {
                  const Real t[2] = {std::fabs(xp[c] - ap[c]),
                                     std::fabs(xp[c] - (ap[c] + 1))};
                  wghts[c][0] = 1 - t[0];
                  wghts[c][1] = 1 - t[1];
                }
                for (int idy = std::max(0, iap[1]);
                     idy < std::min(iap[1] + 2, BS[1]); ++idy)
                  for (int idx = std::max(0, iap[0]);
                       idx < std::min(iap[0] + 2, BS[0]); ++idx) {
                    const int sx = idx - iap[0], sy = idy - iap[1];
                    const Real wxwy = wghts[1][sy] * wghts[0][sx];
                    assert(idx >= 0 && idx < _BS_ && wxwy >= 0);
                    assert(idy >= 0 && idy < _BS_ && wxwy <= 1);
                    o->udef[idy][idx][0] += wxwy * udef[0];
                    o->udef[idy][idx][1] += wxwy * udef[1];
                    o->chi[idy][idx] += wxwy;
                    static constexpr Real EPS =
                        std::numeric_limits<Real>::epsilon();
                    if (std::fabs(o->dist[idy][idx] + 1) < EPS)
                      o->dist[idy][idx] = 1;
                  }
              }
            }
          }
          static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
          for (int iy = 0; iy < _BS_; iy++)
            for (int ix = 0; ix < _BS_; ix++) {
              const Real normfac = o->chi[iy][ix] > EPS ? o->chi[iy][ix] : 1;
              o->udef[iy][ix][0] /= normfac;
              o->udef[iy][ix][1] /= normfac;
              o->dist[iy][ix] = o->dist[iy][ix] >= 0
                                    ? std::sqrt(o->dist[iy][ix])
                                    : -std::sqrt(-o->dist[iy][ix]);
              b(ix, iy).s = std::max(b(ix, iy).s, o->dist[iy][ix]);
              ;
            }
          std::fill(o->chi[0], o->chi[0] + BS[1] * BS[0], 0);
        }
      }
    }
    for (auto &E : vSegments)
      delete E;
    for (auto &E : segmentsPerBlock)
      delete E;
  }
  compute<ScalarLab>(PutChiOnGrid(), sim.tmp);
  compute<ComputeSurfaceNormals, ScalarGrid, ScalarLab, ScalarGrid, ScalarLab>(
      ComputeSurfaceNormals(), *sim.chi, *sim.tmp);
  for (const auto &shape : sim.shapes) {
    Real com[3] = {0.0, 0.0, 0.0};
    const std::vector<ObstacleBlock *> &OBLOCK = shape->obstacleBlocks;
#pragma omp parallel for reduction(+ : com[:3])
    for (size_t i = 0; i < OBLOCK.size(); i++) {
      if (OBLOCK[i] == nullptr)
        continue;
      com[0] += OBLOCK[i]->Mass;
      com[1] += OBLOCK[i]->COM_x;
      com[2] += OBLOCK[i]->COM_y;
    }
    MPI_Allreduce(MPI_IN_PLACE, com, 3, MPI_Real, MPI_SUM, MPI_COMM_WORLD);
    shape->M = com[0];
    shape->centerOfMass[0] += com[1] / com[0];
    shape->centerOfMass[1] += com[2] / com[0];
  }
  for (const auto &shape : sim.shapes) {
    Real _x = 0, _y = 0, _m = 0, _j = 0, _u = 0, _v = 0, _a = 0;
#pragma omp parallel for schedule(dynamic, 1)                                  \
    reduction(+ : _x, _y, _m, _j, _u, _v, _a)
    for (size_t i = 0; i < chiInfo.size(); i++) {
      const Real hsq = std::pow(chiInfo[i].h, 2);
      const auto pos = shape->obstacleBlocks[chiInfo[i].blockID];
      if (pos == nullptr)
        continue;
      const CHI_MAT &__restrict__ CHI = pos->chi;
      const UDEFMAT &__restrict__ UDEF = pos->udef;
      for (int iy = 0; iy < _BS_; ++iy)
        for (int ix = 0; ix < _BS_; ++ix) {
          if (CHI[iy][ix] <= 0)
            continue;
          Real p[2];
          chiInfo[i].pos(p, ix, iy);
          const Real chi = CHI[iy][ix] * hsq;
          p[0] -= shape->centerOfMass[0];
          p[1] -= shape->centerOfMass[1];
          _x += chi * p[0];
          _y += chi * p[1];
          _m += chi;
          _j += chi * (p[0] * p[0] + p[1] * p[1]);
          _u += chi * UDEF[iy][ix][0];
          _v += chi * UDEF[iy][ix][1];
          _a += chi * (p[0] * UDEF[iy][ix][1] - p[1] * UDEF[iy][ix][0]);
        }
    }
    Real quantities[7] = {_x, _y, _m, _j, _u, _v, _a};
    MPI_Allreduce(MPI_IN_PLACE, quantities, 7, MPI_Real, MPI_SUM,
                  MPI_COMM_WORLD);
    _x = quantities[0];
    _y = quantities[1];
    _m = quantities[2];
    _j = quantities[3];
    _u = quantities[4];
    _v = quantities[5];
    _a = quantities[6];
    _u /= _m;
    _v /= _m;
    _a /= _j;
    Integrals I = Integrals(_x, _y, _m, _j, _u, _v, _a);
    shape->M = I.m;
    shape->J = I.j;
    const Real dCx = shape->center[0] - shape->centerOfMass[0];
    const Real dCy = shape->center[1] - shape->centerOfMass[1];
    shape->d_gm[0] =
        dCx * std::cos(shape->orientation) + dCy * std::sin(shape->orientation);
    shape->d_gm[1] = -dCx * std::sin(shape->orientation) +
                     dCy * std::cos(shape->orientation);
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < chiInfo.size(); i++) {
      const auto pos = shape->obstacleBlocks[chiInfo[i].blockID];
      if (pos == nullptr)
        continue;
      for (int iy = 0; iy < _BS_; ++iy)
        for (int ix = 0; ix < _BS_; ++ix) {
          Real p[2];
          chiInfo[i].pos(p, ix, iy);
          p[0] -= shape->centerOfMass[0];
          p[1] -= shape->centerOfMass[1];
          pos->udef[iy][ix][0] -= I.u - I.a * p[1];
          pos->udef[iy][ix][1] -= I.v + I.a * p[0];
        }
    }
    const Real Rmatrix2D[2][2] = {
        {std::cos(shape->orientation), -std::sin(shape->orientation)},
        {std::sin(shape->orientation), std::cos(shape->orientation)}};
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < shape->upperSkin.Npoints; ++i) {
      rotate2D(Rmatrix2D, shape->upperSkin.xSurf[i], shape->upperSkin.ySurf[i]);
      shape->upperSkin.xSurf[i] += shape->centerOfMass[0];
      shape->upperSkin.ySurf[i] += shape->centerOfMass[1];
      rotate2D(Rmatrix2D, shape->lowerSkin.xSurf[i], shape->lowerSkin.ySurf[i]);
      shape->lowerSkin.xSurf[i] += shape->centerOfMass[0];
      shape->lowerSkin.ySurf[i] += shape->centerOfMass[1];
    }
    {
      const Real Rmatrix2D[2][2] = {
          {std::cos(shape->orientation), -std::sin(shape->orientation)},
          {std::sin(shape->orientation), std::cos(shape->orientation)}};
      for (int i = 0; i < shape->Nm; ++i) {
        rotate2D(Rmatrix2D, shape->rX[i], shape->rY[i]);
        rotate2D(Rmatrix2D, shape->norX[i], shape->norY[i]);
        shape->rX[i] += shape->centerOfMass[0];
        shape->rY[i] += shape->centerOfMass[1];
      }
#pragma omp parallel for
      for (size_t i = 0; i < shape->lowerSkin.Npoints - 1; ++i) {
        shape->lowerSkin.midX[i] =
            (shape->lowerSkin.xSurf[i] + shape->lowerSkin.xSurf[i + 1]) / 2;
        shape->upperSkin.midX[i] =
            (shape->upperSkin.xSurf[i] + shape->upperSkin.xSurf[i + 1]) / 2;
        shape->lowerSkin.midY[i] =
            (shape->lowerSkin.ySurf[i] + shape->lowerSkin.ySurf[i + 1]) / 2;
        shape->upperSkin.midY[i] =
            (shape->upperSkin.ySurf[i] + shape->upperSkin.ySurf[i + 1]) / 2;
        shape->lowerSkin.normXSurf[i] =
            (shape->lowerSkin.ySurf[i + 1] - shape->lowerSkin.ySurf[i]);
        shape->upperSkin.normXSurf[i] =
            (shape->upperSkin.ySurf[i + 1] - shape->upperSkin.ySurf[i]);
        shape->lowerSkin.normYSurf[i] =
            -(shape->lowerSkin.xSurf[i + 1] - shape->lowerSkin.xSurf[i]);
        shape->upperSkin.normYSurf[i] =
            -(shape->upperSkin.xSurf[i + 1] - shape->upperSkin.xSurf[i]);
        Real normL = std::sqrt(std::pow(shape->lowerSkin.normXSurf[i], 2) +
                               std::pow(shape->lowerSkin.normYSurf[i], 2));
        Real normU = std::sqrt(std::pow(shape->upperSkin.normXSurf[i], 2) +
                               std::pow(shape->upperSkin.normYSurf[i], 2));
        shape->lowerSkin.normXSurf[i] /= normL;
        shape->upperSkin.normXSurf[i] /= normU;
        shape->lowerSkin.normYSurf[i] /= normL;
        shape->upperSkin.normYSurf[i] /= normU;
        const int ii = (i < 8) ? 8
                               : ((i > shape->lowerSkin.Npoints - 9)
                                      ? shape->lowerSkin.Npoints - 9
                                      : i);
        const Real dirL = shape->lowerSkin.normXSurf[i] *
                              (shape->lowerSkin.midX[i] - shape->rX[ii]) +
                          shape->lowerSkin.normYSurf[i] *
                              (shape->lowerSkin.midY[i] - shape->rY[ii]);
        const Real dirU = shape->upperSkin.normXSurf[i] *
                              (shape->upperSkin.midX[i] - shape->rX[ii]) +
                          shape->upperSkin.normYSurf[i] *
                              (shape->upperSkin.midY[i] - shape->rY[ii]);
        if (dirL < 0) {
          shape->lowerSkin.normXSurf[i] *= -1.0;
          shape->lowerSkin.normYSurf[i] *= -1.0;
        }
        if (dirU < 0) {
          shape->upperSkin.normXSurf[i] *= -1.0;
          shape->upperSkin.normYSurf[i] *= -1.0;
        }
      }
    }
  }
}
struct GradChiOnTmp {
  GradChiOnTmp() {}
  const StencilInfo stencil{-4, -4, 0, 5, 5, 1, true, {0}};
  const std::vector<BlockInfo> &tmpInfo = sim.tmp->m_vInfo;
  void operator()(ScalarLab &lab, const BlockInfo &info) const {
    auto &__restrict__ TMP = *(ScalarBlock *)tmpInfo[info.blockID].ptrBlock;
    const int offset = (info.level == sim.tmp->getlevelMax() - 1) ? 4 : 2;
    const Real threshold = sim.bAdaptChiGradient ? 0.9 : 1e4;
    for (int y = -offset; y < _BS_ + offset; ++y)
      for (int x = -offset; x < _BS_ + offset; ++x) {
        lab(x, y).s = std::min(lab(x, y).s, (Real)1.0);
        lab(x, y).s = std::max(lab(x, y).s, (Real)0.0);
        if (lab(x, y).s > 0.0 && lab(x, y).s < threshold) {
          TMP(_BS_ / 2 - 1, _BS_ / 2).s = 2 * sim.Rtol;
          TMP(_BS_ / 2 - 1, _BS_ / 2 - 1).s = 2 * sim.Rtol;
          TMP(_BS_ / 2, _BS_ / 2).s = 2 * sim.Rtol;
          TMP(_BS_ / 2, _BS_ / 2 - 1).s = 2 * sim.Rtol;
          break;
        }
      }
  }
};
static void adapt() {
  compute<VectorLab>(KernelVorticity(), sim.vel);
  compute<ScalarLab>(GradChiOnTmp(), sim.chi);
  sim.tmp_amr->Tag();
  sim.chi_amr->TagLike(sim.tmp->m_vInfo);
  sim.pres_amr->TagLike(sim.tmp->m_vInfo);
  sim.pold_amr->TagLike(sim.tmp->m_vInfo);
  sim.vel_amr->TagLike(sim.tmp->m_vInfo);
  sim.vOld_amr->TagLike(sim.tmp->m_vInfo);
  sim.tmpV_amr->TagLike(sim.tmp->m_vInfo);
  sim.tmp_amr->Adapt(false);
  sim.chi_amr->Adapt(false);
  sim.vel_amr->Adapt(false);
  sim.vOld_amr->Adapt(false);
  sim.pres_amr->Adapt(false);
  sim.pold_amr->Adapt(false);
  sim.tmpV_amr->Adapt(true);
}
static Real weno5_plus(Real um2, Real um1, Real u, Real up1, Real up2) {
  Real exponent = 2, e = 1e-6;
  Real b1 = 13.0 / 12.0 * pow((um2 + u) - 2 * um1, 2) +
            0.25 * pow((um2 + 3 * u) - 4 * um1, 2);
  Real b2 =
      13.0 / 12.0 * pow((um1 + up1) - 2 * u, 2) + 0.25 * pow(um1 - up1, 2);
  Real b3 = 13.0 / 12.0 * pow((u + up2) - 2 * up1, 2) +
            0.25 * pow((3 * u + up2) - 4 * up1, 2);
  Real g1 = 0.1, g2 = 0.6, g3 = 0.3;
  Real what1 = g1 / pow(b1 + e, exponent);
  Real what2 = g2 / pow(b2 + e, exponent);
  Real what3 = g3 / pow(b3 + e, exponent);
  Real aux = 1.0 / ((what1 + what3) + what2);
  Real w1 = what1 * aux, w2 = what2 * aux, w3 = what3 * aux;
  Real f1 = (11.0 / 6.0) * u + ((1.0 / 3.0) * um2 - (7.0 / 6.0) * um1);
  Real f2 = (5.0 / 6.0) * u + ((-1.0 / 6.0) * um1 + (1.0 / 3.0) * up1);
  Real f3 = (1.0 / 3.0) * u + ((+5.0 / 6.0) * up1 - (1.0 / 6.0) * up2);
  return (w1 * f1 + w3 * f3) + w2 * f2;
}
static Real weno5_minus(Real um2, Real um1, Real u, Real up1, Real up2) {
  Real exponent = 2, e = 1e-6;
  Real b1 = 13.0 / 12.0 * pow((um2 + u) - 2 * um1, 2) +
            0.25 * pow((um2 + 3 * u) - 4 * um1, 2);
  Real b2 =
      13.0 / 12.0 * pow((um1 + up1) - 2 * u, 2) + 0.25 * pow(um1 - up1, 2);
  Real b3 = 13.0 / 12.0 * pow((u + up2) - 2 * up1, 2) +
            0.25 * pow((3 * u + up2) - 4 * up1, 2);
  Real g1 = 0.3, g2 = 0.6, g3 = 0.1;
  Real what1 = g1 / pow(b1 + e, exponent);
  Real what2 = g2 / pow(b2 + e, exponent);
  Real what3 = g3 / pow(b3 + e, exponent);
  Real aux = 1.0 / ((what1 + what3) + what2);
  Real w1 = what1 * aux;
  Real w2 = what2 * aux;
  Real w3 = what3 * aux;
  Real f1 = (1.0 / 3.0) * u + ((-1.0 / 6.0) * um2 + (5.0 / 6.0) * um1);
  Real f2 = (5.0 / 6.0) * u + ((1.0 / 3.0) * um1 - (1.0 / 6.0) * up1);
  Real f3 = (11.0 / 6.0) * u + ((-7.0 / 6.0) * up1 + (1.0 / 3.0) * up2);
  return (w1 * f1 + w3 * f3) + w2 * f2;
}
static Real derivative(const Real U, const Real um3, const Real um2,
                       const Real um1, const Real u, const Real up1,
                       const Real up2, const Real up3) {
  Real fp = 0.0;
  Real fm = 0.0;
  if (U > 0) {
    fp = weno5_plus(um2, um1, u, up1, up2);
    fm = weno5_plus(um3, um2, um1, u, up1);
  } else {
    fp = weno5_minus(um1, u, up1, up2, up3);
    fm = weno5_minus(um2, um1, u, up1, up2);
  }
  return (fp - fm);
}
static Real dU_adv_dif(const VectorLab &V, const Real uinf[2], const Real advF,
                       const Real difF, const int ix, const int iy) {
  const Real u = V(ix, iy).u[0];
  const Real v = V(ix, iy).u[1];
  const Real UU = u + uinf[0];
  const Real VV = v + uinf[1];
  const Real up1x = V(ix + 1, iy).u[0];
  const Real up2x = V(ix + 2, iy).u[0];
  const Real up3x = V(ix + 3, iy).u[0];
  const Real um1x = V(ix - 1, iy).u[0];
  const Real um2x = V(ix - 2, iy).u[0];
  const Real um3x = V(ix - 3, iy).u[0];
  const Real up1y = V(ix, iy + 1).u[0];
  const Real up2y = V(ix, iy + 2).u[0];
  const Real up3y = V(ix, iy + 3).u[0];
  const Real um1y = V(ix, iy - 1).u[0];
  const Real um2y = V(ix, iy - 2).u[0];
  const Real um3y = V(ix, iy - 3).u[0];
  const Real dudx = derivative(UU, um3x, um2x, um1x, u, up1x, up2x, up3x);
  const Real dudy = derivative(VV, um3y, um2y, um1y, u, up1y, up2y, up3y);
  return advF * (UU * dudx + VV * dudy) +
         difF * (((up1x + um1x) + (up1y + um1y)) - 4 * u);
}
static Real dV_adv_dif(const VectorLab &V, const Real uinf[2], const Real advF,
                       const Real difF, const int ix, const int iy) {
  const Real u = V(ix, iy).u[0];
  const Real v = V(ix, iy).u[1];
  const Real UU = u + uinf[0];
  const Real VV = v + uinf[1];
  const Real vp1x = V(ix + 1, iy).u[1];
  const Real vp2x = V(ix + 2, iy).u[1];
  const Real vp3x = V(ix + 3, iy).u[1];
  const Real vm1x = V(ix - 1, iy).u[1];
  const Real vm2x = V(ix - 2, iy).u[1];
  const Real vm3x = V(ix - 3, iy).u[1];
  const Real vp1y = V(ix, iy + 1).u[1];
  const Real vp2y = V(ix, iy + 2).u[1];
  const Real vp3y = V(ix, iy + 3).u[1];
  const Real vm1y = V(ix, iy - 1).u[1];
  const Real vm2y = V(ix, iy - 2).u[1];
  const Real vm3y = V(ix, iy - 3).u[1];
  const Real dvdx = derivative(UU, vm3x, vm2x, vm1x, v, vp1x, vp2x, vp3x);
  const Real dvdy = derivative(VV, vm3y, vm2y, vm1y, v, vp1y, vp2y, vp3y);
  return advF * (UU * dvdx + VV * dvdy) +
         difF * (((vp1x + vm1x) + (vp1y + vm1y)) - 4 * v);
}
template <typename ElementType> struct KernelAdvectDiffuse {
  KernelAdvectDiffuse() {
    uinf[0] = sim.uinfx;
    uinf[1] = sim.uinfy;
  }
  Real uinf[2];
  const StencilInfo stencil{-3, -3, 0, 4, 4, 1, true, {0, 1}};
  const std::vector<BlockInfo> &tmpVInfo = sim.tmpV->m_vInfo;
  void operator()(VectorLab &lab, const BlockInfo &info) const {
    const Real h = info.h;
    const Real dfac = sim.nu * sim.dt;
    const Real afac = -sim.dt * h;
    VectorBlock &__restrict__ TMP =
        *(VectorBlock *)tmpVInfo[info.blockID].ptrBlock;
    for (int iy = 0; iy < _BS_; ++iy)
      for (int ix = 0; ix < _BS_; ++ix) {
        TMP(ix, iy).u[0] = dU_adv_dif(lab, uinf, afac, dfac, ix, iy);
        TMP(ix, iy).u[1] = dV_adv_dif(lab, uinf, afac, dfac, ix, iy);
      }
    BlockCase<VectorBlock, ElementType> *tempCase =
        (BlockCase<VectorBlock, ElementType> *)(tmpVInfo[info.blockID]
                                                    .auxiliary);
    ElementType *faceXm = nullptr;
    ElementType *faceXp = nullptr;
    ElementType *faceYm = nullptr;
    ElementType *faceYp = nullptr;
    const Real aux_coef = dfac;
    if (tempCase != nullptr) {
      faceXm = tempCase->storedFace[0] ? &tempCase->m_pData[0][0] : nullptr;
      faceXp = tempCase->storedFace[1] ? &tempCase->m_pData[1][0] : nullptr;
      faceYm = tempCase->storedFace[2] ? &tempCase->m_pData[2][0] : nullptr;
      faceYp = tempCase->storedFace[3] ? &tempCase->m_pData[3][0] : nullptr;
    }
    if (faceXm != nullptr) {
      int ix = 0;
      for (int iy = 0; iy < _BS_; ++iy) {
        faceXm[iy].u[0] = aux_coef * (lab(ix, iy).u[0] - lab(ix - 1, iy).u[0]);
        faceXm[iy].u[1] = aux_coef * (lab(ix, iy).u[1] - lab(ix - 1, iy).u[1]);
      }
    }
    if (faceXp != nullptr) {
      int ix = _BS_ - 1;
      for (int iy = 0; iy < _BS_; ++iy) {
        faceXp[iy].u[0] = aux_coef * (lab(ix, iy).u[0] - lab(ix + 1, iy).u[0]);
        faceXp[iy].u[1] = aux_coef * (lab(ix, iy).u[1] - lab(ix + 1, iy).u[1]);
      }
    }
    if (faceYm != nullptr) {
      int iy = 0;
      for (int ix = 0; ix < _BS_; ++ix) {
        faceYm[ix].u[0] = aux_coef * (lab(ix, iy).u[0] - lab(ix, iy - 1).u[0]);
        faceYm[ix].u[1] = aux_coef * (lab(ix, iy).u[1] - lab(ix, iy - 1).u[1]);
      }
    }
    if (faceYp != nullptr) {
      int iy = _BS_ - 1;
      for (int ix = 0; ix < _BS_; ++ix) {
        faceYp[ix].u[0] = aux_coef * (lab(ix, iy).u[0] - lab(ix, iy + 1).u[0]);
        faceYp[ix].u[1] = aux_coef * (lab(ix, iy).u[1] - lab(ix, iy + 1).u[1]);
      }
    }
  }
};
struct KernelComputeForces {
  const int big = 5;
  const int small = -4;
  KernelComputeForces(){};
  StencilInfo stencil{small, small, 0, big, big, 1, true, {0, 1}};
  StencilInfo stencil2{small, small, 0, big, big, 1, true, {0}};
  const int bigg = _BS_ + big - 1;
  const int stencil_start[3] = {small, small, small},
            stencil_end[3] = {big, big, big};
  const Real c0 = -137. / 60.;
  const Real c1 = 5.;
  const Real c2 = -5.;
  const Real c3 = 10. / 3.;
  const Real c4 = -5. / 4.;
  const Real c5 = 1. / 5.;
  bool inrange(const int i) const { return (i >= small && i < bigg); }
  const std::vector<BlockInfo> &presInfo = sim.pres->m_vInfo;
  void operator()(VectorLab &lab, ScalarLab &chi, const BlockInfo &info,
                  const BlockInfo &info2) const {
    VectorLab &V = lab;
    ScalarBlock &__restrict__ P =
        *(ScalarBlock *)presInfo[info.blockID].ptrBlock;
    for (const auto &shape : sim.shapes) {
      const std::vector<ObstacleBlock *> &OBLOCK = shape->obstacleBlocks;
      const Real Cx = shape->centerOfMass[0], Cy = shape->centerOfMass[1];
      const Real vel_norm =
          std::sqrt(shape->u * shape->u + shape->v * shape->v);
      const Real vel_unit[2] = {
          vel_norm > 0 ? (Real)shape->u / vel_norm : (Real)0,
          vel_norm > 0 ? (Real)shape->v / vel_norm : (Real)0};
      const Real NUoH = sim.nu / info.h;
      ObstacleBlock *const O = OBLOCK[info.blockID];
      if (O == nullptr)
        continue;
      assert(O->filled);
      for (size_t k = 0; k < O->n_surfPoints; ++k) {
        const int ix = O->surface[k]->ix, iy = O->surface[k]->iy;
        const std::array<Real, 2> p = info.pos(ix, iy);
        const Real normX = O->surface[k]->dchidx;
        const Real normY = O->surface[k]->dchidy;
        const Real norm = 1.0 / std::sqrt(normX * normX + normY * normY);
        const Real dx = normX * norm;
        const Real dy = normY * norm;
        Real DuDx;
        Real DuDy;
        Real DvDx;
        Real DvDy;
        {
          int x = ix;
          int y = iy;
          for (int kk = 0; kk < 5; kk++) {
            const int dxi = round(kk * dx);
            const int dyi = round(kk * dy);
            if (ix + dxi + 1 >= _BS_ + big - 1 || ix + dxi - 1 < small)
              continue;
            if (iy + dyi + 1 >= _BS_ + big - 1 || iy + dyi - 1 < small)
              continue;
            x = ix + dxi;
            y = iy + dyi;
            if (chi(x, y).s < 0.01)
              break;
          }
          const auto &l = lab;
          const int sx = normX > 0 ? +1 : -1;
          const int sy = normY > 0 ? +1 : -1;
          VectorElement dveldx;
          if (inrange(x + 5 * sx))
            dveldx = sx * (c0 * l(x, y) + c1 * l(x + sx, y) +
                           c2 * l(x + 2 * sx, y) + c3 * l(x + 3 * sx, y) +
                           c4 * l(x + 4 * sx, y) + c5 * l(x + 5 * sx, y));
          else if (inrange(x + 2 * sx))
            dveldx = sx * (-1.5 * l(x, y) + 2.0 * l(x + sx, y) -
                           0.5 * l(x + 2 * sx, y));
          else
            dveldx = sx * (l(x + sx, y) - l(x, y));
          VectorElement dveldy;
          if (inrange(y + 5 * sy))
            dveldy = sy * (c0 * l(x, y) + c1 * l(x, y + sy) +
                           c2 * l(x, y + 2 * sy) + c3 * l(x, y + 3 * sy) +
                           c4 * l(x, y + 4 * sy) + c5 * l(x, y + 5 * sy));
          else if (inrange(y + 2 * sy))
            dveldy = sy * (-1.5 * l(x, y) + 2.0 * l(x, y + sy) -
                           0.5 * l(x, y + 2 * sy));
          else
            dveldy = sx * (l(x, y + sy) - l(x, y));
          const VectorElement dveldx2 =
              l(x - 1, y) - 2.0 * l(x, y) + l(x + 1, y);
          const VectorElement dveldy2 =
              l(x, y - 1) - 2.0 * l(x, y) + l(x, y + 1);
          VectorElement dveldxdy;
          if (inrange(x + 2 * sx) && inrange(y + 2 * sy))
            dveldxdy =
                sx * sy *
                (-0.5 * (-1.5 * l(x + 2 * sx, y) + 2 * l(x + 2 * sx, y + sy) -
                         0.5 * l(x + 2 * sx, y + 2 * sy)) +
                 2 * (-1.5 * l(x + sx, y) + 2 * l(x + sx, y + sy) -
                      0.5 * l(x + sx, y + 2 * sy)) -
                 1.5 * (-1.5 * l(x, y) + 2 * l(x, y + sy) -
                        0.5 * l(x, y + 2 * sy)));
          else
            dveldxdy = sx * sy * (l(x + sx, y + sy) - l(x + sx, y)) -
                       (l(x, y + sy) - l(x, y));
          DuDx =
              dveldx.u[0] + dveldx2.u[0] * (ix - x) + dveldxdy.u[0] * (iy - y);
          DvDx =
              dveldx.u[1] + dveldx2.u[1] * (ix - x) + dveldxdy.u[1] * (iy - y);
          DuDy =
              dveldy.u[0] + dveldy2.u[0] * (iy - y) + dveldxdy.u[0] * (ix - x);
          DvDy =
              dveldy.u[1] + dveldy2.u[1] * (iy - y) + dveldxdy.u[1] * (ix - x);
        }
        const Real fXV = NUoH * DuDx * normX + NUoH * DuDy * normY,
                   fXP = -P(ix, iy).s * normX;
        const Real fYV = NUoH * DvDx * normX + NUoH * DvDy * normY,
                   fYP = -P(ix, iy).s * normY;
        const Real fXT = fXV + fXP, fYT = fYV + fYP;
        O->x_s[k] = p[0];
        O->y_s[k] = p[1];
        O->p_s[k] = P(ix, iy).s;
        O->u_s[k] = V(ix, iy).u[0];
        O->v_s[k] = V(ix, iy).u[1];
        O->nx_s[k] = dx;
        O->ny_s[k] = dy;
        O->omega_s[k] = (DvDx - DuDy) / info.h;
        O->uDef_s[k] = O->udef[iy][ix][0];
        O->vDef_s[k] = O->udef[iy][ix][1];
        O->fX_s[k] = -P(ix, iy).s * dx + NUoH * DuDx * dx + NUoH * DuDy * dy;
        O->fY_s[k] = -P(ix, iy).s * dy + NUoH * DvDx * dx + NUoH * DvDy * dy;
        O->fXv_s[k] = NUoH * DuDx * dx + NUoH * DuDy * dy;
        O->fYv_s[k] = NUoH * DvDx * dx + NUoH * DvDy * dy;
        O->perimeter += std::sqrt(normX * normX + normY * normY);
        O->circulation += normX * O->v_s[k] - normY * O->u_s[k];
        O->forcex += fXT;
        O->forcey += fYT;
        O->forcex_V += fXV;
        O->forcey_V += fYV;
        O->forcex_P += fXP;
        O->forcey_P += fYP;
        O->torque += (p[0] - Cx) * fYT - (p[1] - Cy) * fXT;
        O->torque_P += (p[0] - Cx) * fYP - (p[1] - Cy) * fXP;
        O->torque_V += (p[0] - Cx) * fYV - (p[1] - Cy) * fXV;
        const Real forcePar = fXT * vel_unit[0] + fYT * vel_unit[1];
        O->thrust += .5 * (forcePar + std::fabs(forcePar));
        O->drag -= .5 * (forcePar - std::fabs(forcePar));
        const Real forcePerp = fXT * vel_unit[1] - fYT * vel_unit[0];
        O->lift += forcePerp;
        const Real powOut = fXT * O->u_s[k] + fYT * O->v_s[k];
        const Real powDef = fXT * O->uDef_s[k] + fYT * O->vDef_s[k];
        O->Pout += powOut;
        O->defPower += powDef;
        O->PoutBnd += std::min((Real)0, powOut);
        O->defPowerBnd += std::min((Real)0, powDef);
      }
      O->PoutNew = O->forcex * shape->u + O->forcey * shape->v;
    }
  }
};
using UDEFMAT = Real[_BS_][_BS_][2];
struct PoissonSolver {
  struct EdgeCellIndexer;
  std::unique_ptr<LocalSpMatDnVec> LocalLS_;
  std::vector<long long> Nblocks_xcumsum_;
  std::vector<long long> Nrows_xcumsum_;
  PoissonSolver()
      : GenericCell(*this), XminCell(*this), XmaxCell(*this), YminCell(*this),
        YmaxCell(*this), edgeIndexers{&XminCell, &XmaxCell, &YminCell,
                                      &YmaxCell} {
    Nblocks_xcumsum_.resize(sim.size + 1);
    Nrows_xcumsum_.resize(sim.size + 1);
    std::vector<std::vector<double>> L;
    std::vector<std::vector<double>> L_inv;
    L.resize((_BS_ * _BS_));
    L_inv.resize((_BS_ * _BS_));
    for (int i(0); i < (_BS_ * _BS_); i++) {
      L[i].resize(i + 1);
      L_inv[i].resize(i + 1);
      for (int j(0); j <= i; j++) {
        L_inv[i][j] = (i == j) ? 1. : 0.;
      }
    }
    for (int i(0); i < (_BS_ * _BS_); i++) {
      double s1 = 0;
      for (int k(0); k <= i - 1; k++)
        s1 += L[i][k] * L[i][k];
      L[i][i] = sqrt(getA_local(i, i) - s1);
      for (int j(i + 1); j < (_BS_ * _BS_); j++) {
        double s2 = 0;
        for (int k(0); k <= i - 1; k++)
          s2 += L[i][k] * L[j][k];
        L[j][i] = (getA_local(j, i) - s2) / L[i][i];
      }
    }
    for (int br(0); br < (_BS_ * _BS_); br++) {
      const double bsf = 1. / L[br][br];
      for (int c(0); c <= br; c++)
        L_inv[br][c] *= bsf;
      for (int wr(br + 1); wr < (_BS_ * _BS_); wr++) {
        const double wsf = L[wr][br];
        for (int c(0); c <= br; c++)
          L_inv[wr][c] -= (wsf * L_inv[br][c]);
      }
    }
    std::vector<double> P_inv((_BS_ * _BS_) * (_BS_ * _BS_));
    for (int i(0); i < (_BS_ * _BS_); i++)
      for (int j(0); j < (_BS_ * _BS_); j++) {
        double aux = 0.;
        for (int k(0); k < (_BS_ * _BS_); k++)
          aux += (i <= k && j <= k) ? L_inv[k][i] * L_inv[k][j] : 0.;
        P_inv[i * (_BS_ * _BS_) + j] = -aux;
      }
    LocalLS_ = std::make_unique<LocalSpMatDnVec>(MPI_COMM_WORLD, _BS_ * _BS_,
                                                 sim.bMeanConstraint, P_inv);
  }
  void solve(const ScalarGrid *input) {
    const double max_error = sim.step < 10 ? 0.0 : sim.PoissonTol;
    const double max_rel_error = sim.step < 10 ? 0.0 : sim.PoissonTolRel;
    const int max_restarts = sim.step < 10 ? 100 : sim.maxPoissonRestarts;
    if (sim.pres->UpdateFluxCorrection) {
      sim.pres->UpdateFluxCorrection = false;
      this->getMat();
      this->getVec();
      LocalLS_->solveWithUpdate(max_error, max_rel_error, max_restarts);
    } else {
      this->getVec();
      LocalLS_->solveNoUpdate(max_error, max_rel_error, max_restarts);
    }
    std::vector<BlockInfo> &zInfo = sim.pres->m_vInfo;
    const int Nblocks = zInfo.size();
    const std::vector<double> &x = LocalLS_->get_x();
    double avg = 0;
    double avg1 = 0;
#pragma omp parallel for reduction(+ : avg, avg1)
    for (int i = 0; i < Nblocks; i++) {
      ScalarBlock &P = *(ScalarBlock *)zInfo[i].ptrBlock;
      const double vv = zInfo[i].h * zInfo[i].h;
      for (int iy = 0; iy < _BS_; iy++)
        for (int ix = 0; ix < _BS_; ix++) {
          P(ix, iy).s = x[i * _BS_ * _BS_ + iy * _BS_ + ix];
          avg += P(ix, iy).s * vv;
          avg1 += vv;
        }
    }
    double quantities[2] = {avg, avg1};
    MPI_Allreduce(MPI_IN_PLACE, &quantities, 2, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    avg = quantities[0];
    avg1 = quantities[1];
    avg = avg / avg1;
#pragma omp parallel for
    for (int i = 0; i < Nblocks; i++) {
      ScalarBlock &P = *(ScalarBlock *)zInfo[i].ptrBlock;
      for (int iy = 0; iy < _BS_; iy++)
        for (int ix = 0; ix < _BS_; ix++)
          P(ix, iy).s += -avg;
    }
  }
  struct CellIndexer {
    CellIndexer(const PoissonSolver &pSolver) : ps(pSolver) {}
    ~CellIndexer() = default;
    long long This(const BlockInfo &info, const int ix, const int iy) const {
      return blockOffset(info) + (long long)(iy * _BS_ + ix);
    }
    static bool validXm(const int ix, const int iy) { return ix > 0; }
    static bool validXp(const int ix, const int iy) { return ix < _BS_ - 1; }
    static bool validYm(const int ix, const int iy) { return iy > 0; }
    static bool validYp(const int ix, const int iy) { return iy < _BS_ - 1; }
    long long Xmin(const BlockInfo &info, const int ix, const int iy,
                   const int offset = 0) const {
      return blockOffset(info) + (long long)(iy * _BS_ + offset);
    }
    long long Xmax(const BlockInfo &info, const int ix, const int iy,
                   const int offset = 0) const {
      return blockOffset(info) + (long long)(iy * _BS_ + (_BS_ - 1 - offset));
    }
    long long Ymin(const BlockInfo &info, const int ix, const int iy,
                   const int offset = 0) const {
      return blockOffset(info) + (long long)(offset * _BS_ + ix);
    }
    long long Ymax(const BlockInfo &info, const int ix, const int iy,
                   const int offset = 0) const {
      return blockOffset(info) + (long long)((_BS_ - 1 - offset) * _BS_ + ix);
    }
    long long blockOffset(const BlockInfo &info) const {
      return (info.blockID + ps.Nblocks_xcumsum_[sim.tmp->Tree(info).rank()]) *
             (_BS_ * _BS_);
    }
    static int ix_f(const int ix) { return (ix % (_BS_ / 2)) * 2; }
    static int iy_f(const int iy) { return (iy % (_BS_ / 2)) * 2; }
    const PoissonSolver &ps;
  };
  struct EdgeCellIndexer : public CellIndexer {
    EdgeCellIndexer(const PoissonSolver &pSolver) : CellIndexer(pSolver) {}
    virtual long long neiUnif(const BlockInfo &nei_info, const int ix,
                              const int iy) const = 0;
    virtual long long neiInward(const BlockInfo &info, const int ix,
                                const int iy) const = 0;
    virtual double taylorSign(const int ix, const int iy) const = 0;
    virtual int ix_c(const BlockInfo &info, const int ix) const {
      return info.index[0] % 2 == 0 ? ix / 2 : ix / 2 + _BS_ / 2;
    }
    virtual int iy_c(const BlockInfo &info, const int iy) const {
      return info.index[1] % 2 == 0 ? iy / 2 : iy / 2 + _BS_ / 2;
    }
    virtual long long neiFine1(const BlockInfo &nei_info, const int ix,
                               const int iy, const int offset = 0) const = 0;
    virtual long long neiFine2(const BlockInfo &nei_info, const int ix,
                               const int iy, const int offset = 0) const = 0;
    virtual bool isBD(const int ix, const int iy) const = 0;
    virtual bool isFD(const int ix, const int iy) const = 0;
    virtual long long Nei(const BlockInfo &info, const int ix, const int iy,
                          const int dist) const = 0;
    virtual long long Zchild(const BlockInfo &nei_info, const int ix,
                             const int iy) const = 0;
  };
  struct XbaseIndexer : public EdgeCellIndexer {
    XbaseIndexer(const PoissonSolver &pSolver) : EdgeCellIndexer(pSolver) {}
    double taylorSign(const int ix, const int iy) const override {
      return iy % 2 == 0 ? -1. : 1.;
    }
    bool isBD(const int ix, const int iy) const override {
      return iy == _BS_ - 1 || iy == _BS_ / 2 - 1;
    }
    bool isFD(const int ix, const int iy) const override {
      return iy == 0 || iy == _BS_ / 2;
    }
    long long Nei(const BlockInfo &info, const int ix, const int iy,
                  const int dist) const override {
      return This(info, ix, iy + dist);
    }
  };
  struct XminIndexer : public XbaseIndexer {
    XminIndexer(const PoissonSolver &pSolver) : XbaseIndexer(pSolver) {}
    long long neiUnif(const BlockInfo &nei_info, const int ix,
                      const int iy) const override {
      return Xmax(nei_info, ix, iy);
    }
    long long neiInward(const BlockInfo &info, const int ix,
                        const int iy) const override {
      return This(info, ix + 1, iy);
    }
    int ix_c(const BlockInfo &info, const int ix) const override {
      return _BS_ - 1;
    }
    long long neiFine1(const BlockInfo &nei_info, const int ix, const int iy,
                       const int offset = 0) const override {
      return Xmax(nei_info, ix_f(ix), iy_f(iy), offset);
    }
    long long neiFine2(const BlockInfo &nei_info, const int ix, const int iy,
                       const int offset = 0) const override {
      return Xmax(nei_info, ix_f(ix), iy_f(iy) + 1, offset);
    }
    long long Zchild(const BlockInfo &nei_info, const int ix,
                     const int iy) const override {
      return nei_info.Zchild[1][int(iy >= _BS_ / 2)][0];
    }
  };
  struct XmaxIndexer : public XbaseIndexer {
    XmaxIndexer(const PoissonSolver &pSolver) : XbaseIndexer(pSolver) {}
    long long neiUnif(const BlockInfo &nei_info, const int ix,
                      const int iy) const override {
      return Xmin(nei_info, ix, iy);
    }
    long long neiInward(const BlockInfo &info, const int ix,
                        const int iy) const override {
      return This(info, ix - 1, iy);
    }
    int ix_c(const BlockInfo &info, const int ix) const override { return 0; }
    long long neiFine1(const BlockInfo &nei_info, const int ix, const int iy,
                       const int offset = 0) const override {
      return Xmin(nei_info, ix_f(ix), iy_f(iy), offset);
    }
    long long neiFine2(const BlockInfo &nei_info, const int ix, const int iy,
                       const int offset = 0) const override {
      return Xmin(nei_info, ix_f(ix), iy_f(iy) + 1, offset);
    }
    long long Zchild(const BlockInfo &nei_info, const int ix,
                     const int iy) const override {
      return nei_info.Zchild[0][int(iy >= _BS_ / 2)][0];
    }
  };
  struct YbaseIndexer : public EdgeCellIndexer {
    YbaseIndexer(const PoissonSolver &pSolver) : EdgeCellIndexer(pSolver) {}
    double taylorSign(const int ix, const int iy) const override {
      return ix % 2 == 0 ? -1. : 1.;
    }
    bool isBD(const int ix, const int iy) const override {
      return ix == _BS_ - 1 || ix == _BS_ / 2 - 1;
    }
    bool isFD(const int ix, const int iy) const override {
      return ix == 0 || ix == _BS_ / 2;
    }
    long long Nei(const BlockInfo &info, const int ix, const int iy,
                  const int dist) const override {
      return This(info, ix + dist, iy);
    }
  };
  struct YminIndexer : public YbaseIndexer {
    YminIndexer(const PoissonSolver &pSolver) : YbaseIndexer(pSolver) {}
    long long neiUnif(const BlockInfo &nei_info, const int ix,
                      const int iy) const override {
      return Ymax(nei_info, ix, iy);
    }
    long long neiInward(const BlockInfo &info, const int ix,
                        const int iy) const override {
      return This(info, ix, iy + 1);
    }
    int iy_c(const BlockInfo &info, const int iy) const override {
      return _BS_ - 1;
    }
    long long neiFine1(const BlockInfo &nei_info, const int ix, const int iy,
                       const int offset = 0) const override {
      return Ymax(nei_info, ix_f(ix), iy_f(iy), offset);
    }
    long long neiFine2(const BlockInfo &nei_info, const int ix, const int iy,
                       const int offset = 0) const override {
      return Ymax(nei_info, ix_f(ix) + 1, iy_f(iy), offset);
    }
    long long Zchild(const BlockInfo &nei_info, const int ix,
                     const int iy) const override {
      return nei_info.Zchild[int(ix >= _BS_ / 2)][1][0];
    }
  };
  struct YmaxIndexer : public YbaseIndexer {
    YmaxIndexer(const PoissonSolver &pSolver) : YbaseIndexer(pSolver) {}
    long long neiUnif(const BlockInfo &nei_info, const int ix,
                      const int iy) const override {
      return Ymin(nei_info, ix, iy);
    }
    long long neiInward(const BlockInfo &info, const int ix,
                        const int iy) const override {
      return This(info, ix, iy - 1);
    }
    int iy_c(const BlockInfo &info, const int iy) const override { return 0; }
    long long neiFine1(const BlockInfo &nei_info, const int ix, const int iy,
                       const int offset = 0) const override {
      return Ymin(nei_info, ix_f(ix), iy_f(iy), offset);
    }
    long long neiFine2(const BlockInfo &nei_info, const int ix, const int iy,
                       const int offset = 0) const override {
      return Ymin(nei_info, ix_f(ix) + 1, iy_f(iy), offset);
    }
    long long Zchild(const BlockInfo &nei_info, const int ix,
                     const int iy) const override {
      return nei_info.Zchild[int(ix >= _BS_ / 2)][0][0];
    }
  };
  CellIndexer GenericCell;
  XminIndexer XminCell;
  XmaxIndexer XmaxCell;
  YminIndexer YminCell;
  YmaxIndexer YmaxCell;
  std::array<const EdgeCellIndexer *, 4> edgeIndexers;
  std::array<std::pair<long long, double>, 3> D1(const BlockInfo &info,
                                                 const EdgeCellIndexer &indexer,
                                                 const int ix,
                                                 const int iy) const {
    if (indexer.isBD(ix, iy))
      return {{{indexer.Nei(info, ix, iy, -2), 1. / 8.},
               {indexer.Nei(info, ix, iy, -1), -1. / 2.},
               {indexer.This(info, ix, iy), 3. / 8.}}};
    else if (indexer.isFD(ix, iy))
      return {{{indexer.Nei(info, ix, iy, 2), -1. / 8.},
               {indexer.Nei(info, ix, iy, 1), 1. / 2.},
               {indexer.This(info, ix, iy), -3. / 8.}}};
    return {{{indexer.Nei(info, ix, iy, -1), -1. / 8.},
             {indexer.Nei(info, ix, iy, 1), 1. / 8.},
             {indexer.This(info, ix, iy), 0.}}};
  }
  std::array<std::pair<long long, double>, 3> D2(const BlockInfo &info,
                                                 const EdgeCellIndexer &indexer,
                                                 const int ix,
                                                 const int iy) const {
    if (indexer.isBD(ix, iy))
      return {{{indexer.Nei(info, ix, iy, -2), 1. / 32.},
               {indexer.Nei(info, ix, iy, -1), -1. / 16.},
               {indexer.This(info, ix, iy), 1. / 32.}}};
    else if (indexer.isFD(ix, iy))
      return {{{indexer.Nei(info, ix, iy, 2), 1. / 32.},
               {indexer.Nei(info, ix, iy, 1), -1. / 16.},
               {indexer.This(info, ix, iy), 1. / 32.}}};
    return {{{indexer.Nei(info, ix, iy, -1), 1. / 32.},
             {indexer.Nei(info, ix, iy, 1), 1. / 32.},
             {indexer.This(info, ix, iy), -1. / 16.}}};
  }
  void interpolate(const BlockInfo &info_c, const int ix_c, const int iy_c,
                   const BlockInfo &info_f, const long long fine_close_idx,
                   const long long fine_far_idx, const double signInt,
                   const double signTaylor, const EdgeCellIndexer &indexer,
                   SpRowInfo &row) const {
    const int rank_c = sim.tmp->Tree(info_c).rank();
    const int rank_f = sim.tmp->Tree(info_f).rank();
    row.mapColVal(rank_f, fine_close_idx, signInt * 2. / 3.);
    row.mapColVal(rank_f, fine_far_idx, -signInt * 1. / 5.);
    const double tf = signInt * 8. / 15.;
    row.mapColVal(rank_c, indexer.This(info_c, ix_c, iy_c), tf);
    std::array<std::pair<long long, double>, 3> D;
    D = D1(info_c, indexer, ix_c, iy_c);
    for (int i(0); i < 3; i++)
      row.mapColVal(rank_c, D[i].first, signTaylor * tf * D[i].second);
    D = D2(info_c, indexer, ix_c, iy_c);
    for (int i(0); i < 3; i++)
      row.mapColVal(rank_c, D[i].first, tf * D[i].second);
  }
  void makeFlux(const BlockInfo &rhs_info, const int ix, const int iy,
                const BlockInfo &rhsNei, const EdgeCellIndexer &indexer,
                SpRowInfo &row) const {
    const long long sfc_idx = indexer.This(rhs_info, ix, iy);
    if (sim.tmp->Tree(rhsNei).Exists()) {
      const int nei_rank = sim.tmp->Tree(rhsNei).rank();
      const long long nei_idx = indexer.neiUnif(rhsNei, ix, iy);
      row.mapColVal(nei_rank, nei_idx, 1.);
      row.mapColVal(sfc_idx, -1.);
    } else if (sim.tmp->Tree(rhsNei).CheckCoarser()) {
      const BlockInfo &rhsNei_c =
          sim.tmp->getBlockInfoAll(rhs_info.level - 1, rhsNei.Zparent);
      const int ix_c = indexer.ix_c(rhs_info, ix);
      const int iy_c = indexer.iy_c(rhs_info, iy);
      const long long inward_idx = indexer.neiInward(rhs_info, ix, iy);
      const double signTaylor = indexer.taylorSign(ix, iy);
      interpolate(rhsNei_c, ix_c, iy_c, rhs_info, sfc_idx, inward_idx, 1.,
                  signTaylor, indexer, row);
      row.mapColVal(sfc_idx, -1.);
    } else if (sim.tmp->Tree(rhsNei).CheckFiner()) {
      const BlockInfo &rhsNei_f = sim.tmp->getBlockInfoAll(
          rhs_info.level + 1, indexer.Zchild(rhsNei, ix, iy));
      const int nei_rank = sim.tmp->Tree(rhsNei_f).rank();
      long long fine_close_idx = indexer.neiFine1(rhsNei_f, ix, iy, 0);
      long long fine_far_idx = indexer.neiFine1(rhsNei_f, ix, iy, 1);
      row.mapColVal(nei_rank, fine_close_idx, 1.);
      interpolate(rhs_info, ix, iy, rhsNei_f, fine_close_idx, fine_far_idx, -1.,
                  -1., indexer, row);
      fine_close_idx = indexer.neiFine2(rhsNei_f, ix, iy, 0);
      fine_far_idx = indexer.neiFine2(rhsNei_f, ix, iy, 1);
      row.mapColVal(nei_rank, fine_close_idx, 1.);
      interpolate(rhs_info, ix, iy, rhsNei_f, fine_close_idx, fine_far_idx, -1.,
                  1., indexer, row);
    } else {
      throw std::runtime_error(
          "Neighbour doesn't exist, isn't coarser, nor finer...");
    }
  }
  void getMat() {
    std::array<int, 3> blocksPerDim = sim.pres->getMaxBlocks();
    sim.tmp->UpdateBlockInfoAll_States(true);
    std::vector<BlockInfo> &RhsInfo = sim.tmp->m_vInfo;
    const int Nblocks = RhsInfo.size();
    const int N = _BS_ * _BS_ * Nblocks;
    LocalLS_->reserve(N);
    const long long Nblocks_long = Nblocks;
    MPI_Allgather(&Nblocks_long, 1, MPI_LONG_LONG, Nblocks_xcumsum_.data(), 1,
                  MPI_LONG_LONG, MPI_COMM_WORLD);
    for (int i(Nblocks_xcumsum_.size() - 1); i > 0; i--) {
      Nblocks_xcumsum_[i] = Nblocks_xcumsum_[i - 1];
    }
    Nblocks_xcumsum_[0] = 0;
    Nrows_xcumsum_[0] = 0;
    for (size_t i(1); i < Nblocks_xcumsum_.size(); i++) {
      Nblocks_xcumsum_[i] += Nblocks_xcumsum_[i - 1];
      Nrows_xcumsum_[i] = (_BS_ * _BS_) * Nblocks_xcumsum_[i];
    }
    for (int i = 0; i < Nblocks; i++) {
      const BlockInfo &rhs_info = RhsInfo[i];
      const int aux = 1 << rhs_info.level;
      const int MAX_X_BLOCKS = blocksPerDim[0] * aux - 1;
      const int MAX_Y_BLOCKS = blocksPerDim[1] * aux - 1;
      std::array<bool, 4> isBoundary;
      isBoundary[0] = (rhs_info.index[0] == 0);
      isBoundary[1] = (rhs_info.index[0] == MAX_X_BLOCKS);
      isBoundary[2] = (rhs_info.index[1] == 0);
      isBoundary[3] = (rhs_info.index[1] == MAX_Y_BLOCKS);
      std::array<bool, 2> isPeriodic;
      isPeriodic[0] = (cubismBCX == periodic);
      isPeriodic[1] = (cubismBCY == periodic);
      std::array<long long, 4> Z;
      Z[0] = rhs_info.Znei[1 - 1][1][1];
      Z[1] = rhs_info.Znei[1 + 1][1][1];
      Z[2] = rhs_info.Znei[1][1 - 1][1];
      Z[3] = rhs_info.Znei[1][1 + 1][1];
      std::array<const BlockInfo *, 4> rhsNei;
      rhsNei[0] = &(sim.tmp->getBlockInfoAll(rhs_info.level, Z[0]));
      rhsNei[1] = &(sim.tmp->getBlockInfoAll(rhs_info.level, Z[1]));
      rhsNei[2] = &(sim.tmp->getBlockInfoAll(rhs_info.level, Z[2]));
      rhsNei[3] = &(sim.tmp->getBlockInfoAll(rhs_info.level, Z[3]));
      for (int iy = 0; iy < _BS_; iy++)
        for (int ix = 0; ix < _BS_; ix++) {
          const long long sfc_idx = GenericCell.This(rhs_info, ix, iy);
          if ((ix > 0 && ix < _BS_ - 1) && (iy > 0 && iy < _BS_ - 1)) {
            LocalLS_->cooPushBackVal(1, sfc_idx,
                                     GenericCell.This(rhs_info, ix, iy - 1));
            LocalLS_->cooPushBackVal(1, sfc_idx,
                                     GenericCell.This(rhs_info, ix - 1, iy));
            LocalLS_->cooPushBackVal(-4, sfc_idx, sfc_idx);
            LocalLS_->cooPushBackVal(1, sfc_idx,
                                     GenericCell.This(rhs_info, ix + 1, iy));
            LocalLS_->cooPushBackVal(1, sfc_idx,
                                     GenericCell.This(rhs_info, ix, iy + 1));
          } else {
            std::array<bool, 4> validNei;
            validNei[0] = GenericCell.validXm(ix, iy);
            validNei[1] = GenericCell.validXp(ix, iy);
            validNei[2] = GenericCell.validYm(ix, iy);
            validNei[3] = GenericCell.validYp(ix, iy);
            std::array<long long, 4> idxNei;
            idxNei[0] = GenericCell.This(rhs_info, ix - 1, iy);
            idxNei[1] = GenericCell.This(rhs_info, ix + 1, iy);
            idxNei[2] = GenericCell.This(rhs_info, ix, iy - 1);
            idxNei[3] = GenericCell.This(rhs_info, ix, iy + 1);
            SpRowInfo row(sim.tmp->Tree(rhs_info).rank(), sfc_idx, 8);
            for (int j(0); j < 4; j++) {
              if (validNei[j]) {
                row.mapColVal(idxNei[j], 1);
                row.mapColVal(sfc_idx, -1);
              } else if (!isBoundary[j] || (isBoundary[j] && isPeriodic[j / 2]))
                this->makeFlux(rhs_info, ix, iy, *rhsNei[j], *edgeIndexers[j],
                               row);
            }
            LocalLS_->cooPushBackRow(row);
          }
        }
    }
    LocalLS_->make(Nrows_xcumsum_);
  }
  void getVec() {
    std::vector<BlockInfo> &RhsInfo = sim.tmp->m_vInfo;
    std::vector<BlockInfo> &zInfo = sim.pres->m_vInfo;
    const int Nblocks = RhsInfo.size();
    std::vector<double> &x = LocalLS_->get_x();
    std::vector<double> &b = LocalLS_->get_b();
    std::vector<double> &h2 = LocalLS_->get_h2();
    const long long shift = -Nrows_xcumsum_[sim.rank];
#pragma omp parallel for
    for (int i = 0; i < Nblocks; i++) {
      const BlockInfo &rhs_info = RhsInfo[i];
      const ScalarBlock &__restrict__ rhs = *(ScalarBlock *)RhsInfo[i].ptrBlock;
      const ScalarBlock &__restrict__ p = *(ScalarBlock *)zInfo[i].ptrBlock;
      h2[i] = RhsInfo[i].h * RhsInfo[i].h;
      for (int iy = 0; iy < _BS_; iy++)
        for (int ix = 0; ix < _BS_; ix++) {
          const long long sfc_loc = GenericCell.This(rhs_info, ix, iy) + shift;
          if (sim.bMeanConstraint && rhs_info.index[0] == 0 &&
              rhs_info.index[1] == 0 && rhs_info.index[2] == 0 && ix == 0 &&
              iy == 0)
            b[sfc_loc] = 0.;
          else
            b[sfc_loc] = rhs(ix, iy).s;
          x[sfc_loc] = p(ix, iy).s;
        }
    }
  }
};
using CHI_MAT = Real[_BS_][_BS_];
using UDEFMAT = Real[_BS_][_BS_][2];
void ComputeJ(const Real *Rc, const Real *R, const Real *N, const Real *I,
              Real *J) {
  const Real m00 = 1.0;
  const Real m01 = 0.0;
  const Real m02 = 0.0;
  const Real m11 = 1.0;
  const Real m12 = 0.0;
  const Real m22 = I[5];
  Real a00 = m22 * m11 - m12 * m12;
  Real a01 = m02 * m12 - m22 * m01;
  Real a02 = m01 * m12 - m02 * m11;
  Real a11 = m22 * m00 - m02 * m02;
  Real a12 = m01 * m02 - m00 * m12;
  Real a22 = m00 * m11 - m01 * m01;
  const Real determinant = 1.0 / ((m00 * a00) + (m01 * a01) + (m02 * a02));
  a00 *= determinant;
  a01 *= determinant;
  a02 *= determinant;
  a11 *= determinant;
  a12 *= determinant;
  a22 *= determinant;
  const Real aux_0 = (Rc[1] - R[1]) * N[2] - (Rc[2] - R[2]) * N[1];
  const Real aux_1 = (Rc[2] - R[2]) * N[0] - (Rc[0] - R[0]) * N[2];
  const Real aux_2 = (Rc[0] - R[0]) * N[1] - (Rc[1] - R[1]) * N[0];
  J[0] = a00 * aux_0 + a01 * aux_1 + a02 * aux_2;
  J[1] = a01 * aux_0 + a11 * aux_1 + a12 * aux_2;
  J[2] = a02 * aux_0 + a12 * aux_1 + a22 * aux_2;
}
void ElasticCollision(const Real m1, const Real m2, const Real *I1,
                      const Real *I2, const Real *v1, const Real *v2,
                      const Real *o1, const Real *o2, Real *hv1, Real *hv2,
                      Real *ho1, Real *ho2, const Real *C1, const Real *C2,
                      const Real NX, const Real NY, const Real NZ,
                      const Real CX, const Real CY, const Real CZ, Real *vc1,
                      Real *vc2) {
  const Real e = 1.0;
  const Real N[3] = {NX, NY, NZ};
  const Real C[3] = {CX, CY, CZ};
  const Real k1[3] = {N[0] / m1, N[1] / m1, N[2] / m1};
  const Real k2[3] = {-N[0] / m2, -N[1] / m2, -N[2] / m2};
  Real J1[3];
  Real J2[3];
  ComputeJ(C, C1, N, I1, J1);
  ComputeJ(C, C2, N, I2, J2);
  J2[0] = -J2[0];
  J2[1] = -J2[1];
  J2[2] = -J2[2];
  Real u1DEF[3];
  u1DEF[0] = vc1[0] - v1[0] - (o1[1] * (C[2] - C1[2]) - o1[2] * (C[1] - C1[1]));
  u1DEF[1] = vc1[1] - v1[1] - (o1[2] * (C[0] - C1[0]) - o1[0] * (C[2] - C1[2]));
  u1DEF[2] = vc1[2] - v1[2] - (o1[0] * (C[1] - C1[1]) - o1[1] * (C[0] - C1[0]));
  Real u2DEF[3];
  u2DEF[0] = vc2[0] - v2[0] - (o2[1] * (C[2] - C2[2]) - o2[2] * (C[1] - C2[1]));
  u2DEF[1] = vc2[1] - v2[1] - (o2[2] * (C[0] - C2[0]) - o2[0] * (C[2] - C2[2]));
  u2DEF[2] = vc2[2] - v2[2] - (o2[0] * (C[1] - C2[1]) - o2[1] * (C[0] - C2[0]));
  const Real nom = e * ((vc1[0] - vc2[0]) * N[0] + (vc1[1] - vc2[1]) * N[1] +
                        (vc1[2] - vc2[2]) * N[2]) +
                   ((v1[0] - v2[0] + u1DEF[0] - u2DEF[0]) * N[0] +
                    (v1[1] - v2[1] + u1DEF[1] - u2DEF[1]) * N[1] +
                    (v1[2] - v2[2] + u1DEF[2] - u2DEF[2]) * N[2]) +
                   ((o1[1] * (C[2] - C1[2]) - o1[2] * (C[1] - C1[1])) * N[0] +
                    (o1[2] * (C[0] - C1[0]) - o1[0] * (C[2] - C1[2])) * N[1] +
                    (o1[0] * (C[1] - C1[1]) - o1[1] * (C[0] - C1[0])) * N[2]) -
                   ((o2[1] * (C[2] - C2[2]) - o2[2] * (C[1] - C2[1])) * N[0] +
                    (o2[2] * (C[0] - C2[0]) - o2[0] * (C[2] - C2[2])) * N[1] +
                    (o2[0] * (C[1] - C2[1]) - o2[1] * (C[0] - C2[0])) * N[2]);
  const Real denom =
      -(1.0 / m1 + 1.0 / m2) +
      +((J1[1] * (C[2] - C1[2]) - J1[2] * (C[1] - C1[1])) * (-N[0]) +
        (J1[2] * (C[0] - C1[0]) - J1[0] * (C[2] - C1[2])) * (-N[1]) +
        (J1[0] * (C[1] - C1[1]) - J1[1] * (C[0] - C1[0])) * (-N[2])) -
      ((J2[1] * (C[2] - C2[2]) - J2[2] * (C[1] - C2[1])) * (-N[0]) +
       (J2[2] * (C[0] - C2[0]) - J2[0] * (C[2] - C2[2])) * (-N[1]) +
       (J2[0] * (C[1] - C2[1]) - J2[1] * (C[0] - C2[0])) * (-N[2]));
  const Real impulse = nom / (denom + 1e-21);
  hv1[0] = v1[0] + k1[0] * impulse;
  hv1[1] = v1[1] + k1[1] * impulse;
  hv1[2] = v1[2] + k1[2] * impulse;
  hv2[0] = v2[0] + k2[0] * impulse;
  hv2[1] = v2[1] + k2[1] * impulse;
  hv2[2] = v2[2] + k2[2] * impulse;
  ho1[0] = o1[0] + J1[0] * impulse;
  ho1[1] = o1[1] + J1[1] * impulse;
  ho1[2] = o1[2] + J1[2] * impulse;
  ho2[0] = o2[0] + J2[0] * impulse;
  ho2[1] = o2[1] + J2[1] * impulse;
  ho2[2] = o2[2] + J2[2] * impulse;
}
struct pressureCorrectionKernel {
  pressureCorrectionKernel(){};
  const StencilInfo stencil{-1, -1, 0, 2, 2, 1, false, {0}};
  const std::vector<BlockInfo> &tmpVInfo = sim.tmpV->m_vInfo;
  void operator()(ScalarLab &P, const BlockInfo &info) const {
    const Real h = info.h, pFac = -0.5 * sim.dt * h;
    VectorBlock &__restrict__ tmpV =
        *(VectorBlock *)tmpVInfo[info.blockID].ptrBlock;
    for (int iy = 0; iy < _BS_; ++iy)
      for (int ix = 0; ix < _BS_; ++ix) {
        tmpV(ix, iy).u[0] = pFac * (P(ix + 1, iy).s - P(ix - 1, iy).s);
        tmpV(ix, iy).u[1] = pFac * (P(ix, iy + 1).s - P(ix, iy - 1).s);
      }
    BlockCase<VectorBlock, VectorElement> *tempCase =
        (BlockCase<VectorBlock, VectorElement> *)(tmpVInfo[info.blockID]
                                                      .auxiliary);
    VectorElement *faceXm = nullptr;
    VectorElement *faceXp = nullptr;
    VectorElement *faceYm = nullptr;
    VectorElement *faceYp = nullptr;
    if (tempCase != nullptr) {
      faceXm = tempCase->storedFace[0] ? &tempCase->m_pData[0][0] : nullptr;
      faceXp = tempCase->storedFace[1] ? &tempCase->m_pData[1][0] : nullptr;
      faceYm = tempCase->storedFace[2] ? &tempCase->m_pData[2][0] : nullptr;
      faceYp = tempCase->storedFace[3] ? &tempCase->m_pData[3][0] : nullptr;
    }
    if (faceXm != nullptr) {
      int ix = 0;
      for (int iy = 0; iy < _BS_; ++iy) {
        faceXm[iy].u[0] = pFac * (P(ix - 1, iy).s + P(ix, iy).s);
        faceXm[iy].u[1] = 0;
      }
    }
    if (faceXp != nullptr) {
      int ix = _BS_ - 1;
      for (int iy = 0; iy < _BS_; ++iy) {
        faceXp[iy].u[0] = -pFac * (P(ix + 1, iy).s + P(ix, iy).s);
        faceXp[iy].u[1] = 0;
      }
    }
    if (faceYm != nullptr) {
      int iy = 0;
      for (int ix = 0; ix < _BS_; ++ix) {
        faceYm[ix].u[0] = 0;
        faceYm[ix].u[1] = pFac * (P(ix, iy - 1).s + P(ix, iy).s);
      }
    }
    if (faceYp != nullptr) {
      int iy = _BS_ - 1;
      for (int ix = 0; ix < _BS_; ++ix) {
        faceYp[ix].u[0] = 0;
        faceYp[ix].u[1] = -pFac * (P(ix, iy + 1).s + P(ix, iy).s);
      }
    }
  }
};
struct updatePressureRHS {
  updatePressureRHS(){};
  StencilInfo stencil{-1, -1, 0, 2, 2, 1, false, {0, 1}};
  StencilInfo stencil2{-1, -1, 0, 2, 2, 1, false, {0, 1}};
  const std::vector<BlockInfo> &tmpInfo = sim.tmp->m_vInfo;
  const std::vector<BlockInfo> &chiInfo = sim.chi->m_vInfo;
  void operator()(VectorLab &velLab, VectorLab &uDefLab, const BlockInfo &info,
                  const BlockInfo &) const {
    const Real h = info.h;
    const Real facDiv = 0.5 * h / sim.dt;
    ScalarBlock &__restrict__ TMP =
        *(ScalarBlock *)tmpInfo[info.blockID].ptrBlock;
    ScalarBlock &__restrict__ CHI =
        *(ScalarBlock *)chiInfo[info.blockID].ptrBlock;
    for (int iy = 0; iy < _BS_; ++iy)
      for (int ix = 0; ix < _BS_; ++ix) {
        TMP(ix, iy).s =
            facDiv * ((velLab(ix + 1, iy).u[0] - velLab(ix - 1, iy).u[0]) +
                      (velLab(ix, iy + 1).u[1] - velLab(ix, iy - 1).u[1]));
        TMP(ix, iy).s +=
            -facDiv * CHI(ix, iy).s *
            ((uDefLab(ix + 1, iy).u[0] - uDefLab(ix - 1, iy).u[0]) +
             (uDefLab(ix, iy + 1).u[1] - uDefLab(ix, iy - 1).u[1]));
      }
    BlockCase<ScalarBlock, ScalarElement> *tempCase =
        (BlockCase<ScalarBlock, ScalarElement> *)(tmpInfo[info.blockID]
                                                      .auxiliary);
    ScalarElement *faceXm = nullptr;
    ScalarElement *faceXp = nullptr;
    ScalarElement *faceYm = nullptr;
    ScalarElement *faceYp = nullptr;
    if (tempCase != nullptr) {
      faceXm = tempCase->storedFace[0] ? &tempCase->m_pData[0][0] : nullptr;
      faceXp = tempCase->storedFace[1] ? &tempCase->m_pData[1][0] : nullptr;
      faceYm = tempCase->storedFace[2] ? &tempCase->m_pData[2][0] : nullptr;
      faceYp = tempCase->storedFace[3] ? &tempCase->m_pData[3][0] : nullptr;
    }
    if (faceXm != nullptr) {
      int ix = 0;
      for (int iy = 0; iy < _BS_; ++iy) {
        faceXm[iy].s = facDiv * (velLab(ix - 1, iy).u[0] + velLab(ix, iy).u[0]);
        faceXm[iy].s += -(facDiv * CHI(ix, iy).s) *
                        (uDefLab(ix - 1, iy).u[0] + uDefLab(ix, iy).u[0]);
      }
    }
    if (faceXp != nullptr) {
      int ix = _BS_ - 1;
      for (int iy = 0; iy < _BS_; ++iy) {
        faceXp[iy].s =
            -facDiv * (velLab(ix + 1, iy).u[0] + velLab(ix, iy).u[0]);
        faceXp[iy].s -= -(facDiv * CHI(ix, iy).s) *
                        (uDefLab(ix + 1, iy).u[0] + uDefLab(ix, iy).u[0]);
      }
    }
    if (faceYm != nullptr) {
      int iy = 0;
      for (int ix = 0; ix < _BS_; ++ix) {
        faceYm[ix].s = facDiv * (velLab(ix, iy - 1).u[1] + velLab(ix, iy).u[1]);
        faceYm[ix].s += -(facDiv * CHI(ix, iy).s) *
                        (uDefLab(ix, iy - 1).u[1] + uDefLab(ix, iy).u[1]);
      }
    }
    if (faceYp != nullptr) {
      int iy = _BS_ - 1;
      for (int ix = 0; ix < _BS_; ++ix) {
        faceYp[ix].s =
            -facDiv * (velLab(ix, iy + 1).u[1] + velLab(ix, iy).u[1]);
        faceYp[ix].s -= -(facDiv * CHI(ix, iy).s) *
                        (uDefLab(ix, iy + 1).u[1] + uDefLab(ix, iy).u[1]);
      }
    }
  }
};
struct updatePressureRHS1 {
  updatePressureRHS1() {}
  StencilInfo stencil{-1, -1, 0, 2, 2, 1, false, {0}};
  void operator()(ScalarLab &lab, const BlockInfo &info) const {
    ScalarBlock &__restrict__ TMP =
        *(ScalarBlock *)sim.tmp->m_vInfo[info.blockID].ptrBlock;
    for (int iy = 0; iy < _BS_; ++iy)
      for (int ix = 0; ix < _BS_; ++ix)
        TMP(ix, iy).s -= (((lab(ix - 1, iy).s + lab(ix + 1, iy).s) +
                           (lab(ix, iy - 1).s + lab(ix, iy + 1).s)) -
                          4.0 * lab(ix, iy).s);
    BlockCase<ScalarBlock, ScalarElement> *tempCase =
        (BlockCase<ScalarBlock, ScalarElement> *)(sim.tmp->m_vInfo[info.blockID]
                                                    .auxiliary);
    ScalarElement *faceXm = nullptr;
    ScalarElement *faceXp = nullptr;
    ScalarElement *faceYm = nullptr;
    ScalarElement *faceYp = nullptr;
    if (tempCase != nullptr) {
      faceXm = tempCase->storedFace[0] ? &tempCase->m_pData[0][0] : nullptr;
      faceXp = tempCase->storedFace[1] ? &tempCase->m_pData[1][0] : nullptr;
      faceYm = tempCase->storedFace[2] ? &tempCase->m_pData[2][0] : nullptr;
      faceYp = tempCase->storedFace[3] ? &tempCase->m_pData[3][0] : nullptr;
    }
    if (faceXm != nullptr) {
      int ix = 0;
      for (int iy = 0; iy < _BS_; ++iy)
        faceXm[iy] = lab(ix - 1, iy) - lab(ix, iy);
    }
    if (faceXp != nullptr) {
      int ix = _BS_ - 1;
      for (int iy = 0; iy < _BS_; ++iy)
        faceXp[iy] = lab(ix + 1, iy) - lab(ix, iy);
    }
    if (faceYm != nullptr) {
      int iy = 0;
      for (int ix = 0; ix < _BS_; ++ix)
        faceYm[ix] = lab(ix, iy - 1) - lab(ix, iy);
    }
    if (faceYp != nullptr) {
      int iy = _BS_ - 1;
      for (int ix = 0; ix < _BS_; ++ix)
        faceYp[ix] = lab(ix, iy + 1) - lab(ix, iy);
    }
  }
};
struct FactoryFileLineParser : public CommandlineParser {
  std::string &ltrim(std::string &s) {
    s.erase(s.begin(),
            std::find_if(s.begin(), s.end(),
                         std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
  }
  std::string &rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(),
                         std::not1(std::ptr_fun<int, int>(std::isspace)))
                .base(),
            s.end());
    return s;
  }
  std::string &trim(std::string &s) { return ltrim(rtrim(s)); }
  FactoryFileLineParser(std::istringstream &is_line)
      : CommandlineParser(0, NULL) {
    std::string key, value;
    while (std::getline(is_line, key, '=')) {
      if (std::getline(is_line, value, ' ')) {
        mapArguments[trim(key)] = Value(trim(value));
      }
    }
  }
};
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  CommandlineParser parser(argc, argv);
  MPI_Comm_size(MPI_COMM_WORLD, &sim.size);
  MPI_Comm_rank(MPI_COMM_WORLD, &sim.rank);
  if (sim.rank == 0)
    printf("%d ranks\n", sim.size);
#ifdef _OPENMP
#pragma omp parallel
  {
#pragma omp master
    if (sim.rank == 0)
      printf("%d threads\n", omp_get_num_threads());
  }
#endif
  parser.set_strict_mode();
  sim.bpdx = parser("-bpdx").asInt();
  sim.bpdy = parser("-bpdy").asInt();
  sim.levelMax = parser("-levelMax").asInt();
  sim.Rtol = parser("-Rtol").asDouble();
  sim.Ctol = parser("-Ctol").asDouble();
  parser.unset_strict_mode();
  sim.AdaptSteps = parser("-AdaptSteps").asInt(20);
  sim.bAdaptChiGradient = parser("-bAdaptChiGradient").asInt(1);
  sim.levelStart = parser("-levelStart").asInt(-1);
  if (sim.levelStart == -1)
    sim.levelStart = sim.levelMax - 1;
  sim.extent = parser("-extent").asDouble(1);
  sim.dt = parser("-dt").asDouble(0);
  sim.CFL = parser("-CFL").asDouble(0.2);
  sim.nsteps = parser("-nsteps").asInt(0);
  sim.endTime = parser("-tend").asDouble(0);
  sim.lambda = parser("-lambda").asDouble(1e7);
  sim.dlm = parser("-dlm").asDouble(0);
  sim.nu = parser("-nu").asDouble(1e-2);
  std::string BC_x = parser("-BC_x").asString("freespace");
  std::string BC_y = parser("-BC_y").asString("freespace");
  cubismBCX = string2BCflag(BC_x);
  cubismBCY = string2BCflag(BC_y);
  sim.PoissonTol = parser("-poissonTol").asDouble(1e-6);
  sim.PoissonTolRel = parser("-poissonTolRel").asDouble(0);
  sim.maxPoissonRestarts = parser("-maxPoissonRestarts").asInt(30);
  sim.maxPoissonIterations = parser("-maxPoissonIterations").asInt(1000);
  sim.bMeanConstraint = parser("-bMeanConstraint").asInt(0);
  sim.dumpFreq = parser("-fdump").asInt(0);
  sim.dumpTime = parser("-tdump").asDouble(0);
  ScalarLab dummy;
  bool xperiodic = dummy.is_xperiodic();
  bool yperiodic = dummy.is_yperiodic();
  bool zperiodic = dummy.is_zperiodic();
  sim.chi = new ScalarGrid(sim.bpdx, sim.bpdy, 1, sim.extent, sim.levelStart,
                           sim.levelMax, xperiodic, yperiodic, zperiodic);
  sim.vel = new VectorGrid(sim.bpdx, sim.bpdy, 1, sim.extent, sim.levelStart,
                           sim.levelMax, xperiodic, yperiodic, zperiodic);
  sim.vOld = new VectorGrid(sim.bpdx, sim.bpdy, 1, sim.extent, sim.levelStart,
                            sim.levelMax, xperiodic, yperiodic, zperiodic);
  sim.pres = new ScalarGrid(sim.bpdx, sim.bpdy, 1, sim.extent, sim.levelStart,
                            sim.levelMax, xperiodic, yperiodic, zperiodic);
  sim.tmpV = new VectorGrid(sim.bpdx, sim.bpdy, 1, sim.extent, sim.levelStart,
                            sim.levelMax, xperiodic, yperiodic, zperiodic);
  sim.tmp = new ScalarGrid(sim.bpdx, sim.bpdy, 1, sim.extent, sim.levelStart,
                           sim.levelMax, xperiodic, yperiodic, zperiodic);
  sim.pold = new ScalarGrid(sim.bpdx, sim.bpdy, 1, sim.extent, sim.levelStart,
                            sim.levelMax, xperiodic, yperiodic, zperiodic);
  std::vector<BlockInfo> &velInfo = sim.vel->m_vInfo;
  if (velInfo.size() == 0) {
    std::cout << "You are using too many MPI ranks for the given initial "
                 "number of blocks.";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  int aux = pow(2, sim.levelStart);
  sim.extents[0] = aux * sim.bpdx * velInfo[0].h * _BS_;
  sim.extents[1] = aux * sim.bpdy * velInfo[0].h * _BS_;
  int auxMax = pow(2, sim.levelMax - 1);
  sim.minH = sim.extents[0] / (auxMax * sim.bpdx * _BS_);
  std::string shapeArg = parser("-shapes").asString("");
  std::stringstream descriptors(shapeArg);
  std::string lines;
  while (std::getline(descriptors, lines)) {
    std::replace(lines.begin(), lines.end(), '_', ' ');
    std::stringstream ss(lines);
    std::string line;
    while (std::getline(ss, line, ',')) {
      std::istringstream line_stream(line);
      std::string objectName;
      line_stream >> objectName;
      if (objectName.empty() or objectName[0] == '#')
        continue;
      FactoryFileLineParser p(line_stream);
      Real center[2] = {p("-xpos").asDouble(.5 * sim.extents[0]),
                        p("-ypos").asDouble(.5 * sim.extents[1])};
      Shape *shape = new Shape(p, center);
      shape->amplitudeFactor = p("-amplitudeFactor").asDouble(1.0);
      shape->rS[0] = 0;
      int k = 0;
      for (int i = 0; i < shape->Nend; ++i, k++)
        shape->rS[k + 1] =
            shape->rS[k] + shape->dSref +
            (shape->dSmid - shape->dSref) * i / ((Real)shape->Nend - 1.);
      for (int i = 0; i < shape->Nmid; ++i, k++)
        shape->rS[k + 1] = shape->rS[k] + shape->dSmid;
      for (int i = 0; i < shape->Nend; ++i, k++)
        shape->rS[k + 1] = shape->rS[k] + shape->dSref +
                           (shape->dSmid - shape->dSref) *
                               (shape->Nend - i - 1) / ((Real)shape->Nend - 1.);
      shape->rS[k] = std::min(shape->rS[k], (Real)shape->length);
      std::fill(shape->rX, shape->rX + shape->Nm, 0);
      std::fill(shape->rY, shape->rY + shape->Nm, 0);
      std::fill(shape->vX, shape->vX + shape->Nm, 0);
      std::fill(shape->vY, shape->vY + shape->Nm, 0);
      for (int i = 0; i < shape->Nm; ++i) {
        const Real sb = .04 * shape->length, st = .95 * shape->length,
                   wt = .01 * shape->length, wh = .04 * shape->length;
        if (shape->rS[i] < 0 or shape->rS[i] > shape->length)
          shape->width[i] = 0;
        else
          shape->width[i] =
              (shape->rS[i] < sb
                   ? std::sqrt(2 * wh * shape->rS[i] -
                               shape->rS[i] * shape->rS[i])
                   : (shape->rS[i] < st
                          ? wh -
                                (wh - wt) *
                                    std::pow((shape->rS[i] - sb) / (st - sb), 1)
                          : (wt * (shape->length - shape->rS[i]) /
                             (shape->length - st))));
      }
      sim.shapes.push_back(shape);
    }
  }
  PoissonSolver pressureSolver;
#pragma omp parallel for
  for (size_t i = 0; i < velInfo.size(); i++)
    for (int x = 0; x < _BS_; x++)
      for (int y = 0; y < _BS_; y++) {
        ((VectorBlock *)sim.vel->m_vInfo[i].ptrBlock)->data[x][y].clear();
        ((ScalarBlock *)sim.chi->m_vInfo[i].ptrBlock)->data[x][y].clear();
        ((ScalarBlock *)sim.pres->m_vInfo[i].ptrBlock)->data[x][y].clear();
        ((ScalarBlock *)sim.pold->m_vInfo[i].ptrBlock)->data[x][y].clear();
        ((ScalarBlock *)sim.tmp->m_vInfo[i].ptrBlock)->data[x][y].clear();
        ((VectorBlock *)sim.tmpV->m_vInfo[i].ptrBlock)->data[x][y].clear();
        ((VectorBlock *)sim.vOld->m_vInfo[i].ptrBlock)->data[x][y].clear();
      }
  sim.tmp_amr = new ScalarAMR(*sim.tmp, sim.Rtol, sim.Ctol);
  sim.chi_amr = new ScalarAMR(*sim.chi, sim.Rtol, sim.Ctol);
  sim.pres_amr = new ScalarAMR(*sim.pres, sim.Rtol, sim.Ctol);
  sim.pold_amr = new ScalarAMR(*sim.pold, sim.Rtol, sim.Ctol);
  sim.vel_amr = new VectorAMR(*sim.vel, sim.Rtol, sim.Ctol);
  sim.vOld_amr = new VectorAMR(*sim.vOld, sim.Rtol, sim.Ctol);
  sim.tmpV_amr = new VectorAMR(*sim.tmpV, sim.Rtol, sim.Ctol);
  for (int i = 0; i < sim.levelMax; i++) {
    ongrid(0.0);
    adapt();
  }
  ongrid(0.0);
#pragma omp parallel for
  for (size_t i = 0; i < velInfo.size(); i++) {
    for (size_t y = 0; y < _BS_; y++)
      for (size_t x = 0; x < _BS_; x++)
        ((VectorBlock *)sim.tmpV->m_vInfo[i].ptrBlock)->data[y][x].clear();
  }
  for (auto &shape : sim.shapes) {
    std::vector<ObstacleBlock *> &OBLOCK = shape->obstacleBlocks;
#pragma omp parallel for
    for (size_t i = 0; i < velInfo.size(); i++) {
      if (OBLOCK[sim.tmpV->m_vInfo[i].blockID] == nullptr)
        continue;
      UDEFMAT &__restrict__ udef = OBLOCK[sim.tmpV->m_vInfo[i].blockID]->udef;
      CHI_MAT &__restrict__ chi = OBLOCK[sim.tmpV->m_vInfo[i].blockID]->chi;
      auto &__restrict__ UDEF = *(VectorBlock *)sim.tmpV->m_vInfo[i].ptrBlock;
      ScalarBlock &__restrict__ CHI =
          *(ScalarBlock *)sim.chi->m_vInfo[i].ptrBlock;
      for (int iy = 0; iy < _BS_; iy++)
        for (int ix = 0; ix < _BS_; ix++) {
          if (chi[iy][ix] < CHI(ix, iy).s)
            continue;
          Real p[2];
          sim.tmpV->m_vInfo[i].pos(p, ix, iy);
          UDEF(ix, iy).u[0] += udef[iy][ix][0];
          UDEF(ix, iy).u[1] += udef[iy][ix][1];
        }
    }
  }
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < velInfo.size(); i++) {
    VectorBlock &UF = *(VectorBlock *)velInfo[i].ptrBlock;
    VectorBlock &US = *(VectorBlock *)sim.tmpV->m_vInfo[i].ptrBlock;
    ScalarBlock &X = *(ScalarBlock *)sim.chi->m_vInfo[i].ptrBlock;
    for (int iy = 0; iy < _BS_; ++iy)
      for (int ix = 0; ix < _BS_; ++ix) {
        UF(ix, iy).u[0] =
            UF(ix, iy).u[0] * (1 - X(ix, iy).s) + US(ix, iy).u[0] * X(ix, iy).s;
        UF(ix, iy).u[1] =
            UF(ix, iy).u[1] * (1 - X(ix, iy).s) + US(ix, iy).u[1] * X(ix, iy).s;
      }
  }
  while (1) {
    Real CFL = sim.CFL;
    Real h = std::numeric_limits<Real>::infinity();
    for (size_t i = 0; i < sim.vel->m_vInfo.size(); i++)
      h = std::min(sim.vel->m_vInfo[i].h, h);
    MPI_Allreduce(MPI_IN_PLACE, &h, 1, MPI_Real, MPI_MIN, MPI_COMM_WORLD);
    size_t Nblocks = velInfo.size();
    Real umax = 0;
#pragma omp parallel for schedule(static) reduction(max : umax)
    for (size_t i = 0; i < Nblocks; i++) {
      VectorBlock &VEL = *(VectorBlock *)velInfo[i].ptrBlock;
      for (int iy = 0; iy < _BS_; ++iy)
        for (int ix = 0; ix < _BS_; ++ix) {
          umax = std::max(umax, std::fabs(VEL(ix, iy).u[0] + sim.uinfx));
          umax = std::max(umax, std::fabs(VEL(ix, iy).u[1] + sim.uinfy));
          umax = std::max(umax, std::fabs(VEL(ix, iy).u[0]));
          umax = std::max(umax, std::fabs(VEL(ix, iy).u[1]));
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &umax, 1, MPI_Real, MPI_MAX, MPI_COMM_WORLD);
    if (CFL > 0) {
      Real dtDiffusion = 0.25 * h * h / (sim.nu + 0.25 * h * umax);
      Real dtAdvection = h / (umax + 1e-8);
      sim.dt = std::min({dtDiffusion, CFL * dtAdvection});
    }
    if (sim.dt <= 0) {
      std::cout << "[CUP2D] dt <= 0. Aborting..." << std::endl;
      fflush(0);
      abort();
    }
    if (sim.dlm > 0)
      sim.lambda = sim.dlm / sim.dt;
    bool done = false;
    if (!done || sim.dt > 2e-16) {
      bool timeDump = sim.dumpTime > 0 && sim.time >= sim.nextDumpTime;
      bool stepDump = sim.dumpFreq > 0 && (sim.step % sim.dumpFreq) == 0;
      bool bDump = stepDump || timeDump;
      if (bDump) {
        sim.nextDumpTime += sim.dumpTime;
        compute<VectorLab>(KernelVorticity(), sim.vel);
        char path[FILENAME_MAX];
        snprintf(path, sizeof path, "vort.%08d", sim.step);
        dump(sim.time, sim.tmp, path);
      }
      if (sim.step <= 10 || sim.step % sim.AdaptSteps == 0)
        adapt();
      ongrid(sim.dt);
      size_t Nblocks = velInfo.size();
      KernelAdvectDiffuse<VectorElement> Step1;
#pragma omp parallel for
      for (size_t i = 0; i < velInfo.size(); i++) {
        VectorBlock &__restrict__ Vold =
            *(VectorBlock *)sim.vOld->m_vInfo[i].ptrBlock;
        const VectorBlock &__restrict__ V = *(VectorBlock *)velInfo[i].ptrBlock;
        for (int iy = 0; iy < _BS_; ++iy)
          for (int ix = 0; ix < _BS_; ++ix) {
            Vold(ix, iy).u[0] = V(ix, iy).u[0];
            Vold(ix, iy).u[1] = V(ix, iy).u[1];
          }
      }
      compute<VectorLab>(Step1, sim.vel, sim.tmpV);
#pragma omp parallel for
      for (size_t i = 0; i < velInfo.size(); i++) {
        VectorBlock &__restrict__ V = *(VectorBlock *)velInfo[i].ptrBlock;
        const VectorBlock &__restrict__ Vold =
            *(VectorBlock *)sim.vOld->m_vInfo[i].ptrBlock;
        const VectorBlock &__restrict__ tmpV =
            *(VectorBlock *)sim.tmpV->m_vInfo[i].ptrBlock;
        const Real ih2 = 1.0 / (velInfo[i].h * velInfo[i].h);
        for (int iy = 0; iy < _BS_; ++iy)
          for (int ix = 0; ix < _BS_; ++ix) {
            V(ix, iy).u[0] =
                Vold(ix, iy).u[0] + (0.5 * tmpV(ix, iy).u[0]) * ih2;
            V(ix, iy).u[1] =
                Vold(ix, iy).u[1] + (0.5 * tmpV(ix, iy).u[1]) * ih2;
          }
      }
      compute<VectorLab>(Step1, sim.vel, sim.tmpV);
#pragma omp parallel for
      for (size_t i = 0; i < velInfo.size(); i++) {
        VectorBlock &__restrict__ V = *(VectorBlock *)velInfo[i].ptrBlock;
        const VectorBlock &__restrict__ Vold =
            *(VectorBlock *)sim.vOld->m_vInfo[i].ptrBlock;
        const VectorBlock &__restrict__ tmpV =
            *(VectorBlock *)sim.tmpV->m_vInfo[i].ptrBlock;
        const Real ih2 = 1.0 / (velInfo[i].h * velInfo[i].h);
        for (int iy = 0; iy < _BS_; ++iy)
          for (int ix = 0; ix < _BS_; ++ix) {
            V(ix, iy).u[0] = Vold(ix, iy).u[0] + tmpV(ix, iy).u[0] * ih2;
            V(ix, iy).u[1] = Vold(ix, iy).u[1] + tmpV(ix, iy).u[1] * ih2;
          }
      }
      for (const auto &shape : sim.shapes) {
        const std::vector<ObstacleBlock *> &OBLOCK = shape->obstacleBlocks;
        const Real Cx = shape->centerOfMass[0];
        const Real Cy = shape->centerOfMass[1];
        Real PM = 0, PJ = 0, PX = 0, PY = 0, UM = 0, VM = 0, AM = 0;
#pragma omp parallel for reduction(+ : PM, PJ, PX, PY, UM, VM, AM)
        for (size_t i = 0; i < velInfo.size(); i++) {
          const VectorBlock &__restrict__ VEL =
              *(VectorBlock *)velInfo[i].ptrBlock;
          const Real hsq = velInfo[i].h * velInfo[i].h;
          if (OBLOCK[velInfo[i].blockID] == nullptr)
            continue;
          const CHI_MAT &__restrict__ chi = OBLOCK[velInfo[i].blockID]->chi;
          const UDEFMAT &__restrict__ udef = OBLOCK[velInfo[i].blockID]->udef;
          const Real lambdt = sim.lambda * sim.dt;
          for (int iy = 0; iy < _BS_; ++iy)
            for (int ix = 0; ix < _BS_; ++ix) {
              if (chi[iy][ix] <= 0)
                continue;
              const Real udiff[2] = {VEL(ix, iy).u[0] - udef[iy][ix][0],
                                     VEL(ix, iy).u[1] - udef[iy][ix][1]};
              const Real Xlamdt = chi[iy][ix] >= 0.5 ? lambdt : 0.0;
              const Real F = hsq * Xlamdt / (1 + Xlamdt);
              Real p[2];
              velInfo[i].pos(p, ix, iy);
              p[0] -= Cx;
              p[1] -= Cy;
              PM += F;
              PJ += F * (p[0] * p[0] + p[1] * p[1]);
              PX += F * p[0];
              PY += F * p[1];
              UM += F * udiff[0];
              VM += F * udiff[1];
              AM += F * (p[0] * udiff[1] - p[1] * udiff[0]);
            }
        }
        Real quantities[7] = {PM, PJ, PX, PY, UM, VM, AM};
        MPI_Allreduce(MPI_IN_PLACE, quantities, 7, MPI_Real, MPI_SUM,
                      MPI_COMM_WORLD);
        PM = quantities[0];
        PJ = quantities[1];
        PX = quantities[2];
        PY = quantities[3];
        UM = quantities[4];
        VM = quantities[5];
        AM = quantities[6];
        shape->fluidAngMom = AM;
        shape->fluidMomX = UM;
        shape->fluidMomY = VM;
        shape->penalDX = PX;
        shape->penalDY = PY;
        shape->penalM = PM;
        shape->penalJ = PJ;
        double A[3][3] = {
            {(double)shape->penalM, (double)0, (double)-shape->penalDY},
            {(double)0, (double)shape->penalM, (double)shape->penalDX},
            {(double)-shape->penalDY, (double)shape->penalDX,
             (double)shape->penalJ}};
        double b[3] = {
            (double)(shape->fluidMomX + sim.dt * shape->appliedForceX),
            (double)(shape->fluidMomY + sim.dt * shape->appliedForceY),
            (double)(shape->fluidAngMom + sim.dt * shape->appliedTorque)};
        if (shape->bForcedx) {
          A[0][1] = 0;
          A[0][2] = 0;
          b[0] = shape->penalM * shape->forcedu;
        }
        if (shape->bForcedy) {
          A[1][0] = 0;
          A[1][2] = 0;
          b[1] = shape->penalM * shape->forcedv;
        }
        if (shape->bBlockang) {
          A[2][0] = 0;
          A[2][1] = 0;
          b[2] = shape->penalJ * shape->forcedomega;
        }
        gsl_matrix_view Agsl = gsl_matrix_view_array(&A[0][0], 3, 3);
        gsl_vector_view bgsl = gsl_vector_view_array(b, 3);
        gsl_vector *xgsl = gsl_vector_alloc(3);
        int sgsl;
        gsl_permutation *permgsl = gsl_permutation_alloc(3);
        gsl_linalg_LU_decomp(&Agsl.matrix, permgsl, &sgsl);
        gsl_linalg_LU_solve(&Agsl.matrix, permgsl, &bgsl.vector, xgsl);
        if (not shape->bForcedx)
          shape->u = gsl_vector_get(xgsl, 0);
        if (not shape->bForcedy)
          shape->v = gsl_vector_get(xgsl, 1);
        if (not shape->bBlockang)
          shape->omega = gsl_vector_get(xgsl, 2);
        gsl_permutation_free(permgsl);
        gsl_vector_free(xgsl);
      }
      const auto &shapes = sim.shapes;
      const auto &infos = sim.chi->m_vInfo;
      const size_t N = shapes.size();
      sim.bCollisionID.clear();
      struct CollisionInfo {
        Real iM = 0;
        Real iPosX = 0;
        Real iPosY = 0;
        Real iPosZ = 0;
        Real iMomX = 0;
        Real iMomY = 0;
        Real iMomZ = 0;
        Real ivecX = 0;
        Real ivecY = 0;
        Real ivecZ = 0;
        Real jM = 0;
        Real jPosX = 0;
        Real jPosY = 0;
        Real jPosZ = 0;
        Real jMomX = 0;
        Real jMomY = 0;
        Real jMomZ = 0;
        Real jvecX = 0;
        Real jvecY = 0;
        Real jvecZ = 0;
      };
      std::vector<CollisionInfo> collisions(N);
      std::vector<Real> n_vec(3 * N, 0.0);
#pragma omp parallel for schedule(static)
      for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j) {
          if (i == j)
            continue;
          auto &coll = collisions[i];
          const auto &iBlocks = shapes[i]->obstacleBlocks;
          const Real iU0 = shapes[i]->u;
          const Real iU1 = shapes[i]->v;
          const Real iomega2 = shapes[i]->omega;
          const Real iCx = shapes[i]->centerOfMass[0];
          const Real iCy = shapes[i]->centerOfMass[1];
          const auto &jBlocks = shapes[j]->obstacleBlocks;
          const Real jU0 = shapes[j]->u;
          const Real jU1 = shapes[j]->v;
          const Real jomega2 = shapes[j]->omega;
          const Real jCx = shapes[j]->centerOfMass[0];
          const Real jCy = shapes[j]->centerOfMass[1];
          assert(iBlocks.size() == jBlocks.size());
          const size_t nBlocks = iBlocks.size();
          for (size_t k = 0; k < nBlocks; ++k) {
            if (iBlocks[k] == nullptr || jBlocks[k] == nullptr)
              continue;
            const auto &iSDF = iBlocks[k]->dist;
            const auto &jSDF = jBlocks[k]->dist;
            const CHI_MAT &iChi = iBlocks[k]->chi;
            const CHI_MAT &jChi = jBlocks[k]->chi;
            const UDEFMAT &iUDEF = iBlocks[k]->udef;
            const UDEFMAT &jUDEF = jBlocks[k]->udef;
            for (int iy = 0; iy < _BS_; ++iy)
              for (int ix = 0; ix < _BS_; ++ix) {
                if (iChi[iy][ix] <= 0.0 || jChi[iy][ix] <= 0.0)
                  continue;
                const auto pos = infos[k].pos(ix, iy);
                const Real iUr0 = -iomega2 * (pos[1] - iCy);
                const Real iUr1 = iomega2 * (pos[0] - iCx);
                coll.iM += iChi[iy][ix];
                coll.iPosX += iChi[iy][ix] * pos[0];
                coll.iPosY += iChi[iy][ix] * pos[1];
                coll.iMomX += iChi[iy][ix] * (iU0 + iUr0 + iUDEF[iy][ix][0]);
                coll.iMomY += iChi[iy][ix] * (iU1 + iUr1 + iUDEF[iy][ix][1]);
                const Real jUr0 = -jomega2 * (pos[1] - jCy);
                const Real jUr1 = jomega2 * (pos[0] - jCx);
                coll.jM += jChi[iy][ix];
                coll.jPosX += jChi[iy][ix] * pos[0];
                coll.jPosY += jChi[iy][ix] * pos[1];
                coll.jMomX += jChi[iy][ix] * (jU0 + jUr0 + jUDEF[iy][ix][0]);
                coll.jMomY += jChi[iy][ix] * (jU1 + jUr1 + jUDEF[iy][ix][1]);
                Real dSDFdx_i;
                Real dSDFdx_j;
                if (ix == 0) {
                  dSDFdx_i = iSDF[iy][ix + 1] - iSDF[iy][ix];
                  dSDFdx_j = jSDF[iy][ix + 1] - jSDF[iy][ix];
                } else if (ix == _BS_ - 1) {
                  dSDFdx_i = iSDF[iy][ix] - iSDF[iy][ix - 1];
                  dSDFdx_j = jSDF[iy][ix] - jSDF[iy][ix - 1];
                } else {
                  dSDFdx_i = 0.5 * (iSDF[iy][ix + 1] - iSDF[iy][ix - 1]);
                  dSDFdx_j = 0.5 * (jSDF[iy][ix + 1] - jSDF[iy][ix - 1]);
                }
                Real dSDFdy_i;
                Real dSDFdy_j;
                if (iy == 0) {
                  dSDFdy_i = iSDF[iy + 1][ix] - iSDF[iy][ix];
                  dSDFdy_j = jSDF[iy + 1][ix] - jSDF[iy][ix];
                } else if (iy == _BS_ - 1) {
                  dSDFdy_i = iSDF[iy][ix] - iSDF[iy - 1][ix];
                  dSDFdy_j = jSDF[iy][ix] - jSDF[iy - 1][ix];
                } else {
                  dSDFdy_i = 0.5 * (iSDF[iy + 1][ix] - iSDF[iy - 1][ix]);
                  dSDFdy_j = 0.5 * (jSDF[iy + 1][ix] - jSDF[iy - 1][ix]);
                }
                coll.ivecX += iChi[iy][ix] * dSDFdx_i;
                coll.ivecY += iChi[iy][ix] * dSDFdy_i;
                coll.jvecX += jChi[iy][ix] * dSDFdx_j;
                coll.jvecY += jChi[iy][ix] * dSDFdy_j;
              }
          }
        }
      std::vector<Real> buffer(20 * N);
      for (size_t i = 0; i < N; i++) {
        auto &coll = collisions[i];
        buffer[20 * i] = coll.iM;
        buffer[20 * i + 1] = coll.iPosX;
        buffer[20 * i + 2] = coll.iPosY;
        buffer[20 * i + 3] = coll.iPosZ;
        buffer[20 * i + 4] = coll.iMomX;
        buffer[20 * i + 5] = coll.iMomY;
        buffer[20 * i + 6] = coll.iMomZ;
        buffer[20 * i + 7] = coll.ivecX;
        buffer[20 * i + 8] = coll.ivecY;
        buffer[20 * i + 9] = coll.ivecZ;
        buffer[20 * i + 10] = coll.jM;
        buffer[20 * i + 11] = coll.jPosX;
        buffer[20 * i + 12] = coll.jPosY;
        buffer[20 * i + 13] = coll.jPosZ;
        buffer[20 * i + 14] = coll.jMomX;
        buffer[20 * i + 15] = coll.jMomY;
        buffer[20 * i + 16] = coll.jMomZ;
        buffer[20 * i + 17] = coll.jvecX;
        buffer[20 * i + 18] = coll.jvecY;
        buffer[20 * i + 19] = coll.jvecZ;
      }
      MPI_Allreduce(MPI_IN_PLACE, buffer.data(), buffer.size(), MPI_Real,
                    MPI_SUM, MPI_COMM_WORLD);
      for (size_t i = 0; i < N; i++) {
        auto &coll = collisions[i];
        coll.iM = buffer[20 * i];
        coll.iPosX = buffer[20 * i + 1];
        coll.iPosY = buffer[20 * i + 2];
        coll.iPosZ = buffer[20 * i + 3];
        coll.iMomX = buffer[20 * i + 4];
        coll.iMomY = buffer[20 * i + 5];
        coll.iMomZ = buffer[20 * i + 6];
        coll.ivecX = buffer[20 * i + 7];
        coll.ivecY = buffer[20 * i + 8];
        coll.ivecZ = buffer[20 * i + 9];
        coll.jM = buffer[20 * i + 10];
        coll.jPosX = buffer[20 * i + 11];
        coll.jPosY = buffer[20 * i + 12];
        coll.jPosZ = buffer[20 * i + 13];
        coll.jMomX = buffer[20 * i + 14];
        coll.jMomY = buffer[20 * i + 15];
        coll.jMomZ = buffer[20 * i + 16];
        coll.jvecX = buffer[20 * i + 17];
        coll.jvecY = buffer[20 * i + 18];
        coll.jvecZ = buffer[20 * i + 19];
      }
#pragma omp parallel for schedule(static)
      for (size_t i = 0; i < N; ++i)
        for (size_t j = i + 1; j < N; ++j) {
          if (i == j)
            continue;
          Real m1 = shapes[i]->M;
          Real m2 = shapes[j]->M;
          Real v1[3] = {shapes[i]->u, shapes[i]->v, 0.0};
          Real v2[3] = {shapes[j]->u, shapes[j]->v, 0.0};
          Real o1[3] = {0, 0, shapes[i]->omega};
          Real o2[3] = {0, 0, shapes[j]->omega};
          Real C1[3] = {shapes[i]->centerOfMass[0], shapes[i]->centerOfMass[1],
                        0};
          Real C2[3] = {shapes[j]->centerOfMass[0], shapes[j]->centerOfMass[1],
                        0};
          Real I1[6] = {1.0, 0, 0, 0, 0, shapes[i]->J};
          Real I2[6] = {1.0, 0, 0, 0, 0, shapes[j]->J};
          auto &coll = collisions[i];
          auto &coll_other = collisions[j];
          if (coll.iM < 2.0 || coll.jM < 2.0)
            continue;
          if (coll_other.iM < 2.0 || coll_other.jM < 2.0)
            continue;
          if (std::fabs(coll.iPosX / coll.iM -
                        coll_other.iPosX / coll_other.iM) > shapes[i]->length ||
              std::fabs(coll.iPosY / coll.iM -
                        coll_other.iPosY / coll_other.iM) > shapes[i]->length) {
            continue;
          }
#pragma omp critical
          {
            sim.bCollisionID.push_back(i);
            sim.bCollisionID.push_back(j);
          }
          const bool iForced = shapes[i]->bForced;
          const bool jForced = shapes[j]->bForced;
          if (iForced || jForced) {
            std::cout << "[CUP2D] WARNING: Forced objects not supported for "
                         "collision."
                      << std::endl;
          }
          Real ho1[3];
          Real ho2[3];
          Real hv1[3];
          Real hv2[3];
          Real norm_i =
              std::sqrt(coll.ivecX * coll.ivecX + coll.ivecY * coll.ivecY +
                        coll.ivecZ * coll.ivecZ);
          Real norm_j =
              std::sqrt(coll.jvecX * coll.jvecX + coll.jvecY * coll.jvecY +
                        coll.jvecZ * coll.jvecZ);
          Real mX = coll.ivecX / norm_i - coll.jvecX / norm_j;
          Real mY = coll.ivecY / norm_i - coll.jvecY / norm_j;
          Real mZ = coll.ivecZ / norm_i - coll.jvecZ / norm_j;
          Real inorm = 1.0 / std::sqrt(mX * mX + mY * mY + mZ * mZ);
          Real NX = mX * inorm;
          Real NY = mY * inorm;
          Real NZ = mZ * inorm;
          Real hitVelX = coll.jMomX / coll.jM - coll.iMomX / coll.iM;
          Real hitVelY = coll.jMomY / coll.jM - coll.iMomY / coll.iM;
          Real hitVelZ = coll.jMomZ / coll.jM - coll.iMomZ / coll.iM;
          Real projVel = hitVelX * NX + hitVelY * NY + hitVelZ * NZ;
          Real vc1[3] = {coll.iMomX / coll.iM, coll.iMomY / coll.iM,
                         coll.iMomZ / coll.iM};
          Real vc2[3] = {coll.jMomX / coll.jM, coll.jMomY / coll.jM,
                         coll.jMomZ / coll.jM};
          if (projVel <= 0)
            continue;
          Real inv_iM = 1.0 / coll.iM;
          Real inv_jM = 1.0 / coll.jM;
          Real iPX = coll.iPosX * inv_iM;
          Real iPY = coll.iPosY * inv_iM;
          Real iPZ = coll.iPosZ * inv_iM;
          Real jPX = coll.jPosX * inv_jM;
          Real jPY = coll.jPosY * inv_jM;
          Real jPZ = coll.jPosZ * inv_jM;
          Real CX = 0.5 * (iPX + jPX);
          Real CY = 0.5 * (iPY + jPY);
          Real CZ = 0.5 * (iPZ + jPZ);
          ElasticCollision(m1, m2, I1, I2, v1, v2, o1, o2, hv1, hv2, ho1, ho2,
                           C1, C2, NX, NY, NZ, CX, CY, CZ, vc1, vc2);
          shapes[i]->u = hv1[0];
          shapes[i]->v = hv1[1];
          shapes[j]->u = hv2[0];
          shapes[j]->v = hv2[1];
          shapes[i]->omega = ho1[2];
          shapes[j]->omega = ho2[2];
        }
      std::vector<BlockInfo> &chiInfo = sim.chi->m_vInfo;
#pragma omp parallel for
      for (size_t i = 0; i < Nblocks; i++)
        for (auto &shape : sim.shapes) {
          std::vector<ObstacleBlock *> &OBLOCK = shape->obstacleBlocks;
          ObstacleBlock *o = OBLOCK[velInfo[i].blockID];
          if (o == nullptr)
            continue;
          Real u_s = shape->u;
          Real v_s = shape->v;
          Real omega_s = shape->omega;
          Real Cx = shape->centerOfMass[0];
          Real Cy = shape->centerOfMass[1];
          CHI_MAT &__restrict__ X = o->chi;
          UDEFMAT &__restrict__ UDEF = o->udef;
          ScalarBlock &__restrict__ CHI = *(ScalarBlock *)chiInfo[i].ptrBlock;
          VectorBlock &__restrict__ V = *(VectorBlock *)velInfo[i].ptrBlock;
          for (int iy = 0; iy < _BS_; ++iy)
            for (int ix = 0; ix < _BS_; ++ix) {
              if (CHI(ix, iy).s > X[iy][ix])
                continue;
              if (X[iy][ix] <= 0)
                continue;
              Real p[2];
              velInfo[i].pos(p, ix, iy);
              p[0] -= Cx;
              p[1] -= Cy;
              Real alpha = X[iy][ix] > 0.5 ? 1 / (1 + sim.lambda * sim.dt) : 1;
              Real US = u_s - omega_s * p[1] + UDEF[iy][ix][0];
              Real VS = v_s + omega_s * p[0] + UDEF[iy][ix][1];
              V(ix, iy).u[0] = alpha * V(ix, iy).u[0] + (1 - alpha) * US;
              V(ix, iy).u[1] = alpha * V(ix, iy).u[1] + (1 - alpha) * VS;
            }
        }
      std::vector<BlockInfo> &tmpVInfo = sim.tmpV->m_vInfo;
#pragma omp parallel for
      for (size_t i = 0; i < Nblocks; i++) {
        for (size_t y = 0; y < _BS_; y++)
          for (size_t x = 0; x < _BS_; x++)
            ((VectorBlock *)tmpVInfo[i].ptrBlock)->data[y][x].clear();
      }
      for (auto &shape : sim.shapes) {
        std::vector<ObstacleBlock *> &OBLOCK = shape->obstacleBlocks;
#pragma omp parallel for
        for (size_t i = 0; i < Nblocks; i++) {
          if (OBLOCK[tmpVInfo[i].blockID] == nullptr)
            continue;
          UDEFMAT &__restrict__ udef = OBLOCK[tmpVInfo[i].blockID]->udef;
          CHI_MAT &__restrict__ chi = OBLOCK[tmpVInfo[i].blockID]->chi;
          auto &__restrict__ UDEF = *(VectorBlock *)tmpVInfo[i].ptrBlock;
          ScalarBlock &__restrict__ CHI = *(ScalarBlock *)chiInfo[i].ptrBlock;
          for (int iy = 0; iy < _BS_; iy++)
            for (int ix = 0; ix < _BS_; ix++) {
              if (chi[iy][ix] < CHI(ix, iy).s)
                continue;
              Real p[2];
              tmpVInfo[i].pos(p, ix, iy);
              UDEF(ix, iy).u[0] += udef[iy][ix][0];
              UDEF(ix, iy).u[1] += udef[iy][ix][1];
            }
        }
      }
      compute<updatePressureRHS, VectorGrid, VectorLab, VectorGrid, VectorLab,
              ScalarGrid>(updatePressureRHS(), *sim.vel, *sim.tmpV, true,
                          sim.tmp);
      std::vector<BlockInfo> &presInfo = sim.pres->m_vInfo;
      std::vector<BlockInfo> &poldInfo = sim.pold->m_vInfo;
#pragma omp parallel for
      for (size_t i = 0; i < Nblocks; i++) {
        ScalarBlock &__restrict__ PRES = *(ScalarBlock *)presInfo[i].ptrBlock;
        ScalarBlock &__restrict__ POLD = *(ScalarBlock *)poldInfo[i].ptrBlock;
        for (int iy = 0; iy < _BS_; ++iy)
          for (int ix = 0; ix < _BS_; ++ix) {
            POLD(ix, iy).s = PRES(ix, iy).s;
            PRES(ix, iy).s = 0;
          }
      }
      compute<ScalarLab>(updatePressureRHS1(), sim.pold, sim.tmp);
      pressureSolver.solve(sim.tmp);
      Real avg = 0;
      Real avg1 = 0;
#pragma omp parallel for reduction(+ : avg, avg1)
      for (size_t i = 0; i < Nblocks; i++) {
        ScalarBlock &P = *(ScalarBlock *)presInfo[i].ptrBlock;
        const Real vv = presInfo[i].h * presInfo[i].h;
        for (int iy = 0; iy < _BS_; iy++)
          for (int ix = 0; ix < _BS_; ix++) {
            avg += P(ix, iy).s * vv;
            avg1 += vv;
          }
      }
      Real quantities[2] = {avg, avg1};
      MPI_Allreduce(MPI_IN_PLACE, &quantities, 2, MPI_Real, MPI_SUM,
                    MPI_COMM_WORLD);
      avg = quantities[0];
      avg1 = quantities[1];
      avg = avg / avg1;
#pragma omp parallel for
      for (size_t i = 0; i < Nblocks; i++) {
        ScalarBlock &P = *(ScalarBlock *)presInfo[i].ptrBlock;
        const ScalarBlock &__restrict__ POLD =
            *(ScalarBlock *)poldInfo[i].ptrBlock;
        for (int iy = 0; iy < _BS_; iy++)
          for (int ix = 0; ix < _BS_; ix++)
            P(ix, iy).s += POLD(ix, iy).s - avg;
      }
      { compute<ScalarLab>(pressureCorrectionKernel(), sim.pres, sim.tmpV); }
#pragma omp parallel for
      for (size_t i = 0; i < velInfo.size(); i++) {
        const Real ih2 = 1.0 / velInfo[i].h / velInfo[i].h;
        VectorBlock &__restrict__ V = *(VectorBlock *)velInfo[i].ptrBlock;
        VectorBlock &__restrict__ tmpV = *(VectorBlock *)tmpVInfo[i].ptrBlock;
        for (int iy = 0; iy < _BS_; ++iy)
          for (int ix = 0; ix < _BS_; ++ix) {
            V(ix, iy).u[0] += tmpV(ix, iy).u[0] * ih2;
            V(ix, iy).u[1] += tmpV(ix, iy).u[1] * ih2;
          }
      }
      compute<KernelComputeForces, VectorGrid, VectorLab, ScalarGrid,
              ScalarLab>(KernelComputeForces(), *sim.vel, *sim.chi);
      for (const auto &shape : sim.shapes) {
        shape->perimeter = 0;
        shape->forcex = 0;
        shape->forcey = 0;
        shape->forcex_P = 0;
        shape->forcey_P = 0;
        shape->forcex_V = 0;
        shape->forcey_V = 0;
        shape->torque = 0;
        shape->torque_P = 0;
        shape->torque_V = 0;
        shape->drag = 0;
        shape->thrust = 0;
        shape->lift = 0;
        shape->Pout = 0;
        shape->PoutNew = 0;
        shape->PoutBnd = 0;
        shape->defPower = 0;
        shape->defPowerBnd = 0;
        shape->circulation = 0;
        for (auto &block : shape->obstacleBlocks)
          if (block not_eq nullptr) {
            shape->circulation += block->circulation;
            shape->perimeter += block->perimeter;
            shape->torque += block->torque;
            shape->forcex += block->forcex;
            shape->forcey += block->forcey;
            shape->forcex_P += block->forcex_P;
            shape->forcey_P += block->forcey_P;
            shape->forcex_V += block->forcex_V;
            shape->forcey_V += block->forcey_V;
            shape->torque_P += block->torque_P;
            shape->torque_V += block->torque_V;
            shape->drag += block->drag;
            shape->thrust += block->thrust;
            shape->lift += block->lift;
            shape->Pout += block->Pout;
            shape->PoutNew += block->PoutNew;
            shape->defPowerBnd += block->defPowerBnd;
            shape->PoutBnd += block->PoutBnd;
            shape->defPower += block->defPower;
          }
        Real quantities[19];
        quantities[0] = shape->circulation;
        quantities[1] = shape->perimeter;
        quantities[2] = shape->forcex;
        quantities[3] = shape->forcex_P;
        quantities[4] = shape->forcex_V;
        quantities[5] = shape->torque_P;
        quantities[6] = shape->drag;
        quantities[7] = shape->lift;
        quantities[8] = shape->Pout;
        quantities[9] = shape->PoutNew;
        quantities[10] = shape->PoutBnd;
        quantities[11] = shape->torque;
        quantities[12] = shape->forcey;
        quantities[13] = shape->forcey_P;
        quantities[14] = shape->forcey_V;
        quantities[15] = shape->torque_V;
        quantities[16] = shape->thrust;
        quantities[17] = shape->defPowerBnd;
        quantities[18] = shape->defPower;
        MPI_Allreduce(MPI_IN_PLACE, quantities, 19, MPI_Real, MPI_SUM,
                      MPI_COMM_WORLD);
        shape->circulation = quantities[0];
        shape->perimeter = quantities[1];
        shape->forcex = quantities[2];
        shape->forcex_P = quantities[3];
        shape->forcex_V = quantities[4];
        shape->torque_P = quantities[5];
        shape->drag = quantities[6];
        shape->lift = quantities[7];
        shape->Pout = quantities[8];
        shape->PoutNew = quantities[9];
        shape->PoutBnd = quantities[10];
        shape->torque = quantities[11];
        shape->forcey = quantities[12];
        shape->forcey_P = quantities[13];
        shape->forcey_V = quantities[14];
        shape->torque_V = quantities[15];
        shape->thrust = quantities[16];
        shape->defPowerBnd = quantities[17];
        shape->defPower = quantities[18];
        shape->Pthrust = shape->thrust *
                         std::sqrt(shape->u * shape->u + shape->v * shape->v);
        shape->Pdrag =
            shape->drag * std::sqrt(shape->u * shape->u + shape->v * shape->v);
        const Real denUnb = shape->Pthrust - std::min(shape->defPower, (Real)0);
        const Real demBnd = shape->Pthrust - shape->defPowerBnd;
        shape->EffPDef = shape->Pthrust / std::max(denUnb, EPS);
        shape->EffPDefBnd = shape->Pthrust / std::max(demBnd, EPS);
        int tot_blocks = 0;
        int nb = (int)sim.chi->m_vInfo.size();
        MPI_Reduce(&nb, &tot_blocks, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
      }
      sim.time += sim.dt;
      sim.step++;
    }
    if (!done) {
      bool timeEnd = sim.endTime > 0 && sim.time >= sim.endTime;
      bool stepEnd = sim.nsteps > 0 && sim.step >= sim.nsteps;
      done = timeEnd || stepEnd;
    }
    if (done)
      break;
  }
  MPI_Finalize();
}
