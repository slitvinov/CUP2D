static Real dist(Real a[2], Real b[2]) {
  return std::pow(a[0] - b[0], 2) + std::pow(a[1] - b[1], 2);
}
static void rotate2D(const Real Rmatrix2D[2][2], Real *x, Real *y) {
  Real p[2] = {*x, *y};
  *x = Rmatrix2D[0][0] * p[0] + Rmatrix2D[0][1] * p[1];
  *y = Rmatrix2D[1][0] * p[0] + Rmatrix2D[1][1] * p[1];
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
static void pack(Real *srcbase, Real *dst, int dim, int xstart, int ystart,
                 int xend, int yend) {
  if (dim == 1) {
    const int mod = (xend - xstart) % 4;
    int idst = 0;
    for (int iy = ystart; iy < yend; ++iy) {
      for (int ix = xstart; ix < xend - mod; ix += 4, idst += 4) {
        dst[idst + 0] = srcbase[ix + 0 + _BS_ * iy];
        dst[idst + 1] = srcbase[ix + 1 + _BS_ * iy];
        dst[idst + 2] = srcbase[ix + 2 + _BS_ * iy];
        dst[idst + 3] = srcbase[ix + 3 + _BS_ * iy];
      }
      for (int ix = xend - mod; ix < xend; ix++, idst++) {
        dst[idst] = srcbase[ix + _BS_ * iy];
      }
    }
  } else {
    int idst = 0;
    for (int iy = ystart; iy < yend; ++iy)
      for (int ix = xstart; ix < xend; ++ix) {
        const Real *src = srcbase + dim * (ix + _BS_ * iy);
        for (int ic = 0; ic < dim; ic++, idst++)
          dst[idst] = src[ic];
      }
  }
}
static void unpack_subregion(Real *pack, Real *dstbase, int dim, int srcxstart,
                             int srcystart, int LX, int dstxend, int dstyend,
                             int xsize) {
  if (dim == 1) {
    int mod = dstxend % 4;
    for (int yd = 0; yd < dstyend; ++yd) {
      int offset = srcxstart + LX * (yd + srcystart);
      int offset_dst = xsize * yd;
      for (int xd = 0; xd < dstxend - mod; xd += 4) {
        dstbase[xd + 0 + offset_dst] = pack[xd + 0 + offset];
        dstbase[xd + 1 + offset_dst] = pack[xd + 1 + offset];
        dstbase[xd + 2 + offset_dst] = pack[xd + 2 + offset];
        dstbase[xd + 3 + offset_dst] = pack[xd + 3 + offset];
      }
      for (int xd = dstxend - mod; xd < dstxend; ++xd)
        dstbase[xd + offset_dst] = pack[xd + offset];
    }
  } else {
    for (int yd = 0; yd < dstyend; ++yd)
      for (int xd = 0; xd < dstxend; ++xd) {
        Real *dst = dstbase + dim * (xd + xsize * yd);
        Real *src = pack + dim * (xd + srcxstart + LX * (yd + srcystart));
        for (int c = 0; c < dim; ++c)
          dst[c] = src[c];
      }
  }
}
static void if2d_solve(unsigned Nm, Real *rS, Real *rX, Real *rY, Real *vX,
                       Real *vY, Real *norX, Real *norY, Real *vNorX,
                       Real *vNorY) {
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
    Real ds = rS[i] - rS[i - 1];
    rX[i] = rX[i - 1] + ds * ksiX;
    rY[i] = rY[i - 1] + ds * ksiY;
    norX[i] = norX[i - 1];
    norY[i] = norY[i - 1];
    vX[i] = vX[i - 1] + ds * vKsiX;
    vY[i] = vY[i - 1] + ds * vKsiY;
    vNorX[i] = vNorX[i - 1];
    vNorY[i] = vNorY[i - 1];
    Real d1 = ksiX * ksiX + ksiY * ksiY;
    Real d2 = norX[i] * norX[i] + norY[i] * norY[i];
    if (d1 > std::numeric_limits<Real>::epsilon()) {
      Real normfac = 1 / std::sqrt(d1);
      ksiX *= normfac;
      ksiY *= normfac;
    }
    if (d2 > std::numeric_limits<Real>::epsilon()) {
      Real normfac = 1 / std::sqrt(d2);
      norX[i] *= normfac;
      norY[i] *= normfac;
    }
  }
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
static Real derivative(Real U, Real um3, Real um2, Real um1, Real u, Real up1,
                       Real up2, Real up3) {
  return U > 0 ? weno5_plus(um2, um1, u, up1, up2) -
                     weno5_plus(um3, um2, um1, u, up1)
               : weno5_minus(um1, u, up1, up2, up3) -
                     weno5_minus(um2, um1, u, up1, up2);
}
static void compute_j(Real *Rc, Real *R, Real *N, Real *I, Real *J) {
  Real m00 = 1.0;
  Real m01 = 0.0;
  Real m02 = 0.0;
  Real m11 = 1.0;
  Real m12 = 0.0;
  Real m22 = I[5];
  Real a00 = m22 * m11 - m12 * m12;
  Real a01 = m02 * m12 - m22 * m01;
  Real a02 = m01 * m12 - m02 * m11;
  Real a11 = m22 * m00 - m02 * m02;
  Real a12 = m01 * m02 - m00 * m12;
  Real a22 = m00 * m11 - m01 * m01;
  Real determinant = 1.0 / ((m00 * a00) + (m01 * a01) + (m02 * a02));
  a00 *= determinant;
  a01 *= determinant;
  a02 *= determinant;
  a11 *= determinant;
  a12 *= determinant;
  a22 *= determinant;
  Real aux_0 = (Rc[1] - R[1]) * N[2] - (Rc[2] - R[2]) * N[1];
  Real aux_1 = (Rc[2] - R[2]) * N[0] - (Rc[0] - R[0]) * N[2];
  Real aux_2 = (Rc[0] - R[0]) * N[1] - (Rc[1] - R[1]) * N[0];
  J[0] = a00 * aux_0 + a01 * aux_1 + a02 * aux_2;
  J[1] = a01 * aux_0 + a11 * aux_1 + a12 * aux_2;
  J[2] = a02 * aux_0 + a12 * aux_1 + a22 * aux_2;
}
static void collision(Real m1, Real m2, Real *I1, Real *I2, Real *v1, Real *v2,
                      Real *o1, Real *o2, Real *hv1, Real *hv2, Real *ho1,
                      Real *ho2, Real *C1, Real *C2, Real NX, Real NY, Real NZ,
                      Real CX, Real CY, Real CZ, Real *vc1, Real *vc2) {
  Real e = 1.0;
  Real N[3] = {NX, NY, NZ};
  Real C[3] = {CX, CY, CZ};
  Real k1[3] = {N[0] / m1, N[1] / m1, N[2] / m1};
  Real k2[3] = {-N[0] / m2, -N[1] / m2, -N[2] / m2};
  Real J1[3];
  Real J2[3];
  compute_j(C, C1, N, I1, J1);
  compute_j(C, C2, N, I2, J2);
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
  Real nom = e * ((vc1[0] - vc2[0]) * N[0] + (vc1[1] - vc2[1]) * N[1] +
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
  Real denom = -(1.0 / m1 + 1.0 / m2) +
               +((J1[1] * (C[2] - C1[2]) - J1[2] * (C[1] - C1[1])) * (-N[0]) +
                 (J1[2] * (C[0] - C1[0]) - J1[0] * (C[2] - C1[2])) * (-N[1]) +
                 (J1[0] * (C[1] - C1[1]) - J1[1] * (C[0] - C1[0])) * (-N[2])) -
               ((J2[1] * (C[2] - C2[2]) - J2[2] * (C[1] - C2[1])) * (-N[0]) +
                (J2[2] * (C[0] - C2[0]) - J2[0] * (C[2] - C2[2])) * (-N[1]) +
                (J2[0] * (C[1] - C2[1]) - J2[1] * (C[0] - C2[0])) * (-N[2]));
  Real impulse = nom / (denom + 1e-21);
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
struct SpaceCurve {
  int base_level;
  bool isRegular;
  std::vector<std::vector<long long>> Zsave;
  std::vector<std::vector<int>> i_inverse, j_inverse;
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
  long long forward(const int l, const int i, const int j) const {
    const int aux = 1 << l;
    if (l >= sim.levelMax)
      return 0;
    long long retval;
    if (!isRegular) {
      const int I = i / aux;
      const int J = j / aux;
      const int c2_a[2] = {i - I * aux, j - J * aux};
      retval = AxestoTranspose(c2_a, l);
      retval += Zsave[0][J * sim.bpdx + I] * aux * aux;
    } else {
      const int c2_a[2] = {i, j};
      retval = AxestoTranspose(c2_a, l + base_level);
    }
    return retval;
  }
  void inverse(long long Z, int l, int *i, int *j) const {
    if (isRegular) {
      int X[2] = {0, 0};
      TransposetoAxes(Z, X, l + base_level);
      *i = X[0];
      *j = X[1];
    } else {
      int aux = 1 << l;
      long long Zloc = Z % (aux * aux);
      int X[2] = {0, 0};
      TransposetoAxes(Zloc, X, l);
      long long index = Z / (aux * aux);
      int I, J;
      I = i_inverse[0][index];
      J = j_inverse[0][index];
      *i = X[0] + I * aux;
      *j = X[1] + J * aux;
    }
    return;
  }
  long long Encode(int level, int index[2]) {
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
    for (int l = level + 1; l < sim.levelMax; l++) {
      long long Zc = forward(l, ix, iy);
      Zc -= Zc % 4;
      retval += Zc;
      int ix1, iy1;
      inverse(Zc, l, &ix1, &iy1);
      ix = 2 * ix1;
      iy = 2 * iy1;
    }
    retval += level;
    return retval;
  }
};
static long long forward(int level, int i, int j) {
  return sim.space_curve->forward(level, i % (1 << level * sim.bpdx),
                                  j % (1 << level * sim.bpdy));
}
struct Value {
  std::string content;
  Value() = default;
  Value(const std::string &content_) : content(content_) {}
  Real asDouble() { return atof(content.c_str()); }
  int asInt() { return atoi(content.c_str()); }
  std::string asString() { return content; }
};
struct CommandlineParser {
  std::map<std::string, Value> mapArguments;
  CommandlineParser(const int argc, char **argv) {
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
          mapArguments[key] = Value(values);
        } else {
          if (mapArguments.find(key) == mapArguments.end())
            mapArguments[key] = Value(values);
        }
        i += itemCount;
      }
  }
  Value &operator()(std::string key) {
    if (mapArguments.find(key) == mapArguments.end()) {
      fprintf(stderr, "main.cpp: runtime %s is not set\n", key.data());
      abort();
    }
    return mapArguments[key];
  }
};

std::vector<double> precond() {
  std::vector<double> L[_BS_ * _BS_];
  std::vector<double> L_inv[_BS_ * _BS_];
  for (int i = 0; i < _BS_ * _BS_; i++) {
    L[i].resize(i + 1);
    L_inv[i].resize(i + 1);
    for (int j = 0; j <= i; j++)
      L_inv[i][j] = i == j ? 1. : 0.;
  }
  for (int i = 0; i < _BS_ * _BS_; i++) {
    double s1 = 0;
    for (int k = 0; k <= i - 1; k++)
      s1 += L[i][k] * L[i][k];
    L[i][i] = sqrt(getA_local(i, i) - s1);
    for (int j = i + 1; j < _BS_ * _BS_; j++) {
      double s2 = 0;
      for (int k = 0; k <= i - 1; k++)
        s2 += L[i][k] * L[j][k];
      L[j][i] = (getA_local(j, i) - s2) / L[i][i];
    }
  }
  for (int br = 0; br < _BS_ * _BS_; br++) {
    double bsf = 1. / L[br][br];
    for (int c = 0; c <= br; c++)
      L_inv[br][c] *= bsf;
    for (int wr = br + 1; wr < _BS_ * _BS_; wr++) {
      double wsf = L[wr][br];
      for (int c = 0; c <= br; c++)
        L_inv[wr][c] -= wsf * L_inv[br][c];
    }
  }
  std::vector<double> P_inv(_BS_ * _BS_ * _BS_ * _BS_);
  for (int i = 0; i < _BS_ * _BS_; i++)
    for (int j = 0; j < _BS_ * _BS_; j++) {
      double aux = 0.;
      for (int k = 0; k < _BS_ * _BS_; k++)
        aux += i <= k && j <= k ? L_inv[k][i] * L_inv[k][j] : 0.;
      P_inv[i * _BS_ * _BS_ + j] = -aux;
    };

  return P_inv;
}

static Real sdf_circle(Real x, Real y, Real cx, Real cy, Real r) {
  Real dx = x - cx;
  Real dy = y - cy;
  return std::sqrt(dx * dx + dy * dy) - r;
}
