static Real real_min(Real a, Real b) { return a < b ? a : b; }

static void LI(Real *a0, Real *b0, Real *c0) {
  Real a = *a0;
  Real b = *b0;
  Real c = *c0;
  Real kappa = ((4.0 / 15.0) * a + (6.0 / 15.0) * c) + (-10.0 / 15.0) * b;
  Real lambda = (b - c) - kappa;
  *a0 = (4.0 * kappa + 2.0 * lambda) + c;
}
static void LE(Real *a0, Real *b0, Real *c0) {
  Real a = *a0;
  Real b = *b0;
  Real c = *c0;
  Real kappa = ((4.0 / 15.0) * a + (6.0 / 15.0) * c) + (-10.0 / 15.0) * b;
  Real lambda = (b - c) - kappa;
  *a0 = (9.0 * kappa + 3.0 * lambda) + c;
}
static double getA_local(int I1, int I2) {
  int j1 = I1 / BS;
  int i1 = I1 % BS;
  int j2 = I2 / BS;
  int i2 = I2 % BS;
  if (i1 == i2 && j1 == j2)
    return 4.0;
  else if (abs(i1 - i2) + abs(j1 - j2) == 1)
    return -1.0;
  else
    return 0.0;
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
static void sfc_rot(long long n, int *x, int *y, long long rx, long long ry) {
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
static long long sfc_forward(const int l, int i, int j) {
  if (l >= sim.levelMax)
    return 0;
  int n = 1 << l;
  int rx, ry, s, d = 0;
  for (s = n / 2; s > 0; s /= 2) {
    rx = (i & s) > 0;
    ry = (j & s) > 0;
    d += s * s * ((3 * rx) ^ ry);
    sfc_rot(n, &i, &j, rx, ry);
  }
  return d;
}
static void sfc_inverse(long long Z, int l, int *i, int *j) {
  int n = 1 << l;
  long long rx, ry, s;
  *i = 0;
  *j = 0;
  for (s = 1; s < n; s *= 2) {
    rx = 1 & (Z / 2);
    ry = 1 & (Z ^ rx);
    sfc_rot(s, i, j, rx, ry);
    *i += s * rx;
    *j += s * ry;
    Z /= 4;
  }
}
static long long sfc_encode(int level, int index[2]) {
  long long retval = 0;
  int ix = index[0];
  int iy = index[1];
  for (int l = level; l >= 0; l--) {
    long long Zp = sfc_forward(l, ix, iy);
    retval += Zp;
    ix /= 2;
    iy /= 2;
  }
  ix = 2 * index[0];
  iy = 2 * index[1];
  for (int l = level + 1; l < sim.levelMax; l++) {
    long long Zc = sfc_forward(l, ix, iy);
    Zc -= Zc % 4;
    retval += Zc;
    int ix1, iy1;
    sfc_inverse(Zc, l, &ix1, &iy1);
    ix = 2 * ix1;
    iy = 2 * iy1;
  }
  retval += level;
  return retval;
};
static long long forward(int level, int i, int j) {
  return sfc_forward(level, i % (1 << level), j % (1 << level));
}
static const char *arg_find(int argc, char **argv, const char *key) {
  for (int i = 1; i < argc; i++)
    if (argv[i][0] == '-' && strcmp(argv[i] + 1, key) == 0) {
      if (i + 1 < argc)
        return argv[i + 1];
      fprintf(stderr, "main.cpp: error: option -%s has no value\n", key);
      exit(1);
    }
  fprintf(stderr, "main.cpp: error: option -%s is not set\n", key);
  exit(1);
}
static Real arg_real(int argc, char **argv, const char *key) {
  const char *s = arg_find(argc, argv, key);
  char *end;
  Real v = strtod(s, &end);
  if (end == s || *end != '\0') {
    fprintf(stderr, "main.cpp: error: -%s: bad real '%s'\n", key, s);
    exit(1);
  }
  return v;
}
static int arg_int(int argc, char **argv, const char *key) {
  const char *s = arg_find(argc, argv, key);
  char *end;
  long v = strtol(s, &end, 10);
  if (end == s || *end != '\0') {
    fprintf(stderr, "main.cpp: error: -%s: bad integer '%s'\n", key, s);
    exit(1);
  }
  return (int)v;
}
static int kv_sep(int c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }
static const char *kv_find(const char *line, const char *key) {
  const char *p = line;
  size_t klen = strlen(key);
  while (*p) {
    while (kv_sep(*p)) p++;
    if (strncmp(p, key, klen) == 0 && p[klen] == '=')
      return p + klen + 1;
    while (*p && !kv_sep(*p)) p++;
  }
  fprintf(stderr, "main.cpp: error: key '%s' not found in '%s'\n", key, line);
  exit(1);
}
static Real kv_real(const char *line, const char *key) {
  const char *s = kv_find(line, key);
  char *end;
  Real v = strtod(s, &end);
  if (end == s) {
    fprintf(stderr, "main.cpp: error: %s: bad real in '%s'\n", key, line);
    exit(1);
  }
  return v;
}
static const char *kv_str(const char *line, const char *key, char *buf, size_t n) {
  const char *s = kv_find(line, key);
  size_t i = 0;
  while (s[i] && !kv_sep(s[i]) && i < n - 1) { buf[i] = s[i]; i++; }
  buf[i] = '\0';
  return buf;
}
std::vector<double> precond() {
  std::vector<double> L[BS * BS];
  std::vector<double> L_inv[BS * BS];
  for (int i = 0; i < BS * BS; i++) {
    L[i].resize(i + 1);
    L_inv[i].resize(i + 1);
    for (int j = 0; j <= i; j++)
      L_inv[i][j] = i == j ? 1. : 0.;
  }
  for (int i = 0; i < BS * BS; i++) {
    double s1 = 0;
    for (int k = 0; k <= i - 1; k++)
      s1 += L[i][k] * L[i][k];
    L[i][i] = sqrt(getA_local(i, i) - s1);
    for (int j = i + 1; j < BS * BS; j++) {
      double s2 = 0;
      for (int k = 0; k <= i - 1; k++)
        s2 += L[i][k] * L[j][k];
      L[j][i] = (getA_local(j, i) - s2) / L[i][i];
    }
  }
  for (int br = 0; br < BS * BS; br++) {
    double bsf = 1. / L[br][br];
    for (int c = 0; c <= br; c++)
      L_inv[br][c] *= bsf;
    for (int wr = br + 1; wr < BS * BS; wr++) {
      double wsf = L[wr][br];
      for (int c = 0; c <= br; c++)
        L_inv[wr][c] -= wsf * L_inv[br][c];
    }
  }
  std::vector<double> P_inv(BS * BS * BS * BS);
  for (int i = 0; i < BS * BS; i++)
    for (int j = 0; j < BS * BS; j++) {
      double aux = 0.;
      for (int k = 0; k < BS * BS; k++)
        aux += i <= k && j <= k ? L_inv[k][i] * L_inv[k][j] : 0.;
      P_inv[i * BS * BS + j] = -aux;
    };

  return P_inv;
}
