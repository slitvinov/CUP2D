#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "solver.h"

struct Solver {
  int blen, m, nnz, mean_row;
  const double *coo_val;
  const int *coo_row, *coo_col;
  const double *h2;
  double *precond;
  double *r, *rhat, *p, *nu, *t, *z, *x_opt;
};

struct Solver *solver_create(int blen, const double *precond) {
  struct Solver *s = calloc(1, sizeof *s);
  s->blen = blen;
  s->precond = malloc(blen * blen * sizeof(double));
  memcpy(s->precond, precond, blen * blen * sizeof(double));
  return s;
}

void solver_destroy(struct Solver *s) {
  free(s->precond);
  free(s->r);
  free(s->rhat);
  free(s->p);
  free(s->nu);
  free(s->t);
  free(s->z);
  free(s->x_opt);
  free(s);
}

/* y = A * x  (COO SpMV) */
static void spmv(int m, int nnz, const double *val, const int *row,
                 const int *col, const double *x, double *y) {
  memset(y, 0, m * sizeof(double));
  for (int i = 0; i < nnz; i++)
    y[row[i]] += val[i] * x[col[i]];
}

/* y = P_inv * x  (block-diagonal, blen x blen blocks) */
static void precond_apply(int m, int blen, const double *P,
                          const double *x, double *y) {
  int nb = m / blen;
#pragma omp parallel for schedule(static)
  for (int b = 0; b < nb; b++) {
    const double *xb = x + b * blen;
    double *yb = y + b * blen;
    for (int i = 0; i < blen; i++) {
      double s = 0;
      const double *row = P + i * blen;
      for (int j = 0; j < blen; j++)
        s += row[j] * xb[j];
      yb[i] = s;
    }
  }
}

/* y[i] += a * x[i] */
static void axpy(int m, double a, const double *x, double *y) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < m; i++)
    y[i] += a * x[i];
}

/* x[i] *= a */
static void scal(int m, double a, double *x) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < m; i++)
    x[i] *= a;
}

static double dot(int m, const double *a, const double *b) {
  double s = 0;
#pragma omp parallel for reduction(+:s) schedule(static)
  for (int i = 0; i < m; i++)
    s += a[i] * b[i];
  return s;
}

static double amax(int m, const double *x) {
  double mx = 0;
#pragma omp parallel for reduction(max:mx) schedule(static)
  for (int i = 0; i < m; i++) {
    double a = fabs(x[i]);
    if (a > mx) mx = a;
  }
  return mx;
}

/* A*x with mean constraint row */
static void matvec(struct Solver *s, const double *x, double *y) {
  spmv(s->m, s->nnz, s->coo_val, s->coo_row, s->coo_col, x, y);
  if (s->mean_row >= 0) {
    double sum = 0;
    for (int i = 0; i < s->m; i++)
      sum += s->h2[i / s->blen] * x[i];
    y[s->mean_row] = sum;
  }
}

static void bicgstab(struct Solver *s, double *x, double max_error,
                     double max_rel_error, int max_restarts) {
  int m = s->m;
  double *r = s->r, *rhat = s->rhat, *p = s->p;
  double *nu = s->nu, *t = s->t, *z = s->z, *x_opt = s->x_opt;
  double eps = 1e-21;

  /* r = b - A*x  (r was loaded with b, subtract A*x) */
  matvec(s, x, nu);
  axpy(m, -1.0, nu, r);

  double error = amax(m, r);
  double error_init = error;
  double error_opt = error;
  memcpy(x_opt, x, m * sizeof(double));
  memcpy(rhat, r, m * sizeof(double));
  memset(nu, 0, m * sizeof(double));
  memset(p, 0, m * sizeof(double));

  double rho_prev = 1, alpha = 1, omega = 1;
  int restarts = 0;

  for (int k = 0; k < 1000; k++) {
    double rho = dot(m, rhat, r);
    double nr = dot(m, r, r);
    double nrh = dot(m, rhat, rhat);
    int serious_breakdown = rho * rho < 1e-16 * nr * nrh;

    double beta = (rho / (rho_prev + eps)) * (alpha / (omega + eps));

    if (serious_breakdown && max_restarts > 0) {
      restarts++;
      if (restarts >= max_restarts) break;
      memcpy(rhat, r, m * sizeof(double));
      rho = dot(m, r, r);
      memset(nu, 0, m * sizeof(double));
      memset(p, 0, m * sizeof(double));
      rho_prev = 1; alpha = 1; omega = 1;
      beta = (rho / (rho_prev + eps)) * (alpha / (omega + eps));
    }

    /* p = r + beta * (p - omega * nu) */
    axpy(m, -omega, nu, p);
    scal(m, beta, p);
    axpy(m, 1.0, r, p);

    /* z = P_inv * p;  nu = A * z */
    precond_apply(m, s->blen, s->precond, p, z);
    matvec(s, z, nu);

    /* alpha = rho / (rhat . nu) */
    double rhat_nu = dot(m, rhat, nu);
    alpha = rho / (rhat_nu + eps);

    /* x += alpha * z;  r -= alpha * nu */
    axpy(m, alpha, z, x);
    axpy(m, -alpha, nu, r);

    /* z = P_inv * r;  t = A * z */
    precond_apply(m, s->blen, s->precond, r, z);
    matvec(s, z, t);

    /* omega = (t . r) / (t . t) */
    double tr = dot(m, t, r);
    double tt = dot(m, t, t);
    omega = tr / (tt + eps);

    /* x += omega * z;  r -= omega * t */
    axpy(m, omega, z, x);
    axpy(m, -omega, t, r);

    error = amax(m, r);
    if (error < error_opt) {
      error_opt = error;
      memcpy(x_opt, x, m * sizeof(double));
      if (error <= max_error || error / error_init <= max_rel_error)
        break;
    }
    rho_prev = rho;
  }
  memcpy(x, x_opt, m * sizeof(double));
}

void solver_solve(struct Solver *s, int update_matrix,
    int m, int nnz,
    const double *coo_val, const int *coo_row, const int *coo_col,
    double *x, const double *b, const double *h2, int mean_row,
    double tol, double rtol, int restarts) {
  if (update_matrix || s->m != m) {
    free(s->r);     free(s->rhat);  free(s->p);
    free(s->nu);    free(s->t);     free(s->z);
    free(s->x_opt);
    s->r     = malloc(m * sizeof(double));
    s->rhat  = malloc(m * sizeof(double));
    s->p     = malloc(m * sizeof(double));
    s->nu    = malloc(m * sizeof(double));
    s->t     = malloc(m * sizeof(double));
    s->z     = malloc(m * sizeof(double));
    s->x_opt = malloc(m * sizeof(double));
  }
  s->m = m;
  s->nnz = nnz;
  s->coo_val = coo_val;
  s->coo_row = coo_row;
  s->coo_col = coo_col;
  s->h2 = h2;
  s->mean_row = mean_row;
  memcpy(s->r, b, m * sizeof(double));
  bicgstab(s, x, tol, rtol, restarts);
}
