#pragma once
#ifdef __cplusplus
extern "C" {
#endif
struct GpuSolver;
GpuSolver *gpu_solver_create(int blen, const double *precond);
void gpu_solver_destroy(GpuSolver *s);
void gpu_solver_solve(GpuSolver *s, int update_matrix,
    int m, int nnz,
    const double *coo_val, const int *coo_row, const int *coo_col,
    double *x, const double *b, const double *h2, int mean_row,
    double tol, double rtol, int restarts);
#ifdef __cplusplus
}
#endif
