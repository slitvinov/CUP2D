#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

struct BiCGSTABScalars {
  double alpha;
  double beta;
  double omega;
  double eps;
  double rho_prev;
  double rho_curr;
  double buff_1;
  double buff_2;
  int amax_idx;
};

struct GpuSolver {
  cudaStream_t stream;
  cublasHandle_t cublas;
  cusparseHandle_t cusparse;
  int m;
  int nnz;
  int blen;
  int mean_row;
  double *d_consts;   /* [0]=1, [1]=-1, [2]=0 */
  BiCGSTABScalars *h_coeffs;
  BiCGSTABScalars *d_coeffs;
  double *d_coo_val;
  int *d_coo_row;
  int *d_coo_col;
  double *d_x;
  double *d_x_opt;
  double *d_r;
  double *d_P_inv;
  double *d_h2;
  double *d_red;
  double *d_red_res;
  void *d_red_temp;
  size_t red_temp_bytes;
  double *d_rhat;
  double *d_p;
  double *d_nu;
  double *d_t;
  double *d_z;
  cusparseSpMatDescr_t spA;
  cusparseDnVecDescr_t spNu;
  cusparseDnVecDescr_t spT;
  cusparseDnVecDescr_t spZ;
  size_t spmv_buf_sz;
  void *spmv_buf;
};

static void free_matrix(GpuSolver *s) {
  if (s->m == 0) return;
  cudaFree(s->d_coo_val);
  cudaFree(s->d_coo_row);
  cudaFree(s->d_coo_col);
  cudaFree(s->d_x);
  cudaFree(s->d_x_opt);
  cudaFree(s->d_r);
  cudaFree(s->d_rhat);
  cudaFree(s->d_p);
  cudaFree(s->d_nu);
  cudaFree(s->d_t);
  cudaFree(s->d_z);
  cudaFree(s->spmv_buf);
  cusparseDestroySpMat(s->spA);
  cusparseDestroyDnVec(s->spNu);
  cusparseDestroyDnVec(s->spT);
  cusparseDestroyDnVec(s->spZ);
  cudaFree(s->d_h2);
  cudaFree(s->d_red);
  cudaFree(s->d_red_res);
  cudaFree(s->d_red_temp);
  s->m = 0;
}

GpuSolver *gpu_solver_create(int blen, const double *precond) {
  GpuSolver *s = (GpuSolver *)calloc(1, sizeof *s);
  s->blen = blen;
  int device;
  if (cudaGetDevice(&device) != cudaSuccess) {
    fprintf(stderr, "cuda.cu: error: no CUDA-capable devices found\n");
    exit(1);
  }
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  cudaUUID_t u = prop.uuid;
  fprintf(stderr,
      "cuda.cu: %s (UUID: "
      "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x)\n",
      prop.name,
      (unsigned char)u.bytes[0],  (unsigned char)u.bytes[1],
      (unsigned char)u.bytes[2],  (unsigned char)u.bytes[3],
      (unsigned char)u.bytes[4],  (unsigned char)u.bytes[5],
      (unsigned char)u.bytes[6],  (unsigned char)u.bytes[7],
      (unsigned char)u.bytes[8],  (unsigned char)u.bytes[9],
      (unsigned char)u.bytes[10], (unsigned char)u.bytes[11],
      (unsigned char)u.bytes[12], (unsigned char)u.bytes[13],
      (unsigned char)u.bytes[14], (unsigned char)u.bytes[15]);
  cudaStreamCreate(&s->stream);
  cublasCreate(&s->cublas);
  cusparseCreate(&s->cusparse);
  cublasSetStream(s->cublas, s->stream);
  cusparseSetStream(s->cusparse, s->stream);
  cublasSetPointerMode(s->cublas, CUBLAS_POINTER_MODE_DEVICE);
  cusparseSetPointerMode(s->cusparse, CUSPARSE_POINTER_MODE_DEVICE);
  double h_consts[3] = {1., -1., 0.};
  cudaMalloc(&s->d_consts, 3 * sizeof(double));
  cudaMemcpyAsync(s->d_consts, h_consts, 3 * sizeof(double),
                  cudaMemcpyHostToDevice, s->stream);
  cudaMalloc(&s->d_coeffs, sizeof(BiCGSTABScalars));
  cudaMallocHost(&s->h_coeffs, sizeof(BiCGSTABScalars));
  cudaMalloc(&s->d_P_inv, blen * blen * sizeof(double));
  cudaMemcpyAsync(s->d_P_inv, precond, blen * blen * sizeof(double),
                  cudaMemcpyHostToDevice, s->stream);
  return s;
}

void gpu_solver_destroy(GpuSolver *s) {
  free_matrix(s);
  cudaFree(s->d_P_inv);
  cudaFree(s->d_consts);
  cudaFree(s->d_coeffs);
  cudaFreeHost(s->h_coeffs);
  cublasDestroy(s->cublas);
  cusparseDestroy(s->cusparse);
  cudaStreamDestroy(s->stream);
  free(s);
}

__global__ void set_squared(double *const val) { val[0] *= val[0]; }
__global__ void set_amax(double *const dest, const int *const idx,
                         const double *const source) {
  dest[0] = fabs(source[idx[0] - 1]);
}
__global__ void set_negative(double *const dest, double *const source) {
  dest[0] = -source[0];
}
__global__ void breakdown_update(BiCGSTABScalars *coeffs) {
  coeffs->rho_prev = 1.;
  coeffs->alpha = 1.;
  coeffs->omega = 1.;
  coeffs->beta = (coeffs->rho_curr / (coeffs->rho_prev + coeffs->eps)) *
                 (coeffs->alpha / (coeffs->omega + coeffs->eps));
}
__global__ void set_beta(BiCGSTABScalars *coeffs) {
  coeffs->beta = (coeffs->rho_curr / (coeffs->rho_prev + coeffs->eps)) *
                 (coeffs->alpha / (coeffs->omega + coeffs->eps));
}
__global__ void set_alpha(BiCGSTABScalars *coeffs) {
  coeffs->alpha = coeffs->rho_curr / (coeffs->buff_1 + coeffs->eps);
}
__global__ void set_omega(BiCGSTABScalars *coeffs) {
  coeffs->omega = coeffs->buff_1 / (coeffs->buff_2 + coeffs->eps);
}
__global__ void set_rho(BiCGSTABScalars *coeffs) {
  coeffs->rho_prev = coeffs->rho_curr;
}
__global__ void blockDscal(const int m, const int BLEN,
                           const double *__restrict__ const alpha,
                           double *__restrict__ const x) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m;
       i += blockDim.x * gridDim.x)
    x[i] = alpha[i / BLEN] * x[i];
}

static void spmv(GpuSolver *s, double *d_op, cusparseDnVecDescr_t spDescrOp,
                 double *d_res, cusparseDnVecDescr_t spDescrRes) {
  cusparseSpMV(s->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, s->d_consts,
               s->spA, spDescrOp, (s->d_consts + 2), spDescrRes, CUDA_R_64F,
               CUSPARSE_SPMV_ALG_DEFAULT, s->spmv_buf);
  if (s->mean_row >= 0) {
    cudaMemcpyAsync(s->d_red, d_op, s->m * sizeof(double),
                    cudaMemcpyDeviceToDevice, s->stream);
    blockDscal<<<8 * 56, 128, 0, s->stream>>>(s->m, s->blen, s->d_h2, s->d_red);
    cudaGetLastError();
    cub::DeviceReduce::Sum<double *, double *>(s->d_red_temp,
                                               s->red_temp_bytes, s->d_red,
                                               s->d_red_res, s->m, s->stream);
    double h_red_res;
    cudaMemcpyAsync(&h_red_res, s->d_red_res, sizeof(double),
                    cudaMemcpyDeviceToHost, s->stream);
    cudaStreamSynchronize(s->stream);
    cudaMemcpyAsync(&d_res[s->mean_row], &h_red_res, sizeof(double),
                    cudaMemcpyHostToDevice, s->stream);
  }
}

static void upload_matrix(GpuSolver *s, int m, int nnz,
    const double *coo_val, const int *coo_row, const int *coo_col,
    const double *x, const double *b, const double *h2, int mean_row) {
  free_matrix(s);
  s->m = m;
  s->nnz = nnz;
  s->mean_row = mean_row;
  int nblocks = m / s->blen;
  cudaMalloc(&s->d_coo_val, nnz * sizeof(double));
  cudaMalloc(&s->d_coo_row, nnz * sizeof(int));
  cudaMalloc(&s->d_coo_col, nnz * sizeof(int));
  cudaMalloc(&s->d_x, m * sizeof(double));
  cudaMalloc(&s->d_x_opt, m * sizeof(double));
  cudaMalloc(&s->d_r, m * sizeof(double));
  cudaMalloc(&s->d_rhat, m * sizeof(double));
  cudaMalloc(&s->d_p, m * sizeof(double));
  cudaMalloc(&s->d_nu, m * sizeof(double));
  cudaMalloc(&s->d_t, m * sizeof(double));
  cudaMalloc(&s->d_z, m * sizeof(double));
  cudaMalloc(&s->d_h2, nblocks * sizeof(double));
  cudaMalloc(&s->d_red, m * sizeof(double));
  cudaMalloc(&s->d_red_res, sizeof(double));
  s->d_red_temp = NULL;
  s->red_temp_bytes = 0;
  cub::DeviceReduce::Sum<double *, double *>(s->d_red_temp,
                                             s->red_temp_bytes, s->d_red,
                                             s->d_red_res, m, s->stream);
  cudaMalloc(&s->d_red_temp, s->red_temp_bytes);
  cudaMemcpyAsync(s->d_coo_val, coo_val, nnz * sizeof(double),
                  cudaMemcpyHostToDevice, s->stream);
  cudaMemcpyAsync(s->d_coo_row, coo_row, nnz * sizeof(int),
                  cudaMemcpyHostToDevice, s->stream);
  cudaMemcpyAsync(s->d_coo_col, coo_col, nnz * sizeof(int),
                  cudaMemcpyHostToDevice, s->stream);
  cudaMemcpyAsync(s->d_h2, h2, nblocks * sizeof(double),
                  cudaMemcpyHostToDevice, s->stream);
  cusparseCreateCoo(&s->spA, m, m, nnz, s->d_coo_row,
                    s->d_coo_col, s->d_coo_val, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  cusparseCreateDnVec(&s->spNu, m, s->d_nu, CUDA_R_64F);
  cusparseCreateDnVec(&s->spT, m, s->d_t, CUDA_R_64F);
  cusparseCreateDnVec(&s->spZ, m, s->d_z, CUDA_R_64F);
  cusparseSpMV_bufferSize(s->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          s->d_consts, s->spA, s->spZ, (s->d_consts + 2),
                          s->spNu, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                          &s->spmv_buf_sz);
  cudaMalloc(&s->spmv_buf, s->spmv_buf_sz * sizeof(char));
  // upload vectors
  cudaMemcpyAsync(s->d_x, x, m * sizeof(double),
                  cudaMemcpyHostToDevice, s->stream);
  cudaMemcpyAsync(s->d_r, b, m * sizeof(double),
                  cudaMemcpyHostToDevice, s->stream);
}

static void upload_vectors(GpuSolver *s, const double *x, const double *b) {
  cudaMemcpyAsync(s->d_x, x, s->m * sizeof(double),
                  cudaMemcpyHostToDevice, s->stream);
  cudaMemcpyAsync(s->d_r, b, s->m * sizeof(double),
                  cudaMemcpyHostToDevice, s->stream);
}

static void bicgstab(GpuSolver *s, double max_error, double max_rel_error,
                     int max_restarts) {
  int m = s->m;
  double error = 1e50;
  double error_init = 1e50;
  double error_opt = 1e50;
  int restarts = 0;
  *s->h_coeffs = {1., 1., 1., 1e-21, 1., 1., 0., 0., 0};
  cudaMemcpyAsync(s->d_coeffs, s->h_coeffs, sizeof(BiCGSTABScalars),
                  cudaMemcpyHostToDevice, s->stream);
  cudaMemcpyAsync(s->d_z, s->d_x, m * sizeof(double), cudaMemcpyDeviceToDevice,
                  s->stream);
  spmv(s, s->d_z, s->spZ, s->d_nu, s->spNu);
  cublasDaxpy(s->cublas, m, (s->d_consts + 1), s->d_nu, 1, s->d_r, 1);
  cublasIdamax(s->cublas, m, s->d_nu, 1, &(s->d_coeffs->amax_idx));
  set_amax<<<1, 1, 0, s->stream>>>(&(s->d_coeffs->buff_1),
                                           &(s->d_coeffs->amax_idx), s->d_nu);
  cudaGetLastError();
  cublasIdamax(s->cublas, m, s->d_r, 1, &(s->d_coeffs->amax_idx));
  set_amax<<<1, 1, 0, s->stream>>>(&(s->d_coeffs->buff_2),
                                           &(s->d_coeffs->amax_idx), s->d_r);
  cudaGetLastError();
  cudaMemcpyAsync(&(s->h_coeffs->buff_1), &(s->d_coeffs->buff_1),
                  2 * sizeof(double), cudaMemcpyDeviceToHost, s->stream);
  cudaStreamSynchronize(s->stream);
  error = s->h_coeffs->buff_2;
  error_init = error;
  error_opt = error;
  cudaMemcpyAsync(s->d_x_opt, s->d_x, m * sizeof(double), cudaMemcpyDeviceToDevice,
                  s->stream);
  cudaMemcpyAsync(s->d_rhat, s->d_r, m * sizeof(double), cudaMemcpyDeviceToDevice,
                  s->stream);
  cudaMemsetAsync(s->d_nu, 0, m * sizeof(double), s->stream);
  cudaMemsetAsync(s->d_p, 0, m * sizeof(double), s->stream);
  const int max_iter = 1000;
  for (int k = 0; k < max_iter; k++) {
    cublasDdot(s->cublas, m, s->d_rhat, 1, s->d_r, 1, &(s->d_coeffs->rho_curr));
    cublasDnrm2(s->cublas, m, s->d_r, 1, &(s->d_coeffs->buff_1));
    cublasDnrm2(s->cublas, m, s->d_rhat, 1, &(s->d_coeffs->buff_2));
    cudaMemcpyAsync(&(s->h_coeffs->rho_curr), &(s->d_coeffs->rho_curr),
                    3 * sizeof(double), cudaMemcpyDeviceToHost, s->stream);
    cudaStreamSynchronize(s->stream);
    s->h_coeffs->buff_1 *= s->h_coeffs->buff_1;
    s->h_coeffs->buff_2 *= s->h_coeffs->buff_2;
    cudaMemcpyAsync(&(s->d_coeffs->rho_curr), &(s->h_coeffs->rho_curr),
                    sizeof(double), cudaMemcpyHostToDevice, s->stream);
    const bool serious_breakdown =
        s->h_coeffs->rho_curr * s->h_coeffs->rho_curr <
        1e-16 * s->h_coeffs->buff_1 * s->h_coeffs->buff_2;
    set_beta<<<1, 1, 0, s->stream>>>(s->d_coeffs);
    cudaGetLastError();
    if (serious_breakdown && max_restarts > 0) {
      restarts++;
      if (restarts >= max_restarts) break;
      cudaMemcpyAsync(s->d_rhat, s->d_r, m * sizeof(double),
                      cudaMemcpyDeviceToDevice, s->stream);
      cublasDnrm2(s->cublas, m, s->d_rhat, 1, &(s->d_coeffs->rho_curr));
      cudaMemcpyAsync(&(s->h_coeffs->rho_curr), &(s->d_coeffs->rho_curr),
                      sizeof(double), cudaMemcpyDeviceToHost, s->stream);
      cudaStreamSynchronize(s->stream);
      s->h_coeffs->rho_curr *= s->h_coeffs->rho_curr;
      cudaMemcpyAsync(&(s->d_coeffs->rho_curr), &(s->h_coeffs->rho_curr),
                      sizeof(double), cudaMemcpyHostToDevice, s->stream);
      cudaMemsetAsync(s->d_nu, 0, m * sizeof(double), s->stream);
      cudaMemsetAsync(s->d_p, 0, m * sizeof(double), s->stream);
      breakdown_update<<<1, 1, 0, s->stream>>>(s->d_coeffs);
      cudaGetLastError();
    }
    set_negative<<<1, 1, 0, s->stream>>>(&(s->d_coeffs->buff_1),
                                                &(s->d_coeffs->omega));
    cudaGetLastError();
    cublasDaxpy(s->cublas, m, &(s->d_coeffs->buff_1), s->d_nu, 1, s->d_p, 1);
    cublasDscal(s->cublas, m, &(s->d_coeffs->beta), s->d_p, 1);
    cublasDaxpy(s->cublas, m, s->d_consts, s->d_r, 1, s->d_p, 1);
    cublasDgemm(s->cublas, CUBLAS_OP_T, CUBLAS_OP_N, s->blen, m / s->blen,
                s->blen, s->d_consts, s->d_P_inv, s->blen, s->d_p, s->blen,
                (s->d_consts + 2), s->d_z, s->blen);
    spmv(s, s->d_z, s->spZ, s->d_nu, s->spNu);
    cublasDdot(s->cublas, m, s->d_rhat, 1, s->d_nu, 1, &(s->d_coeffs->buff_1));
    cudaMemcpyAsync(&(s->h_coeffs->buff_1), &(s->d_coeffs->buff_1), sizeof(double),
                    cudaMemcpyDeviceToHost, s->stream);
    cudaStreamSynchronize(s->stream);
    cudaMemcpyAsync(&(s->d_coeffs->buff_1), &(s->h_coeffs->buff_1), sizeof(double),
                    cudaMemcpyHostToDevice, s->stream);
    set_alpha<<<1, 1, 0, s->stream>>>(s->d_coeffs);
    cudaGetLastError();
    cublasDaxpy(s->cublas, m, &(s->d_coeffs->alpha), s->d_z, 1, s->d_x, 1);
    set_negative<<<1, 1, 0, s->stream>>>(&(s->d_coeffs->buff_1),
                                                &(s->d_coeffs->alpha));
    cudaGetLastError();
    cublasDaxpy(s->cublas, m, &(s->d_coeffs->buff_1), s->d_nu, 1, s->d_r, 1);
    cublasDgemm(s->cublas, CUBLAS_OP_T, CUBLAS_OP_N, s->blen, m / s->blen,
                s->blen, s->d_consts, s->d_P_inv, s->blen, s->d_r, s->blen,
                (s->d_consts + 2), s->d_z, s->blen);
    spmv(s, s->d_z, s->spZ, s->d_t, s->spT);
    cublasDdot(s->cublas, m, s->d_t, 1, s->d_r, 1, &(s->d_coeffs->buff_1));
    cublasDnrm2(s->cublas, m, s->d_t, 1, &(s->d_coeffs->buff_2));
    set_squared<<<1, 1, 0, s->stream>>>(&(s->d_coeffs->buff_2));
    cudaGetLastError();
    cudaMemcpyAsync(&(s->h_coeffs->buff_1), &(s->d_coeffs->buff_1),
                    2 * sizeof(double), cudaMemcpyDeviceToHost, s->stream);
    cudaStreamSynchronize(s->stream);
    cudaMemcpyAsync(&(s->d_coeffs->buff_1), &(s->h_coeffs->buff_1),
                    2 * sizeof(double), cudaMemcpyHostToDevice, s->stream);
    set_omega<<<1, 1, 0, s->stream>>>(s->d_coeffs);
    cudaGetLastError();
    cublasDaxpy(s->cublas, m, &(s->d_coeffs->omega), s->d_z, 1, s->d_x, 1);
    set_negative<<<1, 1, 0, s->stream>>>(&(s->d_coeffs->buff_1),
                                                &(s->d_coeffs->omega));
    cudaGetLastError();
    cublasDaxpy(s->cublas, m, &(s->d_coeffs->buff_1), s->d_t, 1, s->d_r, 1);
    cublasIdamax(s->cublas, m, s->d_r, 1, &(s->d_coeffs->amax_idx));
    set_amax<<<1, 1, 0, s->stream>>>(&(s->d_coeffs->buff_1),
                                            &(s->d_coeffs->amax_idx), s->d_r);
    cudaGetLastError();
    cudaMemcpyAsync(&error, &(s->d_coeffs->buff_1), sizeof(double),
                    cudaMemcpyDeviceToHost, s->stream);
    cudaMemcpyAsync(s->h_coeffs, s->d_coeffs, sizeof(BiCGSTABScalars),
                    cudaMemcpyDeviceToHost, s->stream);
    cudaStreamSynchronize(s->stream);
    if (error < error_opt) {
      error_opt = error;
      cudaMemcpyAsync(s->d_x_opt, s->d_x, m * sizeof(double),
                      cudaMemcpyDeviceToDevice, s->stream);
      if ((error <= max_error) || (error / error_init <= max_rel_error))
        break;
    }
    set_rho<<<1, 1, 0, s->stream>>>(s->d_coeffs);
    cudaGetLastError();
  }
}

void gpu_solver_solve(GpuSolver *s, int update_matrix,
    int m, int nnz,
    const double *coo_val, const int *coo_row, const int *coo_col,
    double *x, const double *b, const double *h2, int mean_row,
    double tol, double rtol, int restarts) {
  if (update_matrix)
    upload_matrix(s, m, nnz, coo_val, coo_row, coo_col, x, b, h2, mean_row);
  else
    upload_vectors(s, x, b);
  bicgstab(s, tol, rtol, restarts);
  cudaMemcpyAsync(x, s->d_x_opt, m * sizeof(double),
                  cudaMemcpyDeviceToHost, s->stream);
  cudaStreamSynchronize(s->stream);
}
