#include <algorithm>
#include <cassert>
#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <map>
#include <memory>
#include <mpi.h>
#include <set>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <unordered_map>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
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
class BiCGSTABSolver {
public:
  BiCGSTABSolver(MPI_Comm m_comm, LocalSpMatDnVec &LocalLS, const int BLEN,
                 const bool bMeanConstraint, const std::vector<double> &P_inv);
  ~BiCGSTABSolver();
  void solveWithUpdate(const double max_error, const double max_rel_error,
                       const int max_restarts);
  void solveNoUpdate(const double max_error, const double max_rel_error,
                     const int max_restarts);

private:
  void freeLast();
  void updateAll();
  void updateVec();
  void main(const double max_error, const double max_rel_error,
            const int restarts);
  void hd_cusparseSpMV(double *d_op, cusparseDnVecDescr_t spDescrLocOp,
                       cusparseDnVecDescr_t spDescrBdOp, double *d_res,
                       cusparseDnVecDescr_t Res);
  cudaStream_t solver_stream_;
  cudaStream_t copy_stream_;
  cudaEvent_t sync_event_;
  cublasHandle_t cublas_handle_;
  cusparseHandle_t cusparse_handle_;
  bool dirty_ = false;
  int rank_;
  MPI_Comm m_comm_;
  int comm_size_;
  int m_;
  int halo_;
  int loc_nnz_;
  int bd_nnz_;
  int hd_m_;
  const int BLEN_;
  const bool bMeanConstraint_;
  int bMeanRow_;
  LocalSpMatDnVec &LocalLS_;
  int send_buff_sz_;
  int *d_send_pack_idx_;
  double *d_send_buff_;
  double *h_send_buff_;
  double *h_recv_buff_;
  double *d_consts_;
  const double *d_eye_;
  const double *d_nye_;
  const double *d_nil_;
  BiCGSTABScalars *h_coeffs_;
  BiCGSTABScalars *d_coeffs_;
  double *dloc_cooValA_;
  int *dloc_cooRowA_;
  int *dloc_cooColA_;
  double *dbd_cooValA_;
  int *dbd_cooRowA_;
  int *dbd_cooColA_;
  double *d_x_;
  double *d_x_opt_;
  double *d_r_;
  double *d_P_inv_;
  size_t red_temp_storage_bytes_;
  void *d_red_temp_storage_;
  double *d_red_;
  double *d_red_res_;
  double *d_h2_;
  double *d_rhat_;
  double *d_p_;
  double *d_nu_;
  double *d_t_;
  double *d_z_;
  cusparseSpMatDescr_t spDescrLocA_;
  cusparseSpMatDescr_t spDescrBdA_;
  cusparseDnVecDescr_t spDescrNu_;
  cusparseDnVecDescr_t spDescrT_;
  cusparseDnVecDescr_t spDescrLocZ_;
  cusparseDnVecDescr_t spDescrBdZ_;
  size_t locSpMVBuffSz_;
  void *locSpMVBuff_;
  size_t bdSpMVBuffSz_;
  void *bdSpMVBuff_;
};
BiCGSTABSolver::BiCGSTABSolver(MPI_Comm m_comm, LocalSpMatDnVec &LocalLS,
                               const int BLEN, const bool bMeanConstraint,
                               const std::vector<double> &P_inv)
    : m_comm_(m_comm), BLEN_(BLEN), bMeanConstraint_(bMeanConstraint),
      LocalLS_(LocalLS) {
  MPI_Comm_rank(m_comm_, &rank_);
  MPI_Comm_size(m_comm_, &comm_size_);
  cudaStreamCreate(&solver_stream_);
  cudaStreamCreate(&copy_stream_);
  cudaEventCreate(&sync_event_);
  cublasCreate(&cublas_handle_);
  cusparseCreate(&cusparse_handle_);
  cublasSetStream(cublas_handle_, solver_stream_);
  cusparseSetStream(cusparse_handle_, solver_stream_);
  cublasSetPointerMode(cublas_handle_, CUBLAS_POINTER_MODE_DEVICE);
  cusparseSetPointerMode(cusparse_handle_, CUSPARSE_POINTER_MODE_DEVICE);
  double h_consts[3] = {1., -1., 0.};
  cudaMalloc(&d_consts_, 3 * sizeof(double));
  cudaMemcpyAsync(d_consts_, h_consts, 3 * sizeof(double),
                  cudaMemcpyHostToDevice, solver_stream_);
  d_eye_ = d_consts_;
  d_nye_ = d_consts_ + 1;
  d_nil_ = d_consts_ + 2;
  cudaMalloc(&d_coeffs_, sizeof(BiCGSTABScalars));
  cudaMallocHost(&h_coeffs_, sizeof(BiCGSTABScalars));
  cudaMalloc(&d_P_inv_, BLEN_ * BLEN_ * sizeof(double));
  cudaMemcpyAsync(d_P_inv_, P_inv.data(), BLEN_ * BLEN_ * sizeof(double),
                  cudaMemcpyHostToDevice, solver_stream_);
}
BiCGSTABSolver::~BiCGSTABSolver() {
  this->freeLast();
  cudaFree(d_P_inv_);
  cudaFree(d_consts_);
  cudaFree(d_coeffs_);
  cudaFreeHost(h_coeffs_);
  cublasDestroy(cublas_handle_);
  cusparseDestroy(cusparse_handle_);
  cudaEventDestroy(sync_event_);
  cudaStreamDestroy(copy_stream_);
  cudaStreamDestroy(solver_stream_);
}
void BiCGSTABSolver::solveWithUpdate(const double max_error,
                                     const double max_rel_error,
                                     const int max_restarts) {
  this->updateAll();
  this->main(max_error, max_rel_error, max_restarts);
}
void BiCGSTABSolver::solveNoUpdate(const double max_error,
                                   const double max_rel_error,
                                   const int max_restarts) {
  this->updateVec();
  this->main(max_error, max_rel_error, max_restarts);
}
void BiCGSTABSolver::freeLast() {
  if (dirty_) {
    cudaFree(dloc_cooValA_);
    cudaFree(dloc_cooRowA_);
    cudaFree(dloc_cooColA_);
    cudaFree(d_x_);
    cudaFree(d_x_opt_);
    cudaFree(d_r_);
    cudaFree(d_rhat_);
    cudaFree(d_p_);
    cudaFree(d_nu_);
    cudaFree(d_t_);
    cudaFree(d_z_);
    cudaFree(locSpMVBuff_);
    cusparseDestroySpMat(spDescrLocA_);
    cusparseDestroyDnVec(spDescrNu_);
    cusparseDestroyDnVec(spDescrT_);
    cusparseDestroyDnVec(spDescrLocZ_);
    if (comm_size_ > 1) {
      cudaFree(d_send_pack_idx_);
      cudaFree(d_send_buff_);
      cudaFreeHost(h_send_buff_);
      cudaFreeHost(h_recv_buff_);
      cudaFree(dbd_cooValA_);
      cudaFree(dbd_cooRowA_);
      cudaFree(dbd_cooColA_);
      cudaFree(bdSpMVBuff_);
      cusparseDestroySpMat(spDescrBdA_);
      cusparseDestroyDnVec(spDescrBdZ_);
    }
    if (bMeanConstraint_) {
      cudaFree(d_h2_);
      cudaFree(d_red_);
      cudaFree(d_red_res_);
      cudaFree(d_red_temp_storage_);
    }
  }
  dirty_ = true;
}
void BiCGSTABSolver::updateAll() {
  this->freeLast();
  m_ = LocalLS_.m_;
  halo_ = LocalLS_.halo_;
  hd_m_ = m_ + halo_;
  loc_nnz_ = LocalLS_.loc_nnz_;
  bd_nnz_ = LocalLS_.bd_nnz_;
  send_buff_sz_ = LocalLS_.send_pack_idx_.size();
  const int Nblocks = m_ / BLEN_;
  bMeanRow_ = LocalLS_.bMeanRow_;
  cudaMalloc(&dloc_cooValA_, loc_nnz_ * sizeof(double));
  cudaMalloc(&dloc_cooRowA_, loc_nnz_ * sizeof(int));
  cudaMalloc(&dloc_cooColA_, loc_nnz_ * sizeof(int));
  cudaMalloc(&d_x_, m_ * sizeof(double));
  cudaMalloc(&d_x_opt_, m_ * sizeof(double));
  cudaMalloc(&d_r_, m_ * sizeof(double));
  cudaMalloc(&d_rhat_, m_ * sizeof(double));
  cudaMalloc(&d_p_, m_ * sizeof(double));
  cudaMalloc(&d_nu_, m_ * sizeof(double));
  cudaMalloc(&d_t_, m_ * sizeof(double));
  cudaMalloc(&d_z_, hd_m_ * sizeof(double));
  if (bMeanConstraint_) {
    cudaMalloc(&d_h2_, Nblocks * sizeof(double));
    cudaMalloc(&d_red_, m_ * sizeof(double));
    cudaMalloc(&d_red_res_, sizeof(double));
    d_red_temp_storage_ = NULL;
    red_temp_storage_bytes_ = 0;
    cub::DeviceReduce::Sum<double *, double *>(d_red_temp_storage_,
                                               red_temp_storage_bytes_, d_red_,
                                               d_red_res_, m_, solver_stream_);
    cudaMalloc(&d_red_temp_storage_, red_temp_storage_bytes_);
  }
  if (comm_size_ > 1) {
    cudaMalloc(&d_send_pack_idx_, send_buff_sz_ * sizeof(int));
    cudaMalloc(&d_send_buff_, send_buff_sz_ * sizeof(double));
    cudaMallocHost(&h_send_buff_, send_buff_sz_ * sizeof(double));
    cudaMallocHost(&h_recv_buff_, halo_ * sizeof(double));
    cudaMalloc(&dbd_cooValA_, bd_nnz_ * sizeof(double));
    cudaMalloc(&dbd_cooRowA_, bd_nnz_ * sizeof(int));
    cudaMalloc(&dbd_cooColA_, bd_nnz_ * sizeof(int));
  }
  cudaMemcpyAsync(dloc_cooValA_, LocalLS_.loc_cooValA_.data(),
                  loc_nnz_ * sizeof(double), cudaMemcpyHostToDevice,
                  solver_stream_);
  cudaMemcpyAsync(dloc_cooRowA_, LocalLS_.loc_cooRowA_int_.data(),
                  loc_nnz_ * sizeof(int), cudaMemcpyHostToDevice,
                  solver_stream_);
  cudaMemcpyAsync(dloc_cooColA_, LocalLS_.loc_cooColA_int_.data(),
                  loc_nnz_ * sizeof(int), cudaMemcpyHostToDevice,
                  solver_stream_);
  if (comm_size_ > 1) {
    cudaMemcpyAsync(d_send_pack_idx_, LocalLS_.send_pack_idx_.data(),
                    send_buff_sz_ * sizeof(int), cudaMemcpyHostToDevice,
                    solver_stream_);
    cudaMemcpyAsync(dbd_cooValA_, LocalLS_.bd_cooValA_.data(),
                    bd_nnz_ * sizeof(double), cudaMemcpyHostToDevice,
                    solver_stream_);
    cudaMemcpyAsync(dbd_cooRowA_, LocalLS_.bd_cooRowA_int_.data(),
                    bd_nnz_ * sizeof(int), cudaMemcpyHostToDevice,
                    solver_stream_);
    cudaMemcpyAsync(dbd_cooColA_, LocalLS_.bd_cooColA_int_.data(),
                    bd_nnz_ * sizeof(int), cudaMemcpyHostToDevice,
                    solver_stream_);
  }
  if (bMeanConstraint_)
    cudaMemcpyAsync(d_h2_, LocalLS_.h2_.data(), Nblocks * sizeof(double),
                    cudaMemcpyHostToDevice, solver_stream_);
  cusparseCreateCoo(&spDescrLocA_, m_, m_, loc_nnz_, dloc_cooRowA_,
                    dloc_cooColA_, dloc_cooValA_, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  cusparseCreateDnVec(&spDescrNu_, m_, d_nu_, CUDA_R_64F);
  cusparseCreateDnVec(&spDescrT_, m_, d_t_, CUDA_R_64F);
  cusparseCreateDnVec(&spDescrLocZ_, m_, d_z_, CUDA_R_64F);
  cusparseSpMV_bufferSize(cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          d_eye_, spDescrLocA_, spDescrLocZ_, d_nil_,
                          spDescrNu_, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                          &locSpMVBuffSz_);
  cudaMalloc(&locSpMVBuff_, locSpMVBuffSz_ * sizeof(char));
  if (comm_size_ > 1) {
    cusparseCreateCoo(&spDescrBdA_, m_, hd_m_, bd_nnz_, dbd_cooRowA_,
                      dbd_cooColA_, dbd_cooValA_, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseCreateDnVec(&spDescrBdZ_, hd_m_, d_z_, CUDA_R_64F);
    cusparseSpMV_bufferSize(cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            d_eye_, spDescrBdA_, spDescrBdZ_, d_eye_,
                            spDescrNu_, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                            &bdSpMVBuffSz_);
    cudaMalloc(&bdSpMVBuff_, bdSpMVBuffSz_ * sizeof(char));
  }
  this->updateVec();
}
void BiCGSTABSolver::updateVec() {
  cudaMemcpyAsync(d_x_, LocalLS_.x_.data(), m_ * sizeof(double),
                  cudaMemcpyHostToDevice, solver_stream_);
  cudaMemcpyAsync(d_r_, LocalLS_.b_.data(), m_ * sizeof(double),
                  cudaMemcpyHostToDevice, solver_stream_);
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
__global__ void send_buff_pack(int buff_sz, const int *const pack_idx,
                               double *const buff, const double *const source) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < buff_sz;
       i += blockDim.x * gridDim.x)
    buff[i] = source[pack_idx[i]];
}
void BiCGSTABSolver::hd_cusparseSpMV(double *d_op_hd,
                                     cusparseDnVecDescr_t spDescrLocOp,
                                     cusparseDnVecDescr_t spDescrBdOp,
                                     double *d_res_hd,
                                     cusparseDnVecDescr_t spDescrRes) {
  const std::vector<int> &recv_ranks = LocalLS_.recv_ranks_;
  const std::vector<int> &recv_offset = LocalLS_.recv_offset_;
  const std::vector<int> &recv_sz = LocalLS_.recv_sz_;
  const std::vector<int> &send_ranks = LocalLS_.send_ranks_;
  const std::vector<int> &send_offset = LocalLS_.send_offset_;
  const std::vector<int> &send_sz = LocalLS_.send_sz_;
  if (comm_size_ > 1) {
    send_buff_pack<<<8 * 56, 32, 0, solver_stream_>>>(
        send_buff_sz_, d_send_pack_idx_, d_send_buff_, d_op_hd);
    cudaGetLastError();
    cudaEventRecord(sync_event_, solver_stream_);
  }
  cusparseSpMV(cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, d_eye_,
               spDescrLocA_, spDescrLocOp, d_nil_, spDescrRes, CUDA_R_64F,
               CUSPARSE_SPMV_ALG_DEFAULT, locSpMVBuff_);
  if (comm_size_ > 1) {
    cudaStreamWaitEvent(copy_stream_, sync_event_, 0);
    cudaMemcpyAsync(h_send_buff_, d_send_buff_, send_buff_sz_ * sizeof(double),
                    cudaMemcpyDeviceToHost, copy_stream_);
    cudaStreamSynchronize(copy_stream_);
    std::vector<MPI_Request> recv_requests(recv_ranks.size());
    for (size_t i(0); i < recv_ranks.size(); i++)
      MPI_Irecv(&h_recv_buff_[recv_offset[i]], recv_sz[i], MPI_DOUBLE,
                recv_ranks[i], 978, m_comm_, &recv_requests[i]);
    std::vector<MPI_Request> send_requests(send_ranks.size());
    for (size_t i(0); i < send_ranks.size(); i++)
      MPI_Isend(&h_send_buff_[send_offset[i]], send_sz[i], MPI_DOUBLE,
                send_ranks[i], 978, m_comm_, &send_requests[i]);
    MPI_Waitall(send_ranks.size(), send_requests.data(), MPI_STATUS_IGNORE);
    MPI_Waitall(recv_ranks.size(), recv_requests.data(), MPI_STATUS_IGNORE);
    cudaMemcpyAsync(&d_op_hd[m_], h_recv_buff_, halo_ * sizeof(double),
                    cudaMemcpyHostToDevice, solver_stream_);
    cusparseSpMV(cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, d_eye_,
                 spDescrBdA_, spDescrBdOp, d_eye_, spDescrRes, CUDA_R_64F,
                 CUSPARSE_SPMV_ALG_DEFAULT, bdSpMVBuff_);
  }
  if (bMeanConstraint_) {
    cudaMemcpyAsync(d_red_, d_op_hd, m_ * sizeof(double),
                    cudaMemcpyDeviceToDevice, solver_stream_);
    blockDscal<<<8 * 56, 128, 0, solver_stream_>>>(m_, BLEN_, d_h2_, d_red_);
    cudaGetLastError();
    cub::DeviceReduce::Sum<double *, double *>(d_red_temp_storage_,
                                               red_temp_storage_bytes_, d_red_,
                                               d_red_res_, m_, solver_stream_);
    double h_red_res;
    cudaMemcpyAsync(&h_red_res, d_red_res_, sizeof(double),
                    cudaMemcpyDeviceToHost, solver_stream_);
    cudaStreamSynchronize(solver_stream_);
    MPI_Allreduce(MPI_IN_PLACE, &h_red_res, 1, MPI_DOUBLE, MPI_SUM, m_comm_);
    if (bMeanRow_ >= 0)
      cudaMemcpyAsync(&d_res_hd[bMeanRow_], &h_red_res, sizeof(double),
                      cudaMemcpyHostToDevice, solver_stream_);
  }
}
void BiCGSTABSolver::main(const double max_error, const double max_rel_error,
                          const int max_restarts) {
  double error = 1e50;
  double error_init = 1e50;
  double error_opt = 1e50;
  int restarts = 0;
  *h_coeffs_ = {1., 1., 1., 1e-21, 1., 1., 0., 0., 0};
  cudaMemcpyAsync(d_coeffs_, h_coeffs_, sizeof(BiCGSTABScalars),
                  cudaMemcpyHostToDevice, solver_stream_);
  cudaMemcpyAsync(d_z_, d_x_, m_ * sizeof(double), cudaMemcpyDeviceToDevice,
                  solver_stream_);
  hd_cusparseSpMV(d_z_, spDescrLocZ_, spDescrBdZ_, d_nu_, spDescrNu_);
  cublasDaxpy(cublas_handle_, m_, d_nye_, d_nu_, 1, d_r_, 1);
  cublasIdamax(cublas_handle_, m_, d_nu_, 1, &(d_coeffs_->amax_idx));
  set_amax<<<1, 1, 0, solver_stream_>>>(&(d_coeffs_->buff_1),
                                        &(d_coeffs_->amax_idx), d_nu_);
  cudaGetLastError();
  cublasIdamax(cublas_handle_, m_, d_r_, 1, &(d_coeffs_->amax_idx));
  set_amax<<<1, 1, 0, solver_stream_>>>(&(d_coeffs_->buff_2),
                                        &(d_coeffs_->amax_idx), d_r_);
  cudaGetLastError();
  cudaMemcpyAsync(&(h_coeffs_->buff_1), &(d_coeffs_->buff_1),
                  2 * sizeof(double), cudaMemcpyDeviceToHost, solver_stream_);
  cudaStreamSynchronize(solver_stream_);
  MPI_Allreduce(MPI_IN_PLACE, &(h_coeffs_->buff_1), 2, MPI_DOUBLE, MPI_MAX,
                m_comm_);
  error = h_coeffs_->buff_2;
  error_init = error;
  error_opt = error;
  cudaMemcpyAsync(d_x_opt_, d_x_, m_ * sizeof(double), cudaMemcpyDeviceToDevice,
                  solver_stream_);
  cudaMemcpyAsync(d_rhat_, d_r_, m_ * sizeof(double), cudaMemcpyDeviceToDevice,
                  solver_stream_);
  cudaMemsetAsync(d_nu_, 0, m_ * sizeof(double), solver_stream_);
  cudaMemsetAsync(d_p_, 0, m_ * sizeof(double), solver_stream_);
  const size_t max_iter = 1000;
  for (size_t k(0); k < max_iter; k++) {
    cublasDdot(cublas_handle_, m_, d_rhat_, 1, d_r_, 1, &(d_coeffs_->rho_curr));
    cublasDnrm2(cublas_handle_, m_, d_r_, 1, &(d_coeffs_->buff_1));
    cublasDnrm2(cublas_handle_, m_, d_rhat_, 1, &(d_coeffs_->buff_2));
    cudaMemcpyAsync(&(h_coeffs_->rho_curr), &(d_coeffs_->rho_curr),
                    3 * sizeof(double), cudaMemcpyDeviceToHost, solver_stream_);
    cudaStreamSynchronize(solver_stream_);
    h_coeffs_->buff_1 *= h_coeffs_->buff_1;
    h_coeffs_->buff_2 *= h_coeffs_->buff_2;
    MPI_Allreduce(MPI_IN_PLACE, &(h_coeffs_->rho_curr), 3, MPI_DOUBLE, MPI_SUM,
                  m_comm_);
    cudaMemcpyAsync(&(d_coeffs_->rho_curr), &(h_coeffs_->rho_curr),
                    sizeof(double), cudaMemcpyHostToDevice, solver_stream_);
    const bool serious_breakdown =
        h_coeffs_->rho_curr * h_coeffs_->rho_curr <
        1e-16 * h_coeffs_->buff_1 * h_coeffs_->buff_2;
    set_beta<<<1, 1, 0, solver_stream_>>>(d_coeffs_);
    cudaGetLastError();
    if (serious_breakdown && max_restarts > 0) {
      restarts++;
      if (restarts >= max_restarts) {
        break;
      }
      cudaMemcpyAsync(d_rhat_, d_r_, m_ * sizeof(double),
                      cudaMemcpyDeviceToDevice, solver_stream_);
      cublasDnrm2(cublas_handle_, m_, d_rhat_, 1, &(d_coeffs_->rho_curr));
      cudaMemcpyAsync(&(h_coeffs_->rho_curr), &(d_coeffs_->rho_curr),
                      sizeof(double), cudaMemcpyDeviceToHost, solver_stream_);
      cudaStreamSynchronize(solver_stream_);
      h_coeffs_->rho_curr *= h_coeffs_->rho_curr;
      MPI_Allreduce(MPI_IN_PLACE, &(h_coeffs_->rho_curr), 1, MPI_DOUBLE,
                    MPI_SUM, m_comm_);
      cudaMemcpyAsync(&(d_coeffs_->rho_curr), &(h_coeffs_->rho_curr),
                      sizeof(double), cudaMemcpyHostToDevice, solver_stream_);
      cudaMemsetAsync(d_nu_, 0, m_ * sizeof(double), solver_stream_);
      cudaMemsetAsync(d_p_, 0, m_ * sizeof(double), solver_stream_);
      breakdown_update<<<1, 1, 0, solver_stream_>>>(d_coeffs_);
      cudaGetLastError();
    }
    set_negative<<<1, 1, 0, solver_stream_>>>(&(d_coeffs_->buff_1),
                                              &(d_coeffs_->omega));
    cudaGetLastError();
    cublasDaxpy(cublas_handle_, m_, &(d_coeffs_->buff_1), d_nu_, 1, d_p_, 1);
    cublasDscal(cublas_handle_, m_, &(d_coeffs_->beta), d_p_, 1);
    cublasDaxpy(cublas_handle_, m_, d_eye_, d_r_, 1, d_p_, 1);
    cublasDgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N, BLEN_, m_ / BLEN_,
                BLEN_, d_eye_, d_P_inv_, BLEN_, d_p_, BLEN_, d_nil_, d_z_,
                BLEN_);
    hd_cusparseSpMV(d_z_, spDescrLocZ_, spDescrBdZ_, d_nu_, spDescrNu_);
    cublasDdot(cublas_handle_, m_, d_rhat_, 1, d_nu_, 1, &(d_coeffs_->buff_1));
    cudaMemcpyAsync(&(h_coeffs_->buff_1), &(d_coeffs_->buff_1), sizeof(double),
                    cudaMemcpyDeviceToHost, solver_stream_);
    cudaStreamSynchronize(solver_stream_);
    MPI_Allreduce(MPI_IN_PLACE, &(h_coeffs_->buff_1), 1, MPI_DOUBLE, MPI_SUM,
                  m_comm_);
    cudaMemcpyAsync(&(d_coeffs_->buff_1), &(h_coeffs_->buff_1), sizeof(double),
                    cudaMemcpyHostToDevice, solver_stream_);
    set_alpha<<<1, 1, 0, solver_stream_>>>(d_coeffs_);
    cudaGetLastError();
    cublasDaxpy(cublas_handle_, m_, &(d_coeffs_->alpha), d_z_, 1, d_x_, 1);
    set_negative<<<1, 1, 0, solver_stream_>>>(&(d_coeffs_->buff_1),
                                              &(d_coeffs_->alpha));
    cudaGetLastError();
    cublasDaxpy(cublas_handle_, m_, &(d_coeffs_->buff_1), d_nu_, 1, d_r_, 1);
    cublasDgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N, BLEN_, m_ / BLEN_,
                BLEN_, d_eye_, d_P_inv_, BLEN_, d_r_, BLEN_, d_nil_, d_z_,
                BLEN_);
    hd_cusparseSpMV(d_z_, spDescrLocZ_, spDescrBdZ_, d_t_, spDescrT_);
    cublasDdot(cublas_handle_, m_, d_t_, 1, d_r_, 1, &(d_coeffs_->buff_1));
    cublasDnrm2(cublas_handle_, m_, d_t_, 1, &(d_coeffs_->buff_2));
    set_squared<<<1, 1, 0, solver_stream_>>>(&(d_coeffs_->buff_2));
    cudaGetLastError();
    cudaMemcpyAsync(&(h_coeffs_->buff_1), &(d_coeffs_->buff_1),
                    2 * sizeof(double), cudaMemcpyDeviceToHost, solver_stream_);
    cudaStreamSynchronize(solver_stream_);
    MPI_Allreduce(MPI_IN_PLACE, &(h_coeffs_->buff_1), 2, MPI_DOUBLE, MPI_SUM,
                  m_comm_);
    cudaMemcpyAsync(&(d_coeffs_->buff_1), &(h_coeffs_->buff_1),
                    2 * sizeof(double), cudaMemcpyHostToDevice, solver_stream_);
    set_omega<<<1, 1, 0, solver_stream_>>>(d_coeffs_);
    cudaGetLastError();
    cublasDaxpy(cublas_handle_, m_, &(d_coeffs_->omega), d_z_, 1, d_x_, 1);
    set_negative<<<1, 1, 0, solver_stream_>>>(&(d_coeffs_->buff_1),
                                              &(d_coeffs_->omega));
    cudaGetLastError();
    cublasDaxpy(cublas_handle_, m_, &(d_coeffs_->buff_1), d_t_, 1, d_r_, 1);
    cublasIdamax(cublas_handle_, m_, d_r_, 1, &(d_coeffs_->amax_idx));
    set_amax<<<1, 1, 0, solver_stream_>>>(&(d_coeffs_->buff_1),
                                          &(d_coeffs_->amax_idx), d_r_);
    cudaGetLastError();
    cudaMemcpyAsync(&error, &(d_coeffs_->buff_1), sizeof(double),
                    cudaMemcpyDeviceToHost, solver_stream_);
    cudaMemcpyAsync(h_coeffs_, d_coeffs_, sizeof(BiCGSTABScalars),
                    cudaMemcpyDeviceToHost, solver_stream_);
    cudaStreamSynchronize(solver_stream_);
    MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, m_comm_);
    if (error < error_opt) {
      error_opt = error;
      cudaMemcpyAsync(d_x_opt_, d_x_, m_ * sizeof(double),
                      cudaMemcpyDeviceToDevice, solver_stream_);
      if ((error <= max_error) || (error / error_init <= max_rel_error)) {
        break;
      }
    }
    set_rho<<<1, 1, 0, solver_stream_>>>(d_coeffs_);
    cudaGetLastError();
  }
  cudaMemcpyAsync(LocalLS_.x_.data(), d_x_opt_, m_ * sizeof(double),
                  cudaMemcpyDeviceToHost, solver_stream_);
}
LocalSpMatDnVec::LocalSpMatDnVec(MPI_Comm m_comm, const int BLEN,
                                 const bool bMeanConstraint,
                                 const std::vector<double> &P_inv)
    : m_comm_(m_comm), BLEN_(BLEN) {
  MPI_Comm_rank(m_comm_, &rank_);
  MPI_Comm_size(m_comm_, &comm_size_);
  bd_recv_set_.resize(comm_size_);
  bd_recv_vec_.resize(comm_size_);
  recv_ranks_.reserve(comm_size_);
  recv_offset_.reserve(comm_size_);
  recv_sz_.reserve(comm_size_);
  send_ranks_.reserve(comm_size_);
  send_offset_.reserve(comm_size_);
  send_sz_.reserve(comm_size_);
  solver_ = std::make_unique<BiCGSTABSolver>(m_comm, *this, BLEN,
                                             bMeanConstraint, P_inv);
}
LocalSpMatDnVec::~LocalSpMatDnVec() {}
void LocalSpMatDnVec::reserve(const int N) {
  m_ = N;
  bMeanRow_ = -1;
  for (size_t i(0); i < bd_recv_set_.size(); i++)
    bd_recv_set_[i].clear();
  loc_cooValA_.clear();
  loc_cooValA_.reserve(6 * N);
  loc_cooRowA_long_.clear();
  loc_cooRowA_long_.reserve(6 * N);
  loc_cooColA_long_.clear();
  loc_cooColA_long_.reserve(6 * N);
  bd_cooValA_.clear();
  bd_cooValA_.reserve(N);
  bd_cooRowA_long_.clear();
  bd_cooRowA_long_.reserve(N);
  bd_cooColA_long_.clear();
  bd_cooColA_long_.reserve(N);
  x_.resize(N);
  b_.resize(N);
  h2_.resize(N / BLEN_);
}
void LocalSpMatDnVec::cooPushBackVal(const double val, const long long row,
                                     const long long col) {
  loc_cooValA_.push_back(val);
  loc_cooRowA_long_.push_back(row);
  loc_cooColA_long_.push_back(col);
}
void LocalSpMatDnVec::cooPushBackRow(const SpRowInfo &row) {
  for (const auto &[col_idx, val] : row.loc_colval_) {
    loc_cooValA_.push_back(val);
    loc_cooRowA_long_.push_back(row.idx_);
    loc_cooColA_long_.push_back(col_idx);
  }
  if (!row.neirank_cols_.empty()) {
    for (const auto &[col_idx, val] : row.bd_colval_) {
      bd_cooValA_.push_back(val);
      bd_cooRowA_long_.push_back(row.idx_);
      bd_cooColA_long_.push_back(col_idx);
    }
    for (const auto &[rank, col_idx] : row.neirank_cols_) {
      bd_recv_set_[rank].insert(col_idx);
    }
  }
}
void LocalSpMatDnVec::make(const std::vector<long long> &Nrows_xcumsum) {
  loc_nnz_ = loc_cooValA_.size();
  bd_nnz_ = bd_cooValA_.size();
  halo_ = 0;
  std::vector<int> send_sz_allranks(comm_size_);
  std::vector<int> recv_sz_allranks(comm_size_);
  for (int r(0); r < comm_size_; r++)
    recv_sz_allranks[r] = bd_recv_set_[r].size();
  MPI_Alltoall(recv_sz_allranks.data(), 1, MPI_INT, send_sz_allranks.data(), 1,
               MPI_INT, m_comm_);
  recv_ranks_.clear();
  recv_offset_.clear();
  recv_sz_.clear();
  int offset = 0;
  for (int r(0); r < comm_size_; r++) {
    if (r != rank_ && recv_sz_allranks[r] > 0) {
      recv_ranks_.push_back(r);
      recv_offset_.push_back(offset);
      recv_sz_.push_back(recv_sz_allranks[r]);
      offset += recv_sz_allranks[r];
    }
  }
  halo_ = offset;
  send_ranks_.clear();
  send_offset_.clear();
  send_sz_.clear();
  offset = 0;
  for (int r(0); r < comm_size_; r++) {
    if (r != rank_ && send_sz_allranks[r] > 0) {
      send_ranks_.push_back(r);
      send_offset_.push_back(offset);
      send_sz_.push_back(send_sz_allranks[r]);
      offset += send_sz_allranks[r];
    }
  }
  std::vector<long long> send_pack_idx_long(offset);
  send_pack_idx_.resize(offset);
  std::vector<MPI_Request> recv_requests(send_ranks_.size());
  for (size_t i(0); i < send_ranks_.size(); i++)
    MPI_Irecv(&send_pack_idx_long[send_offset_[i]], send_sz_[i], MPI_LONG_LONG,
              send_ranks_[i], 546, m_comm_, &recv_requests[i]);
  std::vector<long long> recv_idx_list(halo_);
  std::vector<MPI_Request> send_requests(recv_ranks_.size());
  for (size_t i(0); i < recv_ranks_.size(); i++) {
    std::copy(bd_recv_set_[recv_ranks_[i]].begin(),
              bd_recv_set_[recv_ranks_[i]].end(),
              &recv_idx_list[recv_offset_[i]]);
    MPI_Isend(&recv_idx_list[recv_offset_[i]], recv_sz_[i], MPI_LONG_LONG,
              recv_ranks_[i], 546, m_comm_, &send_requests[i]);
  }
  const long long shift = -Nrows_xcumsum[rank_];
  loc_cooRowA_int_.resize(loc_nnz_);
  loc_cooColA_int_.resize(loc_nnz_);
  bd_cooRowA_int_.resize(bd_nnz_);
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < loc_nnz_; i++)
      loc_cooRowA_int_[i] = (int)(loc_cooRowA_long_[i] + shift);
#pragma omp for
    for (int i = 0; i < loc_nnz_; i++)
      loc_cooColA_int_[i] = (int)(loc_cooColA_long_[i] + shift);
#pragma omp for
    for (int i = 0; i < bd_nnz_; i++)
      bd_cooRowA_int_[i] = (int)(bd_cooRowA_long_[i] + shift);
  }
  MPI_Waitall(send_ranks_.size(), recv_requests.data(), MPI_STATUS_IGNORE);
  std::unordered_map<long long, int> bd_reindex_map;
  bd_reindex_map.reserve(halo_);
  for (int i(0); i < halo_; i++)
    bd_reindex_map[recv_idx_list[i]] = m_ + i;
  bd_cooColA_int_.resize(bd_nnz_);
  for (int i = 0; i < bd_nnz_; i++)
    bd_cooColA_int_[i] = bd_reindex_map[bd_cooColA_long_[i]];
  MPI_Waitall(recv_ranks_.size(), send_requests.data(), MPI_STATUS_IGNORE);
#pragma omp parallel for
  for (size_t i = 0; i < send_pack_idx_.size(); i++)
    send_pack_idx_[i] = (int)(send_pack_idx_long[i] + shift);
}
void LocalSpMatDnVec::solveWithUpdate(const double max_error,
                                      const double max_rel_error,
                                      const int max_restarts) {
  solver_->solveWithUpdate(max_error, max_rel_error, max_restarts);
}
void LocalSpMatDnVec::solveNoUpdate(const double max_error,
                                    const double max_rel_error,
                                    const int max_restarts) {
  solver_->solveNoUpdate(max_error, max_rel_error, max_restarts);
}
