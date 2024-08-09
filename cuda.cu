#include <map>
#include <set>
#include <vector>
#include <memory>
#include <mpi.h>
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
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <memory>
#include <unordered_map>
#include <mpi.h>
#include "cuda.h"

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#ifdef __DRIVER_TYPES_H__
static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}
#endif

#ifdef CUDA_DRIVER_API
// CUDA Driver API errors
static const char *_cudaGetErrorEnum(CUresult error) {
  static char unknown[] = "<unknown>";
  const char *ret = NULL;
  cuGetErrorName(error, &ret);
  return ret ? ret : unknown;
}
#endif

#ifdef CUBLAS_API_H_
// cuBLAS API errors
static const char *_cudaGetErrorEnum(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";

  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";

  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";

  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";

  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";

  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";

  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";

  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";

  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";

  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  }

  return "<unknown>";
}

#ifdef _CUFFT_H_
// cuFFT API errors
static const char *_cudaGetErrorEnum(cufftResult error) {
  switch (error) {
  case CUFFT_SUCCESS:
    return "CUFFT_SUCCESS";

  case CUFFT_INVALID_PLAN:
    return "CUFFT_INVALID_PLAN";

  case CUFFT_ALLOC_FAILED:
    return "CUFFT_ALLOC_FAILED";

  case CUFFT_INVALID_TYPE:
    return "CUFFT_INVALID_TYPE";

  case CUFFT_INVALID_VALUE:
    return "CUFFT_INVALID_VALUE";

  case CUFFT_INTERNAL_ERROR:
    return "CUFFT_INTERNAL_ERROR";

  case CUFFT_EXEC_FAILED:
    return "CUFFT_EXEC_FAILED";

  case CUFFT_SETUP_FAILED:
    return "CUFFT_SETUP_FAILED";

  case CUFFT_INVALID_SIZE:
    return "CUFFT_INVALID_SIZE";

  case CUFFT_UNALIGNED_DATA:
    return "CUFFT_UNALIGNED_DATA";

  case CUFFT_INCOMPLETE_PARAMETER_LIST:
    return "CUFFT_INCOMPLETE_PARAMETER_LIST";

  case CUFFT_INVALID_DEVICE:
    return "CUFFT_INVALID_DEVICE";

  case CUFFT_PARSE_ERROR:
    return "CUFFT_PARSE_ERROR";

  case CUFFT_NO_WORKSPACE:
    return "CUFFT_NO_WORKSPACE";

  case CUFFT_NOT_IMPLEMENTED:
    return "CUFFT_NOT_IMPLEMENTED";

  case CUFFT_LICENSE_ERROR:
    return "CUFFT_LICENSE_ERROR";

  case CUFFT_NOT_SUPPORTED:
    return "CUFFT_NOT_SUPPORTED";
  }

  return "<unknown>";
}
#endif

#ifdef CUSPARSEAPI
// cuSPARSE API errors
static const char *_cudaGetErrorEnum(cusparseStatus_t error) {
  switch (error) {
  case CUSPARSE_STATUS_SUCCESS:
    return "CUSPARSE_STATUS_SUCCESS";

  case CUSPARSE_STATUS_NOT_INITIALIZED:
    return "CUSPARSE_STATUS_NOT_INITIALIZED";

  case CUSPARSE_STATUS_ALLOC_FAILED:
    return "CUSPARSE_STATUS_ALLOC_FAILED";

  case CUSPARSE_STATUS_INVALID_VALUE:
    return "CUSPARSE_STATUS_INVALID_VALUE";

  case CUSPARSE_STATUS_ARCH_MISMATCH:
    return "CUSPARSE_STATUS_ARCH_MISMATCH";

  case CUSPARSE_STATUS_MAPPING_ERROR:
    return "CUSPARSE_STATUS_MAPPING_ERROR";

  case CUSPARSE_STATUS_EXECUTION_FAILED:
    return "CUSPARSE_STATUS_EXECUTION_FAILED";

  case CUSPARSE_STATUS_INTERNAL_ERROR:
    return "CUSPARSE_STATUS_INTERNAL_ERROR";

  case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
    return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

  default:
    break; // Silence -Wswitch.
  }

  return "<unknown>";
}
#endif

#ifdef CUSOLVER_COMMON_H_
// cuSOLVER API errors
static const char *_cudaGetErrorEnum(cusolverStatus_t error) {
  switch (error) {
  case CUSOLVER_STATUS_SUCCESS:
    return "CUSOLVER_STATUS_SUCCESS";
  case CUSOLVER_STATUS_NOT_INITIALIZED:
    return "CUSOLVER_STATUS_NOT_INITIALIZED";
  case CUSOLVER_STATUS_ALLOC_FAILED:
    return "CUSOLVER_STATUS_ALLOC_FAILED";
  case CUSOLVER_STATUS_INVALID_VALUE:
    return "CUSOLVER_STATUS_INVALID_VALUE";
  case CUSOLVER_STATUS_ARCH_MISMATCH:
    return "CUSOLVER_STATUS_ARCH_MISMATCH";
  case CUSOLVER_STATUS_MAPPING_ERROR:
    return "CUSOLVER_STATUS_MAPPING_ERROR";
  case CUSOLVER_STATUS_EXECUTION_FAILED:
    return "CUSOLVER_STATUS_EXECUTION_FAILED";
  case CUSOLVER_STATUS_INTERNAL_ERROR:
    return "CUSOLVER_STATUS_INTERNAL_ERROR";
  case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
    return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
  case CUSOLVER_STATUS_NOT_SUPPORTED:
    return "CUSOLVER_STATUS_NOT_SUPPORTED ";
  case CUSOLVER_STATUS_ZERO_PIVOT:
    return "CUSOLVER_STATUS_ZERO_PIVOT";
  case CUSOLVER_STATUS_INVALID_LICENSE:
    return "CUSOLVER_STATUS_INVALID_LICENSE";
  }

  return "<unknown>";
}
#endif

#ifdef CURAND_H_
// cuRAND API errors
static const char *_cudaGetErrorEnum(curandStatus_t error) {
  switch (error) {
  case CURAND_STATUS_SUCCESS:
    return "CURAND_STATUS_SUCCESS";

  case CURAND_STATUS_VERSION_MISMATCH:
    return "CURAND_STATUS_VERSION_MISMATCH";

  case CURAND_STATUS_NOT_INITIALIZED:
    return "CURAND_STATUS_NOT_INITIALIZED";

  case CURAND_STATUS_ALLOCATION_FAILED:
    return "CURAND_STATUS_ALLOCATION_FAILED";

  case CURAND_STATUS_TYPE_ERROR:
    return "CURAND_STATUS_TYPE_ERROR";

  case CURAND_STATUS_OUT_OF_RANGE:
    return "CURAND_STATUS_OUT_OF_RANGE";

  case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
    return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

  case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
    return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

  case CURAND_STATUS_LAUNCH_FAILURE:
    return "CURAND_STATUS_LAUNCH_FAILURE";

  case CURAND_STATUS_PREEXISTING_FAILURE:
    return "CURAND_STATUS_PREEXISTING_FAILURE";

  case CURAND_STATUS_INITIALIZATION_FAILED:
    return "CURAND_STATUS_INITIALIZATION_FAILED";

  case CURAND_STATUS_ARCH_MISMATCH:
    return "CURAND_STATUS_ARCH_MISMATCH";

  case CURAND_STATUS_INTERNAL_ERROR:
    return "CURAND_STATUS_INTERNAL_ERROR";
  }

  return "<unknown>";
}
#endif

#ifdef NVJPEGAPI
// nvJPEG API errors
static const char *_cudaGetErrorEnum(nvjpegStatus_t error) {
  switch (error) {
  case NVJPEG_STATUS_SUCCESS:
    return "NVJPEG_STATUS_SUCCESS";

  case NVJPEG_STATUS_NOT_INITIALIZED:
    return "NVJPEG_STATUS_NOT_INITIALIZED";

  case NVJPEG_STATUS_INVALID_PARAMETER:
    return "NVJPEG_STATUS_INVALID_PARAMETER";

  case NVJPEG_STATUS_BAD_JPEG:
    return "NVJPEG_STATUS_BAD_JPEG";

  case NVJPEG_STATUS_JPEG_NOT_SUPPORTED:
    return "NVJPEG_STATUS_JPEG_NOT_SUPPORTED";

  case NVJPEG_STATUS_ALLOCATOR_FAILURE:
    return "NVJPEG_STATUS_ALLOCATOR_FAILURE";

  case NVJPEG_STATUS_EXECUTION_FAILED:
    return "NVJPEG_STATUS_EXECUTION_FAILED";

  case NVJPEG_STATUS_ARCH_MISMATCH:
    return "NVJPEG_STATUS_ARCH_MISMATCH";

  case NVJPEG_STATUS_INTERNAL_ERROR:
    return "NVJPEG_STATUS_INTERNAL_ERROR";
  }

  return "<unknown>";
}
#endif

#ifdef NV_NPPIDEFS_H
// NPP API errors
static const char *_cudaGetErrorEnum(NppStatus error) {
  switch (error) {
  case NPP_NOT_SUPPORTED_MODE_ERROR:
    return "NPP_NOT_SUPPORTED_MODE_ERROR";

  case NPP_ROUND_MODE_NOT_SUPPORTED_ERROR:
    return "NPP_ROUND_MODE_NOT_SUPPORTED_ERROR";

  case NPP_RESIZE_NO_OPERATION_ERROR:
    return "NPP_RESIZE_NO_OPERATION_ERROR";

  case NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY:
    return "NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY";

#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) <= 0x5000

  case NPP_BAD_ARG_ERROR:
    return "NPP_BAD_ARGUMENT_ERROR";

  case NPP_COEFF_ERROR:
    return "NPP_COEFFICIENT_ERROR";

  case NPP_RECT_ERROR:
    return "NPP_RECTANGLE_ERROR";

  case NPP_QUAD_ERROR:
    return "NPP_QUADRANGLE_ERROR";

  case NPP_MEM_ALLOC_ERR:
    return "NPP_MEMORY_ALLOCATION_ERROR";

  case NPP_HISTO_NUMBER_OF_LEVELS_ERROR:
    return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR";

  case NPP_INVALID_INPUT:
    return "NPP_INVALID_INPUT";

  case NPP_POINTER_ERROR:
    return "NPP_POINTER_ERROR";

  case NPP_WARNING:
    return "NPP_WARNING";

  case NPP_ODD_ROI_WARNING:
    return "NPP_ODD_ROI_WARNING";
#else

  // These are for CUDA 5.5 or higher
  case NPP_BAD_ARGUMENT_ERROR:
    return "NPP_BAD_ARGUMENT_ERROR";

  case NPP_COEFFICIENT_ERROR:
    return "NPP_COEFFICIENT_ERROR";

  case NPP_RECTANGLE_ERROR:
    return "NPP_RECTANGLE_ERROR";

  case NPP_QUADRANGLE_ERROR:
    return "NPP_QUADRANGLE_ERROR";

  case NPP_MEMORY_ALLOCATION_ERR:
    return "NPP_MEMORY_ALLOCATION_ERROR";

  case NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR:
    return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR";

  case NPP_INVALID_HOST_POINTER_ERROR:
    return "NPP_INVALID_HOST_POINTER_ERROR";

  case NPP_INVALID_DEVICE_POINTER_ERROR:
    return "NPP_INVALID_DEVICE_POINTER_ERROR";
#endif

  case NPP_LUT_NUMBER_OF_LEVELS_ERROR:
    return "NPP_LUT_NUMBER_OF_LEVELS_ERROR";

  case NPP_TEXTURE_BIND_ERROR:
    return "NPP_TEXTURE_BIND_ERROR";

  case NPP_WRONG_INTERSECTION_ROI_ERROR:
    return "NPP_WRONG_INTERSECTION_ROI_ERROR";

  case NPP_NOT_EVEN_STEP_ERROR:
    return "NPP_NOT_EVEN_STEP_ERROR";

  case NPP_INTERPOLATION_ERROR:
    return "NPP_INTERPOLATION_ERROR";

  case NPP_RESIZE_FACTOR_ERROR:
    return "NPP_RESIZE_FACTOR_ERROR";

  case NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR:
    return "NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR";

#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) <= 0x5000

  case NPP_MEMFREE_ERR:
    return "NPP_MEMFREE_ERR";

  case NPP_MEMSET_ERR:
    return "NPP_MEMSET_ERR";

  case NPP_MEMCPY_ERR:
    return "NPP_MEMCPY_ERROR";

  case NPP_MIRROR_FLIP_ERR:
    return "NPP_MIRROR_FLIP_ERR";
#else

  case NPP_MEMFREE_ERROR:
    return "NPP_MEMFREE_ERROR";

  case NPP_MEMSET_ERROR:
    return "NPP_MEMSET_ERROR";

  case NPP_MEMCPY_ERROR:
    return "NPP_MEMCPY_ERROR";

  case NPP_MIRROR_FLIP_ERROR:
    return "NPP_MIRROR_FLIP_ERROR";
#endif

  case NPP_ALIGNMENT_ERROR:
    return "NPP_ALIGNMENT_ERROR";

  case NPP_STEP_ERROR:
    return "NPP_STEP_ERROR";

  case NPP_SIZE_ERROR:
    return "NPP_SIZE_ERROR";

  case NPP_NULL_POINTER_ERROR:
    return "NPP_NULL_POINTER_ERROR";

  case NPP_CUDA_KERNEL_EXECUTION_ERROR:
    return "NPP_CUDA_KERNEL_EXECUTION_ERROR";

  case NPP_NOT_IMPLEMENTED_ERROR:
    return "NPP_NOT_IMPLEMENTED_ERROR";

  case NPP_ERROR:
    return "NPP_ERROR";

  case NPP_SUCCESS:
    return "NPP_SUCCESS";

  case NPP_WRONG_INTERSECTION_QUAD_WARNING:
    return "NPP_WRONG_INTERSECTION_QUAD_WARNING";

  case NPP_MISALIGNED_DST_ROI_WARNING:
    return "NPP_MISALIGNED_DST_ROI_WARNING";

  case NPP_AFFINE_QUAD_INCORRECT_WARNING:
    return "NPP_AFFINE_QUAD_INCORRECT_WARNING";

  case NPP_DOUBLE_SIZE_WARNING:
    return "NPP_DOUBLE_SIZE_WARNING";

  case NPP_WRONG_INTERSECTION_ROI_WARNING:
    return "NPP_WRONG_INTERSECTION_ROI_WARNING";

#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) >= 0x6000
  /* These are 6.0 or higher */
  case NPP_LUT_PALETTE_BITSIZE_ERROR:
    return "NPP_LUT_PALETTE_BITSIZE_ERROR";

  case NPP_ZC_MODE_NOT_SUPPORTED_ERROR:
    return "NPP_ZC_MODE_NOT_SUPPORTED_ERROR";

  case NPP_QUALITY_INDEX_ERROR:
    return "NPP_QUALITY_INDEX_ERROR";

  case NPP_CHANNEL_ORDER_ERROR:
    return "NPP_CHANNEL_ORDER_ERROR";

  case NPP_ZERO_MASK_VALUE_ERROR:
    return "NPP_ZERO_MASK_VALUE_ERROR";

  case NPP_NUMBER_OF_CHANNELS_ERROR:
    return "NPP_NUMBER_OF_CHANNELS_ERROR";

  case NPP_COI_ERROR:
    return "NPP_COI_ERROR";

  case NPP_DIVISOR_ERROR:
    return "NPP_DIVISOR_ERROR";

  case NPP_CHANNEL_ERROR:
    return "NPP_CHANNEL_ERROR";

  case NPP_STRIDE_ERROR:
    return "NPP_STRIDE_ERROR";

  case NPP_ANCHOR_ERROR:
    return "NPP_ANCHOR_ERROR";

  case NPP_MASK_SIZE_ERROR:
    return "NPP_MASK_SIZE_ERROR";

  case NPP_MOMENT_00_ZERO_ERROR:
    return "NPP_MOMENT_00_ZERO_ERROR";

  case NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR:
    return "NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR";

  case NPP_THRESHOLD_ERROR:
    return "NPP_THRESHOLD_ERROR";

  case NPP_CONTEXT_MATCH_ERROR:
    return "NPP_CONTEXT_MATCH_ERROR";

  case NPP_FFT_FLAG_ERROR:
    return "NPP_FFT_FLAG_ERROR";

  case NPP_FFT_ORDER_ERROR:
    return "NPP_FFT_ORDER_ERROR";

  case NPP_SCALE_RANGE_ERROR:
    return "NPP_SCALE_RANGE_ERROR";

  case NPP_DATA_TYPE_ERROR:
    return "NPP_DATA_TYPE_ERROR";

  case NPP_OUT_OFF_RANGE_ERROR:
    return "NPP_OUT_OFF_RANGE_ERROR";

  case NPP_DIVIDE_BY_ZERO_ERROR:
    return "NPP_DIVIDE_BY_ZERO_ERROR";

  case NPP_RANGE_ERROR:
    return "NPP_RANGE_ERROR";

  case NPP_NO_MEMORY_ERROR:
    return "NPP_NO_MEMORY_ERROR";

  case NPP_ERROR_RESERVED:
    return "NPP_ERROR_RESERVED";

  case NPP_NO_OPERATION_WARNING:
    return "NPP_NO_OPERATION_WARNING";

  case NPP_DIVIDE_BY_ZERO_WARNING:
    return "NPP_DIVIDE_BY_ZERO_WARNING";
#endif

#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) >= 0x7000
  /* These are 7.0 or higher */
  case NPP_OVERFLOW_ERROR:
    return "NPP_OVERFLOW_ERROR";

  case NPP_CORRUPTED_DATA_ERROR:
    return "NPP_CORRUPTED_DATA_ERROR";
#endif
  }

  return "<unknown>";
}
#endif

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

#ifdef __DRIVER_TYPES_H__
// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file,
                               const int line) {
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err) {
    fprintf(stderr,
            "%s(%i) : getLastCudaError() CUDA error :"
            " %s : (%d) %s.\n",
            file, line, errorMessage, static_cast<int>(err),
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// This will only print the proper error string when calling cudaGetLastError
// but not exit program incase error detected.
#define printLastCudaError(msg) __printLastCudaError(msg, __FILE__, __LINE__)

inline void __printLastCudaError(const char *errorMessage, const char *file,
                                 const int line) {
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err) {
    fprintf(stderr,
            "%s(%i) : getLastCudaError() CUDA error :"
            " %s : (%d) %s.\n",
            file, line, errorMessage, static_cast<int>(err),
            cudaGetErrorString(err));
  }
}
#endif

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

// Float To Int conversion
inline int ftoi(float value) {
  return (value >= 0 ? static_cast<int>(value + 0.5)
                     : static_cast<int>(value - 0.5));
}

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM; // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192},
      {0x50, 128}, {0x52, 128}, {0x53, 128}, {0x60, 64},
      {0x61, 128}, {0x62, 128}, {0x70, 64},  {0x72, 64},
      {0x75, 64},  {0x80, 64},  {0x86, 128}, {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  printf("MapSMtoCores for SM %d.%d is undefined."
         "  Default to use %d Cores/SM\n",
         major, minor, nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
}

inline const char *_ConvertSMVer2ArchName(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine
  // the GPU Arch name)
  typedef struct {
    int SM; // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    const char *name;
  } sSMtoArchName;

  sSMtoArchName nGpuArchNameSM[] = {
      {0x30, "Kepler"},       {0x32, "Kepler"},  {0x35, "Kepler"},
      {0x37, "Kepler"},       {0x50, "Maxwell"}, {0x52, "Maxwell"},
      {0x53, "Maxwell"},      {0x60, "Pascal"},  {0x61, "Pascal"},
      {0x62, "Pascal"},       {0x70, "Volta"},   {0x72, "Xavier"},
      {0x75, "Turing"},       {0x80, "Ampere"},  {0x86, "Ampere"},
      {-1, "Graphics Device"}};

  int index = 0;

  while (nGpuArchNameSM[index].SM != -1) {
    if (nGpuArchNameSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchNameSM[index].name;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  printf("MapSMtoArchName for SM %d.%d is undefined."
         "  Default to use %s\n",
         major, minor, nGpuArchNameSM[index - 1].name);
  return nGpuArchNameSM[index - 1].name;
}
// end of GPU Architecture definitions

#ifdef __CUDA_RUNTIME_H__
// General GPU Device CUDA Initialization
inline int gpuDeviceInit(int devID) {
  int device_count;
  checkCudaErrors(cudaGetDeviceCount(&device_count));

  if (device_count == 0) {
    fprintf(stderr, "gpuDeviceInit() CUDA error: "
                    "no devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }

  if (devID < 0) {
    devID = 0;
  }

  if (devID > device_count - 1) {
    fprintf(stderr, "\n");
    fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n",
            device_count);
    fprintf(stderr,
            ">> gpuDeviceInit (-device=%d) is not a valid"
            " GPU device. <<\n",
            devID);
    fprintf(stderr, "\n");
    return -devID;
  }

  int computeMode = -1, major = 0, minor = 0;
  checkCudaErrors(
      cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, devID));
  checkCudaErrors(
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
  checkCudaErrors(
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
  if (computeMode == cudaComputeModeProhibited) {
    fprintf(stderr, "Error: device is running in <Compute Mode "
                    "Prohibited>, no threads can use cudaSetDevice().\n");
    return -1;
  }

  if (major < 1) {
    fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
    exit(EXIT_FAILURE);
  }

  checkCudaErrors(cudaSetDevice(devID));
  printf("gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID,
         _ConvertSMVer2ArchName(major, minor));

  return devID;
}

// This function returns the best GPU (with maximum GFLOPS)
inline int gpuGetMaxGflopsDeviceId() {
  int current_device = 0, sm_per_multiproc = 0;
  int max_perf_device = 0;
  int device_count = 0;
  int devices_prohibited = 0;

  uint64_t max_compute_perf = 0;
  checkCudaErrors(cudaGetDeviceCount(&device_count));

  if (device_count == 0) {
    fprintf(stderr, "gpuGetMaxGflopsDeviceId() CUDA error:"
                    " no devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }

  // Find the best CUDA capable GPU device
  current_device = 0;

  while (current_device < device_count) {
    int computeMode = -1, major = 0, minor = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode,
                                           current_device));
    checkCudaErrors(cudaDeviceGetAttribute(
        &major, cudaDevAttrComputeCapabilityMajor, current_device));
    checkCudaErrors(cudaDeviceGetAttribute(
        &minor, cudaDevAttrComputeCapabilityMinor, current_device));

    // If this GPU is not running on Compute Mode prohibited,
    // then we can add it to the list
    if (computeMode != cudaComputeModeProhibited) {
      if (major == 9999 && minor == 9999) {
        sm_per_multiproc = 1;
      } else {
        sm_per_multiproc = _ConvertSMVer2Cores(major, minor);
      }
      int multiProcessorCount = 0, clockRate = 0;
      checkCudaErrors(cudaDeviceGetAttribute(&multiProcessorCount,
                                             cudaDevAttrMultiProcessorCount,
                                             current_device));
      cudaError_t result = cudaDeviceGetAttribute(
          &clockRate, cudaDevAttrClockRate, current_device);
      if (result != cudaSuccess) {
        // If cudaDevAttrClockRate attribute is not supported we
        // set clockRate as 1, to consider GPU with most SMs and CUDA Cores.
        if (result == cudaErrorInvalidValue) {
          clockRate = 1;
        } else {
          fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \n", __FILE__,
                  __LINE__, static_cast<unsigned int>(result),
                  _cudaGetErrorEnum(result));
          exit(EXIT_FAILURE);
        }
      }
      uint64_t compute_perf =
          (uint64_t)multiProcessorCount * sm_per_multiproc * clockRate;

      if (compute_perf > max_compute_perf) {
        max_compute_perf = compute_perf;
        max_perf_device = current_device;
      }
    } else {
      devices_prohibited++;
    }

    ++current_device;
  }

  if (devices_prohibited == device_count) {
    fprintf(stderr, "gpuGetMaxGflopsDeviceId() CUDA error:"
                    " all devices have compute mode prohibited.\n");
    exit(EXIT_FAILURE);
  }

  return max_perf_device;
}

inline int findIntegratedGPU() {
  int current_device = 0;
  int device_count = 0;
  int devices_prohibited = 0;

  checkCudaErrors(cudaGetDeviceCount(&device_count));

  if (device_count == 0) {
    fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }

  // Find the integrated GPU which is compute capable
  while (current_device < device_count) {
    int computeMode = -1, integrated = -1;
    checkCudaErrors(cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode,
                                           current_device));
    checkCudaErrors(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated,
                                           current_device));
    // If GPU is integrated and is not running on Compute Mode prohibited,
    // then cuda can map to GLES resource
    if (integrated && (computeMode != cudaComputeModeProhibited)) {
      checkCudaErrors(cudaSetDevice(current_device));

      int major = 0, minor = 0;
      checkCudaErrors(cudaDeviceGetAttribute(
          &major, cudaDevAttrComputeCapabilityMajor, current_device));
      checkCudaErrors(cudaDeviceGetAttribute(
          &minor, cudaDevAttrComputeCapabilityMinor, current_device));
      printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
             current_device, _ConvertSMVer2ArchName(major, minor), major,
             minor);

      return current_device;
    } else {
      devices_prohibited++;
    }

    current_device++;
  }

  if (devices_prohibited == device_count) {
    fprintf(stderr, "CUDA error:"
                    " No GLES-CUDA Interop capable GPU found.\n");
    exit(EXIT_FAILURE);
  }

  return -1;
}

// General check for CUDA GPU SM Capabilities
inline bool checkCudaCapabilities(int major_version, int minor_version) {
  int dev;
  int major = 0, minor = 0;

  checkCudaErrors(cudaGetDevice(&dev));
  checkCudaErrors(
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
  checkCudaErrors(
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));

  if ((major > major_version) ||
      (major == major_version && minor >= minor_version)) {
    printf("  Device %d: <%16s >, Compute SM %d.%d detected\n", dev,
           _ConvertSMVer2ArchName(major, minor), major, minor);
    return true;
  } else {
    printf("  No GPU device was found that can support "
           "CUDA compute capability %d.%d.\n",
           major_version, minor_version);
    return false;
  }
}
#endif

// end of CUDA Helper Functions

#endif // COMMON_HELPER_CUDA_H_

// https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf?page=1&tab=scoredesc#tab-top
template <typename... Args>
std::string string_format(const std::string &format, Args... args) {
  int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) +
               1; // Extra space for '\0'
  if (size_s <= 0) {
    throw std::runtime_error("Error during formatting.");
  }
  auto size = static_cast<size_t>(size_s);
  std::unique_ptr<char[]> buf(new char[size]);
  std::snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(buf.get(),
                     buf.get() + size - 1); // We don't want the '\0' inside
}

class DeviceProfiler {
public:
  DeviceProfiler(MPI_Comm m_comm) : m_comm_(m_comm) {
    MPI_Comm_rank(m_comm, &rank_);
  }

  ~DeviceProfiler() {
#ifdef BICGSTAB_PROFILER
    for (auto &[key, prof] : profs_) {
      checkCudaErrors(cudaEventDestroy(prof.start));
      checkCudaErrors(cudaEventDestroy(prof.stop));
    }
#endif
  }

  void startProfiler(std::string tag, cudaStream_t s) {
#ifdef BICGSTAB_PROFILER
    auto search_it = profs_.find(tag);
    if (search_it != profs_.end()) {
      assert(!profs_[tag].started);
      profs_[tag].started = true;
      checkCudaErrors(cudaEventRecord(profs_[tag].start, s));
    } else {
      ProfilerData dat = {true, 0.};
      profs_.emplace(tag, dat);
      checkCudaErrors(cudaEventCreate(&(profs_[tag].start)));
      checkCudaErrors(cudaEventCreate(&(profs_[tag].stop)));
      checkCudaErrors(cudaEventRecord(profs_[tag].start, s));
    }
#endif
  }

  void stopProfiler(std::string tag, cudaStream_t s) {
#ifdef BICGSTAB_PROFILER
    auto search_it = profs_.find(tag);
    assert(search_it != profs_.end());

    assert(profs_[tag].started);
    profs_[tag].started = false;
    checkCudaErrors(cudaEventRecord(profs_[tag].stop, s));
    checkCudaErrors(cudaEventSynchronize(profs_[tag].stop));

    float et = 0.;
    checkCudaErrors(
        cudaEventElapsedTime(&et, profs_[tag].start, profs_[tag].stop));
    profs_[tag].elapsed += et;
#endif
  }

  void print(std::string mt) {
#ifdef BICGSTAB_PROFILER
    std::string out;
    out += string_format("[DeviceProfiler] rank %i:\n", rank_);
    out += string_format("%10s:\t%.4e [ms]\n", mt.c_str(), profs_[mt].elapsed);
    for (auto &[key, prof] : profs_)
      if (key != mt)
        out += string_format("%10s:\t%.4e [ms]\t%6.2f%% of runtime\n",
                             key.c_str(), prof.elapsed,
                             profs_[key].elapsed / profs_[mt].elapsed * 100.);

    std::cout << out;
#endif
  }

protected:
  int rank_;
  MPI_Comm m_comm_;
  struct ProfilerData {
    bool started;
    float elapsed;
    cudaEvent_t start;
    cudaEvent_t stop;
  };
  std::map<std::string, ProfilerData> profs_;
};

struct BiCGSTABScalars {
  double alpha;
  double beta;
  double omega;
  double eps;
  double rho_prev;
  double rho_curr; // reductions happen along these three, make contigious
  double buff_1;
  double buff_2;
  int amax_idx;
};

class BiCGSTABSolver {
public:
  BiCGSTABSolver(MPI_Comm m_comm, LocalSpMatDnVec &LocalLS, const int BLEN,
                 const bool bMeanConstraint, const std::vector<double> &P_inv);
  ~BiCGSTABSolver();

  // Solve method with update to LHS matrix
  void solveWithUpdate(const double max_error, const double max_rel_error,
                       const int max_restarts);

  // Solve method without update to LHS matrix
  void solveNoUpdate(const double max_error, const double max_rel_error,
                     const int max_restarts);

private:
  // Method to free memory allocated by updateAll
  void freeLast();

  // Method to update LS
  void updateAll();

  // Method to set RHS and LHS vec initial guess
  void updateVec();

  // Main BiCGSTAB call
  void main(const double max_error, const double max_rel_error,
            const int restarts);

  // Haloed SpMV
  void hd_cusparseSpMV(double *d_op, // operand vec
                       cusparseDnVecDescr_t spDescrLocOp,
                       cusparseDnVecDescr_t spDescrBdOp,
                       double *d_res, // result vec
                       cusparseDnVecDescr_t Res);

  cudaStream_t solver_stream_;
  cudaStream_t copy_stream_;
  cudaEvent_t sync_event_;
  cublasHandle_t cublas_handle_;
  cusparseHandle_t cusparse_handle_;

  bool dirty_ = false; // dirty "bit" to set after first call to updateAll

  // Sparse linear system metadata
  int rank_;
  MPI_Comm m_comm_;
  int comm_size_;
  int m_;
  int halo_;
  int loc_nnz_;
  int bd_nnz_;
  int hd_m_;       // haloed number of row
  const int BLEN_; // block length (i.e no. of rows in preconditioner)
  const bool bMeanConstraint_;
  int bMeanRow_;

  // Reference to owner LocalLS
  LocalSpMatDnVec &LocalLS_;

  // Send/receive rules and buffers
  int send_buff_sz_;
  int *d_send_pack_idx_;
  double *d_send_buff_;
  double *h_send_buff_;
  double *h_recv_buff_;

  // Device-side constants
  double *d_consts_;
  const double *d_eye_;
  const double *d_nye_;
  const double *d_nil_;

  // Device-side varibles for linear system
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

  // bMeanConstraint buffers
  size_t red_temp_storage_bytes_;
  void *d_red_temp_storage_;
  double *d_red_;     // auxilary buffer for carrying out reductions
  double *d_red_res_; // auxilary buffer for reduction result
  double *d_h2_;

  // Device-side intermediate variables for BiCGSTAB
  double *d_rhat_;
  double *d_p_;
  double *d_nu_;
  double *d_t_;
  double *d_z_; // vec with halos
  // Descriptors for variables that will pass through cuSPARSE
  cusparseSpMatDescr_t spDescrLocA_;
  cusparseSpMatDescr_t spDescrBdA_;
  cusparseDnVecDescr_t spDescrNu_;
  cusparseDnVecDescr_t spDescrT_;
  cusparseDnVecDescr_t spDescrLocZ_;
  cusparseDnVecDescr_t spDescrBdZ_;
  // Work buffer for cusparseSpMV
  size_t locSpMVBuffSz_;
  void *locSpMVBuff_;
  size_t bdSpMVBuffSz_;
  void *bdSpMVBuff_;

  DeviceProfiler prof_;
};

BiCGSTABSolver::BiCGSTABSolver(MPI_Comm m_comm, LocalSpMatDnVec &LocalLS,
                               const int BLEN, const bool bMeanConstraint,
                               const std::vector<double> &P_inv)
    : m_comm_(m_comm), BLEN_(BLEN), bMeanConstraint_(bMeanConstraint),
      LocalLS_(LocalLS), prof_(m_comm) {
  // MPI
  MPI_Comm_rank(m_comm_, &rank_);
  MPI_Comm_size(m_comm_, &comm_size_);

  // Set-up CUDA streams events, and handles
  checkCudaErrors(cudaStreamCreate(&solver_stream_));
  checkCudaErrors(cudaStreamCreate(&copy_stream_));
  checkCudaErrors(cudaEventCreate(&sync_event_));
  checkCudaErrors(cublasCreate(&cublas_handle_));
  checkCudaErrors(cusparseCreate(&cusparse_handle_));
  // Set handles to stream
  checkCudaErrors(cublasSetStream(cublas_handle_, solver_stream_));
  checkCudaErrors(cusparseSetStream(cusparse_handle_, solver_stream_));
  // Set pointer modes to device
  checkCudaErrors(
      cublasSetPointerMode(cublas_handle_, CUBLAS_POINTER_MODE_DEVICE));
  checkCudaErrors(
      cusparseSetPointerMode(cusparse_handle_, CUSPARSE_POINTER_MODE_DEVICE));

  // Set constants and allocate memory for scalars
  double h_consts[3] = {1., -1., 0.};
  checkCudaErrors(cudaMalloc(&d_consts_, 3 * sizeof(double)));
  checkCudaErrors(cudaMemcpyAsync(d_consts_, h_consts, 3 * sizeof(double),
                                  cudaMemcpyHostToDevice, solver_stream_));
  d_eye_ = d_consts_;
  d_nye_ = d_consts_ + 1;
  d_nil_ = d_consts_ + 2;
  checkCudaErrors(cudaMalloc(&d_coeffs_, sizeof(BiCGSTABScalars)));
  checkCudaErrors(cudaMallocHost(&h_coeffs_, sizeof(BiCGSTABScalars)));

  // Copy preconditionner
  checkCudaErrors(cudaMalloc(&d_P_inv_, BLEN_ * BLEN_ * sizeof(double)));
  checkCudaErrors(cudaMemcpyAsync(d_P_inv_, P_inv.data(),
                                  BLEN_ * BLEN_ * sizeof(double),
                                  cudaMemcpyHostToDevice, solver_stream_));
}

BiCGSTABSolver::~BiCGSTABSolver() {
  // Cleanup after last timestep
  this->freeLast();

  prof_.print("Total");

  // Free preconditionner
  checkCudaErrors(cudaFree(d_P_inv_));

  // Free device consants
  checkCudaErrors(cudaFree(d_consts_));
  checkCudaErrors(cudaFree(d_coeffs_));
  checkCudaErrors(cudaFreeHost(h_coeffs_));

  // Destroy CUDA streams and handles
  checkCudaErrors(cublasDestroy(cublas_handle_));
  checkCudaErrors(cusparseDestroy(cusparse_handle_));
  checkCudaErrors(cudaEventDestroy(sync_event_));
  checkCudaErrors(cudaStreamDestroy(copy_stream_));
  checkCudaErrors(cudaStreamDestroy(solver_stream_));
}

// --------------------------------- public class methods
// ------------------------------------

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

// --------------------------------- private class methods
// ------------------------------------

void BiCGSTABSolver::freeLast() {
  if (dirty_) // Previous time-step exists so cleanup first
  {
    // Free device memory allocated for linear system from previous time-step
    checkCudaErrors(cudaFree(dloc_cooValA_));
    checkCudaErrors(cudaFree(dloc_cooRowA_));
    checkCudaErrors(cudaFree(dloc_cooColA_));
    checkCudaErrors(cudaFree(d_x_));
    checkCudaErrors(cudaFree(d_x_opt_));
    checkCudaErrors(cudaFree(d_r_));
    // Cleanup memory allocated for BiCGSTAB arrays
    checkCudaErrors(cudaFree(d_rhat_));
    checkCudaErrors(cudaFree(d_p_));
    checkCudaErrors(cudaFree(d_nu_));
    checkCudaErrors(cudaFree(d_t_));
    checkCudaErrors(cudaFree(d_z_));
    // Free and destroy cuSPARSE memory/descriptors
    checkCudaErrors(cudaFree(locSpMVBuff_));
    checkCudaErrors(cusparseDestroySpMat(spDescrLocA_));
    checkCudaErrors(cusparseDestroyDnVec(spDescrNu_));
    checkCudaErrors(cusparseDestroyDnVec(spDescrT_));
    checkCudaErrors(cusparseDestroyDnVec(spDescrLocZ_));
    if (comm_size_ > 1) {
      checkCudaErrors(cudaFree(d_send_pack_idx_));
      checkCudaErrors(cudaFree(d_send_buff_));
      checkCudaErrors(cudaFreeHost(h_send_buff_));
      checkCudaErrors(cudaFreeHost(h_recv_buff_));
      checkCudaErrors(cudaFree(dbd_cooValA_));
      checkCudaErrors(cudaFree(dbd_cooRowA_));
      checkCudaErrors(cudaFree(dbd_cooColA_));
      checkCudaErrors(cudaFree(bdSpMVBuff_));
      checkCudaErrors(cusparseDestroySpMat(spDescrBdA_));
      checkCudaErrors(cusparseDestroyDnVec(spDescrBdZ_));
    }
    if (bMeanConstraint_) {
      checkCudaErrors(cudaFree(d_h2_));
      checkCudaErrors(cudaFree(d_red_));
      checkCudaErrors(cudaFree(d_red_res_));
      checkCudaErrors(cudaFree(d_red_temp_storage_));
    }
  }
  dirty_ = true;
}

void BiCGSTABSolver::updateAll() {
  this->freeLast();

  // Update LS metadata
  m_ = LocalLS_.m_;
  halo_ = LocalLS_.halo_;
  hd_m_ = m_ + halo_;
  loc_nnz_ = LocalLS_.loc_nnz_;
  bd_nnz_ = LocalLS_.bd_nnz_;
  send_buff_sz_ = LocalLS_.send_pack_idx_.size();
  const int Nblocks = m_ / BLEN_;
  bMeanRow_ = LocalLS_.bMeanRow_;

  // Allocate device memory for local linear system
  checkCudaErrors(cudaMalloc(&dloc_cooValA_, loc_nnz_ * sizeof(double)));
  checkCudaErrors(cudaMalloc(&dloc_cooRowA_, loc_nnz_ * sizeof(int)));
  checkCudaErrors(cudaMalloc(&dloc_cooColA_, loc_nnz_ * sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_x_, m_ * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_x_opt_, m_ * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_r_, m_ * sizeof(double)));
  // Allocate arrays for BiCGSTAB storage
  checkCudaErrors(cudaMalloc(&d_rhat_, m_ * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_p_, m_ * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_nu_, m_ * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_t_, m_ * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_z_, hd_m_ * sizeof(double)));
  if (bMeanConstraint_) {
    checkCudaErrors(cudaMalloc(&d_h2_, Nblocks * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_red_, m_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_red_res_, sizeof(double)));
    // Allocate temporary storage for reductions
    d_red_temp_storage_ = NULL;
    red_temp_storage_bytes_ = 0;
    cub::DeviceReduce::Sum<double *, double *>(d_red_temp_storage_,
                                               red_temp_storage_bytes_, d_red_,
                                               d_red_res_, m_, solver_stream_);
    checkCudaErrors(cudaMalloc(&d_red_temp_storage_, red_temp_storage_bytes_));
  }
  if (comm_size_ > 1) {
    checkCudaErrors(cudaMalloc(&d_send_pack_idx_, send_buff_sz_ * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_send_buff_, send_buff_sz_ * sizeof(double)));
    checkCudaErrors(
        cudaMallocHost(&h_send_buff_, send_buff_sz_ * sizeof(double)));
    checkCudaErrors(cudaMallocHost(&h_recv_buff_, halo_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&dbd_cooValA_, bd_nnz_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&dbd_cooRowA_, bd_nnz_ * sizeof(int)));
    checkCudaErrors(cudaMalloc(&dbd_cooColA_, bd_nnz_ * sizeof(int)));
  }

  prof_.startProfiler("Memcpy", solver_stream_);
  // H2D transfer of linear system
  checkCudaErrors(cudaMemcpyAsync(dloc_cooValA_, LocalLS_.loc_cooValA_.data(),
                                  loc_nnz_ * sizeof(double),
                                  cudaMemcpyHostToDevice, solver_stream_));
  checkCudaErrors(cudaMemcpyAsync(
      dloc_cooRowA_, LocalLS_.loc_cooRowA_int_.data(), loc_nnz_ * sizeof(int),
      cudaMemcpyHostToDevice, solver_stream_));
  checkCudaErrors(cudaMemcpyAsync(
      dloc_cooColA_, LocalLS_.loc_cooColA_int_.data(), loc_nnz_ * sizeof(int),
      cudaMemcpyHostToDevice, solver_stream_));
  if (comm_size_ > 1) {
    checkCudaErrors(cudaMemcpyAsync(
        d_send_pack_idx_, LocalLS_.send_pack_idx_.data(),
        send_buff_sz_ * sizeof(int), cudaMemcpyHostToDevice, solver_stream_));
    checkCudaErrors(cudaMemcpyAsync(dbd_cooValA_, LocalLS_.bd_cooValA_.data(),
                                    bd_nnz_ * sizeof(double),
                                    cudaMemcpyHostToDevice, solver_stream_));
    checkCudaErrors(cudaMemcpyAsync(
        dbd_cooRowA_, LocalLS_.bd_cooRowA_int_.data(), bd_nnz_ * sizeof(int),
        cudaMemcpyHostToDevice, solver_stream_));
    checkCudaErrors(cudaMemcpyAsync(
        dbd_cooColA_, LocalLS_.bd_cooColA_int_.data(), bd_nnz_ * sizeof(int),
        cudaMemcpyHostToDevice, solver_stream_));
  }
  if (bMeanConstraint_)
    checkCudaErrors(cudaMemcpyAsync(d_h2_, LocalLS_.h2_.data(),
                                    Nblocks * sizeof(double),
                                    cudaMemcpyHostToDevice, solver_stream_));
  prof_.stopProfiler("Memcpy", solver_stream_);

  // Create descriptors for variables that will pass through cuSPARSE
  checkCudaErrors(cusparseCreateCoo(
      &spDescrLocA_, m_, m_, loc_nnz_, dloc_cooRowA_, dloc_cooColA_,
      dloc_cooValA_, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
  checkCudaErrors(cusparseCreateDnVec(&spDescrNu_, m_, d_nu_, CUDA_R_64F));
  checkCudaErrors(cusparseCreateDnVec(&spDescrT_, m_, d_t_, CUDA_R_64F));
  checkCudaErrors(cusparseCreateDnVec(&spDescrLocZ_, m_, d_z_, CUDA_R_64F));
  // Allocate work buffer for cusparseSpMV
  checkCudaErrors(cusparseSpMV_bufferSize(
      cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, d_eye_, spDescrLocA_,
      spDescrLocZ_, d_nil_, spDescrNu_, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
      &locSpMVBuffSz_));
  checkCudaErrors(cudaMalloc(&locSpMVBuff_, locSpMVBuffSz_ * sizeof(char)));
  if (comm_size_ > 1) {
    checkCudaErrors(cusparseCreateCoo(&spDescrBdA_, m_, hd_m_, bd_nnz_,
                                      dbd_cooRowA_, dbd_cooColA_, dbd_cooValA_,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&spDescrBdZ_, hd_m_, d_z_, CUDA_R_64F));
    checkCudaErrors(cusparseSpMV_bufferSize(
        cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, d_eye_, spDescrBdA_,
        spDescrBdZ_, d_eye_, spDescrNu_, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
        &bdSpMVBuffSz_));
    checkCudaErrors(cudaMalloc(&bdSpMVBuff_, bdSpMVBuffSz_ * sizeof(char)));
  }

  this->updateVec();
}

void BiCGSTABSolver::updateVec() {
  prof_.startProfiler("Memcpy", solver_stream_);
  // Copy RHS, LHS vec initial guess (to d_z_), if LS was updated, updateAll
  // reallocates sufficient memory
  checkCudaErrors(cudaMemcpyAsync(d_x_, LocalLS_.x_.data(), m_ * sizeof(double),
                                  cudaMemcpyHostToDevice, solver_stream_));
  checkCudaErrors(cudaMemcpyAsync(d_r_, LocalLS_.b_.data(), m_ * sizeof(double),
                                  cudaMemcpyHostToDevice, solver_stream_));
  prof_.stopProfiler("Memcpy", solver_stream_);
}

__global__ void set_squared(double *const val) { val[0] *= val[0]; }

__global__ void set_amax(double *const dest, const int *const idx,
                         const double *const source) {
  // 1-based indexing in cublas API
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

void BiCGSTABSolver::hd_cusparseSpMV(double *d_op_hd, // operand vec
                                     cusparseDnVecDescr_t spDescrLocOp,
                                     cusparseDnVecDescr_t spDescrBdOp,
                                     double *d_res_hd, // result vec
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
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaEventRecord(
        sync_event_, solver_stream_)); // event to sync up for MPI comm
  }

  prof_.startProfiler("KerSpMV", solver_stream_);
  // A*x for local rows
  checkCudaErrors(
      cusparseSpMV(cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, d_eye_,
                   spDescrLocA_, spDescrLocOp, d_nil_, spDescrRes, CUDA_R_64F,
                   CUSPARSE_SPMV_ALG_DEFAULT, locSpMVBuff_));
  prof_.stopProfiler("KerSpMV", solver_stream_);

  if (comm_size_ > 1) {
    prof_.startProfiler("HaloComm", copy_stream_);
    // Wait until copy to buffer has completed
    checkCudaErrors(cudaStreamWaitEvent(copy_stream_, sync_event_, 0));
    checkCudaErrors(cudaMemcpyAsync(h_send_buff_, d_send_buff_,
                                    send_buff_sz_ * sizeof(double),
                                    cudaMemcpyDeviceToHost, copy_stream_));
    checkCudaErrors(cudaStreamSynchronize(copy_stream_));

    // Schedule receives and wait for them to arrive
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
    prof_.stopProfiler("HaloComm", copy_stream_);

    // Use solver stream, just in case... even though the halo doesn't
    // particiapte in SpMV race conditions possible due to coalescing?
    checkCudaErrors(cudaMemcpyAsync(&d_op_hd[m_], h_recv_buff_,
                                    halo_ * sizeof(double),
                                    cudaMemcpyHostToDevice, solver_stream_));

    prof_.startProfiler("HaloSpMV", solver_stream_);
    // A*x for rows with halo elements, axpy with local results
    checkCudaErrors(
        cusparseSpMV(cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, d_eye_,
                     spDescrBdA_, spDescrBdOp, d_eye_, spDescrRes, CUDA_R_64F,
                     CUSPARSE_SPMV_ALG_DEFAULT, bdSpMVBuff_));
    prof_.stopProfiler("HaloSpMV", solver_stream_);
  }

  if (bMeanConstraint_) {
    // Copy result to reduction buffer and scale by h_i^2
    checkCudaErrors(cudaMemcpyAsync(d_red_, d_op_hd, m_ * sizeof(double),
                                    cudaMemcpyDeviceToDevice, solver_stream_));
    blockDscal<<<8 * 56, 128, 0, solver_stream_>>>(m_, BLEN_, d_h2_, d_red_);
    checkCudaErrors(cudaGetLastError());

    cub::DeviceReduce::Sum<double *, double *>(d_red_temp_storage_,
                                               red_temp_storage_bytes_, d_red_,
                                               d_red_res_, m_, solver_stream_);

    double h_red_res;
    checkCudaErrors(cudaMemcpyAsync(&h_red_res, d_red_res_, sizeof(double),
                                    cudaMemcpyDeviceToHost, solver_stream_));
    checkCudaErrors(cudaStreamSynchronize(solver_stream_));
    MPI_Allreduce(MPI_IN_PLACE, &h_red_res, 1, MPI_DOUBLE, MPI_SUM, m_comm_);

    if (bMeanRow_ >= 0)
      checkCudaErrors(cudaMemcpyAsync(&d_res_hd[bMeanRow_], &h_red_res,
                                      sizeof(double), cudaMemcpyHostToDevice,
                                      solver_stream_));
  }
}

void BiCGSTABSolver::main(const double max_error, const double max_rel_error,
                          const int max_restarts) {
  prof_.startProfiler("Total", solver_stream_);

  // Initialize variables to evaluate convergence
  double error = 1e50;
  double error_init = 1e50;
  double error_opt = 1e50;
  bool bConverged = false;
  int restarts = 0;

  // 3. Set initial values to scalars
  *h_coeffs_ = {1., 1., 1., 1e-21, 1., 1., 0., 0., 0};
  checkCudaErrors(cudaMemcpyAsync(d_coeffs_, h_coeffs_, sizeof(BiCGSTABScalars),
                                  cudaMemcpyHostToDevice, solver_stream_));

  // 1. r <- b - A*x_0.  Add bias with cuBLAS like in
  // "NVIDIA_CUDA-11.4_Samples/7_CUDALibraries/conjugateGradient"
  checkCudaErrors(cudaMemcpyAsync(d_z_, d_x_, m_ * sizeof(double),
                                  cudaMemcpyDeviceToDevice, solver_stream_));
  hd_cusparseSpMV(d_z_, spDescrLocZ_, spDescrBdZ_, d_nu_, spDescrNu_);
  checkCudaErrors(cublasDaxpy(cublas_handle_, m_, d_nye_, d_nu_, 1, d_r_,
                              1)); // r <- -A*x_0 + b

  // ||A*x_0||_max
  checkCudaErrors(
      cublasIdamax(cublas_handle_, m_, d_nu_, 1, &(d_coeffs_->amax_idx)));
  set_amax<<<1, 1, 0, solver_stream_>>>(&(d_coeffs_->buff_1),
                                        &(d_coeffs_->amax_idx), d_nu_);
  checkCudaErrors(cudaGetLastError());
  // ||b - A*x_0||_max
  checkCudaErrors(
      cublasIdamax(cublas_handle_, m_, d_r_, 1, &(d_coeffs_->amax_idx)));
  set_amax<<<1, 1, 0, solver_stream_>>>(&(d_coeffs_->buff_2),
                                        &(d_coeffs_->amax_idx), d_r_);
  checkCudaErrors(cudaGetLastError());
  // buff_1 and buff_2 in contigious memory in BiCGSTABScalars
  checkCudaErrors(cudaMemcpyAsync(&(h_coeffs_->buff_1), &(d_coeffs_->buff_1),
                                  2 * sizeof(double), cudaMemcpyDeviceToHost,
                                  solver_stream_));
  checkCudaErrors(cudaStreamSynchronize(solver_stream_));

  MPI_Allreduce(MPI_IN_PLACE, &(h_coeffs_->buff_1), 2, MPI_DOUBLE, MPI_MAX,
                m_comm_);

  if (rank_ == 0) {
    std::cout << "  [BiCGSTAB]: || A*x_0 || = " << h_coeffs_->buff_1 << '\n';
    std::cout << "  [BiCGSTAB]: Initial norm: " << h_coeffs_->buff_2 << '\n';
  }
  // Set initial error and x_opt
  error = h_coeffs_->buff_2;
  error_init = error;
  error_opt = error;
  checkCudaErrors(cudaMemcpyAsync(d_x_opt_, d_x_, m_ * sizeof(double),
                                  cudaMemcpyDeviceToDevice, solver_stream_));

  // 2. Set r_hat = r
  checkCudaErrors(cudaMemcpyAsync(d_rhat_, d_r_, m_ * sizeof(double),
                                  cudaMemcpyDeviceToDevice, solver_stream_));

  // 4. Set initial values of vectors to zero
  checkCudaErrors(
      cudaMemsetAsync(d_nu_, 0, m_ * sizeof(double), solver_stream_));
  checkCudaErrors(
      cudaMemsetAsync(d_p_, 0, m_ * sizeof(double), solver_stream_));

  // 5. Start iterations
  const size_t max_iter = 1000;
  for (size_t k(0); k < max_iter; k++) {
    // 1. rho_i = (r_hat, r)
    checkCudaErrors(cublasDdot(cublas_handle_, m_, d_rhat_, 1, d_r_, 1,
                               &(d_coeffs_->rho_curr)));

    // Numerical convergence trick
    checkCudaErrors(
        cublasDnrm2(cublas_handle_, m_, d_r_, 1, &(d_coeffs_->buff_1)));
    checkCudaErrors(
        cublasDnrm2(cublas_handle_, m_, d_rhat_, 1, &(d_coeffs_->buff_2)));
    checkCudaErrors(cudaMemcpyAsync(&(h_coeffs_->rho_curr),
                                    &(d_coeffs_->rho_curr), 3 * sizeof(double),
                                    cudaMemcpyDeviceToHost, solver_stream_));
    checkCudaErrors(cudaStreamSynchronize(solver_stream_));
    h_coeffs_->buff_1 *= h_coeffs_->buff_1; // get square norm
    h_coeffs_->buff_2 *= h_coeffs_->buff_2;
    MPI_Allreduce(MPI_IN_PLACE, &(h_coeffs_->rho_curr), 3, MPI_DOUBLE, MPI_SUM,
                  m_comm_);
    checkCudaErrors(cudaMemcpyAsync(&(d_coeffs_->rho_curr),
                                    &(h_coeffs_->rho_curr), sizeof(double),
                                    cudaMemcpyHostToDevice, solver_stream_));
    const bool serious_breakdown =
        h_coeffs_->rho_curr * h_coeffs_->rho_curr <
        1e-16 * h_coeffs_->buff_1 * h_coeffs_->buff_2;

    // 2. beta = (rho_i / rho_{i-1}) * (alpha / omega_{i-1})
    set_beta<<<1, 1, 0, solver_stream_>>>(d_coeffs_);
    checkCudaErrors(cudaGetLastError());
    if (serious_breakdown && max_restarts > 0) {
      restarts++;
      if (restarts >= max_restarts) {
        break;
      }
      if (rank_ == 0) {
        std::cout << "  [BiCGSTAB]: Restart at iteration: " << k
                  << " norm: " << error << " Initial norm: " << error_init
                  << '\n';
      }
      checkCudaErrors(cudaMemcpyAsync(d_rhat_, d_r_, m_ * sizeof(double),
                                      cudaMemcpyDeviceToDevice,
                                      solver_stream_));
      checkCudaErrors(
          cublasDnrm2(cublas_handle_, m_, d_rhat_, 1, &(d_coeffs_->rho_curr)));
      checkCudaErrors(cudaMemcpyAsync(&(h_coeffs_->rho_curr),
                                      &(d_coeffs_->rho_curr), sizeof(double),
                                      cudaMemcpyDeviceToHost, solver_stream_));
      checkCudaErrors(cudaStreamSynchronize(solver_stream_));
      h_coeffs_->rho_curr *= h_coeffs_->rho_curr;
      MPI_Allreduce(MPI_IN_PLACE, &(h_coeffs_->rho_curr), 1, MPI_DOUBLE,
                    MPI_SUM, m_comm_);
      checkCudaErrors(cudaMemcpyAsync(&(d_coeffs_->rho_curr),
                                      &(h_coeffs_->rho_curr), sizeof(double),
                                      cudaMemcpyHostToDevice, solver_stream_));
      checkCudaErrors(
          cudaMemsetAsync(d_nu_, 0, m_ * sizeof(double), solver_stream_));
      checkCudaErrors(
          cudaMemsetAsync(d_p_, 0, m_ * sizeof(double), solver_stream_));
      breakdown_update<<<1, 1, 0, solver_stream_>>>(d_coeffs_);
      checkCudaErrors(cudaGetLastError());
    }

    // 3. p_i = r_{i-1} + beta(p_{i-1} - omega_{i-1}*nu_i)
    set_negative<<<1, 1, 0, solver_stream_>>>(&(d_coeffs_->buff_1),
                                              &(d_coeffs_->omega));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cublasDaxpy(cublas_handle_, m_, &(d_coeffs_->buff_1), d_nu_,
                                1, d_p_, 1)); // p <- -omega_{i-1}*nu_i + p
    checkCudaErrors(cublasDscal(cublas_handle_, m_, &(d_coeffs_->beta), d_p_,
                                1)); // p <- beta * p
    checkCudaErrors(cublasDaxpy(cublas_handle_, m_, d_eye_, d_r_, 1, d_p_,
                                1)); // p <- r_{i-1} + p

    // 4. z <- K_2^{-1} * p_i
    prof_.startProfiler("Prec", solver_stream_);
    checkCudaErrors(cublasDgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N, BLEN_,
                                m_ / BLEN_, BLEN_, d_eye_, d_P_inv_, BLEN_,
                                d_p_, BLEN_, d_nil_, d_z_, BLEN_));
    prof_.stopProfiler("Prec", solver_stream_);

    // 5. nu_i = A * z
    hd_cusparseSpMV(d_z_, spDescrLocZ_, spDescrBdZ_, d_nu_, spDescrNu_);

    // 6. alpha = rho_i / (r_hat, nu_i)
    checkCudaErrors(cublasDdot(cublas_handle_, m_, d_rhat_, 1, d_nu_, 1,
                               &(d_coeffs_->buff_1)));
    checkCudaErrors(cudaMemcpyAsync(&(h_coeffs_->buff_1), &(d_coeffs_->buff_1),
                                    sizeof(double), cudaMemcpyDeviceToHost,
                                    solver_stream_));
    checkCudaErrors(cudaStreamSynchronize(solver_stream_));
    MPI_Allreduce(MPI_IN_PLACE, &(h_coeffs_->buff_1), 1, MPI_DOUBLE, MPI_SUM,
                  m_comm_);
    checkCudaErrors(cudaMemcpyAsync(&(d_coeffs_->buff_1), &(h_coeffs_->buff_1),
                                    sizeof(double), cudaMemcpyHostToDevice,
                                    solver_stream_));
    set_alpha<<<1, 1, 0, solver_stream_>>>(d_coeffs_);
    checkCudaErrors(cudaGetLastError());

    // 7. h = alpha*z + x_{i-1}
    checkCudaErrors(
        cublasDaxpy(cublas_handle_, m_, &(d_coeffs_->alpha), d_z_, 1, d_x_, 1));

    // 9. s = -alpha * nu_i + r_{i-1}
    set_negative<<<1, 1, 0, solver_stream_>>>(&(d_coeffs_->buff_1),
                                              &(d_coeffs_->alpha));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cublasDaxpy(cublas_handle_, m_, &(d_coeffs_->buff_1), d_nu_,
                                1, d_r_, 1));

    // 10. z <- K_2^{-1} * s
    prof_.startProfiler("Prec", solver_stream_);
    checkCudaErrors(cublasDgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N, BLEN_,
                                m_ / BLEN_, BLEN_, d_eye_, d_P_inv_, BLEN_,
                                d_r_, BLEN_, d_nil_, d_z_, BLEN_));
    prof_.stopProfiler("Prec", solver_stream_);

    // 11. t = A * z
    hd_cusparseSpMV(d_z_, spDescrLocZ_, spDescrBdZ_, d_t_, spDescrT_);

    // 12. omega_i = (t,s)/(t,t), variables alpha & beta no longer in use this
    // iter
    checkCudaErrors(
        cublasDdot(cublas_handle_, m_, d_t_, 1, d_r_, 1, &(d_coeffs_->buff_1)));
    checkCudaErrors(
        cublasDnrm2(cublas_handle_, m_, d_t_, 1, &(d_coeffs_->buff_2)));
    set_squared<<<1, 1, 0, solver_stream_>>>(&(d_coeffs_->buff_2));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpyAsync(&(h_coeffs_->buff_1), &(d_coeffs_->buff_1),
                                    2 * sizeof(double), cudaMemcpyDeviceToHost,
                                    solver_stream_));
    checkCudaErrors(cudaStreamSynchronize(solver_stream_));
    MPI_Allreduce(MPI_IN_PLACE, &(h_coeffs_->buff_1), 2, MPI_DOUBLE, MPI_SUM,
                  m_comm_);
    checkCudaErrors(cudaMemcpyAsync(&(d_coeffs_->buff_1), &(h_coeffs_->buff_1),
                                    2 * sizeof(double), cudaMemcpyHostToDevice,
                                    solver_stream_));
    set_omega<<<1, 1, 0, solver_stream_>>>(d_coeffs_);
    checkCudaErrors(cudaGetLastError());

    // 13. x_i = omega_i * z + h
    checkCudaErrors(
        cublasDaxpy(cublas_handle_, m_, &(d_coeffs_->omega), d_z_, 1, d_x_, 1));

    // 15. r_i = -omega_i * t + s
    set_negative<<<1, 1, 0, solver_stream_>>>(&(d_coeffs_->buff_1),
                                              &(d_coeffs_->omega));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cublasDaxpy(cublas_handle_, m_, &(d_coeffs_->buff_1), d_t_,
                                1, d_r_, 1));

    // If x_i accurate enough then quit
    checkCudaErrors(
        cublasIdamax(cublas_handle_, m_, d_r_, 1, &(d_coeffs_->amax_idx)));
    set_amax<<<1, 1, 0, solver_stream_>>>(&(d_coeffs_->buff_1),
                                          &(d_coeffs_->amax_idx), d_r_);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpyAsync(&error, &(d_coeffs_->buff_1),
                                    sizeof(double), cudaMemcpyDeviceToHost,
                                    solver_stream_));

    checkCudaErrors(cudaMemcpyAsync(h_coeffs_, d_coeffs_,
                                    sizeof(BiCGSTABScalars),
                                    cudaMemcpyDeviceToHost, solver_stream_));

    checkCudaErrors(cudaStreamSynchronize(solver_stream_));
    MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, m_comm_);

    if (error < error_opt) {
      error_opt = error;
      checkCudaErrors(cudaMemcpyAsync(d_x_opt_, d_x_, m_ * sizeof(double),
                                      cudaMemcpyDeviceToDevice,
                                      solver_stream_));

      if ((error <= max_error) || (error / error_init <= max_rel_error)) {
        if (rank_ == 0)
          std::cout << "  [BiCGSTAB]: Converged after " << k << " iterations\n";

        bConverged = true;
        break;
      }
    }

    // Update *_prev values for next iteration
    set_rho<<<1, 1, 0, solver_stream_>>>(d_coeffs_);
    checkCudaErrors(cudaGetLastError());
  }

  if (rank_ == 0) {
    if (bConverged)
      std::cout << "  [BiCGSTAB] Error norm (relative) = " << error_opt << "/"
                << max_error << " (" << error_opt / error_init << "/"
                << max_rel_error << ")\n"
                << std::flush;
    else
      std::cout << "  [BiCGSTAB]: Iteration " << max_iter
                << ". Error norm (relative) = " << error_opt << "/" << max_error
                << " (" << error_opt / error_init << "/" << max_rel_error
                << ")\n"
                << std::flush;
  }

  prof_.startProfiler("Memcpy", solver_stream_);
  // Copy result back to host
  checkCudaErrors(cudaMemcpyAsync(LocalLS_.x_.data(), d_x_opt_,
                                  m_ * sizeof(double), cudaMemcpyDeviceToHost,
                                  solver_stream_));
  prof_.stopProfiler("Memcpy", solver_stream_);
  prof_.stopProfiler("Total", solver_stream_);
}

LocalSpMatDnVec::LocalSpMatDnVec(MPI_Comm m_comm, const int BLEN,
                                 const bool bMeanConstraint,
                                 const std::vector<double> &P_inv)
    : m_comm_(m_comm), BLEN_(BLEN) {
  // MPI
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
  bMeanRow_ = -1; // init at default value after refinement

  // Clear previous contents and reserve excess memory
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
    // Update recv set
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

  // Exchange message sizes between all ranks
  MPI_Alltoall(recv_sz_allranks.data(), 1, MPI_INT, send_sz_allranks.data(), 1,
               MPI_INT, m_comm_);

  // Set receiving rules into halo
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

  // Set sending rules from a 'send' buffer
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

  // Post receives for column indices from other ranks required for SpMV
  std::vector<MPI_Request> recv_requests(send_ranks_.size());
  for (size_t i(0); i < send_ranks_.size(); i++)
    MPI_Irecv(&send_pack_idx_long[send_offset_[i]], send_sz_[i], MPI_LONG_LONG,
              send_ranks_[i], 546, m_comm_, &recv_requests[i]);

  // Create sends to inform other ranks what they will need to send here
  std::vector<long long> recv_idx_list(halo_);
  std::vector<MPI_Request> send_requests(recv_ranks_.size());
  for (size_t i(0); i < recv_ranks_.size(); i++) {
    std::copy(bd_recv_set_[recv_ranks_[i]].begin(),
              bd_recv_set_[recv_ranks_[i]].end(),
              &recv_idx_list[recv_offset_[i]]);
    MPI_Isend(&recv_idx_list[recv_offset_[i]], recv_sz_[i], MPI_LONG_LONG,
              recv_ranks_[i], 546, m_comm_, &send_requests[i]);
  }

  // Now re-index the linear system from global to local indexing
  const long long shift = -Nrows_xcumsum[rank_];
  loc_cooRowA_int_.resize(loc_nnz_);
  loc_cooColA_int_.resize(loc_nnz_);
  bd_cooRowA_int_.resize(bd_nnz_);
#pragma omp parallel
  {
// Shift rows and columns to local indexing
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

  // Make sure recv_idx_list is safe to use (and not prone to deallocation
  // because it will no longer be in use)
  MPI_Waitall(send_ranks_.size(), recv_requests.data(), MPI_STATUS_IGNORE);

  // Map indices of columns from other ranks to the halo
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

  // if (rank_ == 0)
  //   std::cerr << "  [LocalLS]: Rank: " << rank_ << ", m: " << m_ << ", halo:
  //   " << halo_ << std::endl;
}

// Solve method with update to LHS matrix
void LocalSpMatDnVec::solveWithUpdate(const double max_error,
                                      const double max_rel_error,
                                      const int max_restarts) {
  solver_->solveWithUpdate(max_error, max_rel_error, max_restarts);
}
// Solve method without update to LHS matrix
void LocalSpMatDnVec::solveNoUpdate(const double max_error,
                                    const double max_rel_error,
                                    const int max_restarts) {
  solver_->solveNoUpdate(max_error, max_rel_error, max_restarts);
}
