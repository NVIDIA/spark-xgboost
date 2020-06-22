/*!
 * Copyright 2018 by xgboost contributors
 */

#ifdef XGBOOST_USE_CUDF

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <xgboost/gpu_column.h>

#include "../common/host_device_vector.h"
#include "../common/device_helpers.cuh"

#include "./cudf.cuh"
#include "./simple_csr_source.h"

namespace xgboost {
namespace data {

struct CsrCudf {
  Entry* data;
  size_t* offsets;
  size_t n_nz;
  size_t n_rows;
  size_t n_cols;
  bst_float missing;
};

void CUDFToCSR(std::vector<gpu_column_data *> const& gdfcols, CsrCudf* csr);

//--- private CUDA functions / kernels
__global__ void cuda_create_csr_k
(void const* cudf_data, uint32_t const* valid, int dtype, int col, Entry* data,
 size_t *offsets, size_t n_rows, bool is_nan_missing, bst_float missing);

__global__ void determine_valid_rec_count_k(void const* cudf_data, int dtype, uint32_t const* valid,
    size_t n_rows,size_t n_cols, size_t *offset, bool is_nan_missing, bst_float missing);

__device__ int WhichBitmap(int record) { return record / 32; }
__device__ int WhichBit(int bit) { return bit % 32; }
__device__ int CheckBit(uint32_t data, int bit) {

//  gdf_valid_type bit_mask[8] = {1, 2, 4, 8, 16, 32, 64, 128};
//  return data & bit_mask[bit];
  return data & (1U << bit);
}

__device__ bool IsValid(uint32_t const* valid, int tid) {
  if (valid == nullptr)
    return true;
  int bitmap_idx = WhichBitmap(tid);
  int bit_idx = WhichBit(tid);
  uint32_t bitmap = valid[bitmap_idx];
  return CheckBit(bitmap, bit_idx);
}

// Convert a CUDF into a CSR CUDF
void CUDFHandleMissingValue(std::vector<gpu_column_data *> const& gdfcols,
    int n_cols, CsrCudf* csr) {
  // already check its validity
  size_t n_rows = gdfcols[0]->num_row;

  // the first step is to create an array that counts the number of valid entries per row
  // this is done by each thread looking across its row and checking the valid bits
  int threads = 1024;
  int blocks = (n_rows + threads - 1) / threads;

  size_t* offsets = csr->offsets;
  dh::safe_cuda(cudaMemset(offsets, 0, sizeof(size_t) * (n_rows + 1)));

  if (blocks > 0) {
    for (int i = 0; i < n_cols; ++i) {
      determine_valid_rec_count_k<<<blocks, threads>>>(
          gdfcols[i]->data_ptr, gdfcols[i]->type_id,
          reinterpret_cast<const unsigned int *>(gdfcols[i]->valid_ptr),
          n_rows, n_cols, offsets, isnan(csr->missing), csr->missing);
      dh::safe_cuda(cudaGetLastError());
      dh::safe_cuda(cudaDeviceSynchronize());
    }
  }

  // compute the number of elements
  thrust::device_ptr<size_t> offsets_begin(offsets);
  int64_t n_elements = thrust::reduce
      (offsets_begin, offsets_begin + n_rows, 0ull, thrust::plus<size_t>());

  // now do an exclusive scan to compute the offsets for where to write data
  thrust::exclusive_scan(offsets_begin, offsets_begin + n_rows + 1, offsets_begin);

  csr->n_rows = n_rows;
  csr->n_cols = n_cols;
  csr->n_nz = n_elements;
}

void CUDFToCSR(std::vector<gpu_column_data *> const& gdfcols, CsrCudf* csr) {
  size_t n_cols = csr->n_cols;
  size_t n_rows = csr->n_rows;

  int threads = 256;
  int blocks = (n_rows + threads - 1) / threads;

  // temporary offsets for writing data
  thrust::device_ptr<size_t> offset_begin(csr->offsets);
  thrust::device_vector<size_t> offsets2(offset_begin, offset_begin + n_rows + 1);

  // move the data and create the CSR
  if (blocks > 0) {
    for (int col = 0; col < n_cols; ++col) {
      gpu_column_data *cudf_column = gdfcols[col];
      cuda_create_csr_k<<<blocks, threads>>>(cudf_column->data_ptr,
          reinterpret_cast<const unsigned int *>(cudf_column->valid_ptr),
          cudf_column->type_id, col, csr->data, offsets2.data().get(), n_rows,
          isnan(csr->missing), csr->missing);
      dh::safe_cuda(cudaGetLastError());
    }
  }
}

// move data over into CSR and possibly convert the format
__global__ void cuda_create_csr_k
(void const* cudf_data, uint32_t const* valid, int dtype, int col,
 Entry* data, size_t* offsets, size_t n_rows, bool is_nan_missing, bst_float missing) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= n_rows)
    return;
  size_t offset_idx = offsets[tid];
  if (IsValid(valid, tid)) {
    bst_float v = ConvertDataElement(cudf_data, tid, dtype);
    if (is_nan_missing || v != missing) {
      data[offset_idx].fvalue = v;
      data[offset_idx].index = col;
      ++offsets[tid];
    }
  }
}

// compute the number of valid entries per row
__global__ void determine_valid_rec_count_k(void const* cudf_data, int dtype, uint32_t const* valid,
    size_t n_rows, size_t n_cols, size_t *offset, bool is_nan_missing, bst_float missing) {

  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= n_rows)
    return;
  if (IsValid(valid, tid)) {
    bst_float v = ConvertDataElement(cudf_data, tid, dtype);
    if (is_nan_missing || v != missing) {
      ++offset[tid];
    }
  }
}

void SimpleCSRSource::InitFromCUDF(std::vector<gpu_column_data *> const& gdfcols,
    int gpu_id, bst_float missing) {
  unsigned int n_cols = gdfcols.size();
  CHECK_GT(n_cols, 0);
  int32_t n_rows = gdfcols[0]->num_row;
  CHECK_GE(n_rows, 0U);
  info.num_col_ = n_cols;
  info.num_row_ = n_rows;

  // TODO(canonizer): use the same devices as by the rest of xgboost
  int device_id = 0;
  GPUSet devices;
  if (gpu_id > 0) {
    device_id = gpu_id;
    devices = GPUSet::All(device_id, 1);
  } else {
    devices = GPUSet::Range(device_id, 1);
  }

  page_.offset.Reshard(GPUDistribution::Overlap(devices, 1));
  page_.offset.Resize(n_rows + 1);

  CsrCudf csr;
  csr.offsets = page_.offset.DevicePointer(device_id);
  csr.n_nz = 0;
  csr.n_rows = n_rows;
  csr.n_cols = n_cols;
  csr.missing = missing;

  CUDFHandleMissingValue(gdfcols, n_cols, &csr);

  // TODO(canonizer): use the real row offsets for the multi-GPU case
  info.num_nonzero_ = csr.n_nz;
  std::vector<size_t> device_offsets{0, csr.n_nz};
  page_.data.Reshard(GPUDistribution::Explicit(devices, device_offsets));
  page_.data.Resize(csr.n_nz);

  csr.data = page_.data.DevicePointer(device_id);
  CUDFToCSR(gdfcols, &csr);

  std::vector<float> tmp(10);

  // Since training copies the data back to the host (as it assumes the dataset
  // is on the host always), move the data from the device to the host. There is
  // no use for the data to sit on the device, if training doesn't use it.
  // Effect this by resharding to an empty device set. This will draw the data
  // from the device to the system memory
  page_.data.Reshard(GPUDistribution());
  page_.offset.Reshard(GPUDistribution());

}

}  // namespace data
}  // namespace xgboost
#endif
