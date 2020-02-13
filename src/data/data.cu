/*!
 * Copyright 2018 by xgboost contributors
 */

#ifdef XGBOOST_USE_CUDF
#include <xgboost/data.h>
#include <xgboost/logging.h>

#include <cudf/column/column_view.hpp>

#include "../common/device_helpers.cuh"
#include "../common/host_device_vector.h"
#include "./cudf.cuh"

namespace xgboost {

using namespace data;

__global__ void unpack_cudf_column_k
  (float* data, size_t n_rows, size_t n_cols, void const* colData, type_id type) {
  size_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n_rows)
    return;
  data[n_cols * i] = ConvertDataElement(colData, i, type);
}

void MetaInfo::SetCUDFInfo(const char* key, column_view** cols, size_t n_cols, int gpu_id) {
  this->SetCUDFInfoImpl(key, cols, n_cols, gpu_id, false);
}

void MetaInfo::AppendCUDFInfo(const char* key, column_view** cols, size_t n_cols, int gpu_id) {
  this->SetCUDFInfoImpl(key, cols, n_cols, gpu_id, true);
}

void MetaInfo::SetCUDFInfoImpl(const char* key, column_view** cols, size_t n_cols,
                               int gpu_id, bool append) {
  CHECK_GT(n_cols, 0);
  size_t n_rows = cols[0]->size();
  CHECK_GE(n_rows, 0U);
  for (size_t i = 0; i < n_cols; ++i) {
    CHECK_EQ(cols[i]->null_count(), 0) << "all labels and weights must be valid";
    CHECK_EQ(cols[i]->size(), n_rows) << "all CUDF columns must be of the same size";
  }
  HostDeviceVector<bst_float>* field = nullptr;
  if (!strcmp(key, "label")) {
    field = &labels_;
  } else if (!strcmp(key, "weight")) {
    field = &weights_;
    CHECK_EQ(n_cols, 1) << "only one CUDF column allowed for weights";
  } else {
    LOG(WARNING) << key << ": invalid key value for MetaInfo field";
    return;
  }
  // TODO(canonizer): use the same devices as elsewhere in xgboost
  int device_id = 0;
  GPUSet devices;
  if (gpu_id > 0) {
    device_id = gpu_id;
    devices = GPUSet::All(device_id, 1);
  } else {
    devices = GPUSet::Range(device_id, 1);
  }

  size_t prev_size = (append) ? field->Size() : 0;
  field->Reshard(GPUDistribution::Granular(devices, n_cols));
  field->Resize(prev_size + n_cols * n_rows);
  bst_float* data = field->DevicePointer(device_id);
  data += prev_size;
  for (size_t i = 0; i < n_cols; ++i) {
    int block = 256;
    auto pCol = cols[i];
    unpack_cudf_column_k<<<common::DivRoundUp(n_rows, block), block>>>
      (data + i, n_rows, n_cols, pCol->head(), pCol->type().id());
    dh::safe_cuda(cudaGetLastError());
  }
}
  
}  // namespace xgboost
#endif
