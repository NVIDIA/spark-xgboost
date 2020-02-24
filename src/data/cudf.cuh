/*!
 * Copyright 2018 by xgboost contributors
 */

#include <cudf/types.hpp>

namespace xgboost {
namespace data {

using cudf::type_id; 

/**
 * Convert the data element into a common format
 */
__device__ inline float ConvertDataElement(void const* data, int tid, type_id dtype) {
  switch(dtype) {
    case type_id::INT8: {
      int8_t * d = (int8_t*)data;
      return float(d[tid]);
    }
    case type_id::INT16: {
      int16_t * d = (int16_t*)data;
      return float(d[tid]);
    }
    case type_id::INT32: {
      int32_t * d = (int32_t*)data;
      return float(d[tid]);
    }
    case type_id::INT64: {
      int64_t * d = (int64_t*)data;
      return float(d[tid]);
    }
    case type_id::FLOAT32: {
      float * d = (float *)data;
      return float(d[tid]);
    }
    case type_id::FLOAT64: {
      double * d = (double *)data;
      return float(d[tid]);
    }
  }
  return nanf(nullptr);
}

}  // namespace data
}  // namespace xgboost
