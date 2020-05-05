/*!
 * Copyright 2018 by xgboost contributors
 */

namespace xgboost {
namespace data {

/**
 * Convert the data element into a common format
 */
__device__ inline float ConvertDataElement(void const* data, int tid, int dtype) {
  switch(dtype) {
    case 1: {
      int8_t * d = (int8_t*)data;
      return float(d[tid]);
    }
    case 2: {
      int16_t * d = (int16_t*)data;
      return float(d[tid]);
    }
    case 3: {
      int32_t * d = (int32_t*)data;
      return float(d[tid]);
    }
    case 4: {
      int64_t * d = (int64_t*)data;
      return float(d[tid]);
    }
    case 5: {
      float * d = (float *)data;
      return float(d[tid]);
    }
    case 6: {
      double * d = (double *)data;
      return float(d[tid]);
    }
  }
  return nanf(nullptr);
}

}  // namespace data
}  // namespace xgboost
