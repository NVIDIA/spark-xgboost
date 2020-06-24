/*!
 * Copyright 2020 by Contributors
 * \file gpu_column.h
 * \brief store cuDF column info
 * \author Bobby Wang
 */

#ifndef XGBOOST_GPU_COLUMN_H
#define XGBOOST_GPU_COLUMN_H

struct gpu_column_data {
  long* data_ptr;
  long* valid_ptr;
  int dtype_size_in_bytes;
  long num_row;
  int type_id;
  long null_count;
};

#endif //XGBOOST_GPU_COLUMN_H
