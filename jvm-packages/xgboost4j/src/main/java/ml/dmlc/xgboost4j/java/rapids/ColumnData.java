/*
 Copyright (c) 2014 by Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

package ml.dmlc.xgboost4j.java.rapids;

/**
 * This class is composing of datas from Gpu ColumnVector, and it will be used to get
 * cuda array interface's json format and to build unsafe row.
 */
public class ColumnData {
  private final long dataPtr; //  gpu data buffer address
  private final long shape;   // row count
  private final long validPtr; // gpu valid buffer address
  private final int typeSize; // type size in bytes
  private final int typeId;
  private final long nullCount;

  public ColumnData(long dataPtr, long shape, long validPtr, int typeSize, int typeId,
      long nullCount) {
    this.dataPtr = dataPtr;
    this.shape = shape;
    this.validPtr = validPtr;
    this.typeSize = typeSize;
    this.typeId = typeId;
    this.nullCount = nullCount;
  }

  public long getDataPtr() {
    return dataPtr;
  }

  public long getShape() {
    return shape;
  }

  public long getValidPtr() {
    return validPtr;
  }

  public int getTypeSize() {
    return typeSize;
  }

  public int getTypeId() {
    return typeId;
  }

  public long getNullCount() {
    return nullCount;
  }
}
