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

package ai.rapids.cudf;

import ml.dmlc.xgboost4j.java.rapids.ColumnData;

public class GpuColumnVectorUtils {

  public static ColumnData getColumnData(ColumnVector columnVector) {
    BaseDeviceMemoryBuffer dataBuffer = columnVector.getDeviceBufferFor(BufferType.DATA);
    BaseDeviceMemoryBuffer validBuffer = columnVector.getDeviceBufferFor(BufferType.VALIDITY);
    long validPtr = 0;
    if (validBuffer != null) {
      validPtr = validBuffer.getAddress();
    }
    DType dType = columnVector.getType();

    return new ColumnData(dataBuffer.getAddress(), columnVector.getRowCount(), validPtr,
        dType.sizeInBytes, dType.nativeId, columnVector.getNullCount());
  }
}
