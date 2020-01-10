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

package ml.dmlc.xgboost4j.java;

import ai.rapids.cudf.NativeDepsLoader;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * JNI functions for XGBoost4J-Spark
 */
public class XGBoostSparkJNI {
  private static final Log logger = LogFactory.getLog(XGBoostSparkJNI.class);

  static {
    try {
      logger.info("load cuDF libs");
      NativeDepsLoader.libraryLoaded();
      logger.info("load XGBoost libs");
      NativeLibLoader.initXGBoost();
    } catch (Exception ex) {
      logger.error("Failed to load native library", ex);
      throw new RuntimeException(ex);
    }
  }

  /**
   * Build an array of fixed-length Spark UnsafeRow using the GPU.
   * @param nativeColumnPtrs native address of cudf column pointers
   * @return native address of the UnsafeRow array
   * NOTE: It is the responsibility of the caller to free the native memory
   *       returned by this function (e.g.: using Platform.freeMemory).
   */
  public static native long buildUnsafeRows(long[] nativeColumnPtrs);

  public static native int getGpuDevice();

  public static native int allocateGpuDevice(int gpuId);
}
