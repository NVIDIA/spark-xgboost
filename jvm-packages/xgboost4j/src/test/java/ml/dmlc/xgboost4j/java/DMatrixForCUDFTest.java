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

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.Cuda;

import junit.framework.TestCase;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assume.assumeTrue;

/**
 * test cases for DMatrix using CUDF
 *
 * @author liangcail
 */
public class DMatrixForCUDFTest {

  @Test
  public void testCreateFromCUDF() {
    //create Matrix from CUDF
    ColumnVector featureCol = null, labelCol = null, weightCol = null;
    try {
      float[] infoData = new float[]{5.0f, 6.0f, 7.0f};
      // feature
      featureCol = ColumnVector.fromFloats(1.0f, 2.0f, 3.0f);
      // label
      labelCol = ColumnVector.fromFloats(infoData);
      // weight
      weightCol = ColumnVector.fromFloats(infoData);

      DMatrix dmat = new DMatrix(new long[]{featureCol.getNativeView()});
      dmat.setCUDFInfo("label", new long[]{labelCol.getNativeView()});
      dmat.setCUDFInfo("weight", new long[]{weightCol.getNativeView()});

      TestCase.assertTrue("Wrong label ", Arrays.equals(dmat.getLabel(), infoData));
      TestCase.assertTrue("Wrong weight ", Arrays.equals(dmat.getWeight(), infoData));
    } catch (XGBoostError xe) {
      // In case CUDF is not built
      TestCase.assertTrue("Unexpected error: " + xe,
          "CUDF is not enabled!".equals(xe.getMessage()));
    } finally {
      if (featureCol != null) featureCol.close();
      if (labelCol != null) labelCol.close();
      if (weightCol != null) weightCol.close();
    }
  }

  @Test
  public void testCreateFromCUDFWithMissingValue() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());

    ColumnVector v0 = null, v1 = null, v2 = null, v3 = null, labelCol = null, weightCol = null;
    float[] infoData = new float[]{5.0f, 6.0f, 7.0f};
    final int numColumns = 4;
    try {
      v0 = ColumnVector.fromBoxedFloats(-1.0f, 0.0f, 0.0f);
      v1 = ColumnVector.fromBoxedFloats(-1.0f, -1.0f, -1.0f);
      v2 = ColumnVector.fromBoxedFloats(-1.0f, -1.0f, 3.0f);
      v3 = ColumnVector.fromBoxedFloats(-1.0f, 1.0f, 2.0f);

      long[] nativeCols = new long[numColumns];

      //create Matrix from CUDF
      // label
      labelCol = ColumnVector.fromFloats(infoData);
      // weight
      weightCol = ColumnVector.fromFloats(infoData);

      nativeCols[0] = v0.getNativeView();
      nativeCols[1] = v1.getNativeView();
      nativeCols[2] = v2.getNativeView();
      nativeCols[3] = v3.getNativeView();

      DMatrix dmat = new DMatrix(nativeCols, 0, -1.0f);
      dmat.setCUDFInfo("label", new long[]{labelCol.getNativeView()});
      dmat.setCUDFInfo("weight", new long[]{weightCol.getNativeView()});

      TestCase.assertTrue("Wrong label ", Arrays.equals(dmat.getLabel(), infoData));
      TestCase.assertTrue("Wrong weight ", Arrays.equals(dmat.getWeight(), infoData));

      TestCase.assertTrue(dmat.rowNum() == 3);
    } catch (XGBoostError xe) {
      // In case CUDF is not built
      TestCase.assertTrue("Unexpected error: " + xe,
          "CUDF is not enabled!".equals(xe.getMessage()));
    } finally {
      if (labelCol != null) labelCol.close();
      if (weightCol != null) weightCol.close();
      if (v0 != null) v0.close();
      if (v1 != null) v1.close();
      if (v2 != null) v2.close();
      if (v3 != null) v3.close();
    }
  }
}
