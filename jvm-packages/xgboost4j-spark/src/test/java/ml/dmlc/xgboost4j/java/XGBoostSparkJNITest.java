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

import ai.rapids.cudf.Cuda;
import ml.dmlc.xgboost4j.java.rapids.GpuColumnVectorUtils;
import ai.rapids.cudf.Table;
import ml.dmlc.xgboost4j.java.rapids.ColumnData;

import org.apache.spark.sql.catalyst.expressions.UnsafeRow;
import org.apache.spark.unsafe.Platform;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

/**
 * Test cases for XGBoostSparkJNI
 */
public class XGBoostSparkJNITest {
  @Test
  public void testSimpleBuildUnsafeRows() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());

    try (Table table = new Table.TestBuilder()
      .column(null, 1, 3, 5)
      .column(2L, 4L, null, 8L)
      .column(20.0f, null, null, null)
      .column((short) 200, (short) 150, (short) 100, (short) 25)
      .build()) {

      final int numColumns = 4;
      final int rowSize = UnsafeRow.calculateBitSetWidthInBytes(numColumns)
        + numColumns * 8;
      long rawUnsafeRowData = 0;
      try {
        ColumnData[] cds = new ColumnData[numColumns];
        for (int i = 0; i < table.getNumberOfColumns(); i++) {
          cds[i] = GpuColumnVectorUtils.getColumnData(table.getColumn(i));
        }
        rawUnsafeRowData = XGBoostSparkJNI.buildUnsafeRows(cds);
        assertTrue(rawUnsafeRowData != 0);
        UnsafeRow row = new UnsafeRow(4);

        // verify row 0
        row.pointTo(null, rawUnsafeRowData, rowSize);
        assertTrue(row.isNullAt(0));
        assertFalse(row.isNullAt(1));
        assertEquals(2, row.getLong(1));
        assertFalse(row.isNullAt(2));
        assertEquals(20.0f, row.getFloat(2), 0);
        assertFalse(row.isNullAt(3));
        assertEquals(200, row.getShort(3));

        // verify row 1
        row.pointTo(null, rawUnsafeRowData + rowSize, rowSize);
        assertFalse(row.isNullAt(0));
        assertEquals(1, row.getInt(0));
        assertFalse(row.isNullAt(1));
        assertEquals(4, row.getLong(1));
        assertTrue(row.isNullAt(2));
        assertFalse(row.isNullAt(3));
        assertEquals(150, row.getShort(3));

        // verify row 2
        row.pointTo(null, rawUnsafeRowData + rowSize * 2, rowSize);
        assertFalse(row.isNullAt(0));
        assertEquals(3, row.getInt(0));
        assertTrue(row.isNullAt(1));
        assertTrue(row.isNullAt(2));
        assertFalse(row.isNullAt(3));
        assertEquals(100, row.getShort(3));

        // verify row 3
        row.pointTo(null, rawUnsafeRowData + rowSize * 3, rowSize);
        assertFalse(row.isNullAt(0));
        assertEquals(5, row.getInt(0));
        assertFalse(row.isNullAt(1));
        assertEquals(8, row.getLong(1));
        assertTrue(row.isNullAt(2));
        assertFalse(row.isNullAt(3));
        assertEquals(25, row.getShort(3));
      } finally {
        if (rawUnsafeRowData != 0) {
          Platform.freeMemory(rawUnsafeRowData);
        }
      }
    }
  }

}
