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

package ml.dmlc.xgboost4j.java.spark.rapids;

import ai.rapids.cudf.*;
import org.apache.spark.sql.types.StructType;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.nio.file.Paths;

import static org.apache.spark.sql.types.DataTypes.FloatType;
import static org.apache.spark.sql.types.DataTypes.IntegerType;
import static org.junit.Assert.*;
import static org.junit.Assume.assumeTrue;

/**
 * Test cases for GpuColumnBatchTest
 */
public class GpuColumnBatchTest {

  private final String TEST_CSV_PATH =
          this.getClass().getResource("/rank-weight.csv").getPath();

  // Schemas for spark and CUDF
  private final Schema CUDF_SCHEMA = Schema.builder()
          .column(DType.FLOAT32, "label")
          .column(DType.FLOAT32, "a")
          .column(DType.FLOAT32, "b")
          .column(DType.FLOAT32, "c")
          .column(DType.INT32, "group")
          .column(DType.FLOAT32, "weight")
          .build();
  private final StructType DB_SCHEMA = new StructType()
          .add("label", FloatType)
          .add("a", FloatType)
          .add("b", FloatType)
          .add("c", FloatType)
          .add("group", IntegerType)
          .add("weight", FloatType);

  private Table mTestTable;

  @Before
  public void setup() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    mTestTable = Table.readCSV(CUDF_SCHEMA, Paths.get(TEST_CSV_PATH).toFile());
  }

  @After
  public void tearDown() {
    mTestTable.close();
    mTestTable = null;
  }

  @Test
  public void testCountOnColumnHost() {
    // The expected result is computed from file "rank.test.csv" manually.
    int[] expected = new int[]{7, 5, 9, 6, 6, 8, 7, 6, 5, 7};
    GpuColumnBatch columnBatch = new GpuColumnBatch(mTestTable, DB_SCHEMA);
    int[] groupInfo = columnBatch.groupByColumnWithCountHost(4);
    assertArrayEquals(expected, groupInfo);
  }

  @Test
  public void testOneOnColumn() {
    GpuColumnBatch columnBatch = new GpuColumnBatch(mTestTable, DB_SCHEMA);
    // treat first column as weight for failure case
    long[] handles = columnBatch.groupByColumnWithAggregation(4, 0, true);
    assertNull(handles);
    long[] handlesNoCheck = columnBatch.groupByColumnWithAggregation(4, 0, false);
    assertNotNull(handlesNoCheck);
    // normal case
    long[] handlesWeight = columnBatch.groupByColumnWithAggregation(4, 5, true);
    assertNotNull(handlesWeight);
  }
}
