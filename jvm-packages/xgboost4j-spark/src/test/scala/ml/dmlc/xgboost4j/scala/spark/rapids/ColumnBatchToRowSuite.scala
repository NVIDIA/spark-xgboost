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

package ml.dmlc.xgboost4j.scala.spark.rapids

import ai.rapids.cudf.{Cuda, Table}
import ml.dmlc.xgboost4j.java.spark.rapids.GpuColumnBatch
import ml.dmlc.xgboost4j.scala.spark.PerTest
import org.apache.spark.sql.SparkSession
import org.scalactic.TolerantNumerics
import org.scalatest.FunSuite

class ColumnBatchToRowSuite extends FunSuite with PerTest {
  private lazy val TRAIN_CSV_PATH = getTestDataPath("/rank.train.csv")
  private lazy val TEST_CSV_PATH = getTestDataPath("/rank.test.csv")

  override def sparkSessionBuilder: SparkSession.Builder = SparkSession.builder()
    .master("local[1]")
    .appName("GpuDatasetSuite")
    .config("spark.ui.enabled", false)
    .config("spark.driver.memory", "512m")
    .config("spark.task.cpus", 1)

  test("ColumnToRow") {
    assume(Cuda.isEnvCompatibleForTesting)
    val reader = ss.read
    val csvSchema = "a BOOLEAN, b DOUBLE, c DOUBLE, d DOUBLE, e INT"
    val dataset = reader
      .option("asFloats", "false")
      .schema(csvSchema)
      .csv(TRAIN_CSV_PATH, TEST_CSV_PATH)

    val rdd = PluginUtils.toColumnarRdd(dataset).mapPartitions((iter: Iterator[Table]) => {
      val columnBatchToRow = new ColumnBatchToRow()
      while (iter.hasNext) {
        columnBatchToRow.appendColumnBatch(new GpuColumnBatch(iter.next, dataset.schema))
      }
      columnBatchToRow.toIterator
    })

    val data = rdd.collect()
    val firstRow = data.head
    assertResult(215) { data.length }
    assertResult(firstRow.size) { 5 }

    assertResult(false) { firstRow.getBoolean(0) }
    implicit val doubleEquality = TolerantNumerics.tolerantDoubleEquality(0.000000001)
    assert(firstRow.getDouble(1) === 985.574005058)
    assert(firstRow.getDouble(2) === 320.223538037)
    assert(firstRow.getDouble(3) === 0.621236086198)
    assert(firstRow.getInt(4) == 1)

    val secondRow = data.last
    assertResult(secondRow.size) { 5 }
    assertResult(true) {
      secondRow.getBoolean(0)
    }
    assert(secondRow.getDouble(1) === 1004.1394132)
    assert(secondRow.getDouble(2) === 464.371823646)
    assert(secondRow.getDouble(3) === 0.312492288321)
    assert(secondRow.getInt(4) == 10)
  }

  private def getTestDataPath(resource: String): String = {
    require(resource.startsWith("/"), "resource must start with /")
    getClass.getResource(resource).getPath
  }
}
