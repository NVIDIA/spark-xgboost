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

package ml.dmlc.xgboost4j.scala.spark

import java.io.File

import ai.rapids.cudf.{DType, Table}
import ml.dmlc.xgboost4j.java.spark.rapids.GpuColumnBatch
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{
  BooleanType, ByteType, DataType, DateType, DoubleType,
  FloatType, IntegerType, LongType, ShortType, StructField, StructType, TimestampType
}

object GpuDatasetData {

  private val classifierTrainDataPath = "/iris.data.csv"
  private val classifierTestDataPath = "/iris.test.data.csv"
  private val classifierSchema = new StructType(Array(
    StructField("sepal length", FloatType),
    StructField("sepal width", FloatType),
    StructField("petal length", FloatType),
    StructField("petal width", FloatType),
    StructField("classIndex", FloatType)))
/*
  def getClassifierTrainGpuDataset(spark: SparkSession): GpuDataset = {
    getTrainGpuDataset(spark, classifierSchema, classifierTrainDataPath)
  }

  def getClassifierTestGpuDataset(spark: SparkSession): (GpuDataset, Long) = {
    getTestGpuDataset(spark, classifierSchema, classifierTestDataPath)
  }
*/
  private val regressionTrainDataPath = "/norank.train.csv"
  private val regressionTestDataPath = "/norank.train.csv"
  private val regressionSchema = new StructType(Array(
    StructField("b", DoubleType),
    StructField("c", DoubleType),
    StructField("d", DoubleType),
    StructField("e", DoubleType)))
/*
  def getRegressionTrainGpuDataset(spark: SparkSession): GpuDataset = {
    getTrainGpuDataset(spark, regressionSchema, regressionTrainDataPath)
  }

  def getRegressionTestGpuDataset(spark: SparkSession): (GpuDataset, Long) = {
    getTestGpuDataset(spark, regressionSchema, regressionTestDataPath)
  }
*/
  private val rankingTrainDataPath = "/rank.train.csv"
  private val rankingTestDataPath = "/rank.test.csv"
  private val rankingSchema = new StructType(Array(
    StructField("label", FloatType),
    StructField("b", FloatType),
    StructField("c", FloatType),
    StructField("d", FloatType),
    StructField("group", IntegerType)))
/*
  def getRankingTrainGpuDataset(spark: SparkSession): GpuDataset = {
    getTrainGpuDataset(spark, rankingSchema, rankingTrainDataPath)
  }

  def getRankingTestGpuDataset(spark: SparkSession): (GpuDataset, Long) = {
    getTestGpuDataset(spark, rankingSchema, rankingTestDataPath)
  }

  private def getTrainGpuDataset(
      spark: SparkSession,
      schema: StructType,
      path: String): GpuDataset = {
    new GpuDataReader(spark).schema(schema).csv(getTestDataPath(path))
  }

  private def getTestGpuDataset(
      spark: SparkSession,
      schema: StructType,
      path: String): (GpuDataset, Long) = {
    val test = new GpuDataReader(spark).schema(schema).csv(getTestDataPath(path))
    val counts = test.mapColumnarBatchPerPartition(GpuDataset.getColumnRowNumberMapper).collect()

    (test, counts(0)_2)
  }
*/
  private def getColumns(file: String, schema: StructType, label: String):
  (Table, Array[Long], Array[Long]) = {
    val csvSchemaBuilder = ai.rapids.cudf.Schema.builder
    schema.foreach(f => csvSchemaBuilder.column(toDType(f.dataType), f.name))
    val table = Table.readCSV(csvSchemaBuilder.build(), new File(getTestDataPath(file)))
    val columnBatch = new GpuColumnBatch(table, schema)

    var needTableClose = true
    try {
      val featuresColNames = schema.fieldNames.filter(_ != label)
      val featuresHandle = featuresColNames.map(colName => {
          columnBatch.getColumn(schema.fieldIndex(colName))
      })
      val labelsHandle = Array(label).map(colName => {
          columnBatch.getColumn(schema.fieldIndex(colName))
      })
      needTableClose = false
      (table, featuresHandle, labelsHandle)
    } finally {
      if (needTableClose) {
        table.close()
      }
    }
  }

  private def toDType(dataType: DataType): DType = {
    dataType match {
      case BooleanType | ByteType => DType.INT8
      case ShortType => DType.INT16
      case IntegerType => DType.INT32
      case LongType => DType.INT64
      case FloatType => DType.FLOAT32
      case DoubleType => DType.FLOAT64
      case DateType => DType.TIMESTAMP_DAYS
      // FIXME which one in DType should be used here?
      case TimestampType => DType.TIMESTAMP_MICROSECONDS
      case unknownType => throw new UnsupportedOperationException(
        s"Unsupported Spark SQL type $unknownType")
    }
  }

  private def getTestDataPath(resource: String): String = {
    require(resource.startsWith("/"), "resource must start with /")
    getClass.getResource(resource).getPath
  }

  def classifierCleanUp(): Unit = {
    if (classifierTrainTable != null) {
      classifierTrainTable.close()
      classifierTrainTable = null
    }

    if (classifierTestTable != null) {
      classifierTestTable.close()
      classifierTestTable = null
    }
  }

  def regressionCleanUp(): Unit = {
    if (regressionTestTable != null) {
      regressionTestTable.close()
      regressionTestTable = null
    }

    if (regressionTrainTable != null) {
      regressionTrainTable.close()
      regressionTrainTable = null
    }
  }

  var classifierTestTable: Table = null
  var classifierTrainTable: Table = null

  lazy val classifierTest: (Array[Long], Array[Long]) = {
    val (table, features, labels) = getColumns(classifierTestDataPath,
      classifierSchema, "classIndex")
    classifierTestTable = table;
    (features, labels)
  }
  lazy val classifierTrain: (Array[Long], Array[Long]) = {
    val (table, features, labels) = getColumns(classifierTrainDataPath,
      classifierSchema, "classIndex")
    classifierTrainTable = table
    (features, labels)
  }
  lazy val classifierFeatureCols: Seq[String] =
    classifierSchema.fieldNames.filter(_ != "classIndex")

  var regressionTestTable: Table = null
  var regressionTrainTable: Table = null

  lazy val regressionTest: (Array[Long], Array[Long]) = {
    val (table, features, labels) = getColumns(regressionTestDataPath,
      regressionSchema, "e")
    regressionTestTable = table
    (features, labels)
  }
  lazy val regressionTrain: (Array[Long], Array[Long]) = {
    val (table, features, labels) = getColumns(regressionTrainDataPath,
      regressionSchema, "e")
    regressionTrainTable = table
    (features, labels)
  }

  lazy val regressionFeatureCols: Seq[String] =
    regressionSchema.fieldNames.filter(_ != "e")

  lazy val rankingFeatureCols: Seq[String] =
    rankingSchema.fieldNames.filter(_ != "group").filter(_ != "label")
}

