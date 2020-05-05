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

import ai.rapids.cudf.{GpuColumnVectorUtils, Table}
import ml.dmlc.xgboost4j.java.spark.rapids.GpuColumnBatch
import ml.dmlc.xgboost4j.scala.DMatrix
import ml.dmlc.xgboost4j.scala.spark.rapids.{ColumnBatchToRow, GpuSampler, PluginUtils}
import ml.dmlc.xgboost4j.{LabeledPoint => XGBLabeledPoint}
import org.apache.spark.TaskContext
import org.apache.spark.ml.feature.{LabeledPoint => MLLabeledPoint}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Column, DataFrame, Row}
import org.apache.spark.sql.types.{FloatType, IntegerType, StructType}

object DataUtils extends Serializable {
  private[spark] implicit class XGBLabeledPointFeatures(
      val labeledPoint: XGBLabeledPoint
  ) extends AnyVal {
    /** Converts the point to [[MLLabeledPoint]]. */
    private[spark] def asML: MLLabeledPoint = {
      MLLabeledPoint(labeledPoint.label, labeledPoint.features)
    }

    /**
     * Returns feature of the point as [[org.apache.spark.ml.linalg.Vector]].
     *
     * If the point is sparse, the dimensionality of the resulting sparse
     * vector would be [[Int.MaxValue]]. This is the only safe value, since
     * XGBoost does not store the dimensionality explicitly.
     */
    def features: Vector = if (labeledPoint.indices == null) {
      Vectors.dense(labeledPoint.values.map(_.toDouble))
    } else {
      Vectors.sparse(Int.MaxValue, labeledPoint.indices, labeledPoint.values.map(_.toDouble))
    }
  }

  private[spark] implicit class MLLabeledPointToXGBLabeledPoint(
      val labeledPoint: MLLabeledPoint
  ) extends AnyVal {
    /** Converts an [[MLLabeledPoint]] to an [[XGBLabeledPoint]]. */
    def asXGB: XGBLabeledPoint = {
      labeledPoint.features.asXGB.copy(label = labeledPoint.label.toFloat)
    }
  }

  private[spark] implicit class MLVectorToXGBLabeledPoint(val v: Vector) extends AnyVal {
    /**
     * Converts a [[Vector]] to a data point with a dummy label.
     *
     * This is needed for constructing a [[ml.dmlc.xgboost4j.scala.DMatrix]]
     * for prediction.
     */
    def asXGB: XGBLabeledPoint = v match {
      case v: DenseVector =>
        XGBLabeledPoint(0.0f, null, v.values.map(_.toFloat))
      case v: SparseVector =>
        XGBLabeledPoint(0.0f, v.indices, v.values.map(_.toFloat))
    }
  }

  private[spark] def convertDataFrameToXGBLabeledPointRDDs(
      labelCol: Column,
      featuresCol: Column,
      weight: Column,
      baseMargin: Column,
      group: Option[Column],
      dataFrames: DataFrame*): Array[RDD[XGBLabeledPoint]] = {
    val selectedColumns = group.map(groupCol => Seq(labelCol.cast(FloatType),
      featuresCol,
      weight.cast(FloatType),
      groupCol.cast(IntegerType),
      baseMargin.cast(FloatType))).getOrElse(Seq(labelCol.cast(FloatType),
      featuresCol,
      weight.cast(FloatType),
      baseMargin.cast(FloatType)))
    dataFrames.toArray.map {
      df => df.select(selectedColumns: _*).rdd.map {
        case Row(label: Float, features: Vector, weight: Float, group: Int, baseMargin: Float) =>
          val (indices, values) = features match {
            case v: SparseVector => (v.indices, v.values.map(_.toFloat))
            case v: DenseVector => (null, v.values.map(_.toFloat))
          }
          XGBLabeledPoint(label, indices, values, weight, group, baseMargin)
        case Row(label: Float, features: Vector, weight: Float, baseMargin: Float) =>
          val (indices, values) = features match {
            case v: SparseVector => (v.indices, v.values.map(_.toFloat))
            case v: DenseVector => (null, v.values.map(_.toFloat))
          }
          XGBLabeledPoint(label, indices, values, weight, baseMargin = baseMargin)
      }
    }
  }

  private[spark] def buildDMatrixIncrementally(gpuId: Int, missing: Float,
      featureIndices: Seq[Int], iter: Iterator[Table], schema: StructType,
      sampler: Option[GpuSampler] = None, colNameToBuild: Option[String] = None):
        (DMatrix, ColumnBatchToRow) = {

    var dm: DMatrix = null
    var isFirstBatch = true
    val columnBatchToRow: ColumnBatchToRow = new ColumnBatchToRow

    while (iter.hasNext) {
      val columnBatch = new GpuColumnBatch(iter.next(), schema, sampler.getOrElse(null))
      if (isFirstBatch) {
        isFirstBatch = false
        dm = new DMatrix(gpuId, missing, columnBatch.getAsColumnData(featureIndices: _*): _*)
      } else {
        dm.appendCUDF(columnBatch.getAsColumnData(featureIndices: _*): _*)
      }
      columnBatchToRow.appendColumnBatch(columnBatch, colNameToBuild)
      columnBatch.close()
    }
    if (dm == null) {
      // here we allow empty iter
      // throw new RuntimeException("Can't build Dmatrix from CUDF")
    }
    (dm, columnBatchToRow)
  }

  // called by classifier or regressor
  private[spark] def buildGDFColumnData(
      featuresColNames: Seq[String],
      labelColName: String,
      weightColName: String,
      groupColName: String,
      dataFrame: DataFrame): GDFColumnData = {
    require(featuresColNames.nonEmpty, "No features column is specified!")
    require(labelColName != null && labelColName.nonEmpty, "No label column is specified!")
    val weightAndGroupName = Seq(weightColName, groupColName).map(
      name => if (name == null || name.isEmpty) Array.empty[String] else Array(name)
    )
    // Seq in this order: features, label, weight, group
    val colNames = Seq(featuresColNames.toArray, Array(labelColName)) ++ weightAndGroupName
    // build column indices
    val schema = dataFrame.schema
    val indices = colNames.map(_.filter(schema.fieldNames.contains).map(schema.fieldIndex))
    require(indices.head.length == featuresColNames.length,
      "Features column(s) in schema do NOT match the one(s) in parameters. " +
        s"Expect [${featuresColNames.mkString(", ")}], " +
        s"but found [${indices.head.map(schema.fieldNames).mkString(", ")}]!")
    require(indices(1).nonEmpty, "Missing label column in schema!")
    // Check if has group
    val opGroup = if (colNames(3).nonEmpty) {
      require(indices(3).nonEmpty, "Can not find group column in schema!")
      Some(groupColName)
    } else {
      None
    }

    GDFColumnData(dataFrame, indices, opGroup)
  }

}
