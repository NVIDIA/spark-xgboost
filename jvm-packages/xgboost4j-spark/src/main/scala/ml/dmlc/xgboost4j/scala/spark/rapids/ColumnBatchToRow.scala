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

import ml.dmlc.xgboost4j.java.XGBoostSparkJNI
import ml.dmlc.xgboost4j.java.spark.rapids.GpuColumnBatch
import org.apache.spark.TaskContext
import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.expressions.UnsafeRow
import org.apache.spark.sql.types.StructType
import org.apache.spark.unsafe.Platform

class ColumnBatchToRow {
  private var batches: Seq[ColumnBatchIter] = Seq()
  private lazy val batchIter = batches.toIterator
  private var currentBatchIter: ColumnBatchIter = null

  def appendColumnBatch(batch: GpuColumnBatch, colNameToBuild: Option[String] = None): Unit = {
    batches = batches :+ new ColumnBatchIter(batch, colNameToBuild)
  }

  private[xgboost4j] def toIterator: Iterator[Row] = {
    val taskContext = TaskContext.get
    val iter = new Iterator[Row] with AutoCloseable {

      override def hasNext: Boolean = {
        (currentBatchIter != null && currentBatchIter.hasNext) || nextIterator()
      }

      override def next(): Row = {
        currentBatchIter.next()
      }

      override def close(): Unit = {
        if (currentBatchIter != null) {
          currentBatchIter.close()
        }
      }

      private def nextIterator(): Boolean = {
        if (batchIter.hasNext) {
          close
          currentBatchIter = batchIter.next()
          try {
            hasNext
          } finally {
          }
        } else {
          false
        }
      }
    }
    taskContext.addTaskCompletionListener[Unit](_ => iter.close())
    iter
  }

  class ColumnBatchIter(batch: GpuColumnBatch, colNameToBuild: Option[String] = None) extends
      Iterator[Row] with AutoCloseable {
    val rawSchema = batch.getSchema
    val schema = colNameToBuild.map(name => StructType(rawSchema.fields.filter(x =>
        RowConverter.isSupportingType(x.dataType) && x.name.equals(name))))
      .getOrElse(StructType(
        rawSchema.fields.filter(x => RowConverter.isSupportingType(x.dataType))))

    // schema name maps to index of schema
    val nameToIndex = schema.names.map(rawSchema.fieldIndex)

    val timeUnits = nameToIndex.map(batch.getColumnVector(_).getType())

    private val numRows = batch.getNumRows
    private val converter = new RowConverter(schema, timeUnits)
    private val rowSize = UnsafeRow.calculateBitSetWidthInBytes(schema.length) +
      schema.length * 8
    private var buffer: Long = initBuffer()
    private var nextRow = 0
    private val row = new UnsafeRow(schema.length)

    override def hasNext: Boolean = nextRow < numRows

    override def next(): Row = {
      if (nextRow >= numRows) {
        throw new NoSuchElementException
      }
      row.pointTo(null, buffer + rowSize * nextRow, rowSize)
      nextRow += 1
      converter.toExternalRow(row)
    }

    override def close(): Unit = {
      if (buffer != 0) {
        Platform.freeMemory(buffer)
        buffer = 0
      }
    }

    private def initBuffer(): Long = {
      val colDatas = batch.getAsColumnData(nameToIndex: _*)
      XGBoostSparkJNI.buildUnsafeRows(colDatas: _*)
    }
  }
}
