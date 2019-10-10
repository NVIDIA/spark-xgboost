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

import ml.dmlc.xgboost4j.java.spark.rapids.{GpuColumnBatch, PartitionReader, PartitionReaderFactory}
import org.apache.commons.logging.LogFactory
import org.apache.hadoop.conf.Configuration
import org.apache.spark.{Partition, SerializableWritable, TaskContext}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.execution.datasources.{FilePartition}
import org.apache.spark.sql.types.StructType

private[spark] class GpuDatasetRDD(
    @transient private val sparkSession: SparkSession,
    broadcastedConf: Broadcast[SerializableWritable[Configuration]],
    partitionReaderFactory: PartitionReaderFactory,
    @transient val filePartitions: Seq[FilePartition],
    schema: StructType)
  extends RDD[GpuColumnBatch](sparkSession.sparkContext, Nil) {

  private val logger = LogFactory.getLog(classOf[GpuDatasetRDD])

  // The resulting iterator will only return a single GpuColumnBatch.
  override def compute(split: Partition, context: TaskContext): Iterator[GpuColumnBatch] = {

    val iterator = new Iterator[GpuColumnBatch] with AutoCloseable {
      private[this] var reader: PartitionReader[GpuColumnBatch] = _

      if (partitionReaderFactory != null) {
        reader = partitionReaderFactory.createColumnarReader(split.asInstanceOf[FilePartition])
      }

      override def hasNext: Boolean = {
        reader.next()
      }

      override def next(): GpuColumnBatch = {
        reader.get()
      }

      override def close(): Unit = {
        reader.close()
      }
    }

    // Register an on-task-completion callback to close the input stream.
    context.addTaskCompletionListener[Unit](_ => iterator.close())
    iterator
  }

  override protected def getPartitions: Array[Partition] = filePartitions.toArray

  override protected def getPreferredLocations(split: Partition): Seq[String] = {
    val filePart = split.asInstanceOf[FilePartition]
    val hostToNumBytes = scala.collection.mutable.HashMap.empty[String, Long]
    GpuDataset.getFiles(filePart).foreach { file =>
      try {
        file.locations.filter(_ != "localhost").foreach { host =>
          hostToNumBytes(host) = hostToNumBytes.getOrElse(host, 0L) + file.length
        }
      } catch {
        case e: NoSuchMethodError =>
          // This would fail because some Spark implementations have overloaded FilePartition
          // We handle this in GpuDataset but we never set the locations in those cases
          // so do nothing here
          logger.debug("Not a Spark FilePartition, skipping getting locations")
      }
    }

    // Takes the first 3 hosts with the most data to be retrieved
    hostToNumBytes.toSeq.sortBy {
      case (host, numBytes) => numBytes
    }.reverse.take(3).map {
      case (host, numBytes) => host
    }
  }
}
