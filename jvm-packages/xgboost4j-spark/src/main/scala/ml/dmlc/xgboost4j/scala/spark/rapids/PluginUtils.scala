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

import ai.rapids.cudf.Table
import ml.dmlc.xgboost4j.scala.spark.params.Utils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset}

object PluginUtils extends Serializable {

  // APIs for plugin related
  def isSupportColumnar(data: Dataset[_]): Boolean = {
    val pluginName = data.sparkSession.sparkContext.getConf.get("spark.sql.extensions", "")
    pluginName == "com.nvidia.spark.rapids.SQLExecPlugin"
  }

  def toColumnarRdd(df: DataFrame): RDD[Table] = {
    Utils.classForName("com.nvidia.spark.rapids.ColumnarRdd")
      .getDeclaredMethod("convert", classOf[DataFrame])
      .invoke(null, df)
      .asInstanceOf[RDD[Table]]
  }

  // calculate bench mark
  def time[R](phase: String)(block: => R): (R, Float) = {
    val t0 = System.currentTimeMillis
    val result = block // call-by-name
    val t1 = System.currentTimeMillis
    (result, (t1 - t0).toFloat / 1000)
  }
}
