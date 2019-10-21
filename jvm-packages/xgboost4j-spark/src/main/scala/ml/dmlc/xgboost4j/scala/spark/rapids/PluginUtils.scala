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
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

object PluginUtils extends Serializable {
  // scalastyle:off classforname
  def isSupportColumnar: Boolean = try {
    Class.forName("ai.rapids.spark.ColumnarRdd")
    true
  } catch {
    case _: ClassNotFoundException => false
  }

  def toColumnarRdd(df: DataFrame): RDD[Table] = {
    Class.forName("ai.rapids.spark.ColumnarRdd")
      .getDeclaredMethod("convert", classOf[DataFrame])
      .invoke(null, df)
      .asInstanceOf[RDD[Table]]
  }
  // scalastyle:on classforname

  // calculate bench mark
  def time[R](phase: String)(block: => R): (R, Float) = {
    val t0 = System.currentTimeMillis
    val result = block // call-by-name
    val t1 = System.currentTimeMillis
    (result, (t1 - t0).toFloat / 1000)
  }
}
