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

package ml.dmlc.xgboost4j.scala.spark.params

import org.apache.spark.sql.DataFrame

trait NonParamVariables {
  protected var evalSetsMap: Map[String, AnyRef] = Map.empty
  private val KEY_EVAL_SETS: String = "eval_sets"

  def setEvalSets(evalSets: Map[String, AnyRef]): this.type = {
    evalSetsMap = evalSets
    this
  }

  def getEvalSets(params: Map[String, Any]): Map[String, DataFrame] = {
    val evalDFs = if (params.contains(KEY_EVAL_SETS)) {
      params(KEY_EVAL_SETS).asInstanceOf[Map[String, DataFrame]]
    } else {
      evalSetsMap.asInstanceOf[Map[String, DataFrame]]
    }
    // Do type check for value entry here because the `asInstanceOf` just checks the first
    // layer: Map, even specifying the types for both key and value.
    require(evalDFs.values.forall(_.isInstanceOf[DataFrame]),
      "Wrong type for value! Evaluation sets should be Map(name: String -> DataFrame) for CPU.")
    evalDFs
  }

}
