#
# Copyright (c) 2019 by Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import pyspark.ml.tuning

from pyspark.ml.wrapper import JavaWrapper
from pyspark.ml.tuning import CrossValidatorModel


class CrossValidator(JavaWrapper, pyspark.ml.tuning.CrossValidator):
    def __init__(self):
        super(CrossValidator, self).__init__()
        self._java_obj = self._new_java_obj(
            'ml.dmlc.xgboost4j.scala.spark.rapids.CrossValidator')

    def fit(self, dataset):
        java_estimator, java_epms, java_evaluator = self._to_java_impl()
        self._java_obj.setEstimator(java_estimator)
        self._java_obj.setEvaluator(java_evaluator)
        self._java_obj.setEstimatorParamMaps(java_epms)

        cv_java_model = self._java_obj.fit(dataset._jdf)
        cv_py_model = CrossValidatorModel._from_java(cv_java_model)
        xgbModel = self.getEstimator()._create_model(cv_java_model.bestModel())
        # return CrossValidatorModel
        return CrossValidatorModel(xgbModel, cv_py_model.avgMetrics, cv_py_model.subModels)
