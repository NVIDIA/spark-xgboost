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
import java.nio.file.Files

import ai.rapids.cudf.Table

import scala.collection.{AbstractIterator, mutable}
import scala.util.Random
import ml.dmlc.xgboost4j.java.{IRabitTracker, Rabit, XGBoostError, XGBoostSparkJNI, RabitTracker => PyRabitTracker}
import ml.dmlc.xgboost4j.scala.rabit.RabitTracker
import ml.dmlc.xgboost4j.scala.spark.params.BoosterParams
import ml.dmlc.xgboost4j.scala.spark.params.LearningTaskParams
import ml.dmlc.xgboost4j.scala.spark.rapids.GpuDataset
import ml.dmlc.xgboost4j.scala.{XGBoost => SXGBoost, _}
import ml.dmlc.xgboost4j.{LabeledPoint => XGBLabeledPoint}
import org.apache.commons.io.FileUtils
import org.apache.commons.logging.LogFactory
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkParallelismTracker, TaskContext, TaskFailedListener}
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel

/**
 * Rabit tracker configurations.
 *
 * @param workerConnectionTimeout The timeout for all workers to connect to the tracker.
 *                                Set timeout length to zero to disable timeout.
 *                                Use a finite, non-zero timeout value to prevent tracker from
 *                                hanging indefinitely (in milliseconds)
 *                                (supported by "scala" implementation only.)
 * @param trackerImpl Choice between "python" or "scala". The former utilizes the Java wrapper of
 *                    the Python Rabit tracker (in dmlc_core), whereas the latter is implemented
 *                    in Scala without Python components, and with full support of timeouts.
 *                    The Scala implementation is currently experimental, use at your own risk.
 */
case class TrackerConf(workerConnectionTimeout: Long, trackerImpl: String)

object TrackerConf {
  def apply(): TrackerConf = TrackerConf(0L, "python")
}

/**
 * Traing data group in a RDD partition.
 * @param groupId The group id
 * @param points Array of XGBLabeledPoint within the same group.
 * @param isEdgeGroup whether it is a frist or last group in a RDD partition.
 */
private[spark] case class XGBLabeledPointGroup(
    groupId: Int,
    points: Array[XGBLabeledPoint],
    isEdgeGroup: Boolean)

// ========= GPU Pipeline Start =============
private[spark] case class GDFColumnData(
    rawRDD: RDD[Table],
    colsIndices: Seq[Array[Int]])
// ========= GPU Pipeline End =============

object XGBoost extends Serializable {
  private val logger = LogFactory.getLog("XGBoostSpark")
  private val trainName = "train"

  private def verifyMissingSetting(xgbLabelPoints: Iterator[XGBLabeledPoint], missing: Float):
      Iterator[XGBLabeledPoint] = {
    if (missing != 0.0f) {
      xgbLabelPoints.map(labeledPoint => {
        if (labeledPoint.indices != null) {
            throw new RuntimeException(s"you can only specify missing value as 0.0 (the currently" +
              s" set value $missing) when you have SparseVector or Empty vector as your feature" +
              " format")
        }
        labeledPoint
      })
    } else {
      xgbLabelPoints
    }
  }

  private def removeMissingValues(
      xgbLabelPoints: Iterator[XGBLabeledPoint],
      missing: Float,
      keepCondition: Float => Boolean): Iterator[XGBLabeledPoint] = {
    xgbLabelPoints.map { labeledPoint =>
      val indicesBuilder = new mutable.ArrayBuilder.ofInt()
      val valuesBuilder = new mutable.ArrayBuilder.ofFloat()
      for ((value, i) <- labeledPoint.values.zipWithIndex if keepCondition(value)) {
        indicesBuilder += (if (labeledPoint.indices == null) i else labeledPoint.indices(i))
        valuesBuilder += value
      }
      labeledPoint.copy(indices = indicesBuilder.result(), values = valuesBuilder.result())
    }
  }

  private[spark] def processMissingValues(
      xgbLabelPoints: Iterator[XGBLabeledPoint],
      missing: Float): Iterator[XGBLabeledPoint] = {
    if (!missing.isNaN) {
      removeMissingValues(verifyMissingSetting(xgbLabelPoints, missing),
        missing, (v: Float) => v != missing)
    } else {
      removeMissingValues(verifyMissingSetting(xgbLabelPoints, missing),
        missing, (v: Float) => !v.isNaN)
    }
  }

  private def processMissingValuesWithGroup(
      xgbLabelPointGroups: Iterator[Array[XGBLabeledPoint]],
      missing: Float): Iterator[Array[XGBLabeledPoint]] = {
    if (!missing.isNaN) {
      xgbLabelPointGroups.map {
        labeledPoints => XGBoost.processMissingValues(labeledPoints.iterator, missing).toArray
      }
    } else {
      xgbLabelPointGroups
    }
  }

  private def getCacheDirName(useExternalMemory: Boolean): Option[String] = {
    val taskId = TaskContext.getPartitionId().toString
    if (useExternalMemory) {
      val dir = Files.createTempDirectory(s"${TaskContext.get().stageId()}-cache-$taskId")
      Some(dir.toAbsolutePath.toString)
    } else {
      None
    }
  }

  private def buildDistributedBooster(
      watches: Watches,
      params: Map[String, Any],
      rabitEnv: java.util.Map[String, String],
      round: Int,
      obj: ObjectiveTrait,
      eval: EvalTrait,
      prevBooster: Booster): Iterator[(Booster, Map[String, Array[Float]])] = {
    val taskId = TaskContext.getPartitionId().toString
    rabitEnv.put("DMLC_TASK_ID", taskId)
    rabitEnv.put("DMLC_WORKER_STOP_PROCESS_ON_ERROR", "false")

    try {
      Rabit.init(rabitEnv)
      // to workaround the empty partitions in training dataset,
      // this might not be the best efficient implementation, see
      // (https://github.com/dmlc/xgboost/issues/1277)
      val dmMap = watches.toMap
      if (!dmMap.contains(trainName) || dmMap(trainName).rowNum == 0) {
        throw new XGBoostError(
          s"detected an empty partition in the training data, partition ID:" +
            s" ${TaskContext.getPartitionId()}")
      }

      val numEarlyStoppingRounds = params.get("num_early_stopping_rounds")
        .map(_.toString.toInt).getOrElse(0)
      val overridedParams = if (numEarlyStoppingRounds > 0 &&
          !params.contains("maximize_evaluation_metrics")) {
        if (params.contains("custom_eval")) {
            throw new IllegalArgumentException("maximize_evaluation_metrics has to be "
                + "specified when custom_eval is set")
        }
        val eval_metric = params("eval_metric").toString
        val maximize = LearningTaskParams.evalMetricsToMaximize contains eval_metric
        logger.info("parameter \"maximize_evaluation_metrics\" is set to " + maximize)
        params + ("maximize_evaluation_metrics" -> maximize)
      } else {
        params
      }
      val metrics = Array.tabulate(watches.size)(_ => Array.ofDim[Float](round))

      // infer num class
      if (watches.inferNumClass) {
        val element: Array[Float] = Array(watches.numClass.toFloat)
        val numClassElement = Rabit.allReduce(element, Rabit.OpType.MAX)

        if (numClassElement.length <= 0) {
          throw new XGBoostError("Couldn't infer to class numer")
        }
        val inferedNumClass = numClassElement(0).toInt + 1

        logger.debug("Infered num class: " + inferedNumClass)
        if (params.contains("num_class")) {
          val userNumClass = params("num_class")
          require(userNumClass == inferedNumClass, "The number of classes in Dataset doesn't" +
            " match  \'num_class\' in parameters.")
        }
      }

      val booster = SXGBoost.train(dmMap(trainName), overridedParams, round, dmMap,
          metrics, obj, eval, earlyStoppingRound = numEarlyStoppingRounds, prevBooster)
      Iterator(booster -> watches.toMap.keys.zip(metrics).toMap)
    } catch {
      case xgbException: XGBoostError =>
        logger.error(s"XGBooster worker $taskId has failed due to ", xgbException)
        throw xgbException
    } finally {
      Rabit.shutdown()
      watches.delete()
    }
  }

  private def overrideParamsAccordingToTaskCPUs(
      params: Map[String, Any],
      sc: SparkContext): Map[String, Any] = {
    val coresPerTask = sc.getConf.getInt("spark.task.cpus", 1)
    var overridedParams = params
    if (overridedParams.contains("nthread")) {
      val nThread = overridedParams("nthread").toString.toInt
      require(nThread <= coresPerTask,
        s"the nthread configuration ($nThread) must be no larger than " +
          s"spark.task.cpus ($coresPerTask)")
    } else {
      overridedParams = params + ("nthread" -> coresPerTask)
    }
    overridedParams
  }

  private def startTracker(nWorkers: Int, trackerConf: TrackerConf): IRabitTracker = {
    val tracker: IRabitTracker = trackerConf.trackerImpl match {
      case "scala" => new RabitTracker(nWorkers)
      case "python" => new PyRabitTracker(nWorkers)
      case _ => new PyRabitTracker(nWorkers)
    }

    require(tracker.start(trackerConf.workerConnectionTimeout), "FAULT: Failed to start tracker")
    tracker
  }

  class IteratorWrapper[T](arrayOfXGBLabeledPoints: Array[(String, Iterator[T])])
    extends Iterator[(String, Iterator[T])] {

    private var currentIndex = 0

    override def hasNext: Boolean = currentIndex <= arrayOfXGBLabeledPoints.length - 1

    override def next(): (String, Iterator[T]) = {
      currentIndex += 1
      arrayOfXGBLabeledPoints(currentIndex - 1)
    }
  }

  private def coPartitionNoGroupSets(
      trainingData: RDD[XGBLabeledPoint],
      evalSets: Map[String, RDD[XGBLabeledPoint]],
      nWorkers: Int) = {
    // eval_sets is supposed to be set by the caller of [[trainDistributed]]
    val allDatasets = Map("train" -> trainingData) ++ evalSets
    val repartitionedDatasets = allDatasets.map{case (name, rdd) =>
      if (rdd.getNumPartitions != nWorkers) {
        (name, rdd.repartition(nWorkers))
      } else {
        (name, rdd)
      }
    }
    repartitionedDatasets.foldLeft(trainingData.sparkContext.parallelize(
      Array.fill[(String, Iterator[XGBLabeledPoint])](nWorkers)(null), nWorkers)){
      case (rddOfIterWrapper, (name, rddOfIter)) =>
        rddOfIterWrapper.zipPartitions(rddOfIter){
          (itrWrapper, itr) =>
            if (!itr.hasNext) {
              logger.error("when specifying eval sets as dataframes, you have to ensure that " +
                "the number of elements in each dataframe is larger than the number of workers")
              throw new Exception("too few elements in evaluation sets")
            }
            val itrArray = itrWrapper.toArray
            if (itrArray.head != null) {
              new IteratorWrapper(itrArray :+ (name -> itr))
            } else {
              new IteratorWrapper(Array(name -> itr))
            }
        }
    }
  }

  /**
   * Check to see if Spark expects SSL encryption (`spark.ssl.enabled` set to true).
   * If so, throw an exception unless this safety measure has been explicitly overridden
   * via conf `xgboost.spark.ignoreSsl`.
   *
   * @param sc  SparkContext for the training dataset.  When looking for the confs, this method
   *            first checks for an active SparkSession.  If one is not available, it falls back
   *            to this SparkContext.
   */
  private def validateSparkSslConf(sc: SparkContext): Unit = {
    val (sparkSslEnabled: Boolean, xgboostSparkIgnoreSsl: Boolean) =
      SparkSession.getActiveSession match {
        case Some(ss) =>
          (ss.conf.getOption("spark.ssl.enabled").getOrElse("false").toBoolean,
            ss.conf.getOption("xgboost.spark.ignoreSsl").getOrElse("false").toBoolean)
        case None =>
          (sc.getConf.getBoolean("spark.ssl.enabled", false),
            sc.getConf.getBoolean("xgboost.spark.ignoreSsl", false))
      }
    if (sparkSslEnabled) {
      if (xgboostSparkIgnoreSsl) {
        logger.warn(s"spark-xgboost is being run without encrypting data in transit!  " +
          s"Spark Conf spark.ssl.enabled=true was overridden with xgboost.spark.ignoreSsl=true.")
      } else {
        throw new Exception("xgboost-spark found spark.ssl.enabled=true to encrypt data " +
          "in transit, but xgboost-spark sends non-encrypted data over the wire for efficiency. " +
          "To override this protection and still use xgboost-spark at your own risk, " +
          "you can set the SparkSession conf to use xgboost.spark.ignoreSsl=true.")
      }
    }
  }

  private def parameterFetchAndValidation(params: Map[String, Any], sparkContext: SparkContext) = {
    val nWorkers = params("num_workers").asInstanceOf[Int]
    val round = params("num_round").asInstanceOf[Int]
    val useExternalMemory = params("use_external_memory").asInstanceOf[Boolean]
    val obj = params.getOrElse("custom_obj", null).asInstanceOf[ObjectiveTrait]
    val eval = params.getOrElse("custom_eval", null).asInstanceOf[EvalTrait]
    val missing = params.getOrElse("missing", Float.NaN).asInstanceOf[Float]
    validateSparkSslConf(sparkContext)

    if (params.contains("tree_method")) {
      require(BoosterParams.supportedTreeMethods.contains(params("tree_method")
        .asInstanceOf[String]), "xgboost4j-spark only supports tree_method as [" +
        s"${BoosterParams.supportedTreeMethods.mkString(", ")}]")
    }
    if (params.contains("train_test_ratio")) {
      logger.warn("train_test_ratio is deprecated since XGBoost 0.82, we recommend to explicitly" +
        " pass a training and multiple evaluation datasets by passing 'eval_sets' as" +
        " Map('name'->'Dataset')")
    }
    require(nWorkers > 0, "you must specify more than 0 workers")
    if (obj != null) {
      require(params.get("objective_type").isDefined, "parameter \"objective_type\" is not" +
        " defined, you have to specify the objective type as classification or regression" +
        " with a customized objective function")
    }
    val trackerConf = params.get("tracker_conf") match {
      case None => TrackerConf()
      case Some(conf: TrackerConf) => conf
      case _ => throw new IllegalArgumentException("parameter \"tracker_conf\" must be an " +
        "instance of TrackerConf.")
    }
    val timeoutRequestWorkers: Long = params.get("timeout_request_workers") match {
      case None => 0L
      case Some(interval: Long) => interval
      case _ => throw new IllegalArgumentException("parameter \"timeout_request_workers\" must be" +
        " an instance of Long.")
    }
    val (checkpointPath, checkpointInterval) = CheckpointManager.extractParams(params)
    (nWorkers, round, useExternalMemory, obj, eval, missing, trackerConf, timeoutRequestWorkers,
      checkpointPath, checkpointInterval)
  }

  private def trainForNonRanking(
      trainingData: RDD[XGBLabeledPoint],
      params: Map[String, Any],
      rabitEnv: java.util.Map[String, String],
      checkpointRound: Int,
      prevBooster: Booster,
      evalSetsMap: Map[String, RDD[XGBLabeledPoint]]): RDD[(Booster, Map[String, Array[Float]])] = {
    val (nWorkers, _, useExternalMemory, obj, eval, missing, _, _, _, _) =
      parameterFetchAndValidation(params, trainingData.sparkContext)
    if (evalSetsMap.isEmpty) {
      trainingData.mapPartitions(labeledPoints => {
        val watches = Watches.buildWatches(params,
          processMissingValues(labeledPoints, missing),
          getCacheDirName(useExternalMemory))
        buildDistributedBooster(watches, params, rabitEnv, checkpointRound,
          obj, eval, prevBooster)
      }).cache()
    } else {
      coPartitionNoGroupSets(trainingData, evalSetsMap, nWorkers).mapPartitions {
        nameAndLabeledPointSets =>
          val watches = Watches.buildWatches(
            nameAndLabeledPointSets.map {
              case (name, iter) => (name, processMissingValues(iter, missing))},
            getCacheDirName(useExternalMemory))
          buildDistributedBooster(watches, params, rabitEnv, checkpointRound,
            obj, eval, prevBooster)
      }.cache()
    }
  }

  private def trainForRanking(
      trainingData: RDD[Array[XGBLabeledPoint]],
      params: Map[String, Any],
      rabitEnv: java.util.Map[String, String],
      checkpointRound: Int,
      prevBooster: Booster,
      evalSetsMap: Map[String, RDD[XGBLabeledPoint]]): RDD[(Booster, Map[String, Array[Float]])] = {
    val (nWorkers, _, useExternalMemory, obj, eval, missing, _, _, _, _) =
      parameterFetchAndValidation(params, trainingData.sparkContext)
    if (evalSetsMap.isEmpty) {
      trainingData.mapPartitions(labeledPointGroups => {
        val watches = Watches.buildWatchesWithGroup(params,
          processMissingValuesWithGroup(labeledPointGroups, missing),
          getCacheDirName(useExternalMemory))
        buildDistributedBooster(watches, params, rabitEnv, checkpointRound, obj, eval, prevBooster)
      }).cache()
    } else {
      coPartitionGroupSets(trainingData, evalSetsMap, nWorkers).mapPartitions(
        labeledPointGroupSets => {
          val watches = Watches.buildWatchesWithGroup(
            labeledPointGroupSets.map {
              case (name, iter) => (name, processMissingValuesWithGroup(iter, missing))
            },
            getCacheDirName(useExternalMemory))
          buildDistributedBooster(watches, params, rabitEnv, checkpointRound, obj, eval,
            prevBooster)
        }).cache()
    }
  }

  private def cacheData(ifCacheDataBoolean: Boolean, input: RDD[_]): RDD[_] = {
    if (ifCacheDataBoolean) input.persist(StorageLevel.MEMORY_AND_DISK) else input
  }

  private def composeInputData(
    trainingData: RDD[XGBLabeledPoint],
    ifCacheDataBoolean: Boolean,
    hasGroup: Boolean,
    nWorkers: Int): Either[RDD[Array[XGBLabeledPoint]], RDD[XGBLabeledPoint]] = {
    if (hasGroup) {
      val repartitionedData = repartitionForTrainingGroup(trainingData, nWorkers)
      Left(cacheData(ifCacheDataBoolean, repartitionedData).
        asInstanceOf[RDD[Array[XGBLabeledPoint]]])
    } else {
      val repartitionedData = repartitionForTraining(trainingData, nWorkers)
      Right(cacheData(ifCacheDataBoolean, repartitionedData).asInstanceOf[RDD[XGBLabeledPoint]])
    }
  }

// ========= GPU Pipeline Start =============
  private def parameterOverrideToUseGPU(params: Map[String, Any]): Map[String, Any] = {
    var updatedParams = params
    val treeMethod = "tree_method"
    if(updatedParams.contains(treeMethod)) {
      // This is called after 'parameterFetchAndValidation', so we only need to make sure
      // "tree_method" is gpu_XXX
      val tmValue = updatedParams(treeMethod).asInstanceOf[String]
      if (tmValue == "auto") {
        // Choose "gpu_hist" for GPU training when auto is set
        updatedParams = updatedParams + (treeMethod -> "gpu_hist")
      } else {
        require(tmValue.startsWith("gpu_"),
          "Now for training on GPU, xgboost-spark only supports tree_method as " +
            s"[${BoosterParams.supportedTreeMethods.filter(_.startsWith("gpu_")).mkString(", ")}]" +
            s", but found '$tmValue'")
      }
    } else {
      // Add "gpu_hist" as default for GPU training if not set
      updatedParams = updatedParams + (treeMethod -> "gpu_hist")
    }
    updatedParams
  }

  // repartition all the Columnar RDDs (training and evaluation) to nWorkers,
  // and get the GDF column indices separately from each Columnar RDD.
  // Then wrap this repartitioned RDD and columns indices by a map
  private def prepareDataForGpu(
      trainingData: GDFColumnData,
      evalSetsMap: Map[String, GDFColumnData],
      nWorkers: Int,
      params: Map[String, Any]): Map[String, GDFColumnData] = {
    // Cache is not supported
    val isCacheData = params.getOrElse("cacheTrainingSet", false).asInstanceOf[Boolean]
    if (isCacheData) {
      logger.warn("Data cache is not support for Gpu pipeline!")
    }
    (evalSetsMap + (trainName -> trainingData)).map {
      case (name, colData) =>
        if (colData.rawRDD.getNumPartitions != nWorkers) {
          val newColumnarRDD = colData.rawRDD.repartition(nWorkers)
          name -> GDFColumnData(newColumnarRDD, colData.colsIndices)
        } else {
          (name, colData)
        }
    }
  }

  // zip all the Columnar RDDs into one RDD containing named GDF column batch.
  private def coPartitionForGpu(
      dataMap: Map[String, GDFColumnData],
      sc: SparkContext,
      nWorkers: Int): RDD[(String, Iterator[Table])] = {
    val emptyDataRdd = sc.parallelize(
      Array.fill[(String, Iterator[Table])](nWorkers)(null), nWorkers)

    dataMap.foldLeft(emptyDataRdd) {
      case (zippedRdd, (name, gdfColData)) =>
        zippedRdd.zipPartitions(gdfColData.rawRDD) {
          (itWrapper, itTable) =>
            (itWrapper.toArray :+ (name -> itTable)).filter(x => x != null).toIterator
        }
    }
  }

  // This method is running on executor side
  private def appendGpuIdToParameters(params: Map[String, Any]): (Int, Map[String, Any]) = {
    // Call allocateGpuDevice to force assignment of GPU when in exclusive process mode
    // and pass that as the gpu_id, assumption is that if you are using CUDA_VISIBLE_DEVICES
    // it doesn't hurt to call allocateGpuDevice so just always do it.
    val gpuId = XGBoostSparkJNI.allocateGpuDevice()
    logger.info("XGboost GPU training using device: " + gpuId)
    if (gpuId != 0) {
      (gpuId, params + ("gpu_id" -> gpuId.toString()))
    } else {
      (gpuId, params)
    }
  }

  @throws(classOf[XGBoostError])
  private def trainPreferGpu(
      sc: SparkContext,
      dataMap: Map[String, GDFColumnData],
      noEvalSet: Boolean,
      params: Map[String, Any],
      rabitEnv: java.util.Map[String, String],
      checkpointRound: Int,
      prevBooster: Booster,
      inferNumClass: Boolean = false): RDD[(Booster, Map[String, Array[Float]])] = {
    val updatedParams = parameterOverrideToUseGPU(params)
    val (nWorkers, _, useExternalMemory, obj, eval, missing, _, _, _, _) =
      parameterFetchAndValidation(updatedParams, sc)

    // Start training
    if (noEvalSet) {
      // Get the indices here at driver side to avoid passing the whole Map to executor(s)
      val colIndicesForTrain = dataMap(trainName).colsIndices
      dataMap(trainName).rawRDD.mapPartitions({
        iter: Iterator[Table] =>
          val (gpuId, paramsWithGpuId) = appendGpuIdToParameters(updatedParams)

          val watches = Watches.buildWatches(getCacheDirName(useExternalMemory),
            gpuId, missing, colIndicesForTrain, iter, inferNumClass)
          buildDistributedBooster(watches, paramsWithGpuId, rabitEnv, checkpointRound,
            obj, eval, prevBooster)
      }).cache()
    } else {
      // Train with evaluation sets
      // Get the indices here at driver side to avoid passing the whole Map to executor(s)
      val nameAndColIndices = dataMap.map(nc => (nc._1, nc._2.colsIndices))
      coPartitionForGpu(dataMap, sc, nWorkers).mapPartitions {
        nameAndColumnBatchIter =>
          val (gpuId, paramsWithGpuId) = appendGpuIdToParameters(updatedParams)

          val watches = Watches.buildWatchesWithEval(getCacheDirName(useExternalMemory), gpuId,
            missing, nameAndColIndices, nameAndColumnBatchIter, inferNumClass)
          buildDistributedBooster(watches, paramsWithGpuId, rabitEnv, checkpointRound,
            obj, eval, prevBooster)
      }.cache()
    }
  }

  /**
    * This version is to train on GPU by default
    * @return A tuple of the booster and the metrics used to build training summary
    */
  @throws(classOf[XGBoostError])
  private[spark] def trainDistributedPreferGpu(
      trainingData: GDFColumnData,
      params: Map[String, Any],
      evalSetsMap: Map[String, GDFColumnData] = Map(),
      inferNumClass: Boolean = false): (Booster, Map[String, Array[Float]]) = {
    logger.info(s"Gpu Running XGBoost ${spark.VERSION} with parameters: " +
      s"\n${params.mkString("\n")}")
    // First check and get parameters.
    // Second prepare the training data and run training
    // Then setup checkpoint manager
    val sc = trainingData.rawRDD.sparkContext
    val (nWorkers, round, _, _, _, _, trackerConf, timeoutRequestWorkers,
      checkpointPath, checkpointInterval) = parameterFetchAndValidation(params, sc)
    val dataMap = prepareDataForGpu(trainingData, evalSetsMap, nWorkers, params)
    val checkpointManager = new CheckpointManager(sc, checkpointPath)
    checkpointManager.cleanUpHigherVersions(round.asInstanceOf[Int])
    var prevBooster = checkpointManager.loadCheckpointAsBooster
    try {
      // Train for every ${savingRound} rounds and save the partially completed booster
      checkpointManager.getCheckpointRounds(checkpointInterval, round).map {
        checkpointRound: Int =>
          val tracker = startTracker(nWorkers, trackerConf)
          try {
            val overriddenParams = overrideParamsAccordingToTaskCPUs(params, sc)
            val parallelismTracker = new SparkParallelismTracker(sc, timeoutRequestWorkers,
              nWorkers)
            val boostersAndMetrics = trainPreferGpu(sc, dataMap, evalSetsMap.isEmpty,
              overriddenParams, tracker.getWorkerEnvs, checkpointRound, prevBooster, inferNumClass)
            val sparkJobThread = new Thread() {
              override def run() {
                // force the job
                boostersAndMetrics.foreachPartition(() => _)
              }
            }
            sparkJobThread.setUncaughtExceptionHandler(tracker)
            sparkJobThread.start()
            val trackerReturnVal = parallelismTracker.execute(tracker.waitFor(0L))
            logger.info(s"Gpu Rabit returns with exit code $trackerReturnVal")
            val (booster, metrics) = postTrackerReturnProcessing(trackerReturnVal,
              boostersAndMetrics, sparkJobThread)
            if (checkpointRound < round) {
              prevBooster = booster
              checkpointManager.updateCheckpoint(prevBooster)
            }
            (booster, metrics)
          } finally {
            tracker.stop()
          }
      }.last
    } finally {
      // Cache is not supported
    }
  }
// ========= GPU Pipeline End =============

  /**
   * @return A tuple of the booster and the metrics used to build training summary
   */
  @throws(classOf[XGBoostError])
  private[spark] def trainDistributed(
      trainingData: RDD[XGBLabeledPoint],
      params: Map[String, Any],
      hasGroup: Boolean = false,
      evalSetsMap: Map[String, RDD[XGBLabeledPoint]] = Map()):
    (Booster, Map[String, Array[Float]]) = {
    logger.info(s"Running XGBoost ${spark.VERSION} with parameters:\n${params.mkString("\n")}")
    val (nWorkers, round, _, _, _, _, trackerConf, timeoutRequestWorkers,
      checkpointPath, checkpointInterval) = parameterFetchAndValidation(params,
      trainingData.sparkContext)
    val sc = trainingData.sparkContext
    val checkpointManager = new CheckpointManager(sc, checkpointPath)
    checkpointManager.cleanUpHigherVersions(round.asInstanceOf[Int])
    val transformedTrainingData = composeInputData(trainingData,
      params.getOrElse("cacheTrainingSet", false).asInstanceOf[Boolean], hasGroup, nWorkers)
    var prevBooster = checkpointManager.loadCheckpointAsBooster
    try {
      // Train for every ${savingRound} rounds and save the partially completed booster
      checkpointManager.getCheckpointRounds(checkpointInterval, round).map {
        checkpointRound: Int =>
          val tracker = startTracker(nWorkers, trackerConf)
          try {
            val overriddenParams = overrideParamsAccordingToTaskCPUs(params, sc)
            val parallelismTracker = new SparkParallelismTracker(sc, timeoutRequestWorkers,
              nWorkers)
            val rabitEnv = tracker.getWorkerEnvs
            val boostersAndMetrics = if (hasGroup) {
              trainForRanking(transformedTrainingData.left.get, overriddenParams, rabitEnv,
                checkpointRound, prevBooster, evalSetsMap)
            } else {
              trainForNonRanking(transformedTrainingData.right.get, overriddenParams, rabitEnv,
                checkpointRound, prevBooster, evalSetsMap)
            }
            val sparkJobThread = new Thread() {
              override def run() {
                // force the job
                boostersAndMetrics.foreachPartition(() => _)
              }
            }
            sparkJobThread.setUncaughtExceptionHandler(tracker)
            sparkJobThread.start()
            val trackerReturnVal = parallelismTracker.execute(tracker.waitFor(0L))
            logger.info(s"Rabit returns with exit code $trackerReturnVal")
            val (booster, metrics) = postTrackerReturnProcessing(trackerReturnVal,
              boostersAndMetrics, sparkJobThread)
            if (checkpointRound < round) {
              prevBooster = booster
              checkpointManager.updateCheckpoint(prevBooster)
            }
            (booster, metrics)
          } finally {
            tracker.stop()
          }
      }.last
    } catch {
      case t: Throwable =>
        // if the job was aborted due to an exception
        logger.error("the job was aborted due to ", t)
        trainingData.sparkContext.stop()
        throw t
    } finally {
      uncacheTrainingData(params.getOrElse("cacheTrainingSet", false).asInstanceOf[Boolean],
        transformedTrainingData)
    }
  }

  private def uncacheTrainingData(
      cacheTrainingSet: Boolean,
      transformedTrainingData: Either[RDD[Array[XGBLabeledPoint]], RDD[XGBLabeledPoint]]): Unit = {
    if (cacheTrainingSet) {
      if (transformedTrainingData.isLeft) {
        transformedTrainingData.left.get.unpersist()
      } else {
        transformedTrainingData.right.get.unpersist()
      }
    }
  }

  private[spark] def repartitionForTraining(trainingData: RDD[XGBLabeledPoint], nWorkers: Int) = {
    if (trainingData.getNumPartitions != nWorkers) {
      logger.info(s"repartitioning training set to $nWorkers partitions")
      trainingData.repartition(nWorkers)
    } else {
      trainingData
    }
  }

  private def aggByGroupInfo(trainingData: RDD[XGBLabeledPoint]) = {
    val normalGroups: RDD[Array[XGBLabeledPoint]] = trainingData.mapPartitions(
      // LabeledPointGroupIterator returns (Boolean, Array[XGBLabeledPoint])
      new LabeledPointGroupIterator(_)).filter(!_.isEdgeGroup).map(_.points)

    // edge groups with partition id.
    val edgeGroups: RDD[(Int, XGBLabeledPointGroup)] = trainingData.mapPartitions(
      new LabeledPointGroupIterator(_)).filter(_.isEdgeGroup).map(
      group => (TaskContext.getPartitionId(), group))

    // group chunks from different partitions together by group id in XGBLabeledPoint.
    // use groupBy instead of aggregateBy since all groups within a partition have unique group ids.
    val stitchedGroups: RDD[Array[XGBLabeledPoint]] = edgeGroups.groupBy(_._2.groupId).map(
      groups => {
        val it: Iterable[(Int, XGBLabeledPointGroup)] = groups._2
        // sorted by partition id and merge list of Array[XGBLabeledPoint] into one array
        it.toArray.sortBy(_._1).flatMap(_._2.points)
      })
    normalGroups.union(stitchedGroups)
  }

  private[spark] def repartitionForTrainingGroup(
      trainingData: RDD[XGBLabeledPoint], nWorkers: Int): RDD[Array[XGBLabeledPoint]] = {
    val allGroups = aggByGroupInfo(trainingData)
    logger.info(s"repartitioning training group set to $nWorkers partitions")
    allGroups.repartition(nWorkers)
  }

  private def coPartitionGroupSets(
      aggedTrainingSet: RDD[Array[XGBLabeledPoint]],
      evalSets: Map[String, RDD[XGBLabeledPoint]],
      nWorkers: Int): RDD[(String, Iterator[Array[XGBLabeledPoint]])] = {
    val repartitionedDatasets = Map("train" -> aggedTrainingSet) ++ evalSets.map {
      case (name, rdd) => {
        val aggedRdd = aggByGroupInfo(rdd)
        if (aggedRdd.getNumPartitions != nWorkers) {
          name -> aggedRdd.repartition(nWorkers)
        } else {
          name -> aggedRdd
        }
      }
    }
    repartitionedDatasets.foldLeft(aggedTrainingSet.sparkContext.parallelize(
      Array.fill[(String, Iterator[Array[XGBLabeledPoint]])](nWorkers)(null), nWorkers)){
      case (rddOfIterWrapper, (name, rddOfIter)) =>
        rddOfIterWrapper.zipPartitions(rddOfIter){
          (itrWrapper, itr) =>
            if (!itr.hasNext) {
              logger.error("when specifying eval sets as dataframes, you have to ensure that " +
                "the number of elements in each dataframe is larger than the number of workers")
              throw new Exception("too few elements in evaluation sets")
            }
            val itrArray = itrWrapper.toArray
            if (itrArray.head != null) {
              new IteratorWrapper(itrArray :+ (name -> itr))
            } else {
              new IteratorWrapper(Array(name -> itr))
            }
        }
    }
  }

  private def postTrackerReturnProcessing(
      trackerReturnVal: Int,
      distributedBoostersAndMetrics: RDD[(Booster, Map[String, Array[Float]])],
      sparkJobThread: Thread): (Booster, Map[String, Array[Float]]) = {
    if (trackerReturnVal == 0) {
      // Copies of the final booster and the corresponding metrics
      // reside in each partition of the `distributedBoostersAndMetrics`.
      // Any of them can be used to create the model.
      // it's safe to block here forever, as the tracker has returned successfully, and the Spark
      // job should have finished, there is no reason for the thread cannot return
      sparkJobThread.join()
      val (booster, metrics) = distributedBoostersAndMetrics.first()
      distributedBoostersAndMetrics.unpersist(false)
      (booster, metrics)
    } else {
      try {
        if (sparkJobThread.isAlive) {
          sparkJobThread.interrupt()
        }
      } catch {
        case _: InterruptedException =>
          logger.info("spark job thread is interrupted")
      }
      throw new XGBoostError("XGBoostModel training failed")
    }
  }

}

private class Watches private(
    val datasets: Array[DMatrix],
    val names: Array[String],
    val cacheDirName: Option[String],
    val inferNumClass: Boolean = false,
    val numClass: Double = 0.0) {

  def toMap: Map[String, DMatrix] = {
    names.zip(datasets).toMap.filter { case (_, matrix) => matrix.rowNum > 0 }
  }

  def size: Int = toMap.size

  def delete(): Unit = {
    toMap.values.foreach(_.delete())
    cacheDirName.foreach { name =>
      FileUtils.deleteDirectory(new File(name))
    }
  }

  override def toString: String = toMap.toString
}

private object Watches {
  private val logger = LogFactory.getLog("Watches")

  private def fromBaseMarginsToArray(baseMargins: Iterator[Float]): Option[Array[Float]] = {
    val builder = new mutable.ArrayBuilder.ofFloat()
    var nTotal = 0
    var nUndefined = 0
    while (baseMargins.hasNext) {
      nTotal += 1
      val baseMargin = baseMargins.next()
      if (baseMargin.isNaN) {
        nUndefined += 1  // don't waste space for all-NaNs.
      } else {
        builder += baseMargin
      }
    }
    if (nUndefined == nTotal) {
      None
    } else if (nUndefined == 0) {
      Some(builder.result())
    } else {
      throw new IllegalArgumentException(
        s"Encountered a partition with $nUndefined NaN base margin values. " +
          s"If you want to specify base margin, ensure all values are non-NaN.")
    }
  }

  // ========= GPU Pipeline Begin =============
  private def buildDMatrixIncrementally(gpuId: Int, missing: Float, indices: Seq[Array[Int]],
      iter: Iterator[Table], inferNumClass: Boolean = false): (DMatrix, Double) = {

    if (!missing.isNaN && missing != 0.0f) {
      throw new RuntimeException(s"you can only specify missing value as 0.0 (the currently" +
        s" set value $missing) when you load data from GPU")
    }
    val missingValue = 0.0f

    var isFirstBunch = true
    var dm: DMatrix = null
    var groupInfo: Array[Int] = Array()
    var isLtr = false

    var max: Double = Double.MinValue
    while (iter.hasNext) {
      val table = iter.next()

      // Suppose gdf column indices are given in this order <features, label, weight, group>,
      // Pls refer to method 'DataUtils.buildGDFColumnData'.
      // First transform indices to native handles for (features, label, weight)
      val gdfColsHandles = indices.take(3).map(_.map(table.getColumn(_).getNativeCudfColumnAddress))

      // Then set others (weight && group) as needed
      var weight_ = gdfColsHandles(2)
      val (weightIndices, groupIndices) = (indices(2), indices(3))
      // Disable LTR now
      require(groupIndices.isEmpty, "Learning to rank currently is not supported.")
/**
      if (groupIndices.nonEmpty) {

        isLtr = true
        // It is learning to rank
        logger.info("Learning to rank.")
        // Build and set group info.
        // Now native does not implement the GPU version for group information,
        // so need to fallback to CPU version.
        groupInfo = groupInfo ++ columnBatch.groupByColumnWithCountHost(groupIndices(0))
        if (weightIndices.nonEmpty) {
          val weightIdx = weightIndices(0)
          weight_ = columnBatch.groupByColumnWithAggregation(groupIndices(0),
            weightIdx.toInt, true)
          require(weight_ != null,
            "The instances in the same group have to be assigned with the same weight.")
        }
      }
*/
      if (isFirstBunch) {
        isFirstBunch = false
        // Suppose gdf columns are given in this order: features, label, weight,
        // the same with that in 'DataUtils.buildGDFColumnData'.
        dm = new DMatrix(gdfColsHandles(0), gpuId, missingValue)
        dm.setCUDFInfo("label", gdfColsHandles(1))
        if (weight_.nonEmpty) {
          dm.setCUDFInfo("weight", weight_)
        }
      } else {
        dm.appendCUDF(gdfColsHandles(0))
        dm.appendCUDFInfo("label", gdfColsHandles(1))
        if (weight_.nonEmpty) {
          dm.appendCUDFInfo("weight", weight_)
        }
      }

      if (inferNumClass) {
        // calculate max label to reduce
        indices(1).foreach(index => {
          val scalar = table.getColumn(index).max()
          if (scalar.isValid) {
            val tmp = scalar.getDouble
            max = if (max < tmp) tmp else max
          }
        })
      }
      // Close the table
      table.close
    }
    logger.debug("Num class: " + max)
    if (dm != null && isLtr) {
      dm.setGroup(groupInfo)
    }
    (dm, max)
  }

  def buildWatches(cachedDirName: Option[String], gpuId: Int, missing: Float,
      indices: Seq[Array[Int]], iter: Iterator[Table],
      inferNumClass: Boolean = false): Watches = {
    val ((dm, numClass), time) = GpuDataset.time("Train: Build DMatrix incrementally") {
      buildDMatrixIncrementally(gpuId, missing, indices, iter, inferNumClass)
    }
    logger.debug("Benchmark[Train: Build DMatrix incrementally] " + time)
    if (dm == null) {
      new Watches(Array.empty, Array.empty, cachedDirName, inferNumClass, numClass)
    } else {
      new Watches(Array(dm), Array("train"), cachedDirName, inferNumClass, numClass)
    }
  }

  def buildWatchesWithEval(cachedDirName: Option[String], gpuId: Int, missing: Float,
      indices: Map[String, Seq[Array[Int]]],
      nameAndGdfColumns: Iterator[(String, Iterator[Table])],
      inferNumClass: Boolean = false): Watches = {

    var numClass: Double = 0.0
    val dms = nameAndGdfColumns.map {
      case (name, iter) => (name, {
        val inferring = inferNumClass && name == "train"
        val ((dm, tmpNumClass), time) = GpuDataset.time(s"Train: Build $name DMatrix") {
          buildDMatrixIncrementally(gpuId, missing, indices(name), iter, inferring)
        }
        logger.debug(s"Benchmark[Train build $name DMatrix] " + time)
        if (inferring) {
          numClass = tmpNumClass
        }
        dm
      })
    }.filter(_._2 != null).toArray

    new Watches(dms.map(_._2), dms.map(_._1), cachedDirName, inferNumClass, numClass)
  }
// ========= GPU Pipeline End =============

  def buildWatches(
      nameAndLabeledPointSets: Iterator[(String, Iterator[XGBLabeledPoint])],
      cachedDirName: Option[String]): Watches = {
    val dms = nameAndLabeledPointSets.map {
      case (name, labeledPoints) =>
        val baseMargins = new mutable.ArrayBuilder.ofFloat
        val duplicatedItr = labeledPoints.map(labeledPoint => {
          baseMargins += labeledPoint.baseMargin
          labeledPoint
        })
        val dMatrix = new DMatrix(duplicatedItr, cachedDirName.map(_ + s"/$name").orNull)
        val baseMargin = fromBaseMarginsToArray(baseMargins.result().iterator)
        if (baseMargin.isDefined) {
          dMatrix.setBaseMargin(baseMargin.get)
        }
        (name, dMatrix)
    }.toArray
    new Watches(dms.map(_._2), dms.map(_._1), cachedDirName)
  }

  def buildWatches(
      params: Map[String, Any],
      labeledPoints: Iterator[XGBLabeledPoint],
      cacheDirName: Option[String]): Watches = {
    val trainTestRatio = params.get("train_test_ratio").map(_.toString.toDouble).getOrElse(1.0)
    val seed = params.get("seed").map(_.toString.toLong).getOrElse(System.nanoTime())
    val r = new Random(seed)
    val testPoints = mutable.ArrayBuffer.empty[XGBLabeledPoint]
    val trainBaseMargins = new mutable.ArrayBuilder.ofFloat
    val testBaseMargins = new mutable.ArrayBuilder.ofFloat
    val trainPoints = labeledPoints.filter { labeledPoint =>
      val accepted = r.nextDouble() <= trainTestRatio
      if (!accepted) {
        testPoints += labeledPoint
        testBaseMargins += labeledPoint.baseMargin
      } else {
        trainBaseMargins += labeledPoint.baseMargin
      }
      accepted
    }
    val trainMatrix = new DMatrix(trainPoints, cacheDirName.map(_ + "/train").orNull)
    val testMatrix = new DMatrix(testPoints.iterator, cacheDirName.map(_ + "/test").orNull)

    val trainMargin = fromBaseMarginsToArray(trainBaseMargins.result().iterator)
    val testMargin = fromBaseMarginsToArray(testBaseMargins.result().iterator)
    if (trainMargin.isDefined) trainMatrix.setBaseMargin(trainMargin.get)
    if (testMargin.isDefined) testMatrix.setBaseMargin(testMargin.get)

    new Watches(Array(trainMatrix, testMatrix), Array("train", "test"), cacheDirName)
  }

  def buildWatchesWithGroup(
      nameAndlabeledPointGroupSets: Iterator[(String, Iterator[Array[XGBLabeledPoint]])],
      cachedDirName: Option[String]): Watches = {
    val dms = nameAndlabeledPointGroupSets.map {
      case (name, labeledPointsGroups) =>
        val baseMargins = new mutable.ArrayBuilder.ofFloat
        val groupsInfo = new mutable.ArrayBuilder.ofInt
        val weights = new mutable.ArrayBuilder.ofFloat
        val iter = labeledPointsGroups.filter(labeledPointGroup => {
          var groupWeight = -1.0f
          var groupSize = 0
          labeledPointGroup.map { labeledPoint => {
            if (groupWeight < 0) {
              groupWeight = labeledPoint.weight
            } else if (groupWeight != labeledPoint.weight) {
              throw new IllegalArgumentException("the instances in the same group have to be" +
                s" assigned with the same weight (unexpected weight ${labeledPoint.weight}")
            }
            baseMargins += labeledPoint.baseMargin
            groupSize += 1
            labeledPoint
          }
          }
          weights += groupWeight
          groupsInfo += groupSize
          true
        })
        val dMatrix = new DMatrix(iter.flatMap(_.iterator), cachedDirName.map(_ + s"/$name").orNull)
        val baseMargin = fromBaseMarginsToArray(baseMargins.result().iterator)
        if (baseMargin.isDefined) {
          dMatrix.setBaseMargin(baseMargin.get)
        }
        dMatrix.setGroup(groupsInfo.result())
        dMatrix.setWeight(weights.result())
        (name, dMatrix)
    }.toArray
    new Watches(dms.map(_._2), dms.map(_._1), cachedDirName)
  }

  def buildWatchesWithGroup(
      params: Map[String, Any],
      labeledPointGroups: Iterator[Array[XGBLabeledPoint]],
      cacheDirName: Option[String]): Watches = {
    val trainTestRatio = params.get("train_test_ratio").map(_.toString.toDouble).getOrElse(1.0)
    val seed = params.get("seed").map(_.toString.toLong).getOrElse(System.nanoTime())
    val r = new Random(seed)
    val testPoints = mutable.ArrayBuilder.make[XGBLabeledPoint]
    val trainBaseMargins = new mutable.ArrayBuilder.ofFloat
    val testBaseMargins = new mutable.ArrayBuilder.ofFloat

    val trainGroups = new mutable.ArrayBuilder.ofInt
    val testGroups = new mutable.ArrayBuilder.ofInt

    val trainWeights = new mutable.ArrayBuilder.ofFloat
    val testWeights = new mutable.ArrayBuilder.ofFloat

    val trainLabelPointGroups = labeledPointGroups.filter { labeledPointGroup =>
      val accepted = r.nextDouble() <= trainTestRatio
      if (!accepted) {
        var groupWeight = -1.0f
        var groupSize = 0
        labeledPointGroup.foreach(labeledPoint => {
          testPoints += labeledPoint
          testBaseMargins += labeledPoint.baseMargin
          if (groupWeight < 0) {
            groupWeight = labeledPoint.weight
          } else if (labeledPoint.weight != groupWeight) {
            throw new IllegalArgumentException("the instances in the same group have to be" +
              s" assigned with the same weight (unexpected weight ${labeledPoint.weight}")
          }
          groupSize += 1
        })
        testWeights += groupWeight
        testGroups += groupSize
      } else {
        var groupWeight = -1.0f
        var groupSize = 0
        labeledPointGroup.foreach { labeledPoint => {
          if (groupWeight < 0) {
            groupWeight = labeledPoint.weight
          } else if (labeledPoint.weight != groupWeight) {
            throw new IllegalArgumentException("the instances in the same group have to be" +
              s" assigned with the same weight (unexpected weight ${labeledPoint.weight}")
          }
          trainBaseMargins += labeledPoint.baseMargin
          groupSize += 1
        }}
        trainWeights += groupWeight
        trainGroups += groupSize
      }
      accepted
    }

    val trainPoints = trainLabelPointGroups.flatMap(_.iterator)
    val trainMatrix = new DMatrix(trainPoints, cacheDirName.map(_ + "/train").orNull)
    trainMatrix.setGroup(trainGroups.result())
    trainMatrix.setWeight(trainWeights.result())

    val testMatrix = new DMatrix(testPoints.result().iterator, cacheDirName.map(_ + "/test").orNull)
    if (trainTestRatio < 1.0) {
      testMatrix.setGroup(testGroups.result())
      testMatrix.setWeight(testWeights.result())
    }

    val trainMargin = fromBaseMarginsToArray(trainBaseMargins.result().iterator)
    val testMargin = fromBaseMarginsToArray(testBaseMargins.result().iterator)
    if (trainMargin.isDefined) trainMatrix.setBaseMargin(trainMargin.get)
    if (testMargin.isDefined) testMatrix.setBaseMargin(testMargin.get)

    new Watches(Array(trainMatrix, testMatrix), Array("train", "test"), cacheDirName)
  }
}

/**
 * Within each RDD partition, group the <code>XGBLabeledPoint</code> by group id.</p>
 * And the first and the last groups may not have all the items due to the data partition.
 * <code>LabeledPointGroupIterator</code> orginaizes data in a tuple format:
 * (isFistGroup || isLastGroup, Array[XGBLabeledPoint]).</p>
 * The edge groups across partitions can be stitched together later.
 * @param base collection of <code>XGBLabeledPoint</code>
 */
private[spark] class LabeledPointGroupIterator(base: Iterator[XGBLabeledPoint])
  extends AbstractIterator[XGBLabeledPointGroup] {

  private var firstPointOfNextGroup: XGBLabeledPoint = null
  private var isNewGroup = false

  override def hasNext: Boolean = {
    base.hasNext || isNewGroup
  }

  override def next(): XGBLabeledPointGroup = {
    val builder = mutable.ArrayBuilder.make[XGBLabeledPoint]
    var isFirstGroup = true
    if (firstPointOfNextGroup != null) {
      builder += firstPointOfNextGroup
      isFirstGroup = false
    }

    isNewGroup = false
    while (!isNewGroup && base.hasNext) {
      val point = base.next()
      val groupId = if (firstPointOfNextGroup != null) firstPointOfNextGroup.group else point.group
      firstPointOfNextGroup = point
      if (point.group == groupId) {
        // add to current group
        builder += point
      } else {
        // start a new group
        isNewGroup = true
      }
    }

    val isLastGroup = !isNewGroup
    val result = builder.result()
    val group = XGBLabeledPointGroup(result(0).group, result, isFirstGroup || isLastGroup)

    group
  }
}

