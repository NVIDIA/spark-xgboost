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

import ai.rapids.cudf.Cuda
import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging
import org.apache.spark.resource.ResourceInformation

import scala.collection.mutable.ArrayBuffer
import scala.util.control.NonFatal

object GpuDeviceManager extends Logging {

  def getGpuId(isLocal: Boolean): Int = {
    if (isLocal) {
      // if local mode, force to primary device
      0
    } else {
      // If Spark didn't provide the address we just use the default GPU.
      val resources = getResourcesFromTaskContext
      val addr = initializeGpu(resources)
      addr.getOrElse(findGpuAndAcquire())
    }
  }

  // Initializes the GPU if Spark assigned one.
  // Returns either the GPU addr Spark assigned or None if Spark didn't assign one.
  private def initializeGpu(resources: Map[String, ResourceInformation]): Option[Int] = {
    getGPUAddrFromResources(resources).map(setGpuDeviceAndAcquire(_))
  }

  private def getResourcesFromTaskContext: Map[String, ResourceInformation] = {
    val tc = TaskContext.get()
    if (tc == null) Map.empty[String, ResourceInformation] else tc.resources()
  }

  private def getGPUAddrFromResources(resources: Map[String, ResourceInformation]): Option[Int] = {
    if (resources.contains("gpu")) {
      val addrs = resources("gpu").addresses
      if (addrs.size > 1) {
        // Throw an exception since we assume one GPU per executor.
        // If multiple GPUs are allocated by spark, then different tasks could get assigned
        // different GPUs but RMM would only be initialized for 1. We could also just get
        // weird results that are hard to debug.
        throw new IllegalArgumentException("Spark GPU Plugin only supports 1 gpu per executor")
      }
      Some(addrs.head.toInt)
    } else {
      None
    }
  }

  /**
   * This is just being extra paranoid when we are running on GPUs in exclusive mode. It is not
   * clear exactly what cuda interactions cause us to acquire a device, so this will force us
   * to do an interaction we know will acquire the device.
   */
  private def findGpuAndAcquire(): Int = {
    logInfo("fall back to exclusive mode")
    val deviceCount: Int = Cuda.getDeviceCount()
    // loop multiple times to see if a GPU was released or something unexpected happened that
    // we couldn't acquire on first try
    var numRetries = 2
    val addrsToTry = ArrayBuffer.empty ++= (0 to (deviceCount - 1))
    while (numRetries > 0) {
      val addr = addrsToTry.find(tryToSetGpuDeviceAndAcquire)
      if (addr.isDefined) {
        return addr.get
      }
      numRetries -= 1
    }
    throw new IllegalStateException("Could not find a single GPU to use")
  }

  // Attempt to set and acquire the gpu, return true if acquired, false otherwise
  private def tryToSetGpuDeviceAndAcquire(addr: Int): Boolean = {
    try {
      GpuDeviceManager.setGpuDeviceAndAcquire(addr)
    } catch {
      case NonFatal(e) =>
        // we may have lost a race trying to acquire this addr or GPU is already busy
        return false
    }
    return true
  }

  private def setGpuDeviceAndAcquire(addr: Int): Int = {
    logDebug(s"Initializing GPU device ID to $addr")
    Cuda.setDevice(addr.toInt)
    // cudaFree(0) to actually allocate the set device - no process exclusive required
    // since we are relying on Spark to schedule it properly and not give it to multiple
    // executors
    Cuda.freeZero()
    addr
  }
}
