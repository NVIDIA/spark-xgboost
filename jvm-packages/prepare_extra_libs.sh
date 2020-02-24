#!/bin/bash

# Arguments
MVN_LOCAL_REPO=$1
LIB_CACHE_PATH=$2
CUDF_JAR_VER=$3
CUDF_JAR_CLASS=$4

# Check the cache first
if [ -d ${LIB_CACHE_PATH} ]; then
    echo "Already extracted the libcudf.so and librmm.so."
else
    # CUDF_JAR: Determine the path of cudf jar file.
    CUDF_JAR_PATH=${MVN_LOCAL_REPO}/ai/rapids/cudf/${CUDF_JAR_VER}/cudf-${CUDF_JAR_VER}
    if [ "${CUDF_JAR_CLASS}" == "" ]; then
        CUDF_JAR_PATH=${CUDF_JAR_PATH}.jar
    else
        CUDF_JAR_PATH=${CUDF_JAR_PATH}-${CUDF_JAR_CLASS}.jar
    fi
    echo "Using cudf jar: ${CUDF_JAR_PATH}"

    # Extract the native libraries from jar file.
    unzip -o ${CUDF_JAR_PATH} "*/Linux/*.so" -d "${LIB_CACHE_PATH}"
    mv `find "${LIB_CACHE_PATH}" -name *.so` ${LIB_CACHE_PATH}
    ln -s libboost_filesystem.so ${LIB_CACHE_PATH}/libboost_filesystem.so.1.70.0
fi
