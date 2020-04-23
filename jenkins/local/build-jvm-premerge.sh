#!/bin/bash
##
#
# Script to build xgboost jar files.
#
# Source tree is supposed to be ready by Jenkins
# before starting this script.
#
###
set -e
gcc --version

BUILD_ARG="-Dmaven.repo.local=$WORKSPACE/.m2 -DskipTests -B -s settings.xml -Pmirror-apache-to-gpuwa"

cd jvm-packages
if [ "${CUDA_VER}"x == x ];then
   CUDA_VER="10.1"
fi
. /opt/tools/to_cuda${CUDA_VER}.sh

echo "CUDA_VER: $CUDA_VER, BUILD_ARG: $BUILD_ARG"

rm -rf ../build
mvn $BUILD_ARG clean package
cd ..
