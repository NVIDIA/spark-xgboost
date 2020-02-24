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
. /opt/tools/to_cuda10.0.sh
rm -rf ../build
mvn $BUILD_ARG clean package
cd ..
