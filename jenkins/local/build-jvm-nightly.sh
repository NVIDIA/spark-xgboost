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

BUILD_ARG="-Dmaven.repo.local=$WORKSPACE/.m2 -Dcudf.classifier=cuda10 -DskipTests"
BUILD_MODULE=$WORKSPACE/jenkins/local/module-build-jvm.sh
. $BUILD_MODULE "deploy $BUILD_ARG"



