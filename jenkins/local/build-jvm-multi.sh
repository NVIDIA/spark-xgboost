#!/bin/bash
##
#
# Script to build xgboost jars for deployment
#
# Source tree is supposed to be ready by Jenkins
# before starting this script.
#
###
set -e
gcc --version

stashJars(){
    MODULE_OUT=$OUT/$1 && mkdir -p $MODULE_OUT
    cp -ft $MODULE_OUT $WORKSPACE/jvm-packages/$1/target/xgb*.jar
}

BUILD_MODULE=$WORKSPACE/jenkins/local/module-build-jvm.sh
BUILD_ARG="-s settings.xml -Pmirror-apache-to-gpuwa"

SIGN_FILE=$1 && echo "Sign Jar?: $SIGN_FILE"
if [ "$SIGN_FILE" == true ]; then
    # Build javadoc and sources only when SIGN_JAR
    BUILD_ARG="$BUILD_ARG -Prelease-to-sonatype,sonatype-stage"
fi

###### Build jars ##
. $BUILD_MODULE $BUILD_ARG

###### Stash jars ##
stashJars xgboost4j
stashJars xgboost4j-spark

