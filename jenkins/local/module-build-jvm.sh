###
# Script module to build xgboost jars:
#  1 xgboost4j for all cuda versions
#  2 xgboost4j-spark
#
# Input: <Build Arguments>
# Output: n/a
#
###

MVN_ARG=$*
ORIG_PATH=`pwd`
XGB_ROOT=$WORKSPACE

echo "MVN_ARG: " $MVN_ARG

buildXgboost4j(){
    cd $XGB_ROOT/jvm-packages
    rm -rf ../build
    CUDA_VER=cuda$1
    . /opt/tools/to_$CUDA_VER.sh
    if [ "$CUDA_VER" == cuda10.0 ]; then
        mvn clean package -B $MVN_ARG
    else
        ./create_jni.py $CUDA_VER
    fi
}

####### build xgboost4j .so for and 10.1 ##
buildXgboost4j 10.1

####### build xgboost4j .so for CUDA10.0 and jars ##
buildXgboost4j 10.0

cd $ORIG_PATH
