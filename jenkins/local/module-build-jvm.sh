###
# Script module to build xgboost jars:
#  1 xgboost4j for all cuda versions
#  2 xgboost4j-spark
#
# Input: <Build Arguments>
# Output: n/a
#
###

ORIG_PATH=`pwd`
MVN_ARGS=$@
echo "MVN_ARG: " $MVN_ARGS

## Suppose called under jvm-packages
buildXgboost4j(){
    rm -rf ../build
    CUDA_VER=cuda$1
    . /opt/tools/to_$CUDA_VER.sh
    if [ "$CUDA_VER" == 'cuda10.1' ]; then
        mvn clean package -B -DskipTests -Dmaven.repo.local=$WORKSPACE/.m2 $MVN_ARG
    else
        ./create_jni.py $CUDA_VER
    fi
}

cd $WORKSPACE/jvm-packages

####### build xgboost4j .so for and 10.2 ##
buildXgboost4j 10.2

####### build xgboost4j .so for CUDA10.1 and jars ##
buildXgboost4j 10.1

cd $ORIG_PATH

