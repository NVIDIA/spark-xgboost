###
# Script module to build xgboost jars:
#  1 xgboost4j for all cuda versions
#  2 xgboost4j-spark
#
# Input: <Build Arguments>
# Output: n/a
#
###

MVN_ARG=$1
MVN_LOCAL_REPO=$2
ORIG_PATH=`pwd`
XGB_ROOT=$WORKSPACE
# Place extra libs in folder "build", then will be cleaned before each build.
EXTRA_LIB_PATH=$XGB_ROOT/build/extra-libs

echo "MVN_ARG: " $MVN_ARG

cd $XGB_ROOT/jvm-packages
if [ "$MVN_LOCAL_REPO" == "" ]; then
MVN_LOCAL_REPO=`mvn exec:exec -q -B --non-recursive \
    -Dmaven.repo.local=$MVN_LOCAL_REPO \
    -Dexec.executable=echo \
    -Dexec.args='${settings.localRepository}'`
fi
echo "Maven local directory = $MVN_LOCAL_REPO"

#get cudf version
CUDF_VER=`mvn exec:exec -q -B --non-recursive \
    -Dmaven.repo.local=$MVN_LOCAL_REPO \
    -Dexec.executable=echo \
    -Dexec.args='${cudf.version}'`
echo "Current cudf version in pom = $CUDF_VER"

# Suppose called under jvm-packages
buildXgboost4j(){
    rm -rf ../build
    CUDA_VER=cuda$1
    CUDF_CLASS=$2
    . /opt/tools/to_$CUDA_VER.sh
    if [ "$CUDA_VER" == cuda10.1 ]; then
        mvn clean package -B -DskipTests $MVN_ARG -Dmaven.repo.local=$MVN_LOCAL_REPO
    else
        mvn dependency:get -B -Dmaven.repo.local=$MVN_LOCAL_REPO -Dtransitive=false \
            -DgroupId=ai.rapids \
            -DartifactId=cudf \
            -Dversion=$CUDF_VER \
            -Dclassifier=$CUDF_CLASS
        ./prepare_extra_libs.sh \
            $MVN_LOCAL_REPO \
            $EXTRA_LIB_PATH \
            $CUDF_VER \
            $CUDF_CLASS
        ./create_jni.py $CUDA_VER $EXTRA_LIB_PATH
    fi
}

####### build xgboost4j .so for and 10.2 ##
buildXgboost4j 10.2 cuda10-2

####### build xgboost4j .so for CUDA10.1 and jars ##
buildXgboost4j 10.1

cd $ORIG_PATH
