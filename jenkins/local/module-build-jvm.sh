###
# Script module to build xgboost jars:
#  1 xgboost4j for all cuda versions
#  2 xgboost4j-spark
#
# Input: <Build Arguments>
# Output: n/a
#
###

MVN_EXTRA_ARGS=$@
echo "MVN_ARG: " $MVN_EXTRA_ARGS
WD=$WORKSPACE
MVN="mvn -B -Dmaven.repo.local=$WD/.m2 -DskipTests"
CUDA_UTIL=$WD/jvm-packages/cudautils.py

SUPPORTED_VERS=(`$CUDA_UTIL l`)
NUM_VERS=${#SUPPORTED_VERS[@]}
echo "Supported cuda version: ${SUPPORTED_VERS[@]}"

cd $WD/jvm-packages/

for ((i=$NUM_VERS-1;i>=0;i--)); do
    CU_VER=${SUPPORTED_VERS[i]}
    . /opt/tools/to_cudaver.sh $CU_VER
    rm -rf ../build
    if [ $i -gt 0 ]; then
        ./create_jni.py
    else
        CLASSIFIER=`$CUDA_UTIL g`
        $MVN clean package -Dcudf.classifier=$CLASSIFIER $MVN_EXTRA_ARGS
    fi
done

cd $WD

