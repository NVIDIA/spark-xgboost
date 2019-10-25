###
# Script module to build xgboost jars:
#  1 xgboost4j for all cuda versions
#  2 xgboost4j-spark
#
# Input: <Build Arguments>
# Output: n/a
#
###

BUILD_ARG=$1
ORIGINAL_PATH=`pwd`
XGB_ROOT=$WORKSPACE
RMM_BUILD_DIR="$XGB_ROOT/rmm/build"

####### override the env RMM_ROOT ##
export RMM_ROOT="$XGB_ROOT/rmm_root"

buildRmm(){
    rm -rf $RMM_ROOT && rm -rf $RMM_BUILD_DIR
    mkdir -p $RMM_BUILD_DIR && cd $RMM_BUILD_DIR
    cmake .. -DCMAKE_CXX11_ABI=OFF -DCMAKE_INSTALL_PREFIX=$RMM_ROOT
    make -j6 install
}

buildXgboost4j(){
    cd $XGB_ROOT/jvm-packages
    rm -rf ../build
    if [ "$1" == cuda10.0 ]; then
        mvn clean package -B $BUILD_ARG
    else
        ./create_jni.py $1
    fi
}

####### Main ##
export GIT_COMMITTER_NAME="ci"
export GIT_COMMITTER_EMAIL="ci@nvidia.com"
cd $XGB_ROOT && rm -rf rmm
git clone --recurse-submodules https://github.com/rapidsai/rmm.git -b branch-0.10

####### build rmm and xgboost4j .so for CUDA9.2 ##
. /opt/tools/to_cuda9.2.sh
buildRmm
buildXgboost4j cuda9.2

####### build rmm and xgboost4j .so for CUDA10.1 ##
. /opt/tools/to_cuda10.1.sh
buildRmm
buildXgboost4j cuda10.1

####### build rmm and xgboost4j .so for CUDA10.0 and jars ##
. /opt/tools/to_cuda10.0.sh
buildRmm
buildXgboost4j cuda10.0

cd $ORIGINAL_PATH
unset GIT_COMMITTER_NAME
unset GIT_COMMITTER_EMAIL
