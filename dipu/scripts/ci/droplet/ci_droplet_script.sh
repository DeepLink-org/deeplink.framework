#!/usr/bin/env bash
set -eo pipefail
DIOPI_IMPL=${3:-droplet}
DIPU_DEVICE=${2:-droplet}


function config_dipu_droplet_cmake() {
    mkdir -p build && cd ./build && rm -rf ./*
    echo "config_dipu_nv_cmake PYTORCH_DIR: ${PYTORCH_DIR}"
    echo "config_dipu_nv_cmake PYTHON_INCLUDE_DIR: ${PYTHON_INCLUDE_DIR}"
    cmake ../  -DCMAKE_BUILD_TYPE=Debug \
     -DDEVICE=${DIPU_DEVICE} -DPYTORCH_DIR=${PYTORCH_DIR} \
     -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR}
      # -DCMAKE_C_FLAGS_DEBUG="-g -O0" \
      # -DCMAKE_CXX_FLAGS_DEBUG="-g -O0"
    cd ../
}

function config_all_droplet_cmake() {
    rm -rf ./build
    mkdir -p build
    echo "config_dipu_nv_cmake PYTORCH_DIR: ${PYTORCH_DIR}"
    echo "config_dipu_nv_cmake PYTHON_INCLUDE_DIR: ${PYTHON_INCLUDE_DIR}"

    args=(
        "-DCMAKE_BUILD_TYPE=Debug"
        "-DDEVICE=${DIPU_DEVICE}"
        "-DPYTORCH_DIR=${PYTORCH_DIR}"
        "-DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR}" )
    [[ ! -z "$DIOPI_ROOT" ]] && args+=( "-DWITH_DIOPI_LIBRARY=${DIOPI_ROOT}" )
    [[ ! -z "$DIOPI_PATH" ]] && args+=( "-DWITH_DIOPI_INCLUDE=${DIOPI_PATH}/include" )
    cmake -S . -B build "${args[@]}"
}

function build_diopi_lib() {
    echo "开始编译DIOPI..."
    cd third_party/DIOPI/
    git checkout .
    cd impl
    which cmake
    sh scripts/build_impl.sh clean
    #TODO: need to change if droplet decides to be exposed as shared lib
    sh scripts/build_impl.sh droplet || exit -1

    cd ../../..
}

function build_dipu_lib() { 
    echo "building dipu_lib:$(pwd)"
    echo  "DIOPI_ROOT:${DIOPI_ROOT}"
    echo  "PYTORCH_DIR:${PYTORCH_DIR}"
    echo  "PYTHON_INCLUDE_DIR:${PYTHON_INCLUDE_DIR}"
    export LIBRARY_PATH=$DIOPI_ROOT:$LIBRARY_PATH;
    config_dipu_droplet_cmake 2>&1 | tee ./cmake_droplet.log
    cd build && make -j8  2>&1 | tee ./build.log &&  cd ..
}

function build_all() {
    echo "building dipu_lib:$(pwd)"
    echo  "DIOPI_ROOT:${DIOPI_ROOT}"
    echo  "PYTORCH_DIR:${PYTORCH_DIR}"
    echo  "PYTHON_INCLUDE_DIR:${PYTHON_INCLUDE_DIR}"
    export LIBRARY_PATH=$DIOPI_ROOT:$LIBRARY_PATH;
    config_all_droplet_cmake 2>&1 | tee ./cmake_droplet.log
    cd build && make -j8  2>&1 | tee ./build.log &&  cd ..
}

case $1 in
    build_diopi)
        (
            build_diopi_lib
        ) \
        || exit -1;;
    build_dipu)
        (
            build_all
        ) \
        || exit -1;;
    build_dipu_only)
        (
            build_dipu_lib
        ) \
        || exit -1;;
    *)
        echo -e "[ERROR] Incorrect option:" $1;
esac
exit 0
