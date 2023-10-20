# !/bin/bash
set -e
echo "pwd: $(pwd)"


function config_dipu_nv_cmake() {
    # export NCCL_ROOT="you nccl path should exist"

    mkdir -p build && cd ./build && rm -rf ./*
    cmake ../  -DCMAKE_BUILD_TYPE=Release \
        -DDEVICE=cuda \
        -DENABLE_COVERAGE=${USE_COVERAGE}
    cd ../
}

function build_diopi_lib() {
    cd third_party/DIOPI/
    cd impl
    which cmake
    sh scripts/build_impl.sh clean
    sh scripts/build_impl.sh torch || exit -1

    cd ../../..
    unset Torch_DIR
}

function build_dipu_lib() {
    echo "building dipu_lib:$(pwd)"
    echo  "DIOPI_ROOT:${DIOPI_ROOT}"
    export DIOPI_BUILD_TESTRT=1
    config_dipu_nv_cmake 2>&1 | tee ./cmake_nv.log
    cd build && make -j8  2>&1 | tee ./build.log &&  cd ..
}

case $1 in
    build_dipu)
        (
            build_diopi_lib
            build_dipu_lib
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
