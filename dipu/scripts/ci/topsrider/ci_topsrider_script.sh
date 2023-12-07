#!/usr/bin/env bash
set -eo pipefail

function config_dipu_cmake() {
    mkdir -p build && cd ./build && rm -rf ./*
    cmake ../  -DCMAKE_BUILD_TYPE=Debug \
     -DDEVICE=tops \
     -DWITH_DIOPI=INTERNAL
      # -DCMAKE_C_FLAGS_DEBUG="-g -O0" \
      # -DCMAKE_CXX_FLAGS_DEBUG="-g -O0"

    cd ../
}

function config_all_cmake() {
    mkdir -p build && cd ./build && rm -rf ./*
    cmake ../  -DCMAKE_BUILD_TYPE=Debug \
     -DDEVICE=tops \
     -DWITH_DIOPI=INTERNAL
      # -DCMAKE_C_FLAGS_DEBUG="-g -O0" \
      # -DCMAKE_CXX_FLAGS_DEBUG="-g -O0"
    cd ../
}

function build_dipu_lib() {
    config_dipu_cmake
    cd build && make -j8  2>&1 | tee ./build1.log &&  cd ..
}

function build_all() {
    config_all_cmake
    cd build && make -j8  2>&1 | tee ./build1.log &&  cd ..
}

function build_diopi_lib() {
    cd third_party/DIOPI/impl
    sh scripts/build_impl.sh tops || exit -1
    cd -
}

function build_diopi_clean() {
    cd third_party/DIOPI/impl
    sh scripts/build_impl.sh clean || exit -1
    cd -
}

if [[ "$1" == "builddl" ]]; then
    build_dipu_lib
elif [[ "$1" == "build_dipu" ]]; then
    build_all
elif [[ "$1" == "build_diopi" ]]; then
    build_diopi_lib
elif [[ "$1" == "build" ]]; then
    build_all
elif [[ "$1" == "clean" ]]; then
    build_diopi_clean
fi
