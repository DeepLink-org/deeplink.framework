#!/usr/bin/env bash


function config_dipu_cmake() {
    mkdir -p build && cd ./build && rm -rf ./*
    cmake ../  -DCMAKE_BUILD_TYPE=Debug \
     -DDEVICE=tops \
      # -DCMAKE_C_FLAGS_DEBUG="-g -O0" \
      # -DCMAKE_CXX_FLAGS_DEBUG="-g -O0"

    cd ../
}



function build_dipu_lib() {
    autogen_diopi_wrapper
    config_dipu_cmake
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
    bash scripts/ci/ci_build_third_party.sh
    build_dipu_lib
elif [[ "$1" == "build_diopi" ]]; then
    build_diopi_lib
elif [[ "$1" == "build" ]]; then
    build_diopi_lib
    bash scripts/ci/ci_build_third_party.sh
    build_dipu_lib
elif [[ "$1" == "clean" ]]; then
    build_diopi_clean
fi
