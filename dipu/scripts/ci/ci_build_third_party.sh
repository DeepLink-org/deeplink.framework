# !/bin/bash
set -e

function build_kineto() {
    cd third_party/kineto/libkineto
    rm -rf build && mkdir -p build && cd build
    cmake .. -DKINETO_BUILD_TESTS=OFF -DKINETO_USE_DEVICE_ACTIVITY=ON
    make -j8
    cd ../../../..
}

build_kineto