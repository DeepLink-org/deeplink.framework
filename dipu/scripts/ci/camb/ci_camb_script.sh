# !/bin/bash
set -e
echo "pwd: $(pwd)"


function config_dipu_camb_cmake() {
    mkdir -p build && cd ./build && rm -rf ./*
    cmake ../  -DCMAKE_BUILD_TYPE=Release \
        -DDEVICE=camb  \
        -DENABLE_COVERAGE=${USE_COVERAGE}
    cd ../
}


function build_diopi_lib() {
    cd third_party/DIOPI/impl
    sh scripts/build_impl.sh clean
    sh scripts/build_impl.sh camb || exit -1
    cd -
}

function build_dipu_lib() {
    echo "building dipu_lib:$(pwd)"
    echo  "DIOPI_ROOT:${DIOPI_ROOT}"
    config_dipu_camb_cmake 2>&1 | tee ./cmake_camb.log
    cd build && make -j8  2>&1 | tee ./build.log &&  cd ..
}

case $1 in
    build_dipu)
        (
            bash scripts/ci/ci_build_third_party.sh
            build_diopi_lib
            build_dipu_lib
        ) || exit -1;;
    build_diopi)
        build_diopi_lib || exit -1;;
    build_dipu_only)
        (
            bash scripts/ci/ci_build_third_party.sh
            build_dipu_lib
        ) || exit -1;;
    *)
        echo -e "[ERROR] Incorrect option:" $1;
esac
exit 0
