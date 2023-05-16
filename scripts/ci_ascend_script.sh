# !/bin/bash
set -e
echo "pwd: $(pwd)"

function build_diopi_lib() {
    cd third_party/DIOPI/DIOPI-IMPL
    sh scripts/build_impl.sh clean
    sh scripts/build_impl.sh ascend || exit -1
    cd -
}

function build_dipu_py() {
    echo "building dipu_py:$(pwd)"
    export CMAKE_BUILD_TYPE=debug
    export _GLIBCXX_USE_CXX11_ABI=1
    export MAX_JOBS=12
    python setup.py build_ext 2>&1 | tee ./build1.log
    cp build/python_ext/torch_dipu/_C.cpython-3?-x86_64-linux-gnu.so torch_dipu
}

function config_dipu_ascend_cmake() {
    mkdir -p build && cd ./build && rm -rf ./*
    echo "PYTORCH_DIR: ${PYTORCH_DIR}"
    echo "PYTHON_INCLUDE_DIR: ${PYTHON_INCLUDE_DIR}"
    cmake ../  -DCMAKE_BUILD_TYPE=Debug \
        -DDEVICE=ascend -DPYTORCH_DIR=${PYTORCH_DIR} \
        -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR}
    cd ../
}

function autogen_diopi_wrapper() {
    python scripts/autogen_diopi_wrapper/autogen_diopi_wrapper.py \
        --config scripts/autogen_diopi_wrapper/diopi_functions.yaml \
        --out torch_dipu/csrc_dipu/aten/ops/AutoGenedKernels.cpp
}

function build_dipu_lib() {
    echo "building dipu_lib:$(pwd)"
    export DIOPI_ROOT=$(pwd)/third_party/DIOPI/DIOPI-IMPL/lib
    echo  "DIOPI_ROOT:${DIOPI_ROOT}"
    export LIBRARY_PATH=$DIOPI_ROOT:$LIBRARY_PATH;
    config_dipu_ascend_cmake 2>&1 | tee ./build1.log
    cd build && make -j8  2>&1 | tee ./build1.log &&  cd ..
    cp ./build/torch_dipu/csrc_dipu/libtorch_dipu.so   ./torch_dipu
    cp ./build/torch_dipu/csrc_dipu/libtorch_dipu_python.so   ./torch_dipu
}

case $1 in
    build_dipu)
        (
            build_diopi_lib
            autogen_diopi_wrapper
            build_dipu_lib
            build_dipu_py
        ) \
        || exit -1;;
    *)
        echo -e "[ERROR] Incorrect option:" $1;
esac
exit 0
