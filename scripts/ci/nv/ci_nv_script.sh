# !/bin/bash
set -e
echo "pwd: $(pwd)"

function build_dipu_py() {
    echo "building dipu_py:$(pwd)"
    echo "building dipu_py PYTORCH_DIR: ${PYTORCH_DIR}"
    export CMAKE_BUILD_TYPE=debug
    export _GLIBCXX_USE_CXX11_ABI=1
    export MAX_JOBS=12
    python setup.py build_ext 2>&1 | tee ./setup.log
    cp build/python_ext/torch_dipu/_C.cpython-38-x86_64-linux-gnu.so torch_dipu
}

function config_dipu_nv_cmake() {
    # export NCCL_ROOT="you nccl path should exist"

    mkdir -p build && cd ./build && rm -rf ./*
    echo "config_dipu_nv_cmake PYTORCH_DIR: ${PYTORCH_DIR}"
    echo "config_dipu_nv_cmake PYTHON_INCLUDE_DIR: ${PYTHON_INCLUDE_DIR}"
    cmake ../  -DCMAKE_BUILD_TYPE=Debug \
        -DDEVICE=cuda -DPYTORCH_DIR=${PYTORCH_DIR} \
        -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR}
    cd ../
}

function autogen_diopi_wrapper() {
    python scripts/autogen_diopi_wrapper/autogen_diopi_wrapper.py                   \
        --config scripts/autogen_diopi_wrapper/diopi_functions.yaml                 \
        --out torch_dipu/csrc_dipu/aten/ops/AutoGenedKernels.cpp                    \
        --use_diopi_adapter False                                                   \
        --autocompare False                                                         \
        --print_func_call_info False                                               \
        --print_op_arg True
}

function build_diopi_lib() {
    cd third_party/DIOPI/
    git checkout .
    cd DIOPI-IMPL
    echo "build_diopi_lib PATH: ${PATH}"
    which cmake
    # sed -i "/option(HIP/a set(Torch_DIR $Torch_DIR)" torch/CMakeLists.txt
    sh scripts/build_impl.sh clean
    sh scripts/build_impl.sh torch || exit -1

    cd ../../..
    unset Torch_DIR
}

function build_dipu_lib() {
    echo "building dipu_lib:$(pwd)"
    echo  "DIOPI_ROOT:${DIOPI_ROOT}"
    echo  "PYTORCH_DIR:${PYTORCH_DIR}"
    echo  "PYTHON_INCLUDE_DIR:${PYTHON_INCLUDE_DIR}"
    export LIBRARY_PATH=$DIOPI_ROOT:$LIBRARY_PATH;
    config_dipu_nv_cmake 2>&1 | tee ./cmake_nv.log
    cd build && make -j8  2>&1 | tee ./build.log &&  cd ..
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
    build_dipu_only)
        (
            autogen_diopi_wrapper
            build_dipu_lib
            build_dipu_py
        ) \
        || exit -1;;
    *)
        echo -e "[ERROR] Incorrect option:" $1;
esac
exit 0
