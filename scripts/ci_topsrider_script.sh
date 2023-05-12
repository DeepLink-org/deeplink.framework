#!/usr/bin/env bash

function build_dipu_py() {
    export CMAKE_BUILD_TYPE=debug
    export _GLIBCXX_USE_CXX11_ABI=1
    export MAX_JOBS=12
    # PYTORCH_INSTALL_DIR is /you_pytorch/torch20/pytorch/torch
    # python  setup.py build_clib 2>&1 | tee ./build1.log
    python setup.py build_ext 2>&1 | tee ./build1.log
    cp build/python_ext/torch_dipu/_C.cpython-38-x86_64-linux-gnu.so torch_dipu
}

function config_dipu_cmake() {
    mkdir -p build && cd ./build && rm -rf ./*
    # PYTORCH_DIR="/you_pytorch/torch20/pytorch"
    # PYTHON_INCLUDE_DIR="/you_conda/envs/torch20/include/python3.8"
    cmake ../  -DCMAKE_BUILD_TYPE=Debug \
     -DDEVICE=tops -DPYTORCH_DIR=${PYTORCH_DIR} \
     -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR}
      # -DCMAKE_C_FLAGS_DEBUG="-g -O0" \
      # -DCMAKE_CXX_FLAGS_DEBUG="-g -O0"
    cd ../
}


function autogen_diopi_wrapper() {
    python scripts/autogen_diopi_wrapper/autogen_diopi_wrapper.py \
        --config scripts/autogen_diopi_wrapper/diopi_functions.yaml \
            --out torch_dipu/csrc_dipu/aten/ops/AutoGenedKernels.cpp
}

function build_dipu_lib() {
    autogen_diopi_wrapper
    config_dipu_cmake
    cd build && make -j8  2>&1 | tee ./build1.log &&  cd ..
    cp ./build/torch_dipu/csrc_dipu/libtorch_dipu.so   ./torch_dipu
    cp ./build/torch_dipu/csrc_dipu/libtorch_dipu_python.so   ./torch_dipu
}


if [[ "$1" == "builddl" ]]; then
    build_dipu_lib
elif [[ "$1" == "builddp" ]]; then
    build_dipu_py
fi
