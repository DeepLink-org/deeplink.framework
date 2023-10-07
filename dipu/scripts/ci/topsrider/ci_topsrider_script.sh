#!/usr/bin/env bash

function build_dipu_py() {
    export CMAKE_BUILD_TYPE=debug
    export _GLIBCXX_USE_CXX11_ABI=1
    export MAX_JOBS=12
    # PYTORCH_INSTALL_DIR is /you_pytorch/torch20/pytorch/torch
    # python  setup.py build_clib 2>&1 | tee ./build1.log
    python setup.py build_ext 2>&1 | tee ./build1.log
    mv build/python_ext/torch_dipu/_C.cpython*.so torch_dipu
}

function config_dipu_cmake() {
    mkdir -p build && cd ./build && rm -rf ./*
    # PYTORCH_DIR="/you_pytorch/torch20/pytorch"
    cmake ../  -DCMAKE_BUILD_TYPE=Debug \
     -DDEVICE=tops -DPYTORCH_DIR=${PYTORCH_DIR} \
      # -DCMAKE_C_FLAGS_DEBUG="-g -O0" \
      # -DCMAKE_CXX_FLAGS_DEBUG="-g -O0"
      
    cd ../
}


function autogen_diopi_wrapper() {
    python scripts/autogen_diopi_wrapper/autogen_diopi_wrapper.py                   \
        --config scripts/autogen_diopi_wrapper/diopi_functions.yaml                 \
        --use_diopi_adapter True                                                    \
        --diopi_adapter_header third_party/DIOPI/adaptor/diopi_adaptors.hpp   \
        --autocompare False                                                         \
        --out torch_dipu/csrc_dipu/aten/ops/AutoGenedKernels.cpp                    \
        --fun_config_dict '{"current_device": "topsrider"}'
}

function build_dipu_lib() {
    autogen_diopi_wrapper
    config_dipu_cmake
    cd build && make -j8  2>&1 | tee ./build1.log &&  cd ..
    mv ./build/torch_dipu/csrc_dipu/libtorch_dipu.so   ./torch_dipu
    mv ./build/torch_dipu/csrc_dipu/libtorch_dipu_python.so   ./torch_dipu
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
elif [[ "$1" == "builddp" ]]; then
    build_dipu_py
elif [[ "$1" == "build_dipu" ]]; then
    build_dipu_lib
    build_dipu_py
elif [[ "$1" == "build_diopi" ]]; then
    build_diopi_lib
elif [[ "$1" == "build" ]]; then
    build_diopi_lib
    build_dipu_lib
    build_dipu_py
elif [[ "$1" == "clean" ]]; then
    build_diopi_clean
fi
