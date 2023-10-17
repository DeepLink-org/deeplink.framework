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

function autogen_diopi_wrapper() {
    python scripts/autogen_diopi_wrapper/autogen_diopi_wrapper.py                             \
        --config scripts/autogen_diopi_wrapper/diopi_functions.yaml                           \
        --out torch_dipu/csrc_dipu/aten/ops/AutoGenedKernels.cpp                              \
        --autocompare  False                                                                  \
        --print_func_call_info True                                                           \
        --print_op_arg True                                                                   \
        --fun_config_dict '{"current_device": "camb"}'                                        \
        --use_diopi_adapter False                                                             \
        # --diopi_adapter_header third_party/DIOPI/proto/include/diopi/diopi_adaptors.hpp

    # only test mulity config autogen
    python scripts/autogen_diopi_wrapper/autogen_diopi_wrapper.py                   \
        --config scripts/autogen_diopi_wrapper/custom_diopi_functions.yaml          \
        --out torch_dipu/csrc_dipu/aten/ops/CustomAutoGenedKernels.cpp              \
        --use_diopi_adapter False                                                   \

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
    export LIBRARY_PATH=$DIOPI_ROOT:$LIBRARY_PATH;
    config_dipu_camb_cmake 2>&1 | tee ./cmake_camb.log
    cd build && make -j8  2>&1 | tee ./build.log &&  cd ..
}

case $1 in
    build_dipu)
        (
            build_diopi_lib
            build_dipu_lib
        ) || exit -1;;
    build_diopi)
        build_diopi_lib || exit -1;;
    build_autogen_diopi_wrapper)
        autogen_diopi_wrapper || exit -1;;
    build_dipu_only)
        build_dipu_lib || exit -1;;
    *)
        echo -e "[ERROR] Incorrect option:" $1;
esac
exit 0
