#!/bin/bash

# This script is used by "dipu/torch_dipu/csrc_dipu/CMakeLists.txt", but you can also run it mannualy to generate wrapped_cpp code.

DIPU_DIR=$(readlink -f $(dirname $(readlink -f "$0"))/../..)
AUTOGEN_DIOPI_WRAPPER=$DIPU_DIR/scripts/autogen_diopi_wrapper

UsedVendor=${1:-cuda}
Torch_VERSION=${2:-2.1.0}
GENERATED_KERNELS_SCRIPT=${3:-$AUTOGEN_DIOPI_WRAPPER/autogen_diopi_wrapper.py}
GENERATED_KERNELS_CONFIG=${4:-$AUTOGEN_DIOPI_WRAPPER/diopi_functions.yaml}
GENERATED_KERNELS=${5:-$DIPU_DIR/torch_dipu/csrc_dipu/aten/ops/AutoGenedKernels.cpp}
GENERATE_DEVICE_GUARD=${6:-"True"}
AUTOGEN_CODE_REMOVE_CHECK=${7:-"False"}

GENERATED_KERNELS_VENDOR=${DIPU_DIR}/third_party/DIOPI/impl/${UsedVendor}/convert_config.yaml

PYTHON_CMD="python3 ${GENERATED_KERNELS_SCRIPT} --out=${GENERATED_KERNELS} --config=${GENERATED_KERNELS_CONFIG} \
    --print_op_arg=True --use_diopi_adapter=False --print_func_call_info=True --generate_device_guard=${GENERATE_DEVICE_GUARD} \
    --remove_check_code=${AUTOGEN_CODE_REMOVE_CHECK} --fun_config_dict='{\"current_device\":\"${UsedVendor}\",\"current_torch_ver\":\"${Torch_VERSION}\"}'"

if [ -f "$GENERATED_KERNELS_VENDOR" ]; then
    PYTHON_CMD="$PYTHON_CMD --convert_config=${GENERATED_KERNELS_VENDOR}"
fi

eval "$PYTHON_CMD"
