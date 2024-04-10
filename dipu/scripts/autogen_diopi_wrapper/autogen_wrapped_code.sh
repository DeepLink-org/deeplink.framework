#!/bin/bash

DIPU_DIR=$(readlink -f $(dirname $(readlink -f "$0"))/../..)
AUTOGEN_DIOPI_WRAPPER=$DIPU_DIR/scripts/autogen_diopi_wrapper

USE_AUTOCOMPARE=${1:-OFF}
UsedVendor=${2:-cuda}
Torch_VERSION=${3:-2.1.0}
GENERATED_KERNELS_SCRIPT=${4:-$AUTOGEN_DIOPI_WRAPPER/autogen_diopi_wrapper.py}
GENERATED_KERNELS_CONFIG=${5:-$AUTOGEN_DIOPI_WRAPPER/diopi_functions.yaml}
GENERATED_KERNELS=${6:-$DIPU_DIR/torch_dipu/csrc_dipu/aten/ops/AutoGenedKernels.cpp}

GENERATED_KERNELS_VENDOR=${DIPU_DIR}/third_party/DIOPI/impl/${UsedVendor}/convert_config.yaml

PYTHON_CMD="python3 ${GENERATED_KERNELS_SCRIPT} --out=${GENERATED_KERNELS} --config=${GENERATED_KERNELS_CONFIG} \
    --autocompare=${USE_AUTOCOMPARE} --print_op_arg=True --use_diopi_adapter=False --print_func_call_info=True \
    --fun_config_dict='{\"current_device\":\"${UsedVendor}\",\"current_torch_ver\":\"${Torch_VERSION}\"}'"

if [ -f "$GENERATED_KERNELS_VENDOR" ]; then
    PYTHON_CMD="$PYTHON_CMD --convert_config=${GENERATED_KERNELS_VENDOR}"
fi

eval "$PYTHON_CMD"
