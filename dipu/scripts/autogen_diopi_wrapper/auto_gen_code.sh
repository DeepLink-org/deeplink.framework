DIPU_DIR=$(dirname $(dirname $(dirname "$0")))
USE_AUTOCOMPARE="$1"
UsedVendor="$2"
Torch_VERSION="$3"

GENERATED_KERNELS=${DIPU_DIR}/torch_dipu/csrc_dipu/aten/ops/AutoGenedKernels.cpp
GENERATED_KERNELS_SCRIPT=${DIPU_DIR}/scripts/autogen_diopi_wrapper/autogen_diopi_wrapper.py
GENERATED_KERNELS_CONFIG=${DIPU_DIR}/scripts/autogen_diopi_wrapper/diopi_functions.yaml
GENERATED_KERNELS_VENDOR=${DIPU_DIR}/third_party/DIOPI/impl/${UsedVendor}/convert_config.yaml

PYTHON_CMD="python3 ${GENERATED_KERNELS_SCRIPT} --out='${GENERATED_KERNELS}' --config='${GENERATED_KERNELS_CONFIG}' \
    --autocompare=${USE_AUTOCOMPARE} --print_op_arg=True --use_diopi_adapter=False --print_func_call_info=True \
    --fun_config_dict='{\"current_device\":\"${UsedVendor}\",\"current_torch_ver\":\"${Torch_VERSION}\"}'"

if [ "$GENERATED_KERNELS_VENDOR" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --convert_config=${GENERATED_KERNELS_VENDOR}"
fi

eval "$PYTHON_CMD"
