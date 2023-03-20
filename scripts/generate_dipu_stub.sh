#!/bin/bash

CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
DIPU="$CDIR/.."

cd $DIPU

python3 -m torchgen.gen_dipu_stubs  \
  --output_dir="$DIPU/torch_dipu/csrc_dipu/aten" \
  --source_yaml="$DIPU/torch_dipu/csrc_dipu/aten/dipu_native_functions.yaml" \
  --impl_path="$DIPU/torch_dipu/csrc_dipu/aten"

if [ $? -ne 0 ]; then
  echo "Failed to generate DIPU backend stubs."
  exit 1
fi
