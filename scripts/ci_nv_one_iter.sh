export ONE_ITER_TOOL_STORAGE_PATH=$(pwd)/one_iter_data
sh SMART/tools/one_iter_tool/run_one_iter.sh
export ONE_ITER_TOOL_DEVICE=dipu
export ONE_ITER_TOOL_DEVICE_COMPARE=cpu
sh SMART/tools/one_iter_tool/compare_one_iter.sh