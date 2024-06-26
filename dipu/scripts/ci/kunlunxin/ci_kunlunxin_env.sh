export XDNNTORCH_TOOLKIT_ROOT=/mnt/cache/share/deeplinkci/deps/xdnn_pytorch
export XDNN_TOOLKIT_ROOT=/mnt/cache/share/deeplinkci/deps/xdnn/
export XPURT_TOOLKIT_ROOT=/mnt/cache/share/deeplinkci/deps/xpurt/
export XCCL_TOOLKIT_ROOT=/mnt/cache/share/deeplinkci/deps/xccl/
export LD_LIBRARY_PATH=/mnt/cache/share/deeplinkci/deps/xccl/so/:/usr/local/lib/python3.8/dist-packages:/mnt/cache/share/deeplinkci/deps/xdnn/so/:/mnt/cache/share/deeplinkci/deps/xpurt/so/:/mnt/cache/share/deeplinkci/deps/xdnn_pytorch/so/:/mnt/cache/share/deeplinkci/deps/xdnn_pytorch/so/:$LD_LIBRARY_PATH
export DIPU_DEVICE=kunlunxin
export DIOPI_ROOT=$(pwd)/third_party/DIOPI/impl/lib/
export DIPU_ROOT=$(pwd)/torch_dipu
export DIOPI_PATH=$(pwd)/third_party/DIOPI/proto
export DIPU_PATH=${DIPU_ROOT}
export VENDOR_INCLUDE_DIRS=/mnt/cache/share/deeplinkci/deps/xccl/include:/mnt/cache/share/deeplinkci/deps/xdnn/include:/mnt/cache/share/deeplinkci/deps/xpurt/include:/mnt/cache/share/deeplinkci/deps/xdnn_pytorch/include

export DIPU_CHECK_TENSOR_DEVICE=1