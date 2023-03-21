# !/bin/bash
set -e
echo "pwd: $(pwd)"

function build_dipu_py() {
    echo "building dipu_py:$(pwd)"
    export CMAKE_BUILD_TYPE=debug
    export _GLIBCXX_USE_CXX11_ABI=1
    export MAX_JOBS=12
    # PYTORCH_INSTALL_DIR is /home/fandaoyi.p/torch20/pytorch/torch
    #/mnt/lustre/share_data/caikun/pt2.0/bin/python setup.py build_ext 2>&1 | tee ./build1.log
    python setup.py build_ext 2>&1 | tee ./build1.log
    #/mnt/lustre/share/platform/env/miniconda3.8/envs/pt2.0v1_cpu/bin/python setup.py build_ext 2>&1 | tee ./build1.log
    cp build/python_ext/torch_dipu/_C.cpython-38-x86_64-linux-gnu.so torch_dipu
}

function config_dipu_camb_cmake() {
    mkdir -p build && cd ./build && rm -rf ./*
    PYTORCH_DIR="/mnt/lustre/share/parrotsci/github/cibuild/OpenComputeLab/dipu_poc/33/pytorch"
    echo "PYTORCH_DIR: ${PYTORCH_DIR}"
    #PYTORCH_DIR="/mnt/lustre/share/platform/env/miniconda3.8/envs/pt2.0v1_cpu/lib/python3.8/site-packages"
    #PYTORCH_DIR="/mnt/lustre/share_data/caikun/code/pytorch"
    #echo "PYTHON_INCLUDE_DIR: ${PYTHON_INCLUDE_DIR}"
    #PYTHON_INCLUDE_DIR="/mnt/lustre/share/platform/env/miniconda3.8/envs/pt2.0v1_cpu/include/python3.8"
    PYTHON_INCLUDE_DIR="/mnt/lustre/share_data/caikun/pt2.0/include/python3.8"
    cmake ../  -DCMAKE_BUILD_TYPE=Debug \
        -DCAMB=ON -DPYTORCH_DIR=${PYTORCH_DIR} \
        -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR}
    cd ../
}

function build_dipu_lib() {
    echo "building dipu_lib:$(pwd)"
    export DIOPI_ROOT=/mnt/lustre/share_data/zhaochaoxing/code/dipu_poc/third_party/DIOPI_TEST/lib
    #export DIOPI_ROOT=/mnt/lustre/share_data/caikun/code/ConformanceTest-DIOPI/lib
    export LIBRARY_PATH=$DIOPI_ROOT:$LIBRARY_PATH;
    #export LD_LIBRARY_PATH=/mnt/lustre/share/platform/env/miniconda3.8/envs/pt2.0v1_cpu/lib:$LD_LIBRARY_PATH
    #export PATH=${GCC_ROOT}/bin:/mnt/lustre/share/platform/env/miniconda3.8/envs/pt2.0v1_cpu/bin:${CONDA_ROOT}/bin:${MPI_ROOT}/bin:$PATH
    config_dipu_camb_cmake 2>&1 | tee ./build1.log
    cd build && make -j8  2>&1 | tee ./build1.log &&  cd ..
    cp ./build/torch_dipu/csrc_dipu/libtorch_dipu.so   ./torch_dipu
    cp ./build/torch_dipu/csrc_dipu/libtorch_dipu_python.so   ./torch_dipu
}

case $1 in 
    build_dipu)
        (
            build_dipu_lib
            build_dipu_py
        ) \
        || exit -1;;
    *)
        echo -e "[ERROR] Incorrect option:" $1; 
esac
exit 0
