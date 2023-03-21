# !/bin/bash
set -e
echo "pwd: $(pwd)"

function build_pytorch_source() {
    cd pytorch
    echo "building pytorch:$(pwd)"
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
    BUILD_BINARY=0 USE_PRECOMPILED_HEADERS=1 BUILD_TEST=0 python setup.py install --user
    cd ..
}

function build_diopi_source() {
    cd third_party/DIOPI_TEST
    echo "building DIOPI_TEST:$(pwd)"
    sh scripts/build_impl.sh camb_no_runtime
    cd ../..
}

case $1 in 
    build_pytorch)
        (
            build_pytorch_source
        ) \
        || exit -1;;
    build_diopi)
        (
            build_diopi_source
        ) \
        || exit -1;;
    *)
        echo -e "[ERROR] Incorrect option:" $1; 
esac
exit 0
