CI_BUILD_FLAG=$1 #"ci_build_flag"
PYTORCH_COMMIT=$2  # ${{ vars.PYTORCH_COMMIT }}=c263bd43e8e8502d4726643bc6fd046f0130ac0e
CLUSTER_CAMB=$3 
CAMB_TORCH_BASE_DIR=$4
CAMB_CI_PATH=$5
GITHUB_RUN_NUMBER=$6
MLU_REQUESTS=$7
GITHUB_RUN_NUMBER=$8

function clone() {
    cd /home/autolink/rsync/sourcecode/pytorch && git checkout master && git pull
    && git checkout ${PYTORCH_COMMIT} && git submodule update --init --recursive
}

functions rsync() {
    result=`ssh ${CLUSTER_CAMB} """
            mkdir -p ${CAMB_TORCH_BASE_DIR}/${PYTORCH_COMMIT}
            cd ${CAMB_TORCH_BASE_DIR}/${PYTORCH_COMMIT}
            if [ ! -f ${CI_BUILD_FLAG} ]; then
            touch ${CI_BUILD_FLAG}
            fi
            cat ${CI_BUILD_FLAG}
            """`
    echo "result:${result}"
    if [ "${result}x" = "${PYTORCH_COMMIT}"x ]; then
    echo "pytorch:${CAMB_TORCH_BASE_DIR}/${PYTORCH_COMMIT} exist."
    else
    echo "pytorch not exist, copy pytorch to ${CAMB_TORCH_BASE_DIR}/${PYTORCH_COMMIT}"
    ssh ${CLUSTER_CAMB} "rm -rf ${CAMB_TORCH_BASE_DIR}/${PYTORCH_COMMIT}"
    rsync -a --delete /home/autolink/rsync/sourcecode/pytorch/* ${CLUSTER_CAMB}:${CAMB_TORCH_BASE_DIR}/${PYTORCH_COMMIT}/
    fi
}

function build() {
    ssh ${CLUSTER_CAMB} """
    set -e
    cd ${CAMB_CI_PATH}/${GITHUB_RUN_NUMBER}/source
    source scripts/ci/camb/ci_camb_env.sh
    cd ${CAMB_TORCH_BASE_DIR}/${PYTORCH_COMMIT}
    echo "pwd: $(pwd)"
    if [ -f ${CI_BUILD_FLAG} ]; then
    echo "${CAMB_TORCH_BASE_DIR}/${PYTORCH_COMMIT} has been successfully compiled."
    else
    mkdir -p build && make clean
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
    BUILD_BINARY=0 USE_PRECOMPILED_HEADERS=1 BUILD_TEST=0
    srun --job-name=${GITHUB_JOB} --partition=${SLURM_PAR_CAMB} --time=40 \
    --gres=mlu:${MLU_REQUESTS} python setup.py install --prefix=./install_path
    echo "${PYTORCH_COMMIT}" > ${CI_BUILD_FLAG}
    fi
    """
}