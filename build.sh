/bin/rm -rf build && mkdir -p build && cd build 

export DIOPI_ROOT=/mnt/lustre/caikun/code/tianshu/ConformanceTest-DIOPI/lib/no_runtime
export LIBRARY_PATH=$DIOPI_ROOT:$LIBRARY_PATH; LD_LIBRARY_PATH=$DIOPI_ROOT:$LD_LIBRARY_PATH

cmake .. -DTESTS=ON -DCMAKE_BUILD_TYPE=Debug -DPYTORCH_INSTALL_DIR=`python -c 'import torch;import os;print(os.path.dirname(os.path.abspath(torch.__file__)))'`
# cmake .. -DCMAKE_BUILD_TYPE=Release -DPYTORCH_INSTALL_DIR=`python -c 'import torch;import os;print(os.path.dirname(os.path.abspath(torch.__file__)))'`

make VERBOSE=1 -j10

cd ..
