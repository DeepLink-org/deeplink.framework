#!/bin/bash

chmod 755 dipu_envs.sh

if [ $1 != "init" ] && [ $1 != "deinit" ]; then
    echo "Usage: $0 [-h|--help] [init|deinit]"
    echo "If you want to disable DIPU, could use 'export DIPU=0' to disable it."
    exit 0
fi

python_path=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
if [ $? -ne 0 ]; then
  echo "Couldn't find the python lib(site-package) directory, please check it out!"
fi
sitecustomize_path=${python_path}/sitecustomize.py


if [ "$1" == "init" ]; then
    if [ -f "${sitecustomize_path}" ]; then
        if grep -q "# >>> DIPU Initialize >>>" $sitecustomize_path;then
            echo "DIPU has already loaded!!"
            return
        fi
    fi
 
    cat >> $sitecustomize_path << EOF
# >>> DIPU Initialize >>>
# !! Contents within this block are managed by 'python dipu_init.py'
import os
if not os.environ.get('EN_DIPU', '1') == '0':
    import torch_dipu
    print("DIPU has loaded!")
# <<< DIPU Initialize <<<
EOF

    echo "sitecustomized.py is modified so that dipu will be automatically loaded when python starts"
fi

if [ "$1" == "deinit" ]; then

    sed -i '/# >>> DIPU Initialize >>>/,/# <<< DIPU Initialize <<</d' $sitecustomize_path

    # if sitecustomize is empty, then delete it.
    if [ ! -s $sitecustomize_path ]; then
        rm -rf $sitecustomize_path
        echo "${sitecustomize_path} has been deleted."
    else
        echo "${sitecustomize_path} has been restored."
    fi

    echo "Python environment has restored!"
fi