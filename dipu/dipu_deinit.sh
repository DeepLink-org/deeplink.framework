#!/bin/bash

# The environ used to control whether DIPU is enabled.
# 1 : enable, 0: disable. Enable for default.
unset EN_DIPU

python_path=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
sitecustomize_path=${python_path}/sitecustomize.py

sed -i '/# >>> DIPU Initialize >>>/,/# <<< DIPU Initialize <<</d' $sitecustomize_path

# if sitecustomize is empty, then delete it.
if [ ! -s $sitecustomize_path ]; then
    rm -rf $sitecustomize_path
fi

echo "Python environment has restored !!"