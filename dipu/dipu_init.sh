#!/bin/bash

python_path=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
sitecustomize_path=${python_path}/sitecustomize.py

if [ -f "${sitecustomize_path}" ]; then
    if grep -q "# >>> DIPU Initialize >>>" $sitecustomize_path;then
        echo "DIPU has already loaded!!"
        return
    fi
fi

# The environ used to control whether DIPU is enabled.
# 1 : enable, 0: disable. Enable for default.
export EN_DIPU=1

echo "# >>> DIPU Initialize >>>" >> $sitecustomize_path
echo "# !! Contents within this block are managed by 'python dipu_init.py'" >> $sitecustomize_path
echo "import os" >> $sitecustomize_path
echo "if os.environ.get('EN_DIPU') == '1':" >> $sitecustomize_path
echo "    import torch_dipu" >> $sitecustomize_path
echo "    print(\"DIPU has loaded!\")" >> $sitecustomize_path
echo "# <<< DIPU Initialize <<<" >> $sitecustomize_path

echo "DIPU has loaded successfull !!"