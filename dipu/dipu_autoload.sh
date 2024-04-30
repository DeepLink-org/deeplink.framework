#!/bin/bash

if ! python_site=$(DIPU_AUTOLOAD=0 python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"); then
  echo "Failed to find site-packages path, please check your python installation."
fi
sitecustomize_path=${python_site}/sitecustomize.py

init() {( set -e
  if grep -q "# >>> DIPU Initialize >>>" "${sitecustomize_path}"; then
    echo "DIPU autoload is already configured in ${sitecustomize_path}."
    echo "No action taken."
  else
    echo "Modifying ${sitecustomize_path}..."
    cat >> "${sitecustomize_path}" << EOF
# >>> DIPU Initialize >>>
# !! Contents within this block are managed by 'dipu_autoload.sh init' !!
import os
if os.environ.get('DIPU_AUTOLOAD', '0') != '0':
    print("DIPU is being automatically loaded with sitecustomize.py. To disable this behavior, set \`DIPU_AUTOLOAD=0\` in your environment or execute \`dipu_autoload.sh deinit\`.")
    import torch_dipu
# <<< DIPU Initialize <<<
EOF
   echo "DIPU autoload has been successfully configured in ${sitecustomize_path}."
  fi

  echo "DIPU will be automatically loaded when python starts if DIPU_AUTOLOAD is set to a nonzero value."
)}

deinit() {( set -e
  if ! grep -q "# >>> DIPU Initialize >>>" "${sitecustomize_path}"; then
    echo "DIPU autoload is not configured in ${sitecustomize_path}."
    echo "No action taken."
    return
  fi

  echo "Restoring ${sitecustomize_path}..."
  sed -i '/# >>> DIPU Initialize >>>/,/# <<< DIPU Initialize <<</d' "${sitecustomize_path}"

  # if sitecustomize is empty, then delete it.
  if [ ! -s "${sitecustomize_path}" ]; then
    echo "Deleting empty ${sitecustomize_path}..."
    rm -rf "${sitecustomize_path}"
  fi

  echo "DIPU autoload has been successfully removed from ${sitecustomize_path}."
)}

print_help() {
  echo "Usage: $0 [-h|--help] [init|deinit]"
  echo "  init    : Enable DIPU autoload in sitecustomize.py"
  echo "  deinit  : Disable DIPU autoload in sitecustomize.py"
}

case "$1" in
  init)
    init
  ;;
  deinit)
    deinit
  ;;
  "-h"|"--help")
    print_help
  ;;
  *)
    print_help
    exit 1
  ;;
esac
