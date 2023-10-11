# Copyright (c) 2023, DeepLink.
import glob
import multiprocessing
import multiprocessing.pool
import os
import re
import shutil
import subprocess
import sys
import platform
import setuptools
from skbuild import setup

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VERSION = '0.1'

def get_pytorch_dir():
    import torch
    return os.path.dirname(os.path.abspath(torch.__file__))


def customized_cmake_args():
    cmake_args = list()
    cmake_args.append("-DCMAKE_BUILD_TYPE=Release")
    cmake_args.append("-DDEVICE=cuda")
    cmake_args.append("-DENABLE_COVERAGE=${USE_COVERAGE}")
    return cmake_args

setup(
    name="torch_dipu",
    version=VERSION,
    description='DIPU extension for PyTorch',
    url='https://github.com/DeepLink-org/dipu',
    packages=setuptools.find_packages(),
    cmake_args=customized_cmake_args(),
    cmake_install_target="all",
    package_data={},
    package_dir={},
    ext_modules=[
    ],
    cmdclass={
    })
