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
from setuptools import setup, distutils, Extension

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VERSION = '0.1'

def get_pytorch_dir():
    import torch
    return os.path.dirname(os.path.abspath(torch.__file__))

setup(
    name="torch_dipu",
    version=VERSION,
    description='DIPU extension for PyTorch',
    url='https://github.com/DeepLink-org/dipu',
    packages=["torch_dipu"],
    ext_modules=[
    ],
    cmdclass={
    })
