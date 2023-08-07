# Copyright (c) 2023, DeepLink.
import glob
import multiprocessing
import multiprocessing.pool
import os
import re
import shutil
import subprocess
import sys
import traceback
import platform

import distutils.ccompiler
import distutils.command.clean
from sysconfig import get_paths
from distutils.version import LooseVersion
from distutils.command.build_py import build_py
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from setuptools import setup, distutils, Extension
from setuptools.command.build_clib import build_clib
from setuptools.command.egg_info import egg_info

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VERSION = '0.1'

def build_deps():
    paths = [
        ('third_party/nlohmann/json/single_include/nlohmann/json.hpp', 'dicp/vendor/AscendGraph/codegen/nlohmann/json.hpp')
    ]
    
    for orig_path, new_path in paths:
        if not os.path.exists(new_path):
            os.makedirs(os.path.dirname(new_path), exist_ok=True)

        # Copy the files from the orig location to the new location
        if os.path.isfile(orig_path):
            shutil.copyfile(orig_path, new_path)
            continue
        if os.path.isdir(orig_path):
            if os.path.exists(new_path):
                # copytree fails if the tree exists already, so remove it.
                shutil.rmtree(new_path)
            shutil.copytree(orig_path, new_path)
            continue
        raise RuntimeError("Check the file paths in `build_deps`")        

build_deps()

def which(thefile):
    path = os.environ.get("PATH", os.defpath).split(os.pathsep)
    for d in path:
        fname = os.path.join(d, thefile)
        fnames = [fname]
        if sys.platform == 'win32':
            exts = os.environ.get('PATHEXT', '').split(os.pathsep)
            fnames += [fname + ext for ext in exts]
        for name in fnames:
            if os.access(name, os.F_OK | os.X_OK) and not os.path.isdir(name):
                return name
    return None

def get_cmake_command():
    def _get_version(cmd):
        for line in subprocess.check_output([cmd, '--version']).decode('utf-8').split('\n'):
            if 'version' in line:
                return LooseVersion(line.strip().split(' ')[2])
        raise RuntimeError('no version found')
    "Returns cmake command."
    cmake_command = 'cmake'
    cmake3 = which('cmake3')
    cmake = which('cmake')
    if cmake3 is not None and _get_version(cmake3) >= LooseVersion("3.10.0"):
        cmake_command = 'cmake3'
        return cmake_command
    elif cmake is not None and _get_version(cmake) >= LooseVersion("3.10.0"):
        return cmake_command
    else:
        raise RuntimeError('no cmake or cmake3 with version >= 3.10.0 found')

def get_build_type():
    build_type = 'Debug'
    if os.getenv('Release', default='0').upper() in ['ON', '1', 'YES', 'TRUE', 'Y']:
        build_type = 'Release'

    if  os.getenv('REL_WITH_DEB_INFO', default='0').upper() in ['ON', '1', 'YES', 'TRUE', 'Y']:
        build_type = 'RelWithDebInfo'

    return build_type

def get_pytorch_dir():
    try:
        import torch
        return os.path.dirname(os.path.abspath(torch.__file__))
    except Exception:
        _, _, exc_traceback = sys.exc_info()
        frame_summary = traceback.extract_tb(exc_traceback)[-1]
        return os.path.dirname(frame_summary.filename)

class CPPLibBuild(build_clib, object):
    def run(self):
        cmake = get_cmake_command()

        if cmake is None:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))
        self.cmake = cmake

        build_dir = os.path.join(BASE_DIR, "build")
        build_type_dir = os.path.join(build_dir, get_build_type())
        output_lib_path = os.path.join(build_type_dir, "torch_dipu/lib")
        os.makedirs(build_type_dir, exist_ok=True)
        os.makedirs(output_lib_path, exist_ok=True)
        self.build_lib =  os.path.relpath(os.path.join(build_dir, "torch_dipu"))
        self.build_temp =  os.path.relpath(build_type_dir)
        cmake_args = [
            '-DCMAKE_BUILD_TYPE=' + get_build_type(),
            # '-DCMAKE_INSTALL_PREFIX=' + os.path.abspath(output_lib_path),
            # '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + os.path.abspath(output_lib_path),
            # '-DTORCHDIPU_INSTALL_LIBDIR=' + os.path.abspath(output_lib_path),
            # '-DPYTHON_INCLUDE_DIR=' + get_paths()['include'],
            # '-DPYTORCH_INSTALL_DIR=' + get_pytorch_dir()
        ]
        build_args = ['-j', 12]
        subprocess.check_call([self.cmake, BASE_DIR] + cmake_args, cwd=build_type_dir, env=os.environ)
        subprocess.check_call(['make'] + build_args, cwd=build_type_dir, env=os.environ)

class CppExtensionBuilder:
    include_directories = [
        BASE_DIR +"/torch_dipu",
    ]

    extra_link_args = []

    # DEBUG = (os.getenv('DEBUG', default='').upper() in ['ON', '1', 'YES', 'TRUE', 'Y'])
    # always set on in poc
    DEBUG = "ON"

    extra_compile_args = [
        '-std=c++14',
        '-Wno-sign-compare',
        '-Wno-deprecated-declarations',
        '-Wno-return-type',
    ]
    if DEBUG:
        extra_compile_args += ['-O0', '-g']
        extra_link_args += ['-O0', '-g', '-Wl,-z,now']
    else:
        # extra_compile_args += ['-DNDEBUG']
        extra_link_args += ['-Wl,-z,now,-s']

    @staticmethod
    def genDIPUExt():
        return CppExtensionBuilder.extcamb('torch_dipu._C',  
            sources=["torch_dipu/csrc_dipu/stub.cpp"],
            libraries=["torch_dipu_python", "torch_dipu"],
            include_dirs= CppExtensionBuilder.include_directories,
            extra_compile_args= CppExtensionBuilder.extra_compile_args + ['-fstack-protector-all'],
            library_dirs=["./build/torch_dipu/csrc_dipu"],
            extra_link_args= CppExtensionBuilder.extra_link_args + ['-Wl,-rpath,$ORIGIN/lib'],
        )
    
    @staticmethod
    def extcamb(name, sources, *args, **kwargs):
        r'''
        Creates a :class:`setuptools.Extension` for C++.
        '''
        pytorch_dir = get_pytorch_dir()
        temp_include_dirs = kwargs.get('include_dirs', [])
        temp_include_dirs.append(os.path.join(pytorch_dir, 'include'))
        temp_include_dirs.append(os.path.join(pytorch_dir, 'include/torch/csrc/api/include'))
        kwargs['include_dirs'] = temp_include_dirs

        temp_library_dirs = kwargs.get('library_dirs', [])
        temp_library_dirs.append(os.path.join(pytorch_dir, 'lib'))
        kwargs['library_dirs'] = temp_library_dirs

        libraries = kwargs.get('libraries', [])
        libraries.append('c10')
        libraries.append('torch')
        libraries.append('torch_cpu')
        libraries.append('torch_python')
        kwargs['libraries'] = libraries
        kwargs['language'] = 'c++'
        return Extension(name, sources, *args, **kwargs)


class BuildExt(build_ext, object):
    def run(self):
        self.build_lib =  os.path.relpath(os.path.join(BASE_DIR, f"build/python_ext"))
        self.build_temp =  os.path.relpath(os.path.join(BASE_DIR, f"build/python_ext"))
        super(BuildExt, self).run()


def start_debug():
    rank = 0
    pid1 = os.getpid()
    print("-------------------------print rank,:", rank, "pid1:", pid1)
    if rank == 0:
        #  time.sleep(15)
        import ptvsd
        host = "127.0.0.1" # or "localhost"
        port = 12345
        print("Waiting for debugger attach at %s:%s ......" % (host, port), flush=True)
        ptvsd.enable_attach(address=(host, port), redirect_output=True)
        ptvsd.wait_for_attach()
 
# start_debug()

# placeholder: autogen design.....
# generate_bindings_code()


setup(
    name="torch_dipu",
    version=VERSION,
    description='DIPU extension for PyTorch',
    url='https://github.com/DeepLink-org/dipu',
    packages=["torch_dipu"],
    # libraries=[('torch_dipu', {'sources': list()})],
    # package_dir={'': os.path.relpath(os.path.join(BASE_DIR, f"build/{get_build_type()}/packages"))},
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: POSIX :: Linux"
    ],
    entry_points = {
        'torch_dynamo_backends': [
            'topsgraph = torch_dipu.dicp.vendor.TopsGraph:topsgraph',
            'ascendgraph = torch_dipu.dicp.vendor.AscendGraph:ascendgraph',
        ]
    },
    python_requires=">=3.8",
    install_requires=[
        "torch >= 2.0.0a0"
    ],
    ext_modules=[
        CppExtensionBuilder.genDIPUExt(),
    ],
    package_data={
        "torch_dipu/dicp/vendor": [
            "TopsGraph/codegen/src/*.cpp",
            "TopsGraph/codegen/include/*.h",
            "AscendGraph/codegen/*.cpp",
            "AscendGraph/codegen/*.h",
            "AscendGraph/codegen/nlohmann/json.hpp"
        ],
        'torch_npu': [
            '*.so', 'lib/*.so*',
        ],
    },
    cmdclass={
        # build_clib not work now, need enhance 
        'build_clib': CPPLibBuild,
        'build_ext': BuildExt,
    })
