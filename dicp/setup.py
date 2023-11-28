from setuptools import setup, find_packages
import shutil
import os


def build_deps():
    paths = []

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


def main():
    build_deps()
    setup(
        name="dicp",
        version="0.0.1",
        url="https://github.com/DeepLink-org/DICP",
        packages=find_packages(),
        package_data={"dicp/vendor": [
            "TopsGraph/codegen/src/*.cpp",
            "TopsGraph/codegen/include/*.h",
            "AscendGraph/codegen/*.cpp",
            "AscendGraph/codegen/*.h",
            "AscendGraph/codegen/nlohmann/json.hpp"
        ]},
        classifiers=[
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Operating System :: POSIX :: Linux"
        ],
        entry_points={
            'torch_dynamo_backends': [
                'topsgraph = dicp.vendor.TopsGraph:topsgraph',
                'ascendgraph = dicp.vendor.AscendGraph:ascendgraph',
            ]
        },
        python_requires=">=3.8",
        install_requires=[
            "torch >= 2.0.0a0",
            "torch_dipu == 0.1"
        ]
    )


if __name__ == '__main__':
    main()
