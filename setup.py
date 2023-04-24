from setuptools import setup, find_packages

setup(
    name="dicp",
    version="0.0.1",
    url="https://github.com/OpenComputeLab/DICP",
    packages=find_packages(),
    package_data={"dicp": ["TopsGraph/codegen/src/*.cpp", "TopsGraph/codegen/include/*.h"]},
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: POSIX :: Linux"
    ],
    entry_points = {
        'torch_dynamo_backends': [
             'topsgraph = dicp.TopsGraph:topsgraph',
             'ascendgraph = dicp.AscendGraph:ascendgraph',
        ]
    },
    python_requires=">=3.8",
    install_requires=[
        "torch >= 2.0.0a0"
    ]
)
