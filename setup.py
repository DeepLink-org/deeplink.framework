from setuptools import setup, find_packages

setup(
    name="dicp",
    version="0.0.1",
    url="https://github.com/OpenComputeLab/DICP",
    packages=find_packages(
        include=["dicp"],
        exclude=["test"]
    ),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: POSIX :: Linux"
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch >= 2.0.0a0"
    ]
)
