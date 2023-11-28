import glob
import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)


def get_extensions():
    """Refer to torchvision."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "diso")

    main_file = [os.path.join(this_dir, "src", "pybind.cpp")]
    source_cuda = glob.glob(os.path.join(this_dir, "src", "*.cu"))
    sources = main_file
    extension = CppExtension

    define_macros = []
    extra_compile_args = {}
    if (torch.cuda.is_available() and (CUDA_HOME is not None)) or os.getenv(
        "FORCE_CUDA", "0"
    ) == "1":
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        nvcc_flags = os.getenv("NVCC_FLAGS", "")
        if nvcc_flags == "":
            nvcc_flags = ["-O3"]
        else:
            nvcc_flags = nvcc_flags.split(" ")
        extra_compile_args = {
            "cxx": ["-O3"],
            "nvcc": nvcc_flags,
        }

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir, os.path.join(extensions_dir, "include")]
    print("sources:", sources)

    ext_modules = [
        extension(
            "diso._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules

setup(
    name="diso",
    version="0.0.0",
    author_email="xiwei@ucsd.edu",
    keywords="collision convex decomposition",
    description="Approximate Convex Decomposition for 3D Meshes with Collision-Aware Concavity and Tree Search",
    classifiers=[
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Other Audience",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Framework :: Robot Framework :: Tool",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Education",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
    license="MIT",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.6",
    install_requires=["torch>=1.8.0", "trimesh"],
    ext_modules=get_extensions(),
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
    },
    zip_safe=False
)