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
    packages=find_packages(exclude=["tests"]),
    ext_modules=get_extensions(),
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
    },
)
