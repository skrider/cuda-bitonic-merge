# based off https://github.com/Dao-AILab/flash-attention/blob/main/setup.py
import os
from pathlib import Path
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
)

PACKAGE_NAME = "cbm"
this_dir = os.path.dirname(os.path.abspath(__file__))

ext_modules = []

# Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
# See https://github.com/pytorch/pytorch/pull/70650
generator_flag = []
torch_dir = torch.__path__[0]
if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
    generator_flag = ["-DOLD_GENERATOR_PATH"]

cc_flag = []
cc_flag.append("-gencode")
cc_flag.append("arch=compute_80,code=sm_80")

ext_modules.append(
    CUDAExtension(
        name="cbm_cuda",
        sources=[
            "csrc/cbm/bm_api.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"] + generator_flag,
            "nvcc": [
                    "-O0",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--threads",
                    "4",
                    "-g",
                    "-G",
                ]
                + generator_flag
                + cc_flag,
        },
        include_dirs=[
            Path(this_dir) / "csrc" / "cbm",
            Path(this_dir) / "csrc" / "cbm" / "src",
            Path(this_dir) / "csrc" / "cutlass" / "include",
        ],
    )
)

setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(
        exclude=(
            "csrc",
            "tests",
            "cbm.egg-info",
        )
    ),
    author="Stephen Krider",
    author_email="skrider@berkeley.edu",
    description="cuda bitonic merge",
    url="https://github.com/skrider/cuda-bitonic-merge",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "einops",
        "packaging",
        "ninja",
    ],
)
