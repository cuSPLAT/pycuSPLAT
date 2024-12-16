from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os

this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="ply2splat",
    version="0.1",
    packages=find_packages(), 
    ext_modules=[
        CppExtension(
            name="ply2splat.ply2splat",
            sources=[
                os.path.join("ply2splat", "ply2splat.cpp"),
                os.path.join("ply2splat", "tinyply.cpp")
            ],
            include_dirs=[
                os.path.join(this_dir, "ply2splat")
            ],
            extra_compile_args=["-std=c++17"]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
