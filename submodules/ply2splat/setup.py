from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os

this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="ply2splat",
    packages=["ply2splat"],
    ext_modules=[
        CppExtension(
            name="ply2splat",
            sources=[
                "ply2splat.cpp",
                "tinyply.cpp"
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
