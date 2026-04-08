import sys
import numpy as np
from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize

include_dirs = [np.get_include()]
libraries = []
extra_compile_args = []
extra_link_args = []

if sys.platform == "win32":
    extra_compile_args = ["/O2", "/openmp", "/std:c++17"]
elif sys.platform == "darwin":
    extra_compile_args = ["-O3", "-fPIC", "-std=c++17", "-Xpreprocessor", "-fopenmp"]
    extra_link_args = ["-lomp"]
else:
    extra_compile_args = ["-O3", "-fPIC", "-std=c++17", "-fopenmp"]
    extra_link_args = ["-fopenmp"]

extensions = [
    Extension(
        "*",
        ["kolesky/*.pyx"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        include_dirs=include_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
]

setup(
    name= 'kolesky',
    author = 'Raymond Zeng',
    version = '0.1.4a1',
    packages = find_packages(),
    long_description=open('README.md').read(),
    install_requires = ["numpy", "scipy", "scikit-learn"],
    ext_modules=cythonize(
        extensions,
        annotate=True,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
        },
    ),
)