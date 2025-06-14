from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup, find_packages

path = Path(__file__).parent


include_dirs = [
    str(path / "include"),
    str(path / ".venv/include"),
    np.get_include(),
]
library_dirs = [
    str(path / "lib"),
    str(path / ".venv/lib"),
]
# https://www.hlibpro.com/doc/3.1/install.html
# https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-mkl-for-dpcpp/top.html
libraries = [
    "lapack",
    "blas",
    "boost_filesystem",
    "boost_system",
    "boost_program_options",
    "boost_iostreams",
    "tbb",
    "z",
    "metis",
    "fftw3",
    "gsl",
    "gslcblas",
    "m",
    "hdf5",
    "hdf5_cpp",
    "m",
    "stdc++",
] + [
    "iomp5",
    "pthread",
    "m",
    "dl",
]

extra_compile_args = ["-Wall", "-O3", "-fopenmp", "-fPIC"]

extensions = [
    Extension(
        "*",
        ["kolesky/*.pyx"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name= 'kolesky',
    author = 'Raymond Zeng',
    version = '0.1.3',
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