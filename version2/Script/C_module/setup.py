from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("C_GraphFilterSmoother.pyx"),
    include_dirs=[numpy.get_include()]
)