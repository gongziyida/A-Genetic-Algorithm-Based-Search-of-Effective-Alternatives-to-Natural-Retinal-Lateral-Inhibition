from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        name="*",
        sources=["*.pyx"],
        include_dirs=[numpy.get_include()]
    ),
]

setup(name="Retina Sim", version='0.1', author='Ziyi Gong', author_email='zig9@pitt.edu',
      ext_modules=cythonize(ext_modules))
