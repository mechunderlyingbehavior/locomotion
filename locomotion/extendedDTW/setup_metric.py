from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'custom metric',
  ext_modules = cythonize("metric.pyx"),
)

