### python setup.py build_ext --inplace

from distutils.core import setup, Extension
from distutils.sysconfig import *
from distutils.util import *
import os
import os.path
import numpy

try:
   from distutils.command.build_py import build_py_2to3 \
       as build_py
except ImportError:
   from distutils.command.build_py import build_py
   
# try:
#    from Cython.Distutils import build_ext
# except ImportError:
#    use_cython = False
# else:
#    use_cython = True

from Cython.Distutils import build_ext
use_cython = True
   
#### data files
data_files = []
# Include gsl libs for the win32 distribution
if get_platform() == "win32":
   dlls = ["gsl.lib", "cblas.lib"]
   #data_files += [("Lib\site-packages\dtwext", dlls)]
   
#### libs
if get_platform() == "win32":
   gsl_lib = ['gsl', 'cblas']
   math_lib = []
else:
   gsl_lib = ['gsl', 'gslcblas']
   math_lib = ['m']
   
#### Extra compile args
if get_platform() == "win32":
   extra_compile_args = []
else:
   extra_compile_args = ['-Wno-strict-prototypes']
   
#### Python include
py_inc = [get_python_inc()]

#### NumPy include
np_lib = os.path.dirname(numpy.__file__)
np_inc = [os.path.join(np_lib, 'core/include')]

#### scripts
scripts = []

#### cmdclass
cmdclass = {'build_py': build_py}

#### Extension modules
ext_modules = []
if use_cython:
   print("using cython")
   cmdclass.update({'build_ext': build_ext})
   ext_modules += [Extension("extendedDTW", 
                             ["cdtw.c",
                              "metric.c",
                              "extendedDTW.pyx"],
                             libraries=math_lib,
                             include_dirs=py_inc + np_inc),
                   ]
else:
   ext_modules += [Extension("extendedDTW",
                             ["cdtw.c",
                              "metric.c",
                              "extendedDTW.c"],
                             libraries=math_lib,
                             include_dirs=py_inc + np_inc),
                   ]
   
packages=['extendedDTW']

setup(name = 'extendedDTW',
      #requires=['numpy (>=1.3.0)', 'scipy (>=0.7.0)', 'gsl (>=1.11)'],
      #description='mlpy - Machine Learning Py - ' \
      #   'High-Performance Python Package for Predictive Modeling',
      #author='mlpy Developers',
      #author_email='davide.albanese@gmail.com',
      #maintainer='Davide Albanese',
      #maintainer_email='davide.albanese@gmail.com',
      #packages=packages,
      #url='mlpy.sourceforge.net',
      #download_url='https://sourceforge.net/projects/mlpy/',
      #license='GPLv3',
      #classifiers=classifiers,
      cmdclass=cmdclass,
      ext_modules=ext_modules,
      scripts=scripts,
      data_files=data_files
      )
