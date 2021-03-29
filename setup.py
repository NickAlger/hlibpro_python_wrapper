# from distutils.core import setup, Extension
import subprocess
import os

from setuptools import setup, Extension, find_packages
import glob


HLIBPRO_DIR = '/home/nick/hlibpro-2.8.1'
EIGEN_INCLUDE = '/home/nick/anaconda3/envs/fenics3/include/eigen3'
MARCH_FLAG = '-march=x86-64'


os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

HLIBPRO_LIB = HLIBPRO_DIR + '/lib'
HLIBPRO_INCLUDE = HLIBPRO_DIR + '/include'

CXX_FLAGS = ['-O3', '-Wall', '-shared', '-fPIC', '-std=c++11', MARCH_FLAG]

HLIBPRO_FLAGS = subprocess.run(['/home/nick/hlibpro-2.8.1/bin/hlib-config', '--cflags', '--lflags'],
                               capture_output=True, text=True).stdout.rstrip().split()

PY_FLAGS = subprocess.run(['python3', '-m', 'pybind11', '--includes'],
                          capture_output=True, text=True).stdout.rstrip().split()

LIBS = ['-lhpro', '-Wl,-rpath,' + HLIBPRO_LIB]

LD_FLAGS  = ['-shared', '-L'+HLIBPRO_LIB]

# INCLUDE_COMMANDS = ['-I'+HLIBPRO_INCLUDE, '-I'+EIGEN_INCLUDE]

# ALL_COMPILE_STUFF = CXX_FLAGS + HLIBPRO_FLAGS + PY_FLAGS + INCLUDE_COMMANDS + LD_FLAGS + LIBS

# extra_compile_args = ALL_COMPILE_STUFF
extra_compile_args = CXX_FLAGS + HLIBPRO_FLAGS + PY_FLAGS + LD_FLAGS + LIBS
# extra_link_args = LIBS
extra_link_args = extra_compile_args

_hlibpro_bindings = Extension('_hlibpro_bindings',
                              # include_dirs = [HLIBPRO_INCLUDE],
                              include_dirs = [HLIBPRO_INCLUDE, EIGEN_INCLUDE],
                              # libraries = ['hpro'],
                              # library_dirs = [HLIBPRO_LIB],
                              # runtime_library_dirs=[HLIBPRO_LIB],
                              sources = ['src/grid_interpolate.cpp',
                                         'src/product_convolution_hmatrix.cpp',
                                         'src/hlibpro_bindings.cpp',
                                         ],
                              language='c++',
                              extra_compile_args=extra_compile_args,
                              # extra_link_args=extra_link_args,
                              extra_link_args=extra_link_args,
                              )


setup (name = 'hlibpro_python_wrapper',
       version = '0.1dev',
       description = 'Python wrapper for HLIBPro',
       author = 'Nick Alger (HLIBPro by Dr. Ronald Kriemann)',
       author_email = 'nalger225@gmail.com',
       url = 'https://github.com/NickAlger/hlibpro_python_wrapper',
       long_description = '''
       HLIBPro is a C++ hierarchical matrix package written by Dr. Ronald Kriemann. 
       This library provides a (very incomplete) set of bindings and helper functions 
       for using HLIBPro from within python. 
       ''',
       packages = find_packages(),
       ext_modules = [_hlibpro_bindings])
