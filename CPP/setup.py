import os, sys

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import pybind11

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

class get_pybind_include(object):

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

cpp_args = ['-std=c++11']

ext_modules = [
    Extension(
    'icu',
        ['ICUsim.cpp'],
        include_dirs=[get_pybind_include(), 
        get_pybind_include(user=True),
        os.path.join(os.path.dirname(BASE_DIR), 'eigen'],
    language='c++',
    extra_compile_args = cpp_args,
    ),
    Extension(
    'icuh',
        ['ICUHsim.cpp'],
        include_dirs=[get_pybind_include(), 
        get_pybind_include(user=True),
        os.path.join(os.path.dirname(BASE_DIR), 'eigen'],
    language='c++',
    extra_compile_args = cpp_args,
    ),
]

setup(
    name='Alice_models',
    version='0.0.1',
    author='sanmitra ghosh',
    author_email='sanmitra.ghosh@mrc-bsu.cam.ac.uk',
    description='ICUH',
    ext_modules=ext_modules,
)
