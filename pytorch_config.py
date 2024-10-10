import distutils
import setuptools
import setuptools
from setuptools import msvc

# Specify the platform spec
vc_env = msvc.msvc14_get_vc_env('x64')  # Change 'x64' to match your platform if needed


# Patch distutils to include _msvccompiler
distutils._msvccompiler = setuptools._msvccompiler