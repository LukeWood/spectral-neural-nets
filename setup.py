from setuptools import setup

VERSION = open("VERSION", "r").read()

setup(
    name='kernel-fourier-convolution',
    packages=['kernel_fourier_convolution'],
    version=VERSION,
    description='kernel_fourier_convolution',
    long_description=open("README.md", "r").read(),
    long_description_content_type='text/markdown',
    include_package_data=True,
    url='https://github.com/lukewood/kernel-fourier-convolution',
    author='Luke Wood',
    author_email='lukewoodcs@gmail.com',
)
