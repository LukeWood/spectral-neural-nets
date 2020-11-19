from setuptools import setup

VERSION = open("VERSION", "r").read()

setup(
    name='spectral-neural-nets',
    packages=['spectral_neural_nets'],
    version=VERSION,
    description='spectral_neural_nets',
    long_description=open("README.md", "r").read(),
    long_description_content_type='text/markdown',
    include_package_data=True,
    url='https://github.com/lukewood/spectral-neural-nets',
    author='Luke Wood',
    author_email='lukewoodcs@gmail.com',
)
