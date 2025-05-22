# Create a installation script for the package
from setuptools import setup, find_packages
setup(
    name='HDRangeViT',
    version='0.1',
    author='Hai Dinh',
    packages=find_packages(),
)

# To install the package by
#  pip install -e .
#  Note that the -e flag is used for editable installs, which means that changes to the code will be reflected immediately without needing to reinstall the package.
#  This is useful for development purposes.
#  The . means that the setup.py file is in the current directory.
#  You can also specify the path to the directory containing the setup.py file if it is in a different location.
#  For example, if the setup.py file is in a directory called my_package, you can run:
#  pip install -e my_package
#  This will install the package in editable mode.