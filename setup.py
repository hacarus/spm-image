from setuptools import setup, find_packages
from pathlib import Path
import sys

LICENSE = 'Modified BSD'

if not (3, 5) <= sys.version_info[:2]:
    raise Exception("spm-image requires a Python 3 version newer than 3.5. \n now running on %s" % sys.version)

# load basic information from files
with Path('spmimage', '__init__.py').open() as f:
    for line in f:
        if line.startswith('__version__'):
            VERSION = line.strip().split()[-1][1:-1]
            break

with Path('requirements.txt').open() as f:
    INSTALL_REQUIRES = [line.strip() for line in f.readlines() if line]

setup(
    name='spm-image',
    author='Takashi Someda',
    author_email='takashi@hacarus.com',
    url='https://github.com/hacarus/spm-image',
    description='Sparse modeling and Compressive sensing in Python',
    version=VERSION,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    test_suite='tests',
    license=LICENSE
)
