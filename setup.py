from setuptools import setup, find_packages
from pathlib import Path
import sys

from spmimage import __version__ as version

LICENSE = 'Modified BSD'

if not (3, 5) <= sys.version_info[:2]:
    raise Exception('spm-image requires Python 3.5 or later. \n Now running on {0}'.format(sys.version))

with Path('requirements.txt').open() as f:
    INSTALL_REQUIRES = [line.strip() for line in f.readlines() if line]

setup(
    name='spm-image',
    author='Takashi Someda',
    author_email='takashi@hacarus.com',
    url='https://github.com/hacarus/spm-image',
    description='Sparse modeling and Compressive sensing in Python',
    version=version,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    test_suite='tests',
    license=LICENSE
)
