from setuptools import setup, find_packages

setup(
    name='spm-image',
    author='Takashi Someda',
    author_email='takashi@hacarus.com',
    url='https://github.com/hacarus/spm-image',
    description='Sparse modeling and Compressive sensing in Python',
    version='0.0.1',
    packages=find_packages(),
    test_suite='tests'
)
