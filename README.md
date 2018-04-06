# spm-image : Sparse modeling and Compressive sensing in Python [![Build Status](https://travis-ci.org/hacarus/spm-image.svg?branch=development)](https://travis-ci.org/hacarus/spm-image)

spm-image is a Python library for image analysis using sparse modeling and compressive sensing.

## Requirements

* Python 3.5 or later

## Install

    pip install spm-image

## For developers

To set up development environment, run the following commands.

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Examples

If you want to run examples, create separated venv from one for development above.

```
python -m venv examples_venv
source examples_venv/bin/activate
pip install -r examples_requirements.txt
```

Then add it to jupyter kernels like this.

```
python -m ipykernel install --user --name spm-image-examples --display-name "spm-image Examples"
```

Thereafter, you can run jupyter notebook as follows.

```
jupyter notebook
```

### Testing

You can run all test cases just like this

```
python -m unittest discover
```

Or run specific test case as follows

```
python -m unittest tests.test_decomposition_ksvd.TestKSVD
```

