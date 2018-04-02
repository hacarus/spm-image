# spm-image : Sparse modeling and Compressive sensing in Python

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
To set up examples environment, run the following commands.
```
python -m venv examples_venv
source examples_venv/bin/activate
pip install -r examples_requirements
```

### Testing

You can run all test cases just like this

```
python -m unittest tests/test_*.py
```

Or run specific test case as follows

```
python -m unittest tests.test_decomposition_ksvd.TestKSVD
```

