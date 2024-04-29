# slsvd

Sparse Logistic Singular Value Decomposition (SLSVD) for Binary Matrix Data

<!-- ![CI/CD](https://github.com/UBC-MDS/lr_cd/actions/workflows/ci-cd.yml/badge.svg)
[![codecov](https://codecov.io/gh/UBC-MDS/lr_cd/branch/main/graph/badge.svg)](https://codecov.io/gh/UBC-MDS/lr_cd)
[![Documentation Status](https://readthedocs.org/projects/lr-cd/badge/?version=latest)](https://lr-cd.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![version](https://img.shields.io/github/v/release/UBC-MDS/lr_cd)
[![Python 3.9.0](https://img.shields.io/badge/python-3.9.0-blue.svg)](https://www.python.org/downloads/release/python-390/)
![release](https://img.shields.io/github/release-date/UBC-MDS/lr_cd)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
 -->


## Project Summary

We implement sparse logistic SVD (SLSVD) using the Majorization-Minimization (MM) algorithms in this Python package. 

Our package consists of two major components:

1. Simulated data generation
2. sparse logistic SVD


## Functions

There are three major functions in this package:

- `generate_data(n, d, rank, random_seed=123)`: this function generates many random data points based on the theta coefficients, which will later be used for model fitting.
- `sparse_logistic_pca(dat, lambda_val=0, k=2, quiet=True, max_iters=100, conv_crit=1e-5,
                        randstart=False, procrustes=True, lasso=True, normalize=False,
                        start_A=None, start_B=None, start_mu=None)`: this function performs Majorization-Minimization algorithm to minimize the mean squared error of linear regression and therefore outputs the optimized intercept and coefficients vector.

## Common Parameters

- `n` (integer): Number of data points.
- `n_features` (integer): Number of features to generate, excluding the intercept.
- `theta` (ndarray): True scalar intercept and coefficient weights vector. The first element should always be the intercept.
- `noise` (float): Standard deviation of a normal distribution added to the generated target y array as noise.
- `random_seed` (integer): Random seed to ensure reproducibility.
- `X` (ndarray): Feature data matrix, the independent variable.
<!-- - `y` (ndarray): Response data vector, the dependent variable. Both `X` and `y` should have the same number of observations. -->
<!-- - `ϵ` (float, optional): Stop the algorithm if the change in weights is smaller than the value (default is 1e-6). -->
- `max_iterations` (integer, optional): Maximum number of iterations (default is 1000).
<!-- - `intercept` (float): Optimized intercept. It will be used to calculate the estimated values using observed data `X`. -->
<!-- - `coef` (ndarray): Optimized coefficient weights vector. It will be used to calculate the estimated values using observed data `X`. -->

## Python Ecosystem Context

**SLSVD** establishes itself as a valuable enhancement to the Python ecosystem. There is no function in the Python package `scikit-learn` has similar functionality,  our implementation uses Majorization-Minimization algorithm.

- **Beginner-Friendly** : `SLSVD` is easy to use for beginners in Python and statistics. The well-written functions allow users to play with various simulated data and generate beautiful plots with little effort.

- **Reliable-Alternative** : The Majorization-Minimization algorithm is known for convergence in various optimization problems, making this Python package a reliable alternative to existed packages. 







## Installation

### Prerequisites

Make sure Miniconda or Anaconda is installed on your system

#### Step 1: Clone the Repository

```bash
git clone git@github.com:andyzhangstat/SLSVD.git
cd SLSVD  # Navigate to the cloned repository directory
```

#### Step 2: Create and Activate the Conda Environment

```bash
# Method 1: create Conda Environment from the environment.yml file
conda env create -f environment.yml  # Create Conda environment
conda activate SLSVD  # Activate the Conda environment

# Method 2: create Conda Environment 
conda create --name SLSVD python=3.9 -y
conda activate SLSVD
```

#### Step 3: Install the Package Using Poetry

Ensure the Conda environment is activated (you should see (SLSVD) in the terminal prompt)

```bash
poetry install  # Install the package using Poetry
```

#### Step 4: Get the coverage

```bash
# Check line coverage
pytest --cov=SLSVD

# Check branch coverage
pytest --cov-branch --cov=SLSVD
poetry run pytest --cov-branch --cov=src
poetry run pytest --cov-branch --cov=SLSVD --cov-report html
```

#### Troubleshooting

1. Environment Creation Issues: Ensure environment.yml is in the correct directory and you have the correct Conda version

2. Poetry Installation Issues: Verify Poetry is correctly installed in the Conda environment and your pyproject.toml file is properly configured

## Usage

Use this package to find the optimized score and loading matrices of sparse logistic Singular Value Decomposition. In the following example, we generate a simulated data set with defined size first. By the Majorization-Minimization algorithm, we obtain the optimized score and loading matrices. Finally, we visualize both the simulated data and fitted loadings in one figure.

Example usage:

```python
>>> from slsvd.data_generation import generate_data
>>> import numpy as np
>>> theta = np.array([4, 3])
>>> X, y = generate_data_lr(n=10, n_features=1, theta=theta)
>>> print('Generated X:')
>>> print(X)
>>> print('Generated y:')
>>> print(y)
```

```
Generated X:
[[0.69646919]
 [0.28613933]
 [0.22685145]
 [0.55131477]
 [0.71946897]
 [0.42310646]
 [0.9807642 ]
 [0.68482974]
 [0.4809319 ]
 [0.39211752]]
Generated y:
[[6.34259481]
 [4.68506992]
 [4.54477713]
 [5.63500251]
 [6.45668483]
 [5.14153898]
 [6.8534962 ]
 [5.96761896]
 [5.88398172]
 [5.61370977]]
```



## Documentations

Online documentation is available [readthedocs](https://lr-cd.readthedocs.io/en/latest/?badge=latest).

Publishing on [TestPyPi](https://test.pypi.org/project/lr-cd/) and [PyPi](https://pypi.org/project/lr-cd/).

## Contributors

[Andy Zhang](https://github.com/andyzhangstat) 


## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`SLSVD` was created by Andy Zhang. It is licensed under the terms of the MIT license.

## Credits

`SLSVD` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
