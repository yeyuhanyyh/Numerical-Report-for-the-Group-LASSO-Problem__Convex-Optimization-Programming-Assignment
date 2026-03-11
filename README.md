# Group LASSO Solver Collection

This repository contains MATLAB implementations for a course project on the Group LASSO problem from the Optimization Methods course at Peking University.

## Project Overview

This project studies the matrix-valued Group LASSO problem:

min_X  0.5 * ||A*X - B||_F^2 + mu * sum_i ||X(i,:)||_2

where

- `A` is an `m x n` data matrix,
- `X` is an `n x l` decision matrix,
- `B` is an `m x l` observation matrix,
- `mu > 0` is the regularization parameter.

The row-wise `l2` penalty promotes group sparsity, which means entire rows of `X` tend to become zero together.

## What This Repository Includes

This repository contains:

- several benchmark solvers based on CVX, MOSEK, and Gurobi,
- multiple first-order methods implemented in MATLAB,
- ADMM and augmented Lagrangian based solvers,
- a test script for generating synthetic data and comparing methods,
- a written project report in PDF format.

## Implemented Solvers

### Benchmark / Reference Solvers

- `gl_cvx_mosek.m`
- `gl_cvx_gurobi.m`
- `gl_mosek.m`
- `gl_gurobi.m`

### First-Order Methods

- `gl_SGD_primal.m`
- `gl_GD_primal.m`
- `gl_ProxGD_primal.m`
- `gl_FProxGD_primal.m`

### Augmented Lagrangian / Splitting Methods

- `gl_ALM_dual.m`
- `gl_ADMM_dual.m`
- `gl_ADMM_primal.m`

### Helper Functions

- `sm_grad.m`
- `subgrad.m`

## Repository Structure

- `code/` — MATLAB source files
- `report.pdf` — written project report
- `README.md` — repository overview

## Default Experiment in `code/Test.m`

The default test script does the following:

- sets a fixed random seed for reproducibility,
- generates synthetic data with `n = 512`, `m = 256`, and `l = 2`,
- uses about 10% active groups in the ground-truth solution,
- sets `mu = 1e-2`,
- initializes a random starting point,
- currently runs `gl_ADMM_primal` by default,
- prints CPU time, iteration count, objective value, sparsity, and error-to-exact.

## How to Run

1. Open MATLAB and go to the `code/` folder.
2. Open `Test.m`.
3. Uncomment the solver you want to test and comment out the others.
4. Run `Test.m`.

## External Dependencies

Some solvers require external optimization packages:

- `gl_cvx_mosek.m` requires CVX with MOSEK
- `gl_cvx_gurobi.m` requires CVX with Gurobi
- `gl_mosek.m` uses the MOSEK API directly
- `gl_gurobi.m` uses the Gurobi API directly

The gradient-based and splitting-based methods are implemented directly in MATLAB in this repository.

## Output

The script reports:

- runtime,
- number of iterations,
- objective value,
- solution sparsity,
- relative error to the synthetic ground truth.

This makes it convenient to compare accuracy, sparsity recovery, and computational cost across different methods.

## Notes

- This repository was created for coursework and is intended for educational use.
- It can also serve as a compact reference for comparing different optimization approaches for Group LASSO.
- See `report.pdf` for the accompanying written report.

## Author

Yuhan Ye
