# Numerical Report for the Group LASSO Problem

Course Project for Optimization Methods (Peking University)

This repository contains MATLAB implementations and numerical experiments for the Group LASSO problem. The project compares multiple optimization methods, including commercial solver baselines, first-order primal methods, and augmented Lagrangian / ADMM-based algorithms.

## Project Overview

We study the matrix-valued Group LASSO problem

```math
\min_{X \in \mathbb{R}^{n \times l}}
\frac{1}{2}\|AX-b\|_F^2 + \mu \sum_{i=1}^{n}\|X_{i,:}\|_2.
```

Here

- $A \in \mathbb{R}^{m \times n}$ is the data matrix,
- $X \in \mathbb{R}^{n \times l}$ is the optimization variable,
- $b \in \mathbb{R}^{m \times l}$ is the observation matrix,
- $\mu > 0$ is the regularization parameter.

The row-wise $\ell_2$ regularization promotes **group sparsity**, meaning that entire rows of $X$ are encouraged to become zero together.

The goal of this project is to solve the same optimization problem using different numerical methods and compare them in terms of runtime, convergence behavior, objective value, and sparsity recovery.

## Implemented Methods

### Reference / Solver-Based Baselines

- `gl_cvx_mosek.m`
- `gl_cvx_gurobi.m`
- `gl_mosek.m`
- `gl_gurobi.m`

These solvers provide reference-quality solutions and useful baselines for comparison.

### First-Order Primal Methods

- `gl_SGD_primal.m`
- `gl_GD_primal.m`
- `gl_ProxGD_primal.m`
- `gl_FProxGD_primal.m`

These methods are implemented directly in MATLAB and are useful for studying algorithmic behavior on structured nonsmooth optimization problems.

### Augmented Lagrangian / Splitting Methods

- `gl_ALM_dual.m`
- `gl_ADMM_dual.m`
- `gl_ADMM_primal.m`

These methods reformulate the Group LASSO problem and solve it through splitting or augmented Lagrangian updates.

### Helper Functions

- `sm_grad.m`
- `subgrad.m`

These auxiliary routines are used by the gradient-based and subgradient-based methods.

## Repository Structure

- `code/` — MATLAB source files
- `code/Test.m` — main script for generating data and running experiments
- `report.pdf` — written course report
- `README.md` — repository overview

## Default Experiment

The default experiment in `code/Test.m` generates a synthetic Group LASSO instance and evaluates one solver on it.

Current default settings include:

- `n = 512`
- `m = 256`
- `l = 2`
- about 10% active groups in the ground-truth solution
- `mu = 1e-2`
- default solver: `gl_ADMM_primal`

The script then reports:

- CPU time
- number of iterations
- objective value
- sparsity of the recovered solution
- relative error to the synthetic ground truth

## How to Run

1. Open MATLAB and go to the `code/` folder.
2. Open `Test.m`.
3. Uncomment the solver you want to test.
4. Run the script.

For example:

```matlab
[x, iter, out] = gl_ADMM_primal(x0, A, b, mu, opts);
```

You can replace this line with other solvers in the repository to compare their performance.

## Dependencies

Some solvers require external optimization software:

- `gl_cvx_mosek.m` requires CVX with MOSEK
- `gl_cvx_gurobi.m` requires CVX with Gurobi
- `gl_mosek.m` requires the MOSEK API
- `gl_gurobi.m` requires the Gurobi API

The first-order and splitting-based MATLAB solvers can be run directly once MATLAB is properly configured.

## Output and Comparison

This repository is designed not only to compute a solution, but also to support method comparison. By switching solvers in `Test.m`, you can compare:

- solution quality,
- convergence speed,
- sparsity recovery,
- computational cost.

This makes the repository suitable as both a coursework submission and a compact reference implementation for the Group LASSO problem.

## Notes

This project was created for educational purposes as part of the Optimization Methods course at Peking University.

For more details on the mathematical formulation, derivations, and numerical discussion, please refer to `report.pdf`.

## Author

Yuhan Ye
