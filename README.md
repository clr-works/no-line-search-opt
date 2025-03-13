# No-Line-Search Optimization

An implementation of Davidon's optimally conditioned optimization algorithm without line searches, applied to MNIST digit classification.

## Overview

This repository contains an implementation of the algorithm described in Davidon's 1975 paper: "Optimally Conditioned Optimization Algorithms Without Line Searches" (Mathematical Programming 9, 1–30). The algorithm is applied to train a neural network for MNIST digit classification.

Unlike traditional optimization methods that rely on line searches to determine step sizes, Davidon's approach constructs a quasi-Newton method that automatically adjusts the search direction and step size based on the curvature of the objective function, leading to more efficient convergence.

## Features

- Implementation of Davidon's quasi-Newton update algorithm
- Application to MNIST handwritten digit classification
- Neural network with configurable layer sizes
- Performance metrics including training/test accuracy and cost tracking
- Memory profiling for optimization analysis

## Requirements

- Python 3.x
- NumPy
- memory_profiler
- MNIST dataset (instructions for downloading included below)

## How It Works

The implementation follows Davidon's algorithm for quasi-Newton optimization:

1. Initialize parameters, including the Jacobian matrix
2. Perform forward and backward passes to compute initial cost and gradients
3. Apply Davidon's update strategy to determine search direction and step size
4. Update parameters and adjust based on cost improvements
5. Refine the quasi-Newton approximation based on observation of function behavior
6. Continue until convergence criteria or maximum iterations are reached

The algorithm automatically adjusts step sizes without requiring explicit line searches, which can be computationally expensive in high-dimensional optimization problems.

## Mathematical Background

Davidon's method focuses on constructing positive definite matrices that approximate the Hessian (or its inverse) for Newton-like optimization. Key features include:

- Automatic determination of both search direction and step size
- Adaptation to the local curvature of the objective function
- Convergence properties similar to Newton's method but without requiring second derivatives
- Robust operation without line searches

## Performance

The implementation includes memory profiling to analyze efficiency. The algorithm typically converges in fewer iterations than gradient descent methods, especially for ill-conditioned problems.

## References

- Davidon, W.C.: Optimally Conditioned Optimization Algorithms Without Line Searches. Mathematical Programming 9, 1–30 (1975)

