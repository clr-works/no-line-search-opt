# No-Line-Search Optimization

An implementation of Davidon's optimally conditioned optimization algorithm without line searches, applied to MNIST digit classification.

## Overview

This repository contains an implementation of the algorithm described in Davidon's 1975 paper: "Optimally Conditioned Optimization Algorithms Without Line Searches" (Mathematical Programming 9, 1–30). The algorithm is applied to train a neural network for MNIST digit classification.

Unlike traditional optimization methods that rely on line searches to determine step sizes, Davidon's approach constructs a quasi-Newton method that automatically adjusts the search direction and step size based on the curvature of the objective function, leading to more efficient convergence.

### Why Davidon Struggles with Neural Networks

1. **No Line Search**: Cannot correct for poor J approximations
2. **Accumulating Errors**: Each J update compounds previous inaccuracies  
3. **High Dimensionality**: 4,015 parameters even for tiny network
4. **Non-Convex Landscape**: Neural network loss surfaces are challenging
   
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

Davidon's method is theoretically elegant but practically fragile. While it can achieve impressive initial progress, it lacks the robustness needed for modern machine learning. The method serves as an important historical milestone and educational tool, demonstrating why line searches and limited-memory approaches (like L-BFGS) became standard.

### Hybrid Approach Potential
Our experiments suggest a practical hybrid:
1. Use Davidon for initial rapid descent
2. Switch to Adam/L-BFGS when progress stalls
3. Monitor condition number of J as switching criterion

## References

- Davidon, W.C.: Optimally Conditioned Optimization Algorithms Without Line Searches. Mathematical Programming 9, 1–30 (1975)

