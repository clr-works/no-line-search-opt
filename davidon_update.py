import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass
from os.path  import join
import pickle
from loader import MnistDataloader
from preprocessing import prepare_data, one_hot_encode
from helper_functions import *
from memory_profiler import profile
import time


@dataclass
class DavidonState:
    """Maintains state for Davidon's algorithm"""
    J: np.ndarray  # Jacobian/Hessian approximation
    k0: np.ndarray  # Previous gradient transformed by J
    omega: np.ndarray  # Auxiliary vector
    E0: float  # Previous cost
    E_prime0: float  # Previous directional derivative


class DavidonOptimizer:
    """
    Davidon's Quasi-Newton optimizer without line searches
    Based on "Optimally Conditioned Optimization Algorithms Without Line Searches" (1975)
    """

    def __init__(self, total_params: int, epsilon: float = 1e-6, max_iterations: int = 100):
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.iteration = 0

        # Initialize with scaled identity for better conditioning
        self.state = DavidonState(
            J=np.eye(total_params, dtype=np.float32),  # Use float32 for better precision
            k0=None,
            omega=None,
            E0=float('inf'),
            E_prime0=0
        )

    def step(self, parameters: Dict, grads: Dict, cost: float,
             x_train: np.ndarray, y_train: np.ndarray,
             forward_fn, cost_fn, structure_cache: List) -> Tuple[Dict, float]:
        """
        Perform one optimization step

        Args:
            parameters: Current model parameters
            grads: Current gradients
            cost: Current cost
            x_train, y_train: Training data for re-evaluation
            forward_fn: Function to perform forward pass
            cost_fn: Function to compute cost
            structure_cache: Structure information for parameter updates

        Returns:
            Updated parameters and new cost
        """

        # Flatten gradients
        grad_vector = self._flatten_gradients(grads, structure_cache)

        # First iteration initialization
        if self.state.k0 is None:
            self.state.k0 = self.state.J @ grad_vector
            self.state.omega = self.state.k0.copy()
            self.state.E0 = cost

        # Compute search direction
        if self.iteration == 0:
            s = -self.state.k0
        else:
            s = self._compute_search_direction(grad_vector)

        # Check convergence
        E_prime0 = np.dot(self.state.k0, s)
        if abs(E_prime0) < self.epsilon:
            print(f"Converged at iteration {self.iteration}")
            return parameters, cost

        # Adjust step size if needed (Davidon's condition)
        if 4 * self.state.E0 < -E_prime0:
            s = -4 * s * (self.state.E0 / E_prime0)
            E_prime0 = np.dot(self.state.k0, s)

        # Update parameters
        new_params = self._update_parameters(parameters, structure_cache, s)

        # Evaluate new cost
        AL, _ = forward_fn(x_train, new_params)
        new_cost = cost_fn(AL, y_train)

        # Backtracking if cost increased
        lambda_factor = 1.0
        while new_cost > self.state.E0 and lambda_factor > 1e-8:
            lambda_factor *= 0.5
            s *= 0.5
            E_prime0 *= 0.5

            new_params = self._update_parameters(parameters, structure_cache, s)
            AL, _ = forward_fn(x_train, new_params)
            new_cost = cost_fn(AL, y_train)

        # Update Jacobian approximation
        if new_cost < self.state.E0:
            self._update_jacobian(s, grad_vector, E_prime0)

        # Update state
        self.state.k0 = self.state.J @ grad_vector
        self.state.E0 = new_cost
        self.state.E_prime0 = E_prime0
        self.iteration += 1

        return new_params, new_cost

    def _flatten_gradients(self, grads: Dict, structure_cache: List) -> np.ndarray:
        """Flatten gradients into a single vector"""
        flattened = []
        for grad_key, shape in structure_cache:
            if grad_key in grads:
                flattened.append(grads[grad_key].flatten())
        return np.concatenate(flattened)

    def _update_parameters(self, params: Dict, structure_cache: List,
                           direction: np.ndarray, learning_rate: float = 0.01) -> Dict:
        """Update parameters using the search direction"""
        new_params = params.copy()

        # Remove J from parameters if it exists
        if 'J' in new_params:
            del new_params['J']

        start = 0
        for grad_key, shape in structure_cache:
            param_key = grad_key[1:]  # Remove 'd' prefix

            size = np.prod(shape)
            segment = direction[start:start + size].reshape(shape)

            new_params[param_key] = params[param_key] + learning_rate * segment
            start += size

        return new_params

    def _compute_search_direction(self, grad_vector: np.ndarray) -> np.ndarray:
        """Compute search direction based on current state"""
        k = self.state.J @ grad_vector

        # Simple direction computation for now
        # Full Davidon algorithm would involve more complex calculations
        return -k

    def _update_jacobian(self, s: np.ndarray, grad_vector: np.ndarray,
                         E_prime: float) -> None:
        """
        Update Jacobian approximation using Davidon's formula
        This is a simplified version - full implementation would involve
        the complete update rules from the paper
        """
        k = self.state.J @ grad_vector
        y = k - self.state.k0  # Change in transformed gradient

        # Avoid division by zero
        denominator = np.dot(s, y)
        if abs(denominator) > 1e-10:
            # Rank-2 update (simplified BFGS-like update)
            Bs = self.state.J @ s
            sBs = np.dot(s, Bs)

            if sBs > 1e-10:
                self.state.J -= np.outer(Bs, Bs) / sBs

            self.state.J += np.outer(y, y) / denominator


def train_with_davidon(model_forward, model_backward, compute_cost,
                       x_train, y_train, x_test, y_test,
                       parameters, units_in_layer,
                       epsilon=1e-6, max_iterations=100):
    """
    Train neural network using Davidon's method

    Returns:
        parameters: Optimized parameters
        train_costs: Training cost history
        test_costs: Test cost history
    """

    # Get initial cost and gradients
    AL, caches = model_forward(x_train, parameters)
    cost = compute_cost(AL, y_train)
    grads = model_backward(AL, y_train, caches)

    # Create structure cache
    structure_cache = []
    L = len(units_in_layer)
    for l in range(1, L):
        structure_cache.append((f'dW{l}', grads[f'dW{l}'].shape))
        structure_cache.append((f'db{l}', grads[f'db{l}'].shape))

    # Calculate total parameters
    total_params = sum(np.prod(shape) for _, shape in structure_cache)

    # Initialize optimizer
    optimizer = DavidonOptimizer(total_params, epsilon, max_iterations)

    # Training history
    train_costs = [cost]
    test_costs = []

    # Evaluate initial test cost
    AL_test, _ = model_forward(x_test, parameters)
    test_cost = compute_cost(AL_test, y_test)
    test_costs.append(test_cost)

    print(f"Initial - Train Cost: {cost:.4f}, Test Cost: {test_cost:.4f}")

    # Training loop
    for iteration in range(max_iterations):
        # Forward pass
        AL, caches = model_forward(x_train, parameters)
        cost = compute_cost(AL, y_train)

        # Backward pass
        grads = model_backward(AL, y_train, caches)

        # Optimization step
        parameters, cost = optimizer.step(
            parameters, grads, cost,
            x_train, y_train,
            model_forward, compute_cost,
            structure_cache
        )

        # Record costs
        train_costs.append(cost)

        # Evaluate on test set
        AL_test, _ = model_forward(x_test, parameters)
        test_cost = compute_cost(AL_test, y_test)
        test_costs.append(test_cost)

        # Print progress
        if iteration % 10 == 0:
            print(f"Iteration {iteration} - Train Cost: {cost:.4f}, Test Cost: {test_cost:.4f}")

        # Check convergence
        if len(train_costs) > 1 and abs(train_costs[-1] - train_costs[-2]) < epsilon:
            print(f"Converged at iteration {iteration}")
            break

    return parameters, train_costs, test_costs


# Example usage in main script:
if __name__ == '__main__':

    #################### Input data: #############################
    input_path = '/home/corina/Documents/Math_Machine_Learning/minst'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                       test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    # Preprocess datasets
    x_train_flattened, x_test_flattened, y_train_flattened, y_test_flattened = prepare_data(x_train, y_train, x_test,
                                                                                            y_test)
    # One hot encode Y ground true values
    one_hot_encoded_y_train = one_hot_encode(y_train_flattened)
    one_hot_encoded_y_test = one_hot_encode(y_test_flattened)

    ############ Define the number of units in each layer of the network ###############
    units_in_layer = [784, 5, 5, 10]

    # Initialize the parameters
    parameters = initialize_parameters_davidon(units_in_layer)

    # Train with Davidon's method
    optimized_params, train_history, test_history = train_with_davidon(
        Model_forward, Model_backward, compute_cost,
        x_train_flattened, one_hot_encoded_y_train.T,
        x_test_flattened, one_hot_encoded_y_test.T,
        parameters, units_in_layer,
        epsilon=1e-6, max_iterations=100
    )

    # Evaluate final accuracy
    predictions_train = predict(x_train_flattened, optimized_params)
    predictions_test = predict(x_test_flattened, optimized_params)

    accuracy_train = compute_accuracy(predictions_train, y_train_flattened)
    accuracy_test = compute_accuracy(predictions_test, y_test_flattened)

    print(f"\nFinal Accuracy - Train: {accuracy_train:.4f}, Test: {accuracy_test:.4f}")