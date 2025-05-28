import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass
from os.path import join
import pickle
from loader import MnistDataloader
from preprocessing import prepare_data, one_hot_encode
from helper_functions import *
from memory_profiler import profile
import time
import pandas as pd


@dataclass
class DavidonState:
    """Maintains state for Davidon's algorithm"""
    J: np.ndarray  # Jacobian/Hessian approximation
    k0: np.ndarray  # Previous gradient transformed by J
    omega: np.ndarray  # Auxiliary vector
    E0: float  # Previous cost
    E_prime0: float  # Previous directional derivative


class DavidonDebugger:
    """Debug logger for Davidon optimizer"""

    def __init__(self):
        self.history = {
            'iteration': [],
            'cost': [],
            'grad_norm': [],
            'direction_norm': [],
            'step_size': [],
            'E_prime0': [],
            'backtrack_count': [],
            'update_type': [],  # 'davidon' or 'gradient_descent'
            'J_condition_number': [],
            'param_norms': {},
            'grad_norms': {},
            'update_norms': {}
        }

    def log_iteration(self, iteration: int, cost: float, grad_vector: np.ndarray,
                      direction: np.ndarray, step_size: float, E_prime0: float,
                      backtrack_count: int, update_type: str, J: np.ndarray):
        """Log data for current iteration"""
        self.history['iteration'].append(iteration)
        self.history['cost'].append(cost)
        self.history['grad_norm'].append(np.linalg.norm(grad_vector))
        self.history['direction_norm'].append(np.linalg.norm(direction))
        self.history['step_size'].append(step_size)
        self.history['E_prime0'].append(E_prime0)
        self.history['backtrack_count'].append(backtrack_count)
        self.history['update_type'].append(update_type)

        # Compute condition number of J (expensive, do sparingly)
        if iteration % 10 == 0:
            try:
                cond = np.linalg.cond(J[:100, :100])  # Use subset for efficiency
            except:
                cond = np.inf
            self.history['J_condition_number'].append(cond)
        else:
            self.history['J_condition_number'].append(None)

    def log_parameters(self, iteration: int, params: Dict, grads: Dict,
                       updates: Dict[str, np.ndarray]):
        """Log parameter-specific data"""
        for key in params:
            if key == 'J':
                continue

            # Initialize if needed
            if key not in self.history['param_norms']:
                self.history['param_norms'][key] = []
                self.history['grad_norms'][f'd{key}'] = []
                self.history['update_norms'][key] = []

            # Log norms
            self.history['param_norms'][key].append(np.linalg.norm(params[key]))

            grad_key = f'd{key}'
            if grad_key in grads:
                self.history['grad_norms'][grad_key].append(np.linalg.norm(grads[grad_key]))

            if key in updates:
                self.history['update_norms'][key].append(np.linalg.norm(updates[key]))

    def save_to_csv(self, filename_prefix: str = "davidon_debug"):
        """Save debugging data to CSV files"""
        # Main iteration data
        main_df = pd.DataFrame({
            'iteration': self.history['iteration'],
            'cost': self.history['cost'],
            'grad_norm': self.history['grad_norm'],
            'direction_norm': self.history['direction_norm'],
            'step_size': self.history['step_size'],
            'E_prime0': self.history['E_prime0'],
            'backtrack_count': self.history['backtrack_count'],
            'update_type': self.history['update_type'],
            'J_condition_number': self.history['J_condition_number']
        })
        main_df.to_csv(f"{filename_prefix}_main.csv", index=False)

        # Parameter norms
        param_df = pd.DataFrame(self.history['param_norms'])
        param_df['iteration'] = range(len(param_df))
        param_df.to_csv(f"{filename_prefix}_params.csv", index=False)

        # Gradient norms
        grad_df = pd.DataFrame(self.history['grad_norms'])
        grad_df['iteration'] = range(len(grad_df))
        grad_df.to_csv(f"{filename_prefix}_grads.csv", index=False)

        # Update norms
        update_df = pd.DataFrame(self.history['update_norms'])
        update_df['iteration'] = range(len(update_df))
        update_df.to_csv(f"{filename_prefix}_updates.csv", index=False)

        print(f"Debug data saved to {filename_prefix}_*.csv files")


def analyze_davidon_behavior(optimizer, grad_vector, J):
    """Analyze why Davidon updates might be failing"""
    print("\n=== Davidon Analysis ===")

    # Check gradient magnitude
    print(f"Gradient norm: {np.linalg.norm(grad_vector):.6f}")

    # Check J properties
    print(f"J shape: {J.shape}")
    print(f"J diagonal sample: {np.diag(J)[:5]}")

    # Check if J is well-conditioned
    try:
        eigenvalues = np.linalg.eigvalsh(J[:100, :100])  # Check small submatrix
        print(f"J eigenvalue range: [{eigenvalues.min():.6f}, {eigenvalues.max():.6f}]")
        print(f"J condition number (subset): {eigenvalues.max() / eigenvalues.min():.2e}")
    except:
        print("Could not compute eigenvalues")

    # Check search direction
    k = J @ grad_vector
    print(f"k = J @ grad norm: {np.linalg.norm(k):.6f}")
    angle = np.arccos(np.clip(np.dot(grad_vector, k) / (np.linalg.norm(grad_vector) * np.linalg.norm(k)), -1, 1))
    print(f"Angle between grad and k: {angle * 180 / np.pi:.2f} degrees")

    return k


class DavidonOptimizer:
    """
    Davidon's Quasi-Newton optimizer without line searches
    Based on "Optimally Conditioned Optimization Algorithms Without Line Searches" (1975)
    """

    def __init__(self, total_params: int, epsilon: float = 1e-6, max_iterations: int = 100, debug: bool = True):
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.iteration = 0
        self.min_step_size = 1e-8  # Minimum step size to prevent vanishing updates
        self.max_backtracks = 10  # Limit backtracking iterations
        self.gradient_descent_count = 0  # Track fallbacks
        self.debug = debug

        if self.debug:
            self.debugger = DavidonDebugger()

        # Initialize with scaled identity for better conditioning
        # Scale based on expected gradient magnitude
        scale = 0.1  # Conservative initial scale
        self.state = DavidonState(
            J=np.eye(total_params, dtype=np.float32) * scale,
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

        # DEBUG: Analyze gradient and J
        if self.debug and self.iteration < 5:
            print(f"\n=== Iteration {self.iteration} ===")
            print(f"Initial cost: {cost:.6f}")
            analyze_davidon_behavior(self, grad_vector, self.state.J)

        # First iteration initialization
        if self.state.k0 is None:
            self.state.k0 = self.state.J @ grad_vector
            self.state.omega = self.state.k0.copy()
            self.state.E0 = cost
            if self.debug:
                print(f"Initial k0 norm: {np.linalg.norm(self.state.k0):.6f}")

        # Compute search direction
        if self.iteration == 0:
            s = -self.state.k0
        else:
            s = self._compute_search_direction(grad_vector)

        if self.debug:
            print(f"Search direction norm: {np.linalg.norm(s):.6f}")

        # Check convergence
        E_prime0 = np.dot(self.state.k0, s)
        if self.debug:
            print(f"E_prime0 (directional derivative): {E_prime0:.6f}")

        if abs(E_prime0) < self.epsilon:
            print(f"Converged at iteration {self.iteration}")
            return parameters, cost

        # Adjust step size if needed (Davidon's condition)
        if 4 * self.state.E0 < -E_prime0:
            s = -4 * s * (self.state.E0 / E_prime0)
            E_prime0 = np.dot(self.state.k0, s)
            if self.debug:
                print(f"Adjusted step, new E_prime0: {E_prime0:.6f}")

        # Update parameters
        new_params = self._update_parameters(parameters, structure_cache, s)

        # Evaluate new cost
        AL, _ = forward_fn(x_train, new_params)
        new_cost = cost_fn(AL, y_train)

        # Backtracking if cost increased
        lambda_factor = 1.0
        backtrack_count = 0
        original_s = s.copy()  # Save original direction
        update_type = 'davidon'

        while new_cost > self.state.E0 and backtrack_count < self.max_backtracks:
            lambda_factor *= 0.5
            s = original_s * lambda_factor  # Scale from original

            new_params = self._update_parameters(parameters, structure_cache, s)
            AL, _ = forward_fn(x_train, new_params)
            new_cost = cost_fn(AL, y_train)

            backtrack_count += 1
            if self.debug and backtrack_count <= 3:
                print(f"  Backtrack {backtrack_count}: lambda={lambda_factor:.6f}, cost={new_cost:.6f}")

        # If still no improvement, use gradient descent step
        if new_cost > self.state.E0 and backtrack_count >= self.max_backtracks:
            self.gradient_descent_count += 1
            update_type = 'gradient_descent'
            print(f"  WARNING: Davidon step failed after {backtrack_count} backtracks")
            print(f"  Falling back to gradient descent (total fallbacks: {self.gradient_descent_count})")
            s = -grad_vector * 0.01  # Small gradient step
            new_params = self._update_parameters(parameters, structure_cache, s)
            AL, _ = forward_fn(x_train, new_params)
            new_cost = cost_fn(AL, y_train)
            print(f"  Gradient descent cost: {new_cost:.6f} (vs Davidon attempt: {self.state.E0:.6f})")

        # Log debug information
        if self.debug:
            self.debugger.log_iteration(
                self.iteration, new_cost, grad_vector, s, lambda_factor,
                E_prime0, backtrack_count, update_type, self.state.J
            )

            # Extract update information
            updates = {}
            start = 0
            for grad_key, shape in structure_cache:
                param_key = grad_key[1:]
                size = np.prod(shape)
                updates[param_key] = s[start:start + size].reshape(shape)
                start += size

            self.debugger.log_parameters(self.iteration, new_params, grads, updates)

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
                           direction: np.ndarray, step_size: float = 1.0) -> Dict:
        """Update parameters using the search direction"""
        import copy
        new_params = copy.deepcopy(params)

        # Remove J from parameters if it exists
        if 'J' in new_params:
            del new_params['J']

        # DEBUG: Check direction magnitude
        dir_norm = np.linalg.norm(direction)
        if self.iteration < 5:  # Only print first few iterations
            print(f"Update direction norm: {dir_norm:.6f}, step_size: {step_size:.6f}")

        start = 0
        total_update_norm = 0
        for grad_key, shape in structure_cache:
            param_key = grad_key[1:]  # Remove 'd' prefix

            size = np.prod(shape)
            segment = direction[start:start + size].reshape(shape)

            if self.iteration < 5:  # Detailed logging for first few iterations
                update_norm = np.linalg.norm(segment)
                param_norm = np.linalg.norm(params[param_key])
                # Avoid division by zero for bias terms
                relative_update = update_norm / (param_norm + 1e-8)

                print(
                    f"  {param_key}: update_norm={update_norm:.6f}, param_norm={param_norm:.6f}, relative={relative_update:.6f}")

            # Apply update with step size
            new_params[param_key] = params[param_key] + step_size * segment

            total_update_norm += np.linalg.norm(segment) ** 2
            start += size

        if self.iteration < 5:
            print(f"Total update norm: {np.sqrt(total_update_norm):.6f}")

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
        optimizer: The optimizer object (for debugging)
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

    # Initialize optimizer with adjusted epsilon for convergence
    # Use larger epsilon to avoid premature convergence
    optimizer = DavidonOptimizer(total_params, epsilon=1e-4, max_iterations=max_iterations, debug=True)

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

        # Check convergence based on cost change, not just parameter change
        if len(train_costs) > 10:  # Wait for some iterations
            recent_costs = train_costs[-10:]
            cost_variance = np.var(recent_costs)
            if cost_variance < epsilon ** 2:
                print(f"Converged at iteration {iteration} - cost variance: {cost_variance:.2e}")
                break

    # Save debug data before returning
    if hasattr(optimizer, 'debugger'):
        print("\nSaving debug data...")
        optimizer.debugger.save_to_csv("davidon_debug")

    return parameters, train_costs, test_costs, optimizer  # Return optimizer too


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

    # Train with Davidon's method - now captures the optimizer too
    optimized_params, train_history, test_history, optimizer = train_with_davidon(
        Model_forward, Model_backward, compute_cost,
        x_train_flattened, one_hot_encoded_y_train.T,
        x_test_flattened, one_hot_encoded_y_test.T,
        parameters, units_in_layer,
        epsilon=1e-6, max_iterations=100
    )

    # Print training summary
    print(f"\nTraining Summary:")
    print(f"Total iterations: {optimizer.iteration}")
    print(f"Gradient descent fallbacks: {optimizer.gradient_descent_count}")
    if optimizer.iteration > 0:
        print(
            f"Davidon success rate: {(optimizer.iteration - optimizer.gradient_descent_count) / optimizer.iteration * 100:.1f}%")

    # Evaluate final accuracy
    predictions_train = predict(x_train_flattened, optimized_params)
    predictions_test = predict(x_test_flattened, optimized_params)

    accuracy_train = compute_accuracy(predictions_train, y_train_flattened)
    accuracy_test = compute_accuracy(predictions_test, y_test_flattened)

    print(f"\nFinal Accuracy - Train: {accuracy_train:.4f}, Test: {accuracy_test:.4f}")

    # Save results
    Data_to_save = {
        'parameters': optimized_params,
        'train_costs': train_history,
        'test_costs': test_history,
        'final_train_accuracy': accuracy_train,
        'final_test_accuracy': accuracy_test
    }

    with open(f'davidon_results_{units_in_layer}.pickle', 'wb') as file:
        pickle.dump(Data_to_save, file)