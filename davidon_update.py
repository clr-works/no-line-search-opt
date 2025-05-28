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
    omega: np.ndarray  # Auxiliary vector for Davidon's algorithm
    E0: float  # Previous cost
    E_prime0: float  # Previous directional derivative
    s0: np.ndarray  # Previous search direction
    g0: np.ndarray  # Previous gradient


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
        self.J_initialized = False  # Track if J has been properly scaled

        if self.debug:
            self.debugger = DavidonDebugger()

        # Don't initialize J yet - wait for first gradient
        self.total_params = total_params
        self.state = DavidonState(
            J=None,  # Will initialize based on first gradient
            k0=None,
            omega=None,
            E0=float('inf'),
            E_prime0=0,
            s0=None,
            g0=None
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

        # Initialize J based on first gradient if needed
        if not self.J_initialized:
            grad_norm = np.linalg.norm(grad_vector)
            if grad_norm > 1e-10:
                # Scale J so that ||Jg|| ≈ ||g|| for initial step
                # This gives us a reasonable initial step size
                scale = 1.0 / grad_norm  # This makes ||Jg|| = 1 initially
                self.state.J = np.eye(self.total_params, dtype=np.float64) * scale
                self.J_initialized = True
                if self.debug:
                    print(f"Initialized J with scale={scale:.6f} based on grad_norm={grad_norm:.6f}")
            else:
                # Gradient too small, use default
                self.state.J = np.eye(self.total_params, dtype=np.float64) * 0.1
                self.J_initialized = True

        # DEBUG: Analyze gradient and J
        if self.debug and self.iteration < 5:
            print(f"\n=== Iteration {self.iteration} ===")
            print(f"Initial cost: {cost:.6f}")
            analyze_davidon_behavior(self, grad_vector, self.state.J)

        # First iteration initialization
        if self.state.k0 is None:
            self.state.g0 = grad_vector.copy()
            self.state.k0 = self.state.J @ grad_vector
            self.state.omega = self.state.k0.copy()
            self.state.E0 = cost
            if self.debug:
                print(f"Initial k0 norm: {np.linalg.norm(self.state.k0):.6f}")
                print(f"Initial J scaling: {np.mean(np.abs(np.diag(self.state.J))):.6f}")

        # Compute search direction
        k = self.state.J @ grad_vector

        # Debug: check if k is reasonable
        if self.debug and (np.linalg.norm(k) > 1000 * np.linalg.norm(grad_vector) or
                           np.linalg.norm(k) < 1e-6 * np.linalg.norm(grad_vector)):
            print(
                f"WARNING: k norm={np.linalg.norm(k):.6f} is unreasonable compared to grad norm={np.linalg.norm(grad_vector):.6f}")
            print(f"J matrix may be ill-conditioned")

            # Reset J to scaled identity if it's too bad
            if np.linalg.norm(k) > 10000 * np.linalg.norm(grad_vector):
                print("RESETTING J matrix to scaled identity")
                scale = 0.1 / np.linalg.norm(grad_vector)
                self.state.J = np.eye(len(grad_vector), dtype=np.float64) * scale
                k = self.state.J @ grad_vector

        if self.iteration == 0:
            s = -k  # First iteration: simple gradient descent direction
        else:
            s = -k  # Use transformed gradient as search direction

        if self.debug:
            print(f"Search direction norm: {np.linalg.norm(s):.6f}")

        # Check convergence
        E_prime0 = np.dot(grad_vector, s)
        if self.debug:
            print(f"E_prime0 (directional derivative): {E_prime0:.6f}")

        if abs(E_prime0) < self.epsilon:
            print(f"Converged at iteration {self.iteration}")
            return parameters, cost

        # Adjust step size if needed (Davidon's condition)
        if self.state.E0 < float('inf') and 4 * self.state.E0 < -E_prime0:
            adjustment = -4 * self.state.E0 / E_prime0
            s = s * adjustment
            E_prime0 = np.dot(grad_vector, s)
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
        if new_cost < self.state.E0 and self.iteration > 0 and self.state.g0 is not None:
            self._update_jacobian_davidon(
                s, grad_vector, self.state.g0,
                new_cost, self.state.E0, E_prime0, self.state.E_prime0
            )

        # Update state
        self.state.k0 = k.copy()
        self.state.E0 = new_cost
        self.state.E_prime0 = E_prime0
        self.state.s0 = s.copy()
        self.state.g0 = grad_vector.copy()
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

    def _update_jacobian_davidon(self, s: np.ndarray, g: np.ndarray, g_prev: np.ndarray,
                                 E: float, E_prev: float, E_prime: float, E_prime_prev: float) -> None:
        """
        Update Jacobian using Davidon's formula from the 1975 paper.

        The key insight is that Davidon's method maintains a matrix J such that
        J approximates (inverse Hessian)^{-1}, and updates it to maintain this property.
        """

        # Compute key vectors
        y = g - g_prev  # Change in gradient
        k = self.state.J @ g  # Current transformed gradient
        k_prev = self.state.k0  # Previous transformed gradient

        # Davidon's parameters (from the paper)
        # These ensure the update maintains positive definiteness

        # Step 1: Compute m = s + k_prev - k
        m = s + k_prev - k
        m_norm_sq = np.dot(m, m)

        # Avoid numerical issues
        if m_norm_sq < 1e-16:
            if self.debug:
                print("  Warning: m vector too small, skipping J update")
            return

        # Step 2: Compute v and μ
        v = np.dot(m, s)
        mu = v - m_norm_sq

        # Step 3: Update omega (auxiliary vector)
        if self.state.omega is None:
            self.state.omega = k_prev.copy()

        # Compute u = omega - (m^T omega / m^T m) * m
        m_omega = np.dot(m, self.state.omega)
        u = self.state.omega - (m_omega / m_norm_sq) * m
        u_norm_sq = np.dot(u, u)

        # Step 4: Compute parameters for the update
        n_sq = 0.0
        if u_norm_sq > 1e-16:
            # Check if m and u are sufficiently orthogonal
            m_u = np.dot(m, u)
            if 1e6 * (m_u ** 2) < m_norm_sq * u_norm_sq:
                # Compute n = u - (u^T s / u^T u) * u
                u_s = np.dot(u, s)
                n = s - (u_s / u_norm_sq) * u
                n_sq = np.dot(u, s) ** 2 / u_norm_sq

        # Step 5: Compute b parameter
        b = n_sq - (mu * v) / m_norm_sq

        # Step 6: Determine update parameters α, γ, δ
        if b > self.epsilon:
            # Case 1: b is sufficiently positive
            alpha = v / m_norm_sq
            gamma = 0
            delta = np.sqrt(v / mu) if mu > 0 else 0
        else:
            # Case 2: b is small or negative
            a = b - mu
            c = b + v

            if abs(a) > 1e-16 and mu * v < m_norm_sq * n_sq:
                # Compute gamma to ensure positive definiteness
                discriminant = 1 - (mu * v) / (m_norm_sq * n_sq)
                if discriminant > 0:
                    gamma = np.sqrt(discriminant) / abs(a)
                else:
                    gamma = 0

                if c < a:
                    gamma = -gamma

                delta = 0 if abs(mu) < 1e-16 else np.sqrt(abs(v / mu))
                alpha = (v + mu * delta) / (m_norm_sq * (1 + gamma * n_sq))
            else:
                # Fallback to simple update
                gamma = 0
                delta = 0
                alpha = v / m_norm_sq if m_norm_sq > 1e-16 else 0

        # Step 7: Compute update vectors p and q
        p = alpha * m + gamma * (n if 'n' in locals() else np.zeros_like(m))
        q = m / m_norm_sq if m_norm_sq > 1e-16 else np.zeros_like(m)

        # Step 8: Update J matrix
        # Davidon's formula: J_new = J + p⊗q
        # where ⊗ represents a specific matrix operation

        # The correct Davidon update maintains J as approximation to Hessian
        # Key insight: We need to update J such that J(y) ≈ s

        # First, apply rank-one update
        if abs(np.dot(q, k_prev)) > 1e-10:
            # J = J + (p - Jq) ⊗ q / (q^T k_prev)
            Jq = self.state.J @ q
            update_vec = p - Jq
            denominator = np.dot(q, k_prev)
            self.state.J = self.state.J + np.outer(update_vec, q) / denominator
        else:
            # Simpler update when q^T k_prev is too small
            if m_norm_sq > 1e-10:
                # Use Sherman-Morrison-like update
                self.state.J = self.state.J + alpha * np.outer(m, m) / m_norm_sq

        # Step 9: Update omega for next iteration
        q_k_prev = np.dot(q, k_prev)
        self.state.omega = k_prev + p * q_k_prev

        if self.debug and self.iteration % 10 == 0:
            # Check J properties
            try:
                # Check condition number on small subset
                subset_size = min(50, self.state.J.shape[0])
                J_subset = self.state.J[:subset_size, :subset_size]
                eigvals = np.linalg.eigvalsh(J_subset)
                cond_number = eigvals.max() / eigvals.min() if eigvals.min() > 0 else np.inf
                print(f"  J eigenvalue range: [{eigvals.min():.2e}, {eigvals.max():.2e}]")
                print(f"  J condition number: {cond_number:.2e}")

                # Reset J if it becomes too ill-conditioned
                if cond_number > 1e10:
                    print("  WARNING: J is ill-conditioned, resetting to scaled identity")
                    grad_norm = np.linalg.norm(g)
                    scale = 1.0 / (grad_norm + 1e-10)
                    self.state.J = np.eye(self.total_params, dtype=np.float64) * scale
                    self.state.omega = None  # Reset omega too
            except:
                pass


def train_with_davidon(model_forward, model_backward, compute_cost,
                       x_train, y_train, x_test, y_test,
                       parameters, units_in_layer,
                       epsilon=1e-6, max_iterations=100,
                       optimizer_mode="davidon_with_fallback",
                       learning_rate=0.01):
    """
    Train neural network using Davidon's method or pure gradient descent

    Args:
        optimizer_mode: "davidon_with_fallback" or "gradient_descent"
        learning_rate: Learning rate for gradient descent mode

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

    # Training history
    train_costs = [cost]
    test_costs = []

    # Evaluate initial test cost
    AL_test, _ = model_forward(x_test, parameters)
    test_cost = compute_cost(AL_test, y_test)
    test_costs.append(test_cost)

    print(f"Initial - Train Cost: {cost:.4f}, Test Cost: {test_cost:.4f}")
    print(f"Optimizer Mode: {optimizer_mode}")
    if optimizer_mode == "gradient_descent":
        print(f"Learning Rate: {learning_rate}")

    # Choose optimizer based on mode
    if optimizer_mode == "gradient_descent":
        # Pure gradient descent mode
        print("\nUsing Pure Gradient Descent")

        for iteration in range(max_iterations):
            # Forward pass
            AL, caches = model_forward(x_train, parameters)
            cost = compute_cost(AL, y_train)

            # Backward pass
            grads = model_backward(AL, y_train, caches)

            # Flatten gradients
            grad_list = []
            for grad_key, shape in structure_cache:
                if grad_key in grads:
                    grad_list.append(grads[grad_key].flatten())
            grad_vector = np.concatenate(grad_list)

            # Gradient descent step
            direction = -grad_vector * learning_rate

            # Update parameters
            import copy
            new_params = copy.deepcopy(parameters)
            if 'J' in new_params:
                del new_params['J']

            start = 0
            for grad_key, shape in structure_cache:
                param_key = grad_key[1:]
                size = np.prod(shape)
                segment = direction[start:start + size].reshape(shape)
                new_params[param_key] = parameters[param_key] + segment
                start += size

            parameters = new_params

            # Record costs
            train_costs.append(cost)

            # Evaluate on test set
            AL_test, _ = model_forward(x_test, parameters)
            test_cost = compute_cost(AL_test, y_test)
            test_costs.append(test_cost)

            # Print progress
            if iteration % 10 == 0:
                grad_norm = np.linalg.norm(grad_vector)
                print(
                    f"Iteration {iteration} - Train Cost: {cost:.4f}, Test Cost: {test_cost:.4f}, Grad Norm: {grad_norm:.6f}")

            # Check convergence
            if len(train_costs) > 10:
                recent_costs = train_costs[-10:]
                cost_variance = np.var(recent_costs)
                if cost_variance < epsilon ** 2:
                    print(f"Converged at iteration {iteration} - cost variance: {cost_variance:.2e}")
                    break

        # Create dummy optimizer for compatibility
        class DummyOptimizer:
            def __init__(self):
                self.iteration = iteration
                self.gradient_descent_count = iteration  # All iterations were GD

        optimizer = DummyOptimizer()

    else:  # davidon_with_fallback mode
        # Initialize optimizer with adjusted epsilon for convergence
        optimizer = DavidonOptimizer(total_params, epsilon=1e-4, max_iterations=max_iterations, debug=True)

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
    # Choose optimizer mode: "davidon_with_fallback" or "gradient_descent"
    OPTIMIZER_MODE = "davidon_with_fallback"  # Change this to "gradient_descent" for pure GD
    LEARNING_RATE = 0.1  # Learning rate for gradient descent mode

    optimized_params, train_history, test_history, optimizer = train_with_davidon(
        Model_forward, Model_backward, compute_cost,
        x_train_flattened, one_hot_encoded_y_train.T,
        x_test_flattened, one_hot_encoded_y_test.T,
        parameters, units_in_layer,
        epsilon=1e-6,
        max_iterations=1000,
        optimizer_mode=OPTIMIZER_MODE,
        learning_rate=LEARNING_RATE
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