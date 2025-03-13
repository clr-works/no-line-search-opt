import numpy as np
import copy
from memory_profiler import profile
#from scipy.sparse import identity
import time
import pickle

@profile
def initialize_parameters_davidon(units_in_layer, dtype=np.float16):
    """
    Initializes network parameters using He initialization, and prepares for Davidson's algorithm

    Args:
    units_in_layer -- python list containing the dimensions of each layer in the network

    Returns:
    parameters -- dictionary containing:
                  Wl, bl -- weight matrix and bias vector for layer l
                  J -- Initial Jacobian or Hessian approximation
    """

    np.random.seed(45)
    parameters = {}
    L = len(units_in_layer)  # number of layers in the network

    total_params = 0  # To calculate the total number of parameters
    start = time.time()
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(units_in_layer[l], units_in_layer[l - 1]) * np.sqrt(2. / units_in_layer[l - 1]).astype(dtype)
        parameters['b' + str(l)] = np.zeros((units_in_layer[l], 1), dtype=dtype)

        print("W" + str(l) + " shape: " + str(parameters['W' + str(l)].shape))
        print("b" + str(l) + " shape: " + str(parameters['b' + str(l)].shape))
        total_params += units_in_layer[l] * units_in_layer[l - 1] + units_in_layer[l]
    end = time.time()
    elapsed = end - start
    print("Elapsed time for initialization of w and b: ", elapsed)

    start = time.time()
    # Initialize the Jacobian/Hessian approximation
    # For simplicity, starting with an identity matrix
    parameters['J'] = np.identity(total_params, dtype=dtype)
    end = time.time()
    elapsed = end - start
    print("Elapsed time for initialization of J: ", elapsed)
    print("J shape: " + str(parameters['J'].shape))

    return parameters

def initialize_parameters(units_in_layer, dtype=np.float16):
    """
    Initializes network parameters using He initialization

    Args:
    units_in_layer -- python list containing the dimensions of each layer in the network

    Returns:
    parameters -- dictionary containing:
                  Wl, bl -- weight matrix and bias vector for layer l
    """

    np.random.seed(45)
    parameters = {}
    L = len(units_in_layer)  # number of layers in the network

    total_params = 0  # To calculate the total number of parameters
    start = time.time()
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(units_in_layer[l], units_in_layer[l - 1]) * np.sqrt(2. / units_in_layer[l - 1]).astype(dtype)
        parameters['b' + str(l)] = np.zeros((units_in_layer[l], 1), dtype=dtype)

        print("W" + str(l) + " shape: " + str(parameters['W' + str(l)].shape))
        print("b" + str(l) + " shape: " + str(parameters['b' + str(l)].shape))

    end = time.time()
    elapsed = end - start
    print("Elapsed time for initialization of w and b: ", elapsed)
    return parameters


def relu(Z):
    """
    Compute the ReLU activation for an input array Z.

    Arguments:
    Z -- Input array, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    """
    A = np.maximum(0, Z)
    return A


def softmax(Z):
    """
    Compute the softmax activation for an input array Z.

    Arguments:
    Z -- Input array, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    """
    # Ensure numerical stability by subtracting the maximum value in Z from each entry in Z
    shiftZ = Z - np.max(Z, axis=0, keepdims=True)
    exps = np.exp(shiftZ)
    A = exps / np.sum(exps, axis=0, keepdims=True)

    return A


def linear_forward(A, W, b):
    """
    Implement pre activations of forward part.

    Args:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- pre-activation parameter ready to be processed by the selected activation.
    a cached tuple (A, W, b) ready to compute backward propagation.
    """
    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer
    W -- weights matrix
    b -- bias vector
    activation --  whether it will be  softmax (last layer) or relu (hidden layers)

    Returns:
    A -- the output of the activation function
    a cached tuple that contains "linear_cache" and "activation_cache"
    """

    if activation == "softmax":

        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z), Z


    elif activation == "relu":

        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = np.maximum(0, Z), Z

    cache = (linear_cache, activation_cache)

    return A, cache


def Model_forward(X, parameters):
    """
    Implement forward propagation for the all layers

    Arguments:
    X -- data, numpy array of shape (num_features, num_examples)
    parameters -- output of initialize_parameters()

    Returns:
    AL -- activation value from the output (last) layer
    caches -- list of caches indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A

        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters['b' + str(l)],activation="relu")
        caches.append(cache)


    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation="softmax")
    caches.append(cache)


    return AL, caches


def compute_cost(AL, Y):
    """
    Implement the cross entropy cost function for classification.

    Arguments:
    AL -- probability vector, shape (1, number of examples)
    Y -- true "label" vector shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]

    epsilon = 1e-15
    cost = -1 / m * np.sum(Y * np.log(AL + epsilon))

    cost = np.squeeze(cost)
    print("cost at current iteration is:", cost)

    return cost


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer l

    Parameters:
    dZ -- Derivative of the cost in relation to the linear outcome of the present layer l.
    cache -- tuple holding values (A_prev, W, b) derived from the forward propagation of the current layer.

    Outputs:
    dA_prev -- Derivative of the cost with respect to the activation (from the preceding layer l-1).
    dW -- Derivative of the cost concerning W (of the current layer l).
    db -- Derivative of the cost concerning b (of the current layer l).
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation, Y=None):
   """Parameters:

    dA -- post-activation gradient for the existing layer l.
    cache -- tuple containing values (linear_cache, activation_cache).
    activation -- the type of activation utilized in this layer.

    Outputs:
    dA_prev -- Derivative of the cost related to the activation (from the prior layer l-1), has the same shape as A_prev.
    dW -- Derivative of the cost concerning W (of the current layer l), has the same shape as W.
    db -- Derivative of the cost concerning b (of the current layer l), has the same shape as b.
   """
   linear_cache, activation_cache = cache
   Z = activation_cache

   if activation == "relu":

        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

   elif activation == "softmax":
        if np.isnan(Z).any():
            raise ValueError("NaN detected in Z")

        Z_max = np.max(Z, axis=0, keepdims=True)
        if np.isnan(Z_max).any():
            raise ValueError("NaN detected in Z_max")

        shifted_Z = Z - Z_max
        if np.isnan(shifted_Z).any():
            raise ValueError("NaN introduced after subtraction")

        s = np.exp(shifted_Z) / np.sum(np.exp(shifted_Z), axis=0)
        if np.isnan(s).any():
            raise ValueError("NaN introduced in softmax")

        dZ = s - Y
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

   return dA_prev, dW, db

def Model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SOFTMAX group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector
    caches -- list of caches containing linear activations forward.

    Returns:
    grads -- A dictionary with the gradients
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    # Setting up an epsilon value for numerical stability
    epsilon = 1e-15
    dAL = -(np.divide(Y, AL + epsilon))

    # Lth layer (SOFTMAX -> LINEAR) gradients.

    current_cache = caches[L - 1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, activation="softmax", Y=Y)
    grads["dA" + str(L - 1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    print("dA" + str(L - 1) + " shape: " + str(grads["dA" + str(L - 1)].shape))
    print("dW" + str(L) + " shape: " + str(grads["dW" + str(L)].shape))
    print("db" + str(L) + " shape: " + str(grads["db" + str(L)].shape))


    # Loop from l=L-2 to l=0
    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.

        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache,
                                                                    activation="relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

        print("dA" + str(l) + " shape: " + str(grads["dA" + str(l)].shape))
        print("dW" + str(l + 1) + " shape: " + str(grads["dW" + str(l + 1)].shape))
        print("db" + str(l + 1) + " shape: " + str(grads["db" + str(l + 1)].shape))

    return grads


def update_parameters(params, grads, learning_rate):
    """
    Update parameters using gradient descent.

    Arguments:
    params -- python dictionary containing learnt parameters
    grads -- python dictionary containing gradients

    Returns:
    parameters -- python dictionary containing updated parameters
    """
    parameters = copy.deepcopy(params)
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters

def flatten_gradients_for_jacobian(grads, units_in_layer):
    """
    Flatten and concatenate gradients in the specified order for Jacobian multiplication,
    and create a cache for the structure of gradients.

    Args:
    grads -- Dictionary containing gradient arrays for 'dW' and 'db' for each layer
    units_in_layer -- Number of units in each layer (including input layer)

    Returns:
    flattened_gradient -- Flattened and concatenated gradient vector
    structure_cache -- Cache containing the shapes of each gradient array
    """
    L = len(units_in_layer)  # number of layers in the network
    flattened_gradient = []
    structure_cache = []

    # Iterate over layers to concatenate gradients in the specified order
    for l in range(1, L):
        dw_key = 'dW' + str(l)
        db_key = 'db' + str(l)

        # Check if the keys exist in the dictionary and append the gradients
        if dw_key in grads and db_key in grads:
            # Append shape information to the cache
            structure_cache.append((dw_key, grads[dw_key].shape))
            structure_cache.append((db_key, grads[db_key].shape))

            # Flatten and append the gradients
            flattened_gradient.append(grads[dw_key].flatten())
            flattened_gradient.append(grads[db_key].flatten())
        else:
            raise ValueError(f"Gradient for layer {l} not found in the dictionary")

    size = np.concatenate(flattened_gradient).shape[0]

    # Concatenate all flattened gradients into a single vector
    return np.concatenate(flattened_gradient), structure_cache, size

import numpy as np

def update_parameters_with_jacobian(params, structure_cache, s):
    """
    Update parameters using a modification of gradient descent that incorporates
    a Jacobian matrix.

    Arguments:
    params -- python dictionary containing learned parameters
    structure_cache -- list containing the structure (shape and key) of each gradient
    s -- search direction (flattened vector)
    J -- numpy array representing the Jacobian matrix, it is within the parameters dictionary

    Returns:
    params -- python dictionary containing updated parameters
    """

    # Calculate the update direction using the Jacobian
    J = params["J"]
    start_time = time.time()
    update_direction = np.dot(J, s)
    end = time.time()
    elapsed= end - start_time
    print("elapsed time for update direction in update parameters: ", elapsed)
    # Initialize the starting index for slicing update_direction
    start = 0

    # Iterate over the structure_cache to update each parameter
    start_time = time.time()
    for grad_key, shape in structure_cache:
        # Convert gradient key to parameter key (e.g., 'dW1' to 'W1')
        param_key = grad_key[1:]  # Remove the 'd' from the gradient key

        # Ensure the key exists in the params dictionary
        if param_key not in params:
            raise KeyError(f"Parameter {param_key} not found in params dictionary.")

        # Calculate the size of the current parameter
        size = np.prod(shape)

        # Slice the corresponding segment from update_direction
        segment = update_direction[start:start + size]

        segment_reshape = segment.reshape(shape)

        # Reshape and update the parameter in the dictionary
        params[param_key] += 0.01*segment_reshape

        # Update the start index for the next parameter
        start += size
    end = time.time()
    elapsed = end - start_time
    print("elapsed time for updating parameters: ", elapsed)
    return params

def calculate_alpha_p_q_omega(v, mu, m, n, gamma, delta):

    """ Calculates intermediate values as part of Davidon's update rule"""

    alpha = v + mu * delta + m**2 * n**2 * gamma
    p = (delta - n**2 * gamma) * m + gamma * v * n
    q = m / (1 + n**2 * gamma) - (mu * v * n) / alpha
    omega = n**2 * (1 + gamma) * m - (1 + delta) * (mu * v * n) / alpha
    return alpha, p, q, omega

def compute(m,omega):# Calculate the scalar m^T * m

    # compute m_square
    mT_m = np.dot(m.T, m)

    # Calculate the matrix m * m^T
    m_mT = np.outer(m, m)

    # Perform the matrix-vector multiplication (m * m^T) * omega
    m_mT_omega = np.dot(m_mT, omega)

    # Calculate u by subtracting the scaled vector from omega
    u = omega - (m_mT_omega / mT_m)
    return u

def calculate_gamma(mu, v, m_square, n_squared, a, b):

    """" Calculates the scalar gamma as part of Davidon's update rule"""

    if mu * v < m_square * n_squared and a != 0 and b != 0:
        gamma = np.sqrt(1 - (mu * v) / (m_square * n_squared/ (a*b)))
    return gamma

def predict(X, parameters):
    """
    Given input features and parameters, it predicts the class labels

    Arguments:
    X -- input features, numpy array of shape (number of features, number of examples)
    parameters -- python dictionary containing the updated parameters of the model

    Returns:
    predictions -- vector of predicted labels for the examples in X
    """

    # Forward propagation
    AL, caches = Model_forward(X, parameters)

    # Convert probabilities AL into a prediction by taking the class with the highest probability
    predictions = np.argmax(AL, axis=0)

    return predictions


def compute_accuracy(predictions, Y):
    """
    Computes the accuracy of the predictions against the true labels

    Arguments:
    predictions -- predicted labels, a numpy array of shape (1, number of examples)
    Y -- true "label" vector (containing labels from 0 to 9), of shape (1, number of examples)

    Returns:
    accuracy -- accuracy of the predictions
    """
    accuracy = np.mean(predictions == Y)
    return accuracy

def load_pickle(object):
    with open(object, 'rb') as handle:
        return pickle.load(handle)

