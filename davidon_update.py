import numpy as np
from os.path import join
import pickle
from loader import MnistDataloader
from preprocessing import prepare_data, one_hot_encode
from helper_functions import *
from memory_profiler import profile
import time

@profile
def davidon_quasi_newton_update(x_train_flattened, x_test_flattened, parameters, E, cost_list, cost_list_test, k, units_in_layer, epsilon=0.01, max_iterations=10):
    """ Performs davidson quasi newton update

    Args: Training and testing sets,
          initial parameters,
          initial cost,
          initial cost list,
          initial cost list test,
          initial gradient vector,
          units in layer,
          epsilon,
          max iterations

    Returns: parameters, cost_list, cost_list_test"""

    # Initialize variables only once
    parameters0 = parameters
    J = parameters0['J'] # initial parameters
    E0 = cost # initial cost before the first forward pass, just the cost related to the initial parameters
    grad_vector, structure_cache, size = flatten_gradients_for_jacobian(grads, units_in_layer) # grads after first backward pass
    # Check if sizes are compatible for matrix multiplication
    if grad_vector.shape[0] != size:
        raise ValueError(f"The gradient vector is expected to be of size 4015, but got size {grad_vector.shape[0]}")
    start = time.time()
    k0 = np.dot(J,grad_vector) # perform matrix multiplication to get k0, grad_vector after first backward pass
    end = time.time()
    elapsed = end - start
    print("elapsed time for matrix multiplication: ", elapsed)
    k0 = k0.reshape(grad_vector.shape[0],1)
    omega = k0

    iteration_counter = 0

    while True:

        iteration_counter += 1

        if iteration_counter >= max_iterations:
            print("Maximum number of iterations reached, exiting the algorithm")
            return parameters, cost_list, cost_list_test
        if iteration_counter == 1:
            s = -k0
        else:
            s = k0

        E_prime0 = np.dot(k0.T,s)
        lambda_factor = 2


        # Iterative update of s until the condition 4 * E0 > - E_prime0 is true
        if 4 * E0 < - E_prime0:
            s = -4 * s * (E0 / E_prime0)

        # Update parameters using the current Jacobian and search direction
        parameters = update_parameters_with_jacobian(parameters0,structure_cache, s)

        # Break the loop if the update is smaller than epsilon to prevent infinite loops
        if iteration_counter == 1:
            if -E_prime0 < epsilon:
                print("Stopping criteria met, exiting the algorithm")
                return parameters, cost_list, cost_list_test
        else:
            if E_prime0 < epsilon:
                print("Stopping criteria met, exiting the algorithm")
                return parameters, cost_list, cost_list_test
        # Calculate a new forward pass and then the cost
        AL, caches = Model_forward(x_train_flattened, parameters)
        AL_test, caches_test = Model_forward(x_test_flattened, parameters)
        # Compute the cost
        E = compute_cost(AL, one_hot_encoded_y_train.T) # new cost with new parameters after first update
        E_test = compute_cost(AL_test, one_hot_encoded_y_test.T)
        cost_list.append(E)
        cost_list_test.append(E_test)
        # Check if the update is sufficient, otherwise adjust
        while E > E0:
            s = s/2
            E_prime0 = E_prime0/2
            lambda_factor = 1/2
            parameters = update_parameters_with_jacobian(parameters, structure_cache,s)
            if -E_prime0 < epsilon:
                print("Stopping criteria met, exiting the algorithm")
                return parameters, cost_list, cost_list_test
            AL, caches = Model_forward(x_train_flattened, parameters)
            AL_test, caches_test = Model_forward(x_test_flattened, parameters)
            # Compute cost
            E = compute_cost(AL, one_hot_encoded_y_train.T)
            E_test = compute_cost(AL_test, one_hot_encoded_y_test.T)
            cost_list.append(E)
            cost_list.append(E_test)

        while True:
            # Update parameters using the current Jacobian and search direction
            parameters = update_parameters_with_jacobian(parameters, structure_cache, s)
            AL, caches = Model_forward(x_train_flattened, parameters)
            AL_test, caches_test = Model_forward(x_test_flattened, parameters)
            # Compute cost
            E = compute_cost(AL, one_hot_encoded_y_train.T)
            E_test = compute_cost(AL_test, one_hot_encoded_y_test.T)
            cost_list.append(E)
            k = np.dot(J.T, s)
            E_prime = np.dot(k.T, s)
            b0 = E_prime - E_prime0
            m = s + k0 - k
            parameters0 = parameters
            E0 = E
            k0 = k
            E_prime0 = E_prime

            if b0 >= epsilon:

                m_square = np.dot(m.T, m)
                # Check if the norm of m is sufficiently small
                if (np.linalg.norm(m) ** 2 < epsilon):
                    print("Convergence criteria met, exit")
                    return parameters, cost_list, cost_list_test
                else:
                    v = np.dot(m.T, s)
                    mu = v - m_square

                    # compute u
                    u = omega - compute(m,omega)

                    # Check if m'u is sufficiently small compared to u'u
                    if 1e6 * (np.dot(m.T, u) ** 2 >= m_square * np.dot(u.T, u)):
                        n = np.zeros_like(s) # assuming size of s
                        n_squared = 0
                    else:
                        n = compute(u,s)
                        n_squared = (np.dot(u.T, s) ** 2) / np.dot(u.T, u)

                    b = n_squared - ((mu*v)/m_square)

                    # Check if b is sufficiently large
                    alpha = p = omega = delta = None
                    if b <= epsilon:
                        if mu*v < m_square * n_squared:
                            a = b - mu
                            c = b + v
                            gamma = calculate_gamma(mu, v, m_square, n_squared, a, b)
                            d = c/a if a!=0 else None

                            if c < a:
                                delta = np.sqrt(v / mu)
                                gamma = -gamma
                                alpha, p, q, omega = calculate_alpha_p_q_omega(v, mu, m, n, gamma, delta)
                            else:
                                alpha, p, q, omega = calculate_alpha_p_q_omega(v, mu, m, n, gamma, delta)
                        else:
                            gamma = 0
                            delta = 0
                            alpha, p, q, omega = calculate_alpha_p_q_omega(v, mu, m, n, gamma, delta)
                    else:
                        n = s - (v * m)/m_square
                        n_squared = b0 - (mu*v)/m_square
                        b = b0
                        if mu*v < m_square * n_squared:
                            a = b - mu
                            c = b + v
                            gamma = calculate_gamma(mu, v, m_square, n_squared, a, b)
                            d = c/a if a!=0 else None

                            if c < a:
                                gamma = -gamma
                                alpha, p, q, omega = calculate_alpha_p_q_omega(v, mu, m, n, gamma, delta)
                            else:
                                alpha, p, q, omega = calculate_alpha_p_q_omega(v, mu, m, n, gamma, delta)
                        else:
                            gamma = 0
                            delta = np.sqrt(v/mu)
                            alpha, p, q, omega = calculate_alpha_p_q_omega(v, mu, m, n, gamma, delta)

                    # Update parameters
                    qTk0 = np.dot(q.T, k0)
                    k0 = k0 + p * qTk0
                    qpT = np.dot(q, p.T)
                    J = J +  J * qpT
                    # save cost and parameters for next iteration
                    J = parameters['J']

                    if n_squared > 0:
                        break
                    else:
                        omega = k0
                        break
            else:

                s = lambda_factor * s
                E_prime0 = lambda_factor* E_prime0
                continue # go back to the beginning of while loop 2


    return parameters, cost_list, cost_list_test

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

    # take a sample:
    """x_train_flattened = x_train_flattened[:, :64]
    y_train_flattened = y_train_flattened[:, :64]
    x_test_flattened = x_test_flattened[:, :64]
    y_test_flattened = y_test_flattened[:, :64]"""

    # One hot encode Y ground true values
    one_hot_encoded_y_train = one_hot_encode(y_train_flattened)
    one_hot_encoded_y_test = one_hot_encode(y_test_flattened)

    ############ Define the number of units in each layer of the network ###############
    units_in_layer = [784, 5, 5, 10]

    # Initialize the parameters
    parameters = initialize_parameters_davidon(units_in_layer)

    # First forward pass of the neural network
    AL, caches = Model_forward(x_train_flattened, parameters)

    cost_list_test = []
    AL_test, caches_test = Model_forward(x_test_flattened, parameters)
    cost = compute_cost(AL_test, one_hot_encoded_y_test.T)
    cost_list_test.append(cost)

    # Compute the cost
    print("this is the initial cost computed")
    cost_list = []
    cost = compute_cost(AL, one_hot_encoded_y_train.T)
    cost_list.append(cost)
    # Compute the gradients
    grads = Model_backward(AL, one_hot_encoded_y_train.T, caches)

    parameters, cost_list, cost_list_test = davidon_quasi_newton_update(x_train_flattened, x_test_flattened, parameters, cost, cost_list, cost_list_test, grads, units_in_layer, epsilon=0.01, max_iterations=100)

    # Structure of the data to save: (parameters, cost_list, cost_list_test)
    Data_to_save ={
        'cost_list': cost_list,
        'cost_list_test': cost_list_test
    }

    # Save the parameters in a pickle file
    with open('cost_list_' + str(units_in_layer) + '.pickle', 'wb') as file:
        pickle.dump(Data_to_save, file)

    # Get predictions for the training and test sets
    predictions_train = predict(x_train_flattened, parameters)
    predictions_test = predict(x_test_flattened, parameters)

    # Compute the accuracy of the predictions
    accuracy_train = compute_accuracy(predictions_train, y_train_flattened)
    accuracy_test = compute_accuracy(predictions_test, y_test_flattened)

    print(f"Accuracy on the training set: {accuracy_train}")
    print(f"Accuracy on the test set: {accuracy_test}")