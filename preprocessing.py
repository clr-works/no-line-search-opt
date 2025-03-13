from loader import MnistDataloader
import numpy as np

def prepare_data(x_train, y_train, x_test, y_test):
    """
    Reads the images and labels from the given file paths.

    Args:
    - images (list): List of images, each represented as a 2D list.
    - labels (list): List of integer labels
    Returns:
    - four numpy arrays flattened as follows:
    x_train_flattened with dimension (num_features, num_examples_in_train)
    y_train_flattened with dimension (1, num_examples_in_train)
    x_test_flattened with dimension (num_features, num_examples_in_test)
    y_test_flattened with dimension (1, num_examples_in_test)
    """

    x_train_np = np.array([np.vstack(img) for img in x_train])
    x_test_np = np.array([np.vstack(img) for img in x_test])
    y_train_np = np.array(y_train)
    y_test_np = np.array(y_test)

    print("Shapes BEFORE using prepare_data:")
    print(f"x_train_np shape: {x_train_np.shape}")
    print(f"y_train_np shape: {y_train_np.shape}")
    print(f"x_test_np shape: {x_test_np.shape}")
    print(f"y_test_np shape: {y_test_np.shape}")

    x_train_flattened = x_train_np.reshape(x_train_np.shape[0], -1).T / 255.
    x_test_flattened = x_test_np.reshape(x_test_np.shape[0], -1).T / 255.
    y_train_flattened = y_train_np.reshape(1, y_train_np.shape[0])
    y_test_flattened = y_test_np.reshape(1, y_test_np.shape[0])

    print("\nShapes AFTER using prepare_data:")
    print(f"x_train_flattened shape: {x_train_flattened.shape}")
    print(f"y_train_flattened shape: {y_train_flattened.shape}")
    print(f"x_test_flattened shape: {x_test_flattened.shape}")
    print(f"y_test_flattened shape: {y_test_flattened.shape}")

    return x_train_flattened, x_test_flattened, y_train_flattened, y_test_flattened


def one_hot_encode(Y, num_classes=10):
    """
    One-hot encode a numpy array with the given number of classes.

    Args:
    - Y: numpy array to be one-hot encoded
    - num_classes: number of possible categories

    Returns:
    - one-hot encoded array
    """
    one_hot = np.zeros((Y.shape[1], num_classes))
    for i, y in enumerate(Y[0]):
        one_hot[i, y] = 1

    return one_hot
