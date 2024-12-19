"""
Utility functions for data preparation and processing
"""
import numpy as np
import tensorflow as tf

def create_training_data(equation_type='ODE', domain=None, num_points=100, noise_level=0.0, ground_truth_fn=None):
    """
    Create training data for Physics-Informed Neural Networks.

    Args:
        equation_type (str): Type of equation ('ODE', 'PDE', etc.).
        domain (list or tuple): Domain boundaries [(x_start, x_end), (y_start, y_end)] for 1D or 2D problems.
        num_points (int): Number of collocation points.
        noise_level (float): Percentage of noise to add to data.
        ground_truth_fn (callable, optional): Custom ground truth function for generating target values.

    Returns:
        tuple: (X_train, y_train) data for PINN training.
    """
    if domain is None:
        domain = [0, 1] if equation_type == 'ODE' else [(0, 1), (0, 1)]

    if equation_type == 'ODE':
        # Generate 1D collocation points
        X_train = np.linspace(domain[0], domain[1], num_points).reshape(-1, 1)

        # Default ground truth: sine function
        if ground_truth_fn is None:
            ground_truth_fn = lambda x: np.sin(x)
        y_train = ground_truth_fn(X_train)

        # Add noise if specified
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, y_train.shape)
            y_train += noise

    elif equation_type == 'PDE':
        # Generate 2D collocation grid
        x_start, x_end = domain[0]
        y_start, y_end = domain[1]
        x = np.linspace(x_start, x_end, int(np.sqrt(num_points)))
        y = np.linspace(y_start, y_end, int(np.sqrt(num_points)))
        X, Y = np.meshgrid(x, y)

        # Reshape into collocation points
        X_train = np.column_stack([X.ravel(), Y.ravel()])

        # Default ground truth: example Laplace solution
        if ground_truth_fn is None:
            ground_truth_fn = lambda x, y: np.sin(np.pi * x) * np.cos(np.pi * y)
        y_train = ground_truth_fn(X_train[:, 0], X_train[:, 1])

        # Add noise if specified
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, y_train.shape)
            y_train += noise

    else:
        raise ValueError(f"Unsupported equation type: {equation_type}")

    # Convert to tensorflow tensors
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

    return X_train, y_train

def custom_ground_truth_example(x):
    """
    Example custom ground truth function for ODE problems.

    Args:
        x (np.ndarray): Input values.

    Returns:
        np.ndarray: Ground truth values.
    """
    return np.exp(-x) * np.sin(2 * np.pi * x)
