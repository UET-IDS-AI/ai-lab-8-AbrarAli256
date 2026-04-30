"""
AI_stats_lab.py

Neural Networks Lab: 3-Layer Forward Pass and Backpropagation

Implement all functions.
Do NOT change function names.
Do NOT print inside functions.
"""

import numpy as np


def sigmoid(z):
    """
    sigmoid(z) = 1 / (1 + exp(-z))
    """
    return 1 / (1 + np.exp(-z))


def forward_pass(X, W1, W2, W3):
    """
    3-layer neural network forward pass.

    Layer 1:
        h1 = sigmoid(XW1)

    Layer 2:
        h2 = sigmoid(h1W2)

    Output layer:
        y = sigmoid(h2W3)

    Returns:
        h1, h2, y
    """
    h1 = sigmoid(np.dot(X,  W1))
    h2 = sigmoid(np.dot(h1, W2))
    y  = sigmoid(np.dot(h2, W3))
    return h1, h2, y


def backward_pass(X, h1, h2, y, label, W1, W2, W3):
    """
    Backpropagation for a 3-layer sigmoid neural network.

    Returns:
        dW1, dW2, dW3, loss
    """
    if label == 1:
        loss   = -np.log(y)
        dJ_dy  = -1.0 / y
    else:
        loss   = -np.log(1 - y)
        dJ_dy  = 1.0 / (1 - y)
 
    # Output layer gradient
    dy_dz3 = y * (1 - y)
    delta3  = dJ_dy * dy_dz3
 
    # Hidden layer 2 gradient
    delta2 = np.dot(delta3, W3.T) * h2 * (1 - h2)
 
    # Hidden layer 1 gradient
    delta1 = np.dot(delta2, W2.T) * h1 * (1 - h1)
 
    # Weight gradients
    dW3 = np.dot(h2.T, delta3)
    dW2 = np.dot(h1.T, delta2)
    dW1 = np.dot(X.T,  delta1)
 
    return dW1, dW2, dW3, loss
