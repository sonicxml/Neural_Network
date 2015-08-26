__author__ = 'Trevin Gandhi'

import numpy as np


class Neural_Network(object):
    def __init__(self, sizes):
        # Initialize Hyperparameters
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:file - 1], sizes[1:])]

    def feedforward(self, x):
        """ Run the input through the network with the current settings of
        weights and biases according to the following equation:
        a' = sigmoid(wa + b)"""
        activations = []
        activations[1] = x
        for w, b in zip(self.weights, self.biases):
            activations.append(sigmoid(np.dot(w, activations[-1]) + b))
        return activations

    def backpropagate(self, activations, y):
        error = y - activations[-1]
        delta = error * sigmoid_deriv(activations[-1])
        self.biases[-1] += delta
        self.weights[-1] += activations[-1].T.dot(delta)
        for i in xrange(2, len(activations)):
            error = self.weights[-i + 1].T.dot(delta)
            delta = error * sigmoid_deriv(activations[-i])
            self.biases[-i] += delta
            self.weights[-i] += activations[-i].T.dot(delta)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return x * (1 - x)