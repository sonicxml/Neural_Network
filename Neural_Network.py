__author__ = 'Trevin Gandhi'

import random
import numpy as np


class NeuralNetwork(object):
    def __init__(self, sizes):
        """Initialize Hyperparameters"""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, x):
        """ Run the input through the network with the current settings of
        weights and biases according to the following equation:
        a' = sigmoid(wa + b)"""
        activations = [x]
        for b, w in zip(self.biases, self.weights):
            activations.append(sigmoid(w.dot(activations[-1]) + b))
        return activations

    def backpropagate(self, activations, y):
        """Given the activations matrix, find the error associated with
        the predicted output for each level of the network"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        error = activations[-1] - y
        delta = error * sigmoid_deriv(activations[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        for i in xrange(2, self.num_layers):
            error = np.dot(self.weights[-i + 1].T, delta)
            delta = error * sigmoid_deriv(activations[-i])
            nabla_b[-i] = delta
            nabla_w[-i] = np.dot(delta, activations[-i - 1].T)
        return nabla_b, nabla_w

    def grad_descent(self, training_data, epochs,
                     mini_batch_size, learning_rate, test_data=None):
        """Mini-batch gradient descent algorithm"""
        num_training = len(training_data)
        num_tests = len(test_data) if test_data else 0
        for i in xrange(epochs):
            # Get mini-batches
            random.shuffle(training_data)
            mini_batches = [training_data[j:j + mini_batch_size]
                            for j in xrange(0, num_training, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_batch(mini_batch, learning_rate)
            if test_data:
                num_correct = self.evaluate(test_data)
                print "Epoch {0}: {1} / {2} = {3} %".format(
                    i, num_correct, num_tests, num_correct / num_tests)
            else:
                print "Epoch {0} complete".format(i)

    def update_batch(self, batch, learning_rate):
        """Use a given batch (or mini-batch) to update the 
        networks weights and biases
        The batch is a list of tuples (x, y)"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in batch:
            nb, nw = self.backpropagate(self.feedforward(x), y)
            nabla_b = sum_lists(nabla_b, nb)
            nabla_w = sum_lists(nabla_w, nw)

        sz = np.size(batch)
        multiplier = -1 * (learning_rate / sz)
        self.biases = sum_lists(self.biases,
                                [multiplier * nb for nb in nabla_b])
        self.weights = sum_lists(self.weights,
                                 [multiplier * nw for nw in nabla_w])

    def evaluate(self, test_data):
        """Given test data, returns how many of the data points the network
        predicts correctly"""
        test_results = [(np.argmax(self.feedforward(x)[-1]), y) for x, y in
                        test_data]
        return sum(int(x == y) for (x, y) in test_results)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_deriv(x):
    return x * (1 - x)


def sum_lists(l1, l2):
    return [x1 + x2 for x1, x2 in zip(l1, l2)]