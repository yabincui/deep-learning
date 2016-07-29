#!/usr/bin/bash

"""
A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network. Gradients are calculated
using backpropagation.
"""

import numpy as np
import random

class Network(object):
    def __init__(self, sizes):
        # sizes is the size of different layers of the neural network.
        # sizes[0] is the input size, sizes[-1] is the output size.
        self.num_layers = len(sizes)
        self.sizes = sizes
        # randomize the biases and weights.
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.weight_num = sum([x * y for x, y in zip(sizes[:-1], sizes[1:])])

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None,
            lmbda=0.0):
        """Train the neural network using mini-batch stochastic
        gradient descent. The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired
        outputs. eta is the learning rate. The other non-optional
        parameters are self-explanatory. If "test_data" is provided
        then the network will be evaluated against the test data
        after each epoch, and partial progress printed out. This is
        useful for tracking progress, but slows things down
        substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                    training_data[k:k+mini_batch_size]
                    for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda)
            if test_data:
                print "Epoch {0}: cost {1}  {2} / {3}".format(
                        j, self.evaluate_cost(training_data),
                        self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta, lmbda):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single
        mini batch. The "mini_batch" is a list of tuples
        "(x, y)", and "eta" is the learning rate."""
        x = np.asarray([_x.ravel() for _x, _y in mini_batch]).transpose()
        y = np.asarray([_y.ravel() for _x, _y in mini_batch]).transpose()
        nabla_b, nabla_w = self.backprop(x, y)
        weight_decay = 1 - lmbda / self.weight_num
        self.weights = [weight_decay * w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                        for b, nb in zip(self.biases, nabla_b)]
    

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x. ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays,
        similar to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y)
        nabla_b[-1] = delta.sum(1).reshape([len(delta), 1])
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for i in xrange(2, self.num_layers):
            z = zs[-i]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-i+1].transpose(), delta) * sp
            nabla_b[-i] = delta.sum(1).reshape([len(delta), 1])
            nabla_w[-i] = np.dot(delta, activations[-i-1].transpose())
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_z /
        \partial a for the output activations. The cost function
        is C(z) = -1/n * (ylna + (1-y)ln(1-a)), so the derivative is
        C' = a - y."""
        return (output_activations-y)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural network
        outputs the correct result. Note that the neural network's
        output is assumed to be the index of whichever neuron
        in the final layer has the highest activation."""
        test_result = [(np.argmax(self.feedforward(x)), y)
                for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_result)

    def evaluate_cost(self, training_data):
        """Return the average difference of activation[-1] and y for
        training_data."""
        test_result = [self.feedforward(x) - y for (x, y) in training_data]
        cost = [(a*a).sum() for a in test_result]
        n = len(training_data)
        return 1.0 / n * sum(cost)

### Miscellaneous functions
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
