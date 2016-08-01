#!/usr/bin/bash

"""
A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network. Gradients are calculated
using backpropagation.
"""

import numpy as np
import random

import sys
sys.path.append("../ch1")

from mnist_loader import vectorized_result

#### Define the quadratic and cross-entropy cost functions.

class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output ``y``."""
        return 0.5 * np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * sigmoid_prime(z)

class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output ``y``.
        Note that np.nan_to_num is used to ensure numerical stability. In particular,
        if both ``a`` and ``y`` have a 1.0 in the same slot, then the expression
        (1-y)*np.log(1-a) returns nan. The np.nan_to_num ensures that that is converted
        to the correct value (0.0).
        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer. Note that the parameter
        ``z`` is not used by the method. It is included in the method's parameter
        in order to make the interface consistent with the delta method for other
        cost classes.
        """
        return (a-y)


class Network(object):
    def __init__(self, sizes, cost=CrossEntropyCost):
        # sizes is the size of different layers of the neural network.
        # sizes[0] is the input size, sizes[-1] is the output size.
        self.num_layers = len(sizes)
        self.sizes = sizes
        # randomize the biases and weights.
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.cost = cost

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            early_stop=0,
            learning_schedule=False):
        """Train the neural network using mini-batch stochastic
        gradient descent. The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired
        outputs. eta is the learning rate. The other non-optional
        parameters are self-explanatory.
        If early_stop is not zero and evaluation_data is not None,
        stop the iteration if the accuracy of evaluation_data hasn't
        been improved for [early_stop] epochs.
        If learning_schedule is True, half the learning rate when
        early_stop situation is met. Terminate until we reach 1/128
        of the original [eta] value.
        Return a tuple of (evaluation_cost, evaluation_accuracy,
        training_cost, training_accuracy), each item in the tuple
        is a list of cost or accuracy value.
        """
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        if learning_schedule:
            if early_stop == 0:
                early_stop = 10
            origin_eta = eta

        use_early_stop = False
        if early_stop != 0 and evaluation_data is not None:
            use_early_stop = True
            max_accuracy = 0
            max_accuracy_updated_place = -1
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                    training_data[k:k+mini_batch_size]
                    for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
            print "Epoch %s training complete" % j
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda, True)
                training_cost.append(cost)
                print "Cost on training_data: {}".format(cost)
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, True)
                training_accuracy.append(accuracy)
                print 'Accuracy on training_data: {} / {}'.format(accuracy, n)
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, False)
                evaluation_cost.append(cost)
                print 'Cost on evaluation data: {}'.format(cost)
            if monitor_evaluation_accuracy or use_early_stop:
                accuracy = self.accuracy(evaluation_data)
                if monitor_evaluation_accuracy:
                    evaluation_accuracy.append(accuracy)
                print 'Accuracy on evaluation data: {} / {}'.format(accuracy, n_data)
                if use_early_stop:
                    if max_accuracy < accuracy:
                        max_accuracy = accuracy
                        max_accuracy_updated_place = j
                    elif j - max_accuracy_updated_place >= early_stop:
                        if origin_eta <= 128 * eta:
                            eta = eta / 2.0
                            print "Use learning rate: {}".format(eta)
                            max_accuracy = -1
                        else:
                            print "Stop because evaluation accuracy " + \
                                  "hasn't been updated for {} cycles".format(early_stop)
                            print "Max accuracy is {} / {}".format(max_accuracy, n_data)
                            break

            print
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy


    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single
        mini batch. The "mini_batch" is a list of tuples
        "(x, y)", and "eta" is the learning rate."""
        x = np.asarray([_x.ravel() for _x, _y in mini_batch]).transpose()
        y = np.asarray([_y.ravel() for _x, _y in mini_batch]).transpose()
        nabla_b, nabla_w = self.backprop(x, y)
        weight_decay = 1 - eta * lmbda / n
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
        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta.sum(1).reshape([len(delta), 1])
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for i in xrange(2, self.num_layers):
            z = zs[-i]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-i+1].transpose(), delta) * sp
            nabla_b[-i] = delta.sum(1).reshape([len(delta), 1])
            nabla_w[-i] = np.dot(delta, activations[-i-1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, isTrainingData=False):
        """Return the number of test inputs for which the neural network
        outputs the correct result. Note that the neural network's
        output is assumed to be the index of whichever neuron
        in the final layer has the highest activation."""
        if isTrainingData:
            test_result = [(np.argmax(self.feedforward(x)), np.argmax(y))
                    for (x, y) in data]
        else:
            test_result = [(np.argmax(self.feedforward(x)), y)
                    for (x, y) in data]
        return sum(int(x == y) for (x, y) in test_result)

    def total_cost(self, data, lmbda, isTrainingData=False):
        """Return the total cost for the data set ``data``."""
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if not isTrainingData: y = vectorized_result(y)
            cost += self.cost.fn(a, y) / len(data)
        cost += 0.5 * (lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost

### Miscellaneous functions
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
