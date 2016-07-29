#!/usr/bin/python

import sys
sys.path.append("../ch1")

import mnist_loader
import network_l2_regularization as network

net = network.Network([784, 100, 10])
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net.SGD(training_data[:50000], 60, 10, 0.1, test_data=test_data, lmbda=5)
