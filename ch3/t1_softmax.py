#!/usr/bin/python

import sys
sys.path.append("../ch1")

import mnist_loader
import network_softmax as network

net = network.Network([784, 100, 10])
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net.SGD(training_data, 30, 10, 0.5, test_data=test_data)

