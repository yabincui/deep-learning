#!/usr/bin/python

import mnist_loader
import network_fast as network

net = network.Network([784, 30, 10])
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
