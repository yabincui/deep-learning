#!/usr/bin/python

import sys
sys.path.append("../ch1")

import mnist_loader
import network2 as network

net = network.Network([784, 30, 10])
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net.SGD(training_data[:], 60, 10, 0.1, lmbda=5, evaluation_data=validation_data[:],
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True,
        early_stop=10)
