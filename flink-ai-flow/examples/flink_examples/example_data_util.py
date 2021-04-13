"""
Utilities to get the path of the data sets that are used by the examples
"""

import os


def get_example_data():
    return os.path.dirname(os.path.abspath(__file__)) + '/example_data'


def get_mnist_train_data():
    return get_example_data() + '/mnist_train.npz'


def get_mnist_evaluate_data():
    return get_example_data() + '/mnist_evaluate.npz'


def get_mnist_predict_data():
    return get_example_data() + '/mnist_predict.npz'
