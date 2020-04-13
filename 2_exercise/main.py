import copy
from math import e
from pprint import pprint

import numpy as np
from collections import deque

np.random.seed(seed=2)

learningRate = 0.1

sigmoid = np.vectorize(lambda x: 1 / (1 + e ** -x))

sigmoid_derivative = np.vectorize(lambda x: 1 / (1 + e ** -x) * (1 - 1 / (1 + e ** -x)))

pseudo_sigmoid_derivative = np.vectorize(lambda x: x * (1 - x))

input_vector = np.array([[-1.54632465], [-0.55020734]])


def __is_in_unit_circle(vector): return vector[0] ** 2 + vector[1] ** 2 <= 1


def __desired_result(_network):
    return 0.8 if __is_in_unit_circle(np.array([_network[0][0][1], _network[0][0][2]])) else 0


def __calculate_error(point_vector, result):
    result = result[-1][0]
    temp = []
    for _neuron_value in result:
        _error = _neuron_value[0] - 0.8 if __is_in_unit_circle(point_vector) else _neuron_value
        temp.append(_error)
    return np.array(temp)


def __randomize_weights(origin_size, destiny_size): return np.random.uniform(-1, 1, (destiny_size, origin_size))


def __small_delta_output(_network_input, _desired_result):
    return sigmoid_derivative(_network_input) * (_desired_result - _network_input)


# deltas is a queue of deltas (FIFO)
def __small_delta_hidden(_network_input, _this_weights, _neuron_number, _deltas):
    _temp_deltas = copy.deepcopy(_deltas)
    _sum = 0
    for i in range(len(_temp_deltas)):
        _current_delta = _temp_deltas.popleft()
        _current_weight = _this_weights[i][_neuron_number]
        print(f"small_delta_hidden current_delta: {_current_delta}, current_weight: {_current_weight}")
        _sum += _current_delta * _current_weight

    _delta = sigmoid_derivative(_network_input) * _sum

    print(f"Neurons weights delta: {_delta}")
    return _delta


def __calculate_delta(_network, _network_output):
    _delta_matrix = []
    deltas = deque()
    # for every layer starting from the last layer
    for i in reversed(range(len(_network[0]) + 1)):
        # check for output layer
        if i == len(_network[0]):
            # __small_delta_output
            for j in range(len(_network[i][0])):
                print(f"Calculating delta for Layer: {i}, Neuron: {j}")
                deltas.append(__small_delta_output(_network_output, __desired_result(_network)))
                print(f"small delta output {deltas}")
        else:
            # __small_delta_hidden
            temp_deltas = deque()
            # for every neuron (k) in the layer (i)
            for k in range(len(_network[i][0])):
                print(f"Calculating delta for Layer: {i}, Neuron: {k}")
                temp_deltas.append(__small_delta_hidden(_network[i][0][k], _network[i][1], k, deltas))
            deltas = copy.deepcopy(temp_deltas)
            _delta_matrix.insert(0, list(deltas))
            deltas.popleft()
            print(f"Layer: {i} finished")
    print(f"Delta Matrix {_delta_matrix}")

    for i in range(len(_delta_matrix)):
        layer_delta = np.concatenate(_delta_matrix[i])
        print(sigmoid(_network[i][0]))
        print(np.dot(layer_delta, np.transpose(sigmoid(_network[i][0]))))


# initializes network with a given structure
# structure: [n0_count, n1_count,...nn_count]
# returns array filled with a matrix for each layer and it's weights
# bias neurons are added automatically!
def init_network(structure):
    network_structure = []
    for i in range(len(structure)):
        if i + 1 < len(structure):
            layer_neurons = np.ones((structure[i] + 1, 1))
            layer_weights = __randomize_weights(structure[i] + 1, structure[i + 1] + 1)
        # output layer
        else:
            layer_neurons = layer_weights = np.ones((structure[i] + 1, 1))

        network_structure.append([layer_neurons, layer_weights])

    return network_structure


def __feed_forward(_input_vector, _network):
    # input values into first layer
    for neuron_index in range(len(_input_vector)):
        _network[0][0][neuron_index + 1] = _input_vector[neuron_index]

    # for each layer
    for layer_index in range(len(_network) - 1):
        _neurons = _network[layer_index][0]
        _weights = _network[layer_index][1]
        network_input = _weights.dot(_neurons)
        activated_network_input = sigmoid(network_input)

        # make sure bias is one
        if len(activated_network_input) > 1:
            activated_network_input[0] = 1
        # input values into next layer
        _network[layer_index + 1][0] = activated_network_input
        print(f"Next layer:  {_network[layer_index + 1][0]}")
    else:
        return _network


# def backpropagation(_network):
#     for i in reversed(range(len(_network[0]))):


network = init_network([2, 4, 1])

network[0][1] = np.array([
    [0.79593196, 0.03079537, 0.00940831],
    [0.79593196, -0.0558573, 0.00940831],
    [3.12866633, 2.17995175, 3.47417767],
    [-2.93184716, 3.86271963, -0.24278788],
    [-2.94459101, -1.85508791, 3.598483]
])

network[1][1] = np.array([
    [-2.27532725, -1.4844733, 7.93574257, -7.95186599, -8.15543886]
])
print(f"Network input weights: {network[0][1]}")
print(f"Network hidden weights: {network[1][1]}")

network_output = __feed_forward(input_vector, network)

pprint(network_output)
print(network_output[2][0])
# print(f"Network output after feed forward: {network_output}")
#
# __calculate_delta(network, network_output)
