from math import e

import matplotlib.pyplot as plt
import numpy as np
import atexit

# np.random.seed(seed=2)

learningRate = 0.001

sigmoid = np.vectorize(lambda x: 1 / (1 + e ** -x))

sigmoid_derivative = np.vectorize(lambda x: 1 / (1 + e ** -x) * (1 - 1 / (1 + e ** -x)))

pseudo_sigmoid_derivative = np.vectorize(lambda x: x * (1 - x))

value_range = 2


def __is_in_unit_circle(vector): return vector[0] ** 2 + vector[1] ** 2 <= 1


def __desired_result(_network):
    return 0.8 if __is_in_unit_circle(np.array([_network[0][0][1], _network[0][0][2]])) else 0


def __randomize_weights(origin_size, destiny_size): return np.random.uniform(-1, 1, (destiny_size, origin_size))


# Calculates gradients of the output layer
def __gradient_o(_layer_activation_values, _desired_result):
    _layer_deltas = []
    for _activation_value_index in range(len(_layer_activation_values)):
        # skip bias neuron
        if _activation_value_index == 0:
            continue
        _layer_deltas.append(_layer_activation_values[_activation_value_index] * (
                _desired_result - _layer_activation_values[_activation_value_index]))
    return _layer_deltas


# Calculates gradients of hidden layers
def __gradient_h(_previous_layer_deltas, _layer_activation_values, _layer_weights):
    _layer_deltas = []
    _sum = 0
    for _weight_index in range(len(_layer_weights[0])):
        # skip bias neuron
        if _weight_index == 0:
            continue
        for _delta_index in range(len(_previous_layer_deltas)):
            _sum += _previous_layer_deltas[_delta_index] * _layer_weights[_delta_index][_weight_index]

        _current_delta_h = pseudo_sigmoid_derivative(_layer_activation_values[_weight_index]) * _sum
        _layer_deltas.append(_current_delta_h)

    return _layer_deltas


# Calculates and returns a gradient matrix, which is used by
def __calculate_gradients(_network):
    _gradient_matrix = []
    _gradients = []
    # for every layer starting from the last layer
    for _layer_index in reversed(range(len(_network))):
        # skip input layer
        if _layer_index == 0:
            break
        _layer_activation_values = _network[_layer_index][0]
        _layer_weights = _network[_layer_index][1]

        if _layer_index == len(_network) - 1:
            # output layer
            _gradients = __gradient_o(_layer_activation_values, __desired_result(_network))
        else:
            # hidden layer
            _gradients = __gradient_h(_gradients, _layer_activation_values, _layer_weights)

        _gradient_matrix.insert(0, list(_gradients))

    return _gradient_matrix


# Calculates the error gradients and changes the weights accordingly.
def __back_propagation(_network):
    _error_gradient_matrix = __calculate_gradients(_network)
    for _layer_index in range(len(_network)):
        if _layer_index == len(_network) - 1:
            break
        for _weight_index in range(len(_network[_layer_index][1])):
            for _neuron_index in range(len(_network[_layer_index][0])):
                _weight = _network[_layer_index][1][_weight_index][_neuron_index]
                _error_gradient = _error_gradient_matrix[_layer_index][_weight_index]
                _current_delta = learningRate * _error_gradient * _network[_layer_index][0][_neuron_index]
                _network[_layer_index][1][_weight_index][_neuron_index] = _weight + _current_delta[0]
    return _network


# Initializes network with a given structure
# Structure: [Layer 1 neuron count, Layer 2 neuron count,...Layer n neuron count]
# Returns array filled with a matrix for each layer and it's weights
# Bias neurons are added automatically!
def init_network(structure):
    network_structure = []
    for i in range(len(structure)):
        if i + 1 < len(structure):
            layer_neurons = np.ones((structure[i] + 1, 1))
            layer_weights = __randomize_weights(structure[i] + 1, structure[i + 1])
        # output layer
        else:
            layer_neurons = layer_weights = np.ones((structure[i] + 1, 1))

        network_structure.append([layer_neurons, layer_weights])

    return network_structure


# Calculate the output of the network for a given input vector
def feed_forward(_input_vector, _network):
    # input values into first layer
    for neuron_index in range(len(_input_vector)):
        _network[0][0][neuron_index + 1] = _input_vector[neuron_index]

    # for each layer
    for layer_index in range(len(_network) - 1):
        _neurons = _network[layer_index][0]
        _weights = _network[layer_index][1]
        network_input = _weights.dot(_neurons)
        activated_network_input = sigmoid(network_input)
        # append bias
        activated_network_input = np.insert(activated_network_input, 0, [1], axis=0)
        # input values into next layer
        _network[layer_index + 1][0] = activated_network_input
    else:
        return _network


# Trains a given network n times.
# You can choose to visualize the performance of the network
# _visualize: None, "show", "save", "both"
# _visualize_step: The network performance will be visualized every n percentages.
def train(_network, _count, _visualize=None, _visualize_step=5):
    for i in range(_count + 1):
        print(str(int((i / _count) * 100)) + "%") if (i % 10000) == 0 else 0
        _input_vector = np.random.uniform(low=-value_range, high=value_range, size=2)
        feed_forward(_input_vector, _network)
        __back_propagation(_network)

        if _visualize is not None:
            _percentage = (i / _count) * 100
            if _percentage % _visualize_step == 0:
                visualize_performance(_network, percentage=_percentage, _visualize=_visualize)

    return _network


# Visualize the performance of a network
# _visualize: "show", "save", "both"
def visualize_performance(_network, percentage=None, _visualize="print"):
    x_range = np.arange(-value_range * 2, value_range * 2, 0.05)
    y_range = np.arange(-value_range * 2, value_range * 2, 0.05)
    x, y = np.meshgrid(x_range, y_range)
    z = []
    for x_coordinate in x_range:
        z_row = []
        for y_coordinate in y_range:
            z_row.append(feed_forward(np.array([x_coordinate, y_coordinate]), _network)[-1][0][1][0])
        z.append(z_row)
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    p = ax.pcolor(x, y, z)
    fig.colorbar(p)

    if percentage is not None:
        percentage = int(percentage)
        plt.title(f"Learning rate: {learningRate}, Percentage: {percentage}")
    else:
        plt.title(f"Learning rate: {learningRate}")
        percentage = 0

    if _visualize == "save":
        plt.savefig(f"{percentage}.png", bbox_inches='tight')
        plt.close(fig)
    elif _visualize == "show":
        plt.show()
        plt.close(fig)
    elif _visualize == "both":
        plt.savefig(f"{percentage}.png", bbox_inches='tight')
        plt.show()
        plt.close(fig)


# Prints the network in a more readable way than simple pprint
def print_network(_network):
    for _layer_index in range(len(_network)):
        print("Layer " + str(_layer_index))
        for _neuron_index in range(len(_network[_layer_index][0])):
            # We don't want to print the weights of the last layer
            if _layer_index == len(_network) - 1:
                # We can also skip the bias neuron
                if _neuron_index > 0:
                    print("Neuron " + str(_neuron_index) + ": " + str(
                        _network[_layer_index][0][_neuron_index]))
            else:
                _weights = []
                for _weight_index in range(len(_network[_layer_index][1])):
                    _weights.append(_network[_layer_index][1][_weight_index][_neuron_index])
                print("Neuron " + str(_neuron_index) + ": " + str(
                    _network[_layer_index][0][_neuron_index]) + " Weights: " + str(_weights))


# Exit handler should make one able to abort the program and preserve the current progress
def exit_handler():
    print_network(network)


# Initialize network
# Any structure of hidden layers is possible
network = init_network([2, 4, 1])
print_network(network)

atexit.register(exit_handler)

train(network, 1000000, _visualize="show", _visualize_step=10)
