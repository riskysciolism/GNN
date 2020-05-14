import numpy as np
from math import e

np.random.seed(seed=2)

sigmoid = np.vectorize(lambda x: 1 / (1 + e ** -x))
WEIGHT_MULTIPLIER = 5


class Neuron:
    def __init__(self, number_of_inputs_from_previous):
        self._value = 1
        self._weights = np.random.random(number_of_inputs_from_previous) * WEIGHT_MULTIPLIER

    def get_weight(self, input_neuron_index):
        return self._weights[input_neuron_index]

    def get_weights(self):
        return self._weights

    def set_weight(self, input_neuron_index, value):
        self._weights[input_neuron_index] = value

    def get_value(self):
        return self._value

    def activate(self, input_activations):
        self._value = sigmoid(input_activations.dot(self._weights))

    def print(self):
        print(f"Value: {self.get_value()}")
        print(f"Weight: {self.get_weights()}")


class InputNeuron:
    def __init__(self):
        self.value = 0

    def set_value(self, value):
        self.value = value

    def get_value(self):
        return self.value

    def print(self):
        print(f"Value: {self.get_value()}")
