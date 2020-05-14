from .neuron import Neuron, InputNeuron
import numpy as np


class Layer:
    def __init__(self, number_of_neurons, input_layer=None):
        neurons = []
        # hidden layer
        for neuron_index in range(number_of_neurons):
            if input_layer is not None:
                neurons.append(Neuron(input_layer.get_neuron_count()))
            else:
                neurons.append(InputNeuron())

        self.neurons = neurons

    def get_neuron_count(self):
        return len(self.neurons)

    def get_neurons(self):
        return self.neurons

    def activate(self, previous_layer):
        activations = []
        for neuron in previous_layer.get_neurons():
            activations.append(neuron.get_value())

        for neuron in self.neurons:
            neuron.activate(np.array(activations))

    def print(self):
        for index, neuron in enumerate(self.neurons):
            print(f"Neuron {index}")
            neuron.print()

    def set_input(self, _input):
        neurons = self.get_neurons()

        if len(neurons) == len(_input):
            for index, neuron in enumerate(neurons):
                neuron.set_value(_input[index])
        else:
            print(f"Wrong input size!")
            print(f"Input size: {len(_input)}")
            print(f"Layer size: {len(neurons)}")
            raise Exception()
