from .layer import Layer


class Network:
    def __init__(self, structure):
        _network = []
        # for every layer
        for i in range(len(structure)):
            # input layer
            if i == 0:
                _network.append(Layer(number_of_neurons=structure[i]))
            else:
                _network.append(Layer(number_of_neurons=structure[i], input_layer=_network[-1]))

        self.net = _network

    def set_input(self, _input):
        input_layer = self.net[0]
        input_layer.set_input(_input)

    def feed_forward(self):
        for index, layer in enumerate(self.net):
            if index > 0:
                # activate layer with previous layer
                layer.activate(self.net[index - 1])

    def feed_backward(self):
        for index, layer in enumerate(reversed(self.net)):
            pass

    def print(self):
        print("Network:")
        for index, layer in enumerate(self.net):
            print("---------------------------------------------------------------------------------------------------")
            print(f"Layer {index}:")
            print("----------")
            layer.print()
