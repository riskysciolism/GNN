from math import e, sqrt
import numpy as np

sigmoid = np.vectorize(lambda x: 1 / (1 + e ** -x))
input_vector = np.array([[0], [1]])


def __is_in_unit_circle(vector): return vector[0] ** 2 + vector[1] ** 2 <= 1


def __calculate_error(vector, result):
    return abs(result - 0.8) if __is_in_unit_circle(vector) else result


def __randomize_biases(size): return np.random.rand(size, 1) * 1


def __randomize_weights(origin_size, destiny_size): return np.random.rand(destiny_size, origin_size)


# activation function per layer
def __activation(weights, neurons, biases, do_sigmoid):
    return sigmoid(weights.dot(np.add(neurons, biases))) if do_sigmoid else weights.dot(np.add(neurons, biases))


weights_n0 = __randomize_weights(2, 4)
weights_n1 = __randomize_weights(4, 1)
biases_n0 = __randomize_biases(2)
biases_n1 = __randomize_biases(4)

print(f"Weights_n0: {weights_n0}")
print()
print(f"Weights_n1: {weights_n1}")
print()
print(f"Biases_n0: {biases_n0}")
print()
print(f"Biases_n1: {biases_n1}")
print()

input_layer = __activation(weights_n0, input_vector, biases_n0, False)
second_layer = __activation(weights_n1, input_layer, biases_n1, True)
third_layer = sigmoid(second_layer)

print(input_layer)
print()
print(second_layer)
print()
print(third_layer)
print()

print(f"Vector: {input_vector}, is in unit circle: {__is_in_unit_circle(input_vector)}")
print(f"Error: {__calculate_error(input_vector, third_layer)}")

