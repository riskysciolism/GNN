from entities.network import Network
import numpy as np

network = Network([5, 3])
network.set_input([1, 2, 3, 4, 5])

network.feed_forward()

network.print()
