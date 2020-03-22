import math
import matplotlib.pyplot as plt


# Function: dx/dt=x-x^3
# Step size: 0.01
# Start values: x=-7, x=-0.2, x=8
def plotting_function(x): return x - math.pow(x, 3)


def plot_euler(function, start_x, step_size, count, pos):
    x = start_x
    result = [(0, start_x)]

    for i in range(1, count):
        x += step_size * function(x)
        result.append((i, x))

    plt.subplot(2, 2, pos)
    plt.title("Start value: " + str(start_x))
    plt.ylabel("f(x)")
    plt.xlabel("Iterations")
    plt.grid(color='b', alpha=0.4, linestyle='dashed', linewidth=0.5)
    plt.plot(*zip(*result))
    print(result)


plt.subplots(figsize=(10, 7))
plot_euler(plotting_function, -7, 0.01, 1500, 1)
plot_euler(plotting_function, -0.2, 0.01, 1500, 2)
plot_euler(plotting_function, 8, 0.01, 1500, 3)
plot_euler(plotting_function, 0.2, 0.01, 1500, 4)
plt.show()
