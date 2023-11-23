import random

def randomize(point, x_range, y_range, z_range):

    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range

    x = random.uniform(x_min, x_max)
    x += point[0]
    y = random.uniform(y_min, y_max)
    y += point[1]
    z = random.uniform(z_min, z_max)
    z += point[2]

    return (x, y, z)

# Example usage:
# initial_point = (10, 20, 30)
# x_range = (0, 5)
# y_range = (0, 5)
# z_range = (0, 5)

# randomized_point = randomize(initial_point, x_range, y_range, z_range)
# print("Randomized Point:", randomized_point)
