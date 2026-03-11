import numpy as np
from numba import njit

@njit
def mandelbrot_numba(width, height, max_iter):

    x_min, x_max = -2, 1
    y_min, y_max = -1.5, 1.5

    image = np.zeros((height, width))

    for i in range(height):
        for j in range(width):

            x = x_min + (x_max - x_min) * j / width
            y = y_min + (y_max - y_min) * i / height

            c = complex(x, y)
            z = 0

            iteration = 0

            while abs(z) <= 2 and iteration < max_iter:
                z = z*z + c
                iteration += 1

            image[i,j] = iteration/max_iter

    return image
