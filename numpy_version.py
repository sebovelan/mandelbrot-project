import numpy as np

def mandelbrot_numpy(width, height, max_iter):

    x = np.linspace(-2, 1, width)
    y = np.linspace(-1.5, 1.5, height)

    X, Y = np.meshgrid(x, y)
    C = X + 1j*Y

    Z = np.zeros_like(C)
    output = np.zeros(C.shape, dtype=int)

    for i in range(max_iter+1):

        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + C[mask]
        output[mask] = i

    return output/max_iter
