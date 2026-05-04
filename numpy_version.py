import numpy as np

def mandelbrot_numpy(width, height, max_iter):
    """
    Computes the Mandelbrot set using a vectorized NumPy approach.
    
    This function creates a 2D complex grid and iteratively applies the 
    Mandelbrot formula (z = z^2 + c) to all points simultaneously using 
    NumPy arrays. It tracks how many iterations it takes for each point 
    to escape a radius of 2.
    
    Args:
        width (int): The width of the output image in pixels.
        height (int): The height of the output image in pixels.
        max_iter (int): The maximum number of iterations to test for escape.
        
    Returns:
        numpy.ndarray: A 2D array of shape (height, width) containing the 
        normalized iteration count (from 0 to 1) for each pixel.
    """

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
