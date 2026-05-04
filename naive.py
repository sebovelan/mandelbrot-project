import numpy as np

def mandelbrot_naive(width, height, max_iter):
    """
    Computes the Mandelbrot set using a naive, pure Python nested loop approach.
    
    This function iterates through each pixel in the specified width and height,
    maps it to the complex plane, and computes the escape time using a while loop.
    It is useful as a baseline for performance comparisons.
    
    Args:
        width (int): The width of the output image in pixels.
        height (int): The height of the output image in pixels.
        max_iter (int): The maximum number of iterations to test for escape.
        
    Returns:
        numpy.ndarray: A 2D array of shape (height, width) containing the 
        normalized iteration count (from 0 to 1) for each pixel.
    """

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

            image[i, j] = iteration/max_iter

    return image
