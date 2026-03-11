import matplotlib.pyplot as plt
from naive import mandelbrot_naive
from numpy_version import mandelbrot_numpy

width = 2048
height = 2048
max_iter = 100

#image = mandelbrot_naive(width, height, max_iter)
image = mandelbrot_numpy(width, height, max_iter)

plt.imshow(image, cmap="inferno")
plt.colorbar()
plt.title("Mandelbrot Set (Naive Implementation)")
plt.show()
