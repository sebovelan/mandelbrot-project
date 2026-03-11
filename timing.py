import time
from naive import mandelbrot_naive
from numpy_version import mandelbrot_numpy
from njit_version import mandelbrot_numba

sizes = [1024,2048,4096]

max_iter = 100

for s in sizes:

    start = time.time()
    #mandelbrot_naive(s, s, max_iter)
    #mandelbrot_numpy(s, s, max_iter)
    
    mandelbrot_numba(s, s, max_iter)
    end = time.time()

    print(f"Size {s}x{s} -> {end-start:.3f} seconds")
