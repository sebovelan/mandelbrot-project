import time
from naive import mandelbrot_naive

sizes = [1024,2048,4096]

max_iter = 100

for s in sizes:

    start = time.time()

    mandelbrot_naive(s, s, max_iter)

    end = time.time()

    print(f"Size {s}x{s} -> {end-start:.3f} seconds")
