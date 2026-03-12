import time
import pandas as pd
from naive import mandelbrot_naive
from numpy_version import mandelbrot_numpy
from njit_version import mandelbrot_numba

sizes = [1024,2048, 4096,8192]
max_iter = 100

results = []

for size in sizes:

    start = time.time()
    mandelbrot_naive(size, size, max_iter)
    naive_time = time.time() - start

    start = time.time()
    mandelbrot_numpy(size, size, max_iter)
    numpy_time = time.time() - start

    start = time.time()
    mandelbrot_numba(size, size, max_iter)
    numba_time = time.time() - start

    results.append([size, naive_time, numpy_time, numba_time])

df = pd.DataFrame(results, columns=[
    "size",
    "naive_time",
    "numpy_time",
    "numba_time"
])
df["numpy_speedup"] = df["naive_time"] / df["numpy_time"]
df["numba_speedup"] = df["naive_time"] / df["numba_time"]
df.to_csv("timings.csv", index=False)
print(df)
