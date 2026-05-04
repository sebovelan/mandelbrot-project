import time
import numpy as np
from naive import mandelbrot_naive
from numpy_version import mandelbrot_numpy
from cuda_version import run_cuda_mandelbrot
try:
    from dask_version import run_dask_mandelbrot
except ImportError:
    run_dask_mandelbrot = None

def analyze_block_sizes():
    print("--- 1. BLOCK SIZE ANALYSIS ---")
    width, height = 2048, 2048
    max_iter = 100
    
    block_sizes = [(8, 8), (16, 16), (32, 32), (32, 8), (8, 32)]
    
    for bs in block_sizes:
        # Run multiple times and average to be accurate
        times = []
        for _ in range(5):
            ms, _ = run_cuda_mandelbrot(width, height, max_iter, threads_per_block=bs)
            times.append(ms)
        
        avg_time = sum(times) / len(times)
        total_threads = bs[0] * bs[1]
        print(f"Block Size: {bs} (Total threads: {total_threads:4d}) -> Avg Time: {avg_time:.3f} ms")
    print()

def benchmark_all():
    print("--- 2. PERFORMANCE ANALYSIS (Consistent Parameters) ---")
    width, height = 1024, 1024
    max_iter = 100
    print(f"Resolution: {width}x{height}, Max Iterations: {max_iter}")
    
    # Naive
    start = time.time()
    mandelbrot_naive(width, height, max_iter)
    naive_time = time.time() - start
    print(f"Naive Python Time:  {naive_time:.4f} s")
    
    # NumPy
    start = time.time()
    mandelbrot_numpy(width, height, max_iter)
    numpy_time = time.time() - start
    print(f"NumPy Time:         {numpy_time:.4f} s")
    
    # Dask (Local)
    dask_time = float('inf')
    if run_dask_mandelbrot is not None:
        dask_t, _ = run_dask_mandelbrot(width, height, max_iter, chunks_x=4, chunks_y=4)
        dask_time = dask_t
        print(f"Dask Time (Local):  {dask_time:.4f} s")
    
    # CUDA
    cuda_ms, _ = run_cuda_mandelbrot(width, height, max_iter, threads_per_block=(16, 16))
    cuda_time = cuda_ms / 1000.0 # Convert to seconds
    print(f"CUDA Time (Kernel): {cuda_time:.4f} s")
    
    print("\nSpeedups (relative to CUDA):")
    print(f"vs Naive: {naive_time / cuda_time:.1f}x")
    print(f"vs NumPy: {numpy_time / cuda_time:.1f}x")
    if run_dask_mandelbrot is not None:
        print(f"vs Dask:  {dask_time / cuda_time:.1f}x")
    print()

def scale_analysis():
    print("--- 3. SCALING ANALYSIS ---")
    sizes = [256, 512, 1024, 2048, 4096]
    max_iter = 100
    
    print(f"{'Size':<10} | {'NumPy Time (s)':<15} | {'CUDA Time (s)':<15} | {'Speedup':<10}")
    print("-" * 55)
    
    for size in sizes:
        # NumPy
        start = time.time()
        mandelbrot_numpy(size, size, max_iter)
        numpy_time = time.time() - start
        
        # CUDA
        cuda_ms, _ = run_cuda_mandelbrot(size, size, max_iter, threads_per_block=(16, 16))
        cuda_time = cuda_ms / 1000.0
        
        speedup = numpy_time / cuda_time if cuda_time > 0 else 0
        print(f"{size}x{size:<5} | {numpy_time:<15.4f} | {cuda_time:<15.4f} | {speedup:.1f}x")
    print()

if __name__ == "__main__":
    analyze_block_sizes()
    benchmark_all()
    scale_analysis()
