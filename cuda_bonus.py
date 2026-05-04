import numpy as np
from numba import cuda
import math

# Thread block size for reduction (1D)
THREADS = 256

@cuda.jit
def mandelbrot_shared_reduction_kernel(mean_iters, width, height, max_iter, x_min, x_max, y_min, y_max):
    """
    Computes Mandelbrot set and uses shared memory to compute the sum of iterations 
    per block. This is a reduction example for bonus points.
    """
    # Allocate shared memory for the block
    shared_sum = cuda.shared.array(shape=(THREADS,), dtype=np.float64)
    
    # 1D index since reduction is easier in 1D
    tid = cuda.threadIdx.x
    i = cuda.grid(1)
    
    # Compute row and col from 1D grid index
    y = i // width
    x = i % width
    
    iteration = 0
    if x < width and y < height:
        real = x_min + (x_max - x_min) * x / width
        imag = y_min + (y_max - y_min) * y / height

        c = complex(real, imag)
        z = 0.0j

        while abs(z) <= 2.0 and iteration < max_iter:
            z = z * z + c
            iteration += 1
            
    # Load into shared memory
    shared_sum[tid] = iteration
    cuda.syncthreads()
    
    # Reduction in shared memory
    s = 1
    while s < cuda.blockDim.x:
        if tid % (2 * s) == 0 and tid + s < cuda.blockDim.x:
            shared_sum[tid] += shared_sum[tid + s]
        s *= 2
        cuda.syncthreads()
        
    # Thread 0 writes the block's sum to global memory
    if tid == 0:
        mean_iters[cuda.blockIdx.x] = shared_sum[0]

def compute_mean_iterations(width, height, max_iter):
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    
    total_pixels = width * height
    blocks = math.ceil(total_pixels / THREADS)
    
    mean_iters_device = cuda.device_array(blocks, dtype=np.float64)
    
    start = cuda.event(timing=True)
    end = cuda.event(timing=True)
    
    start.record()
    mandelbrot_shared_reduction_kernel[blocks, THREADS](
        mean_iters_device, width, height, max_iter, x_min, x_max, y_min, y_max
    )
    end.record()
    end.synchronize()
    
    # Copy block sums back to CPU
    block_sums = mean_iters_device.copy_to_host()
    
    # Final reduction on CPU
    total_sum = np.sum(block_sums)
    mean_iteration = total_sum / total_pixels
    
    elapsed_ms = cuda.event_elapsed_time(start, end)
    
    return mean_iteration, elapsed_ms

if __name__ == "__main__":
    w, h = 1024, 1024
    m_iter = 100
    print(f"Computing Mean Iterations with Shared Memory Reduction ({w}x{h})")
    mean_val, time_ms = compute_mean_iterations(w, h, m_iter)
    print(f"Mean Iterations: {mean_val:.2f}")
    print(f"Time (ms): {time_ms:.3f}")
