import numpy as np
from numba import cuda
import math

@cuda.jit
def mandelbrot_kernel(image, width, height, max_iter, x_min, x_max, y_min, y_max):
    """
    CUDA kernel to compute the Mandelbrot set.
    Each thread computes the escape iterations for one pixel.
    """
    # 2D coordinates for the thread
    x, y = cuda.grid(2)

    # Boundary check: ensure thread is within image dimensions
    if x < width and y < height:
        # Map pixel coordinate to complex plane
        real = x_min + (x_max - x_min) * x / width
        imag = y_min + (y_max - y_min) * y / height

        c = complex(real, imag)
        z = 0.0j
        iteration = 0

        # Mandelbrot formula
        while abs(z) <= 2.0 and iteration < max_iter:
            z = z * z + c
            iteration += 1

        # Normalize the result (0.0 to 1.0)
        image[y, x] = iteration / max_iter

def run_cuda_mandelbrot(width, height, max_iter, threads_per_block=(16, 16)):
    """
    Launcher for the CUDA Mandelbrot kernel.
    Handles memory transfer, grid/block dimension calculation, and timing.
    
    Args:
        width (int): Image width.
        height (int): Image height.
        max_iter (int): Maximum iterations.
        threads_per_block (tuple): Block size configuration.
        
    Returns:
        tuple: (execution_time_ms, resulting_image)
    """
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5

    # Allocate memory on the host and then transfer/allocate on device
    image_host = np.zeros((height, width), dtype=np.float64)
    image_device = cuda.to_device(image_host)

    # Compute grid size
    blocks_x = math.ceil(width / threads_per_block[0])
    blocks_y = math.ceil(height / threads_per_block[1])
    blocks_per_grid = (blocks_x, blocks_y)

    # Setup timing events
    start = cuda.event(timing=True)
    end = cuda.event(timing=True)

    # --- WARMUP ---
    # To ignore JIT compilation time, we run a tiny warmup
    warmup_device = cuda.device_array((1, 1), dtype=np.float64)
    mandelbrot_kernel[(1,1), (1,1)](warmup_device, 1, 1, 1, x_min, x_max, y_min, y_max)
    cuda.synchronize()

    # --- ACTUAL COMPUTATION ---
    start.record()
    
    # Launch the kernel
    mandelbrot_kernel[blocks_per_grid, threads_per_block](
        image_device, width, height, max_iter, x_min, x_max, y_min, y_max
    )
    
    end.record()
    end.synchronize()
    
    # Get elapsed time in milliseconds
    elapsed_ms = cuda.event_elapsed_time(start, end)

    # Copy the result back to the host
    image_host = image_device.copy_to_host()

    return elapsed_ms, image_host

if __name__ == "__main__":
    # Simple test run
    width, height = 1024, 1024
    max_iter = 100
    threads = (16, 16)
    
    print(f"Running CUDA Mandelbrot with {width}x{height}, max_iter={max_iter}")
    print(f"Block size: {threads}")
    time_ms, img = run_cuda_mandelbrot(width, height, max_iter, threads)
    
    print(f"Kernel Execution Time (excluding data transfer): {time_ms:.3f} ms")
    print(f"Center pixel value (normalized): {img[height//2, width//2]}")
