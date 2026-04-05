import numpy as np
import dask
from dask import delayed
import dask.array as da
from dask.distributed import Client, LocalCluster
import time
import matplotlib.pyplot as plt

# --- 1. CORE NUMPY KERNEL ---
def compute_mandelbrot_chunk(x_min, x_max, y_min, y_max, width, height, max_iter):
    """
    Standard NumPy vectorized implementation applied to a sub-region (chunk).
    """
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros_like(C)

    fractal = np.zeros(C.shape, dtype=np.int32) + max_iter

    for i in range(max_iter):
        mask = np.abs(Z) <= 2.0
        Z[mask] = Z[mask]**2 + C[mask]

        # Mark points that just escaped in this iteration
        escaped_this_iter = (np.abs(Z) > 2.0) & mask
        fractal[escaped_this_iter] = i

    return fractal

# --- 2. DASK ORCHESTRATOR ---
def run_dask_mandelbrot(full_width, full_height, max_iter, chunks_x, chunks_y):
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5

    chunk_width = full_width // chunks_x
    chunk_height = full_height // chunks_y

    x_edges = np.linspace(x_min, x_max, chunks_x + 1)
    y_edges = np.linspace(y_min, y_max, chunks_y + 1)

    tasks = []
    for j in range(chunks_y):
        row_tasks = []
        for i in range(chunks_x):
            c_x_min, c_x_max = x_edges[i], x_edges[i+1]
            c_y_min, c_y_max = y_edges[j], y_edges[j+1]

            # Wrap the NumPy function in dask.delayed
            task = delayed(compute_mandelbrot_chunk)(
                c_x_min, c_x_max, c_y_min, c_y_max,
                chunk_width, chunk_height, max_iter
            )
            row_tasks.append(task)
        tasks.append(row_tasks)

    # Assemble the delayed chunks into a single Dask Array
    blocks = [[da.from_delayed(t, shape=(chunk_height, chunk_width), dtype=np.int32) for t in row] for row in tasks]
    fractal_dask = da.block(blocks)

    # Compute triggers the actual execution across the cluster/cores
    start_time = time.time()
    result = fractal_dask.compute()
    exec_time = time.time() - start_time

    return exec_time, result

if __name__ == '__main__':
    RESOLUTION = 8192 # Use a large resolution so Dask can shine
    MAX_ITER = 100

    # --- LOCAL EXECUTION ---
    print("Setting up Local Dask Cluster...")
    # This automatically detects your CPU cores
    local_cluster = LocalCluster()
    client = Client(local_cluster)
    print(f"Dashboard link: {client.dashboard_link}")

    # Test different chunking strategies (e.g., 4x4 grid = 16 chunks)
    chunks_x, chunks_y = 4, 4

    print(f"Computing Dask Mandelbrot ({RESOLUTION}x{RESOLUTION}) on Local Cluster...")
    time_local, result_local = run_dask_mandelbrot(RESOLUTION, RESOLUTION, MAX_ITER, chunks_x, chunks_y)
    print(f"Local Execution Time: {time_local:.4f} seconds")

    client.close()
    local_cluster.close()

    # --- CLUSTER EXECUTION (Commented out until you set it up) ---


    print("\nConnecting to Remote Cluster...")
    # Replace with your Scheduler's IP address and port
    client = Client('tcp://192.168.0.35:8786')

    time_cluster, result_cluster = run_dask_mandelbrot(RESOLUTION, RESOLUTION, MAX_ITER, chunks_x, chunks_y)
    print(f"Cluster Execution Time: {time_cluster:.4f} seconds")
    client.close()
