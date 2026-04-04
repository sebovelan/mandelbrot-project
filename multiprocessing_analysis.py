import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import os

# --- 1. CORE COMPUTATION (KERNEL) ---
def compute_row(args):
    """
    Computes a single row of the Mandelbrot set.
    Separated to the top level so it can be pickled by multiprocessing.Pool.
    """
    y_idx, y_val, width, x_min, x_max, max_iter = args
    x_vals = np.linspace(x_min, x_max, width)

    row_result = np.zeros(width, dtype=np.int32)
    for x_idx, x_val in enumerate(x_vals):
        c = complex(x_val, y_val)
        z = 0.0j
        for i in range(max_iter):
            if abs(z) > 2.0:
                row_result[x_idx] = i
                break
            z = z * z + c
        else:
            row_result[x_idx] = max_iter

    return y_idx, row_result

def mandelbrot_multiprocessing(width, height, max_iter, num_processes, chunk_size):
    """
    Parallel implementation distributing rows among worker processes.
    """
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    y_vals = np.linspace(y_min, y_max, height)

    # Prepare arguments for each row
    tasks = [(y, y_vals[y], width, x_min, x_max, max_iter) for y in range(height)]
    result_matrix = np.zeros((height, width), dtype=np.int32)

    start_time = time.time()

    # Use a Pool to parallelize execution
    with mp.Pool(processes=num_processes) as pool:
        # imap_unordered is efficient; workers return results as soon as they finish
        results = pool.imap_unordered(compute_row, tasks, chunksize=chunk_size)
        for y_idx, row_result in results:
            result_matrix[y_idx] = row_result

    exec_time = time.time() - start_time
    return exec_time, result_matrix

# --- 2. BENCHMARKING AND ANALYSIS ---
def analyze_chunk_sizes(width=1024, height=1024, max_iter=256):
    print("--- Analyzing Chunk Sizes ---")
    processes_list = [2, 4, 8]  # Test for different P
    chunk_sizes = [1, 2, 5, 10, 25, 50, 100, 200]

    plt.figure(figsize=(10, 6))

    for p in processes_list:
        times = []
        for cs in chunk_sizes:
            t, _ = mandelbrot_multiprocessing(width, height, max_iter, p, cs)
            times.append(t)
            print(f"P={p}, ChunkSize={cs:3d} | Time: {t:.4f} sec")
        plt.plot(chunk_sizes, times, marker='o', label=f'P={p}')

    plt.title('Execution Time vs Chunk Size')
    plt.xlabel('Chunk Size (Number of rows per task)')
    plt.ylabel('Execution Time (Seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig('chunk_size_analysis.png')
    plt.show()

def analyze_speedup(width=1024, height=1024, max_iter=256, optimal_chunk=10):
    print("\n--- Analyzing Speedup and Execution Time ---")
    max_cores = mp.cpu_count()
    processes_list = list(range(1, max_cores + 1))

    times = []

    for p in processes_list:
        t, _ = mandelbrot_multiprocessing(width, height, max_iter, p, optimal_chunk)
        times.append(t)
        print(f"Processes={p:2d} | Time: {t:.4f} sec")

    # Baseline is P=1 (Sequential equivalent via Pool)
    baseline_time = times[0]
    speedups = [baseline_time / t for t in times]

    # Plot Execution Time
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Number of Processes (P)')
    ax1.set_ylabel('Execution Time (s)', color=color)
    ax1.plot(processes_list, times, color=color, marker='o', label="Execution Time")
    ax1.tick_params(axis='y', labelcolor=color)

    # Plot Speedup on a secondary axis
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Speed-up', color=color)
    ax2.plot(processes_list, speedups, color=color, marker='s', label="Actual Speedup")
    ax2.plot(processes_list, processes_list, color='gray', linestyle='--', label="Ideal Speedup")
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Execution Time and Speed-up vs Number of Processes')
    fig.tight_layout()
    ax2.legend(loc='upper left')
    plt.grid(True)
    plt.savefig('speedup_analysis.png')
    plt.show()

if __name__ == '__main__':
    # Adjust resolutions for a quicker test or to match your other files (e.g. 4096)
    RESOLUTION = 1024
    MAX_ITER = 256

    # 1. Run Chunk Size Analysis
    analyze_chunk_sizes(width=RESOLUTION, height=RESOLUTION, max_iter=MAX_ITER)

    # 2. Run Speedup Analysis (Pick the best chunk size observed in the previous step)
    # E.g., if 10 was optimal, pass optimal_chunk=10
    analyze_speedup(width=RESOLUTION, height=RESOLUTION, max_iter=MAX_ITER, optimal_chunk=10)
