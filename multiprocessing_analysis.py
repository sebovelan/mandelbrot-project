import numpy as np
import multiprocessing as mp
import time
import pandas as pd
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
def analyze_chunk_sizes(sizes, max_iter=100):
    print("--- Analyzing Chunk Sizes for Multiple Resolutions ---")
    processes_list = [2, 4, 8]  # Test for different P
    chunk_sizes = [1, 2, 5, 10, 16, 25, 50, 100, 200] # Kept 16 as it's often an optimal sweet spot

    results = []

    for size in sizes:
        print(f"\n--- Testing Size: {size}x{size} ---")
        for p in processes_list:
            for cs in chunk_sizes:
                t, _ = mandelbrot_multiprocessing(size, size, max_iter, p, cs)
                results.append({
                    "size": size,
                    "processes": p,
                    "chunk_size": cs,
                    "execution_time": t
                })
                print(f"Size={size}, P={p}, ChunkSize={cs:3d} | Time: {t:.4f} sec")

    df = pd.DataFrame(results)
    df.to_csv('chunk_analysis_all_sizes.csv', index=False)
    print("\n-> Saved to chunk_analysis_all_sizes.csv")
    return df

def analyze_speedup(sizes, max_iter=100, optimal_chunk=16):
    print("\n--- Analyzing Speedup and Execution Time for Multiple Resolutions ---")
    max_cores = mp.cpu_count()
    processes_list = list(range(1, max_cores + 1))

    results = []

    for size in sizes:
        print(f"\n--- Testing Size: {size}x{size} ---")
        times_for_size = []

        for p in processes_list:
            t, _ = mandelbrot_multiprocessing(size, size, max_iter, p, optimal_chunk)
            times_for_size.append(t)
            print(f"Size={size}, Processes={p:2d} | Time: {t:.4f} sec")

        # Baseline is P=1 (Sequential equivalent via Pool) for this specific size
        baseline_time = times_for_size[0]

        for p, t in zip(processes_list, times_for_size):
            results.append({
                "size": size,
                "processes": p,
                "execution_time": t,
                "actual_speedup": baseline_time / t,
                "ideal_speedup": p
            })

    df = pd.DataFrame(results)
    df.to_csv('speedup_analysis_all_sizes.csv', index=False)
    print("\n-> Saved to speedup_analysis_all_sizes.csv")
    return df

if __name__ == '__main__':
    # All required scaling sizes
    SIZES = [1024, 2048, 4096, 8192]
    MAX_ITER = 100
    OPTIMAL_CHUNK = 16 # Adjust this if chunk analysis reveals a better number

    # 1. Run Chunk Size Analysis across all sizes
    analyze_chunk_sizes(sizes=SIZES, max_iter=MAX_ITER)

    # 2. Run Speedup Analysis across all sizes
    analyze_speedup(sizes=SIZES, max_iter=MAX_ITER, optimal_chunk=OPTIMAL_CHUNK)
