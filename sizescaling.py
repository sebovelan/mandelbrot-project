import time
import pandas as pd
from dask.distributed import Client, LocalCluster

# Import your existing functions
from naive import mandelbrot_naive
from numpy_version import mandelbrot_numpy
from njit_version import mandelbrot_numba
from dask_version import run_dask_mandelbrot

def run_benchmarks():
    sizes = [1024, 2048, 4096, 8192] # Left out 8192 to save you from waiting hours on the naive version
    max_iter = 100

    # --- TOGGLE THIS TO TRUE WHEN YOU ARE IN THE GROUP ROOM ---
    TEST_REMOTE_CLUSTER = True
    REMOTE_IP = 'tcp://192.168.0.35:8786' # Replace with your laptop's IP

    results = []

    print("Setting up Local Dask Cluster for benchmarking...")
    # Using 'with' safely manages the startup and shutdown of the cluster
    with LocalCluster() as cluster, Client(cluster) as client:
        print(f"Local Cluster Dashboard: {client.dashboard_link}")

        for size in sizes:
            print(f"\nBenchmarking size: {size}x{size}...")

            start = time.time()
            mandelbrot_naive(size, size, max_iter)
            naive_time = time.time() - start

            start = time.time()
            mandelbrot_numpy(size, size, max_iter)
            numpy_time = time.time() - start

            start = time.time()
            mandelbrot_numba(size, size, max_iter)
            numba_time = time.time() - start

            # Run Local Dask
            start = time.time()
            run_dask_mandelbrot(size, size, max_iter, 4, 4)
            dask_local_time = time.time() - start

            # Default value for remote cluster time
            dask_remote_time = None
            results.append([size, naive_time, numpy_time, numba_time, dask_local_time, dask_remote_time])

    # --- REMOTE CLUSTER BENCHMARKING ---
    if TEST_REMOTE_CLUSTER:
        print(f"\nConnecting to Remote Cluster at {REMOTE_IP}...")
        with Client(REMOTE_IP) as client:
            for idx, size in enumerate(sizes):
                print(f"Benchmarking Remote Dask size: {size}x{size}...")
                start = time.time()
                run_dask_mandelbrot(size, size, max_iter, 4, 4)
                # Update the specific row in our results list with the remote time
                results[idx][5] = time.time() - start

    # Create the DataFrame
    df = pd.DataFrame(results, columns=[
        "size",
        "naive_time",
        "numpy_time",
        "numba_time",
        "dask_local_time",
        "dask_remote_time"
    ])

    # Calculate Speedups
    df["numpy_speedup"] = df["naive_time"] / df["numpy_time"]
    df["numba_speedup"] = df["naive_time"] / df["numba_time"]
    df["dask_local_speedup"] = df["naive_time"] / df["dask_local_time"]

    if TEST_REMOTE_CLUSTER:
        df["dask_remote_speedup"] = df["naive_time"] / df["dask_remote_time"]

    df.to_csv("timings.csv", index=False)
    print("\n--- Final Results ---")
    print(df)

# This block is MANDATORY on Windows to prevent the RuntimeError
if __name__ == '__main__':
    run_benchmarks()
