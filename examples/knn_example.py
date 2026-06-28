"""KNN example — find nearest neighbours using GPU-accelerated Metal kernels."""
import time
import numpy as np
from metal_kmeans import MetalKNeighbors

def benchmark(corpus_n, query_n, d, k, trials=5):
    corpus = np.random.randn(corpus_n * d).astype(np.float32)
    queries = np.random.randn(query_n * d).astype(np.float32)

    knn = MetalKNeighbors(n_neighbors=k)
    knn.fit(corpus, corpus_n, d)

    # Warmup
    knn.kneighbors(queries, query_n)

    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        dist, idx = knn.kneighbors(queries, query_n)
        times.append(time.perf_counter() - t0)

    avg = np.mean(times) * 1000
    print(f"  {corpus_n:>6} corpus × {query_n:>5} queries × d={d:>2} × k={k:>2}: "
          f"{avg:>7.2f} ms  ({corpus_n * 2 * d / avg / 1e6:.1f} GFLOPS/s)")
    return times

if __name__ == "__main__":
    print("Metal KNN Benchmark\n")
    print(f"{'Corpus':>6} × {'Queries':>5} × dim × k →  {'Time (ms)':>10}  {'Throughput':>10}")
    print("-" * 65)
    for d in [8, 32]:
        for k in [5, 10]:
            benchmark(10_000, 1_000, d, k)
    print("\nDone.")
