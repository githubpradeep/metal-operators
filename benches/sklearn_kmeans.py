#!/usr/bin/env python3
"""Time sklearn KMeans from Rust subprocess.

Input (stdin, binary):
  header:  n d k max_iter tol byte_len  (ASCII line)
  data:    n*d float32 bytes
  init:    1\n + k*d float32 bytes  (or just \n for no init)

Output (stdout, binary):
  float32:  median time in ms across 5 runs
"""
import struct, sys, time, os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import numpy as np
from sklearn.cluster import KMeans

def main():
    line = sys.stdin.buffer.readline()
    n, d, k, max_iter, tol_i, byte_len = map(int, line.split())
    tol = float(tol_i)

    raw = sys.stdin.buffer.read(byte_len)
    data = np.frombuffer(raw, dtype=np.float32).reshape(n, d)

    init_line = sys.stdin.buffer.readline()
    if init_line.strip() == b"1":
        init_raw = sys.stdin.buffer.read(k * d * 4)
        init = np.frombuffer(init_raw, dtype=np.float32).reshape(k, d)
    else:
        init = None

    times = []
    for i in range(6):  # 1 warmup + 5 timed
        km = KMeans(n_clusters=k, n_init=1, max_iter=max_iter, tol=tol,
                    init=init if init is not None else "k-means++",
                    random_state=0)
        t0 = time.perf_counter()
        km.fit(data)
        t = (time.perf_counter() - t0) * 1000.0
        if i > 0:
            times.append(t)

    median_ms = float(np.median(times))
    sys.stdout.buffer.write(struct.pack("f", median_ms))

if __name__ == "__main__":
    main()
