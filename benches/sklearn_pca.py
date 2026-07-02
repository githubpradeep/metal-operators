#!/usr/bin/env python3
"""Time sklearn PCA from Rust subprocess.

Input (stdin, binary):
  header:  n d k byte_len  (ASCII line)
  data:    n*d float32 bytes

Output (stdout, binary):
  float32:  median fit time in ms across 5 runs
"""
import struct, sys, time, os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import numpy as np
from sklearn.decomposition import PCA

def main():
    line = sys.stdin.buffer.readline()
    n, d, k, byte_len = map(int, line.split())

    raw = sys.stdin.buffer.read(byte_len)
    data = np.frombuffer(raw, dtype=np.float32).reshape(n, d)

    times = []
    for i in range(6):  # 1 warmup + 5 timed
        pca = PCA(n_components=k)
        t0 = time.perf_counter()
        pca.fit(data)
        t = (time.perf_counter() - t0) * 1000.0
        if i > 0:
            times.append(t)

    median_ms = float(np.median(times))
    sys.stdout.buffer.write(struct.pack("f", median_ms))

if __name__ == "__main__":
    main()
