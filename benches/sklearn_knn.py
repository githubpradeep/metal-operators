#!/usr/bin/env python3
"""KNN benchmark helper: reads query+corpus from stdin, runs sklearn, writes elapsed ms."""

import struct
import sys
import time
import numpy as np
from sklearn.neighbors import NearestNeighbors


def main():
    header = sys.stdin.buffer.read(4 * 5)
    nq, nc, d, k, _npasses = struct.unpack("IIIII", header)

    qbytes = nq * d * 4
    cbytes = nc * d * 4

    queries = np.frombuffer(sys.stdin.buffer.read(qbytes), dtype=np.float32).reshape(nq, d)
    corpus = np.frombuffer(sys.stdin.buffer.read(cbytes), dtype=np.float32).reshape(nc, d)

    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="brute")
    nn.fit(corpus)

    t0 = time.perf_counter()
    nn.kneighbors(queries)
    elapsed = (time.perf_counter() - t0) * 1000  # ms

    sys.stdout.buffer.write(struct.pack("f", elapsed))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
