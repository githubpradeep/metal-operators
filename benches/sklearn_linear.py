#!/usr/bin/env python3
"""sklearn LinearRegression timing via stdin/stdout."""
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
import time


def main():
    # Read the header line as text from the binary stream
    header = sys.stdin.buffer.readline().decode().strip()
    parts = header.split()
    n = int(parts[0])
    d = int(parts[1])
    alpha = float(parts[2])
    fit_intercept = int(parts[3]) == 1
    nbytes = int(parts[4])

    # Read remaining bytes as float32 data
    raw = sys.stdin.buffer.read()
    data = np.frombuffer(raw[: n * d * 4], dtype=np.float32)
    y = np.frombuffer(raw[n * d * 4 : (n * d + n) * 4], dtype=np.float32)

    data = data.reshape(n, d)

    start = time.perf_counter()
    if alpha > 0.0:
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=alpha, fit_intercept=fit_intercept, solver="cholesky")
    else:
        model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(data, y)
    elapsed = time.perf_counter() - start

    # Write time in ms as f32 little-endian
    sys.stdout.buffer.write(np.float32(elapsed * 1000.0).tobytes())


if __name__ == "__main__":
    main()
