#!/usr/bin/env python3
"""sklearn LogisticRegression timing via stdin/stdout."""
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
import time

def main():
    # Read the header line as text from the binary stream
    header = sys.stdin.buffer.readline().decode().strip()
    parts = header.split()
    n = int(parts[0])
    d = int(parts[1])
    c = float(parts[2])
    lr = float(parts[3])
    mom = float(parts[4])
    max_iter = int(parts[5])
    batch = int(parts[6])
    tol = float(parts[7])
    nbytes = int(parts[8])
    
    # Read remaining bytes as float32 data
    raw = sys.stdin.buffer.read()
    data = np.frombuffer(raw[:n*d*4], dtype=np.float32)
    labels = np.frombuffer(raw[n*d*4:(n*d+n)*4], dtype=np.float32)
    
    data = data.reshape(n, d)
    
    start = time.perf_counter()
    model = LogisticRegression(
        C=c,
        max_iter=max_iter,
        tol=tol,
        solver='lbfgs',
        fit_intercept=True,
        penalty='l2'
    )
    model.fit(data, labels)
    elapsed = time.perf_counter() - start
    
    # Write time in ms as f32 little-endian
    sys.stdout.buffer.write(np.float32(elapsed * 1000.0).tobytes())

if __name__ == '__main__':
    main()
