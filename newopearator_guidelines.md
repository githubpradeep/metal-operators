# New Operator Implementation Guidelines

## Overview
This document provides a complete end-to-end workflow for implementing new GPU-accelerated operators in the metal-operators project. The project follows a consistent pattern for KMeans, KNN, PCA, and LogisticRegression operators.

## Project Structure

```
metal-operators/
├── Cargo.toml                    # Rust package configuration
├── src/
│   ├── metal/                    # Metal GPU context & utilities
│   ├── kmeans/                    # KMeans implementation
│   ├── knn/                       # KNN implementation (3 kernels)
│   ├── pca/                       # PCA implementation
│   ├── logistic_regression/       # Logistic Regression
│   ├── python.rs                 # pyo3 bindings
│   └── lib.rs                    # Public API & module declarations
├── examples/                     # Python smoke tests & benchmarks
├── tests/                        # Unit tests
├── benches/                      # Performance benchmarks
├── reference/                     # Documentation & comparison plots
└── pyproject.toml               # maturin build config
```

## Phase 1: Design & Planning

### 1.1 Understand the Metal GPU Pipeline
```
CPU (Rust) → Metal Shaders → GPU Acceleration → Python API
```

**Key Components:**
- **MetalContext**: Manages GPU device, command queues, and shader compilation
- **Config structs**: Hold operator-specific parameters
- **Operator structs**: Hold state between `fit()` and `predict()/transform()`
- **Shaders**: Metal GPU kernels (in `src/shaders/`)
- **Buffers**: GPU memory for data exchange

### 1.2 Review Existing Patterns

**KMeans Pattern Example:**
```rust
pub struct KMeansConfig {
    pub k: usize,
    pub max_iterations: usize,
    pub tolerance: f32,
    pub seed: u64,
    pub init_centroids: Option<Vec<f32>>,
}

pub struct KMeans {
    config: KMeansConfig,
    centroids: Vec<f32>,
    inertia: f32,
    n_iter: usize,
    labels: Vec<usize>,
}
```

**Key Pattern**: State variables stored on CPU/GPU, accessible via getters.

## Phase 2: Core Implementation

### 2.1 Create Config & Struct Definition

```rust
// src/your_operator/mod.rs
pub struct YourOperatorConfig {
    // Your parameters here
    pub param1: usize,
    pub param2: f32,
    // ... other config
}

impl Default for YourOperatorConfig {
    fn default() -> Self {
        Self {
            param1: 8,
            param2: 1e-4,
            // ...
        }
    }
}

pub struct YourOperator {
    config: YourOperatorConfig,
    // State variables for fitted model
    state_var1: Vec<f32>,
    state_var2: Vec<usize>,
    // ... other state
}
```

### 2.2 Implement Core Methods

**A. Constructor:**
```rust
impl YourOperator {
    pub fn new(config: YourOperatorConfig) -> Self {
        Self {
            config,
            state_var1: Vec::new(),
            state_var2: Vec::new(),
            // ...
        }
    }
}
```

**B. Getters:**
```rust
impl YourOperator {
    pub fn get_state_var1(&self) -> &[f32] { &self.state_var1 }
    pub fn get_state_var2(&self) -> &[usize] { &self.state_var2 }
    // ... other getters
}
```

**C. Main `fit()` Method:**
```rust
pub fn fit(
    &mut self,
    ctx: &MetalContext,
    data: &[f32],
    n: usize,
    d: usize,
) -> anyhow::Result<()> {
    // Validate inputs
    anyhow::ensure!(data.len() == n * d, "Data length mismatch");
    
    // Compile Metal shader
    let pipeline = ctx.compile_kernel(SHADER_SRC, "your_shader"?);
    
    // Create GPU buffers
    let data_buf = ctx.new_buffer(data);
    // ... other buffers
    
    // Execute kernel(s)
    for iteration in 0..self.config.max_iterations {
        // Dispatch workgroups
        // Read results back if needed
    }
    
    // Store results in state
    self.state_var1 = computed_values;
    // ...
    
    Ok(())
}
```

### 2.3 Add Helper Functions

```rust
fn compute_something(data: &[f32], n: usize, d: usize) -> Vec<f32> {
    // CPU-based helper
    (0..n)
        .map(|i| data[i * d..(i + 1) * d].iter().map(|x| x * x).sum())
        .collect()
}

fn pick_kernel(d: usize) -> (&'static str, ComputePipelineState) {
    // Kernel selection logic
}
```

## Phase 3: Metal Shader Development

### 3.1 Create Shader File

```metal
// src/shaders/your_operator.metal
#include <metal_stdlib>
using namespace metal;

// Main computation kernel
kernel void your_operator_kernel(
    device float* data [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& d [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    // Your GPU algorithm here
    float sum = 0.0f;
    for (uint i = 0; i < d; i++) {
        sum += data[gid * d + i] * data[gid * d + i];
    }
    output[gid] = sqrt(sum);
}

// Additional kernels as needed
kernel void your_operator_centroid_update(
    device float* centroids [[buffer(0)]],
    // ... other parameters
) {
    // Update logic
}
```

### 3.2 Shader Compilation

```rust
const SHADER_SRC: &str = include_str!("../../shaders/your_operator.metal");

fn compile_shaders(ctx: &MetalContext) -> anyhow::Result<()> {
    ctx.compile_kernel(SHADER_SRC, "your_operator_kernel"?);
    ctx.compile_kernel(SHADER_SRC, "your_operator_centroid_update"?);
    Ok(())
}
```

## Phase 4: Rust Module Integration

### 4.1 Update Module Structure

```rust
// src/your_operator/mod.rs
use crate::metal::MetalContext;
use metal::*;

// Add to lib.rs
pub mod your_operator;
```

### 4.2 Re-export in Main lib.rs

```rust
// src/lib.rs
#[cfg(feature = "python")]
mod python;

// In python.rs or new bindings file
use crate::your_operator::{YourOperator, YourOperatorConfig};
```

## Phase 5: Python Bindings

### 5.1 Add Python Struct

```rust
// In python.rs or new file
#[pyclass(name = "MetalYourOperator")]
pub struct PyMetalYourOperator {
    inner: YourOperator,
}

#[pymethods]
impl PyMetalYourOperator {
    #[new]
    #[pyo3(signature = (param1, param2=1e-4))]
    fn new(param1: usize, param2: f32) -> Self {
        let config = YourOperatorConfig { param1, param2 };
        Self { inner: YourOperator::new(config) }
    }
    
    fn fit(&mut self, data: Vec<f32>, n: usize, d: usize) -> PyResult<()> {
        let ctx = get_context()?;
        self.inner.fit(ctx, &data, n, d)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
    
    // Add other methods (predict, transform, etc.)
    
    #[getter]
    fn some_state(&self) -> Vec<f32> {
        self.inner.get_state_var1().to_vec()
    }
}
```

### 5.2 Add Functional API

```rust
#[pyfunction]
#[pyo3(signature = (data, n, d, param1, param2=1e-4))]
pub fn your_operator_fit(
    data: Vec<f32>,
    n: usize,
    d: usize,
    param1: usize,
    param2: f32,
) -> PyResult<(Vec<f32>, Vec<usize>)> { // Adjust return type
    let ctx = get_context()?;
    let config = YourOperatorConfig { param1, param2 };
    let mut op = YourOperator::new(config);
    op.fit(ctx, &data, n, d)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    
    Ok((op.get_state_var1().to_vec(), op.get_state_var2().to_vec()))
}
```

### 5.3 Update Module Exports

```rust
// In lib.rs - add new exports
#[cfg(feature = "python")]
pub use python::PyMetalYourOperator;
#[cfg(feature = "python")]
pub use python::your_operator_fit;
```

## Phase 6: Testing & Verification

### 6.1 Write Unit Tests

```rust
// tests/your_operator_test.rs
use super::*;

#[test]
fn test_basic_fit() {
    let config = YourOperatorConfig { param1: 4, param2: 1e-4 };
    let mut op = YourOperator::new(config);
    
    let n = 100;
    let d = 2;
    let mut data = vec![0.0; n * d];
    for i in 0..n {
        data[i * d] = i as f32;
        data[i * d + 1] = (i * 2) as f32;
    }
    
    let ctx = MetalContext::new().unwrap();
    op.fit(&ctx, &data, n, d).unwrap();
    
    // Verify results
    assert_eq!(op.get_state_var1().len(), expected_len);
}
```

### 6.2 Add Performance Benchmarks

```rust
// benches/your_operator_benchmark.rs
use criterion::{black_box, Criterion};
use crate::your_operator::{YourOperator, YourOperatorConfig};

fn bench_fit(c: &mut Criterion) {
    let config = YourOperatorConfig { param1: 8, param2: 1e-4 };
    let mut op = YourOperator::new(config);
    let ctx = MetalContext::new().unwrap();
    
    c.bench_function("YourOperator fit", |b| {
        let n = 1000;
        let d = 32;
        let mut data = vec![0.0; n * d];
        
        b.iter(|| {
            let mut data_clone = data.clone();
            op.fit(&ctx, &mut data_clone, n, d).unwrap();
        });
    });
}
```

## Phase 7: Documentation & Integration

### 7.1 Update README

```markdown
## Your New Operator

Supports GPU-accelerated [brief description].

### Python API

```python
# Based on actual patterns from metal_kmeans, metal_kneighbors, metal_pca

# Functional API (matches metal_kmeans_fit, metal_kneighbors, metal_pca_fit)
from metal_your_operator import metal_your_operator
results = metal_your_operator(data, n, d, param1=8, param2=1e-4)

# sklearn-style API (matches MetalKMeans, MetalKNeighbors, MetalPCA)
from metal_your_operator import MetalYourOperator

# Initialize with parameters
op = MetalYourOperator(param1=8, param2=1e-4)

# Fit the model
op.fit(data, n, d)  # data can be numpy array, accepts raveled list

# Predict on new data
predictions = op.predict(new_data, n_new, d)

# Access model state
state_var1 = op.state_var1  # matches centroids, components_, etc.
state_var2 = op.state_var2  # matches labels, indices, etc.
```

### 7.2 Example Script

```python
// examples/your_operator_example.py
"""metal_your_operator example: realistic use case with smoke test and benchmark."""

import numpy as np
from metal_your_operator import metal_your_operator, MetalYourOperator


def smoke_test():
    """Small example demonstrating typical usage."""
    # Create synthetic 3D data with clear clusters (similar to example.py pattern)
    rng = np.random.RandomState(42)
    n, d, k = 500, 3, 3
    
    # Generate clustered data (similar to metal_kmeans example.py)
    data = np.vstack([
        rng.randn(200, d) + [5, 5, 5],    # Cluster 1
        rng.randn(200, d) + [-5, -5, -5], # Cluster 2  
        rng.randn(100, d) + [0, 0, 0],    # Cluster 3
    ]).astype(np.float32)
    n, d = data.shape
    
    # ── Functional API ──
    results = metal_your_operator(
        data.ravel().tolist(), n, d, 
        param1=k, param2=1e-4
    )
    print("[functional] results:", results)
    
    # ── sklearn-style API ──
    op = MetalYourOperator(param1=k, param2=1e-4)
    op.fit(data, n, d)
    print("[sklearn] model fitted, state_var1 shape:", len(op.state_var1))
    print("[sklearn] state_var2 shape:", len(op.state_var2))
    
    # Make predictions on new data
    new_points = np.array([[1., 1., 1.], [6., 6., 6.], [-2., -2., -2.]], dtype=np.float32)
    predictions = op.predict(new_points.ravel().tolist(), 3, d)
    print("  predictions for new points:", predictions)


def benchmark():
    """Larger scale performance test (similar to example.py benchmark)."""
    n, d, k = 50_000, 64, 128
    rng = np.random.RandomState(7)
    data = rng.randn(n, d).astype(np.float32)
    
    import time
    op = MetalYourOperator(param1=k, param2=1e-4)
    t0 = time.perf_counter()
    results = op.fit(data, n, d)
    elapsed = time.perf_counter() - t0
    
    print("\n[benchmark]  {}×{} k={}  {:.0f} ms".format(
        n, d, k, elapsed * 1000))


if __name__ == "__main__":
    print("=" * 50)
    print("metal_your_operator example")
    print("=" * 50)
    smoke_test()
    benchmark()
```

## Phase 8: Build & Test

```bash
# Build with Python bindings (library check only; see note below)
cargo build --features python

# Run tests
cargo test

# Run benchmarks
cargo bench

# Python smoke test
cd examples
python3 your_operator_example.py
```

> **Note on `cargo build --features python`:** because the crate uses the
> `pyo3/extension-module` feature, Python symbols are resolved at import time
> inside the interpreter, not at link time. A plain `cargo build --features python`
> may fail at the *linking* step (`cc` / undefined symbols for arm64) — this is
> expected and does NOT indicate a code error. To build and install the Python
> extension, always use maturin:
>
> ```bash
> pip install maturin
> source .venv/bin/activate
> maturin develop          # builds wheel + installs into the venv
> ```

## Common Pitfalls & Solutions

1. **Memory leaks**: Use `Drop` implementations for Metal resources
2. **Synchronization**: Always `wait_until_completed()` after GPU work
3. **Thread safety**: Use `Mutex` for shared resources (like KNN does)
4. **Error handling**: Use `anyhow::Result` consistently
5. **Shader debugging**: Add print statements or use Xcode Metal debugger

## Next Steps After Implementation

1. **Integration**: Add to `pyproject.toml` maturin build
2. **Documentation**: Update `README.md` and algorithm docs
3. **Performance tuning**: Optimize workroup sizes and memory usage
4. **Edge cases**: Handle empty data, invalid parameters, large dimensions
5. **Feature flags**: Wrap expensive operations behind optional features

## Key Files to Modify

- `src/your_operator/mod.rs` - Core implementation
- `src/shaders/your_operator.metal` - Metal shader code
- `src/python.rs` - Python bindings (or new file)
- `src/lib.rs` - Module declarations
- `Cargo.toml` - Add dependencies if needed
- `examples/your_operator_example.py` - Smoke test
- `tests/your_operator_test.rs` - Unit tests
- `benches/your_operator_benchmark.rs` - Performance benchmarks

## Validation Checklist

- [ ] Library compiles: `cargo build`
- [ ] All unit tests pass: `cargo test`
- [ ] Performance benchmarks work: `cargo bench`
- [ ] Python bindings build & import: `maturin develop`, then `python3 -c "import metal_kmeans"`
- [ ] Performance benchmarks work
- [ ] Python bindings import correctly
- [ ] Example script runs successfully
- [ ] Shader compilation works on target hardware
- [ ] Memory usage is reasonable
- [ ] Error handling is robust
- [ ] Documentation is complete

This workflow follows the exact patterns established in the existing KMeans, KNN, and PCA operators, ensuring consistency with the rest of the codebase while providing a clear path from concept to production-ready GPU-accelerated Python operator.