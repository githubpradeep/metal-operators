use crate::kmeans::{KMeans, KMeansConfig};
use crate::knn::{KNN, KNNConfig};
use crate::metal::MetalContext;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

static CONTEXT: std::sync::OnceLock<Result<MetalContext, String>> = std::sync::OnceLock::new();

fn get_context() -> PyResult<&'static MetalContext> {
    CONTEXT
        .get_or_init(|| MetalContext::new().map_err(|e| format!("Failed to create MetalContext: {}", e)))
        .as_ref()
        .map_err(|e| PyRuntimeError::new_err(e.clone()))
}

#[pyclass(name = "MetalKMeans")]
pub struct PyMetalKMeans {
    inner: KMeans,
}

#[pymethods]
impl PyMetalKMeans {
    #[new]
    #[pyo3(signature = (n_clusters, max_iterations=100, tolerance=1e-4, seed=42))]
    fn new(n_clusters: usize, max_iterations: usize, tolerance: f32, seed: u64) -> Self {
        let config = KMeansConfig {
            k: n_clusters,
            max_iterations,
            tolerance,
            seed,
            init_centroids: None,
        };
        Self { inner: KMeans::new(config) }
    }

    fn fit(&mut self, data: Vec<f32>, n: usize, d: usize) -> PyResult<()> {
        let ctx = get_context()?;
        self.inner.fit(ctx, &data, n, d).map_err(|e| {
            PyRuntimeError::new_err(format!("KMeans fit failed: {}", e))
        })
    }

    fn predict(&self, data: Vec<f32>, n: usize, d: usize) -> PyResult<Vec<usize>> {
        let ctx = get_context()?;
        self.inner.predict(ctx, &data, n, d).map_err(|e| {
            PyRuntimeError::new_err(format!("KMeans predict failed: {}", e))
        })
    }

    #[getter]
    fn centroids(&self) -> Vec<f32> {
        self.inner.centroids().to_vec()
    }

    #[getter]
    fn labels(&self) -> Vec<usize> {
        self.inner.labels().to_vec()
    }

    #[getter]
    fn inertia(&self) -> f32 {
        self.inner.inertia()
    }

    #[getter]
    fn n_iter(&self) -> usize {
        self.inner.n_iter()
    }
}

#[pyfunction]
#[pyo3(signature = (data, n, d, n_clusters, max_iterations=100, tolerance=1e-4, seed=42))]
pub fn metal_kmeans_fit(
    data: Vec<f32>,
    n: usize,
    d: usize,
    n_clusters: usize,
    max_iterations: usize,
    tolerance: f32,
    seed: u64,
) -> PyResult<(Vec<usize>, Vec<f32>, usize, f32)> {
    let ctx = get_context()?;
    let config = KMeansConfig {
        k: n_clusters,
        max_iterations,
        tolerance,
        seed,
        init_centroids: None,
    };
    let mut km = KMeans::new(config);
    km.fit(ctx, &data, n, d).map_err(|e| {
        PyRuntimeError::new_err(format!("KMeans fit failed: {}", e))
    })?;
    Ok((
        km.labels().to_vec(),
        km.centroids().to_vec(),
        km.n_iter(),
        km.inertia(),
    ))
}

// ── KNN ─────────────────────────────────────────────────────────

#[pyclass(name = "MetalKNeighbors")]
pub struct PyMetalKNeighbors {
    inner: KNN,
}

#[pymethods]
impl PyMetalKNeighbors {
    #[new]
    #[pyo3(signature = (n_neighbors=5))]
    fn new(n_neighbors: usize) -> Self {
        let config = KNNConfig { k: n_neighbors };
        Self { inner: KNN::new(config) }
    }

    fn fit(&mut self, data: Vec<f32>, n: usize, d: usize) -> PyResult<()> {
        let ctx = get_context()?;
        self.inner.fit(ctx, &data, n, d).map_err(|e| {
            PyRuntimeError::new_err(format!("KNeighbors fit failed: {}", e))
        })
    }

    fn kneighbors(
        &self, queries: Vec<f32>, nq: usize,
    ) -> PyResult<(Vec<f32>, Vec<u32>)> {
        let ctx = get_context()?;
        self.inner.kneighbors(ctx, &queries, nq).map_err(|e| {
            PyRuntimeError::new_err(format!("KNeighbors kneighbors failed: {}", e))
        })
    }
}

#[pyfunction]
#[pyo3(signature = (corpus, n_corpus, d, queries, n_queries, n_neighbors=5))]
pub fn metal_kneighbors(
    corpus: Vec<f32>,
    n_corpus: usize,
    d: usize,
    queries: Vec<f32>,
    n_queries: usize,
    n_neighbors: usize,
) -> PyResult<(Vec<f32>, Vec<u32>)> {
    let ctx = get_context()?;
    let config = KNNConfig { k: n_neighbors };
    let mut knn = KNN::new(config);
    knn.fit(ctx, &corpus, n_corpus, d).map_err(|e| {
        PyRuntimeError::new_err(format!("KNeighbors fit failed: {}", e))
    })?;
    knn.kneighbors(ctx, &queries, n_queries).map_err(|e| {
        PyRuntimeError::new_err(format!("KNeighbors kneighbors failed: {}", e))
    })
}
