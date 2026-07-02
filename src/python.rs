use crate::kmeans::{KMeans, KMeansConfig};
use crate::knn::{KNN, KNNConfig};
use crate::metal::MetalContext;
use crate::pca::{PCA, PCAConfig};
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

// ── PCA ─────────────────────────────────────────────────────────

#[pyclass(name = "MetalPCA")]
pub struct PyMetalPCA {
    inner: PCA,
}

#[pymethods]
impl PyMetalPCA {
    #[new]
    fn new(n_components: usize) -> Self {
        let config = PCAConfig { n_components };
        Self { inner: PCA::new(config) }
    }

    fn fit(&mut self, data: Vec<f32>, n: usize, d: usize) -> PyResult<()> {
        let ctx = get_context()?;
        self.inner.fit(ctx, &data, n, d).map_err(|e| {
            PyRuntimeError::new_err(format!("PCA fit failed: {}", e))
        })
    }

    fn transform(&self, data: Vec<f32>, n: usize, d: usize) -> PyResult<Vec<f32>> {
        let ctx = get_context()?;
        self.inner.transform(ctx, &data, n, d).map_err(|e| {
            PyRuntimeError::new_err(format!("PCA transform failed: {}", e))
        })
    }

    fn fit_transform(&mut self, data: Vec<f32>, n: usize, d: usize) -> PyResult<Vec<f32>> {
        let ctx = get_context()?;
        self.inner.fit_transform(ctx, &data, n, d).map_err(|e| {
            PyRuntimeError::new_err(format!("PCA fit_transform failed: {}", e))
        })
    }

    #[getter]
    fn components(&self) -> Vec<f32> {
        self.inner.components().to_vec()
    }

    #[getter]
    fn explained_variance(&self) -> Vec<f32> {
        self.inner.explained_variance().to_vec()
    }

    #[getter]
    fn explained_variance_ratio(&self) -> Vec<f32> {
        self.inner.explained_variance_ratio().to_vec()
    }

    #[getter]
    fn singular_values(&self) -> Vec<f32> {
        self.inner.singular_values().to_vec()
    }

    #[getter]
    fn mean(&self) -> Vec<f32> {
        self.inner.mean().to_vec()
    }

    #[getter]
    fn noise_variance(&self) -> f32 {
        self.inner.noise_variance()
    }
}

#[pyfunction]
#[pyo3(signature = (data, n, d, n_components))]
pub fn metal_pca_fit(
    data: Vec<f32>,
    n: usize,
    d: usize,
    n_components: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let ctx = get_context()?;
    let config = PCAConfig { n_components };
    let mut pca = PCA::new(config);
    pca.fit(ctx, &data, n, d).map_err(|e| {
        PyRuntimeError::new_err(format!("PCA fit failed: {}", e))
    })?;
    Ok((
        pca.components().to_vec(),
        pca.explained_variance().to_vec(),
        pca.explained_variance_ratio().to_vec(),
    ))
}
