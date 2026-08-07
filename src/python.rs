use crate::dbscan::{DBSCANConfig, DBSCAN};
use crate::kmeans::{KMeans, KMeansConfig};
use crate::knn::{KNNConfig, KNN};
use crate::lda::{LDAConfig, LDA};
use crate::linear_regression::{LinearRegression, LinearRegressionConfig};
use crate::logistic_regression::{LogisticRegression, LogisticRegressionConfig};
use crate::metal::MetalContext;
use crate::naive_bayes::{GaussianNB, GaussianNBConfig};
use crate::nmf::{NMFConfig, NMF};
use crate::pca::{PCAConfig, PCA};
use crate::tsne::{TSNEConfig, TSNE};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

static CONTEXT: std::sync::OnceLock<Result<MetalContext, String>> = std::sync::OnceLock::new();

/// Convert raw little-endian float32 bytes into a `Vec<f32>` (fast memcpy;
/// used by the buffer-protocol `_bytes` bindings).
fn bytes_to_vec_f32(buf: &[u8], what: &str) -> PyResult<Vec<f32>> {
    if buf.len() % 4 != 0 {
        return Err(PyValueError::new_err(format!(
            "{} byte length must be a multiple of 4 (float32), got {}",
            what,
            buf.len()
        )));
    }
    let mut v = vec![0.0f32; buf.len() / 4];
    unsafe {
        std::ptr::copy_nonoverlapping(buf.as_ptr(), v.as_mut_ptr() as *mut u8, buf.len());
    }
    Ok(v)
}

fn get_context() -> PyResult<&'static MetalContext> {
    CONTEXT
        .get_or_init(|| {
            MetalContext::new().map_err(|e| format!("Failed to create MetalContext: {}", e))
        })
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
        Self {
            inner: KMeans::new(config),
        }
    }

    fn fit(&mut self, data: Vec<f32>, n: usize, d: usize) -> PyResult<()> {
        let ctx = get_context()?;
        self.inner
            .fit(ctx, &data, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("KMeans fit failed: {}", e)))
    }

    fn predict(&self, data: Vec<f32>, n: usize, d: usize) -> PyResult<Vec<usize>> {
        let ctx = get_context()?;
        self.inner
            .predict(ctx, &data, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("KMeans predict failed: {}", e)))
    }

    // ── Bytes-based variants (zero-copy for numpy inputs) ──

    fn fit_bytes(&mut self, data: &[u8], n: usize, d: usize) -> PyResult<()> {
        let data = bytes_to_vec_f32(data, "data")?;
        let ctx = get_context()?;
        self.inner
            .fit(ctx, &data, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("KMeans fit failed: {}", e)))
    }

    fn predict_bytes(&self, data: &[u8], n: usize, d: usize) -> PyResult<Vec<usize>> {
        let data = bytes_to_vec_f32(data, "data")?;
        let ctx = get_context()?;
        self.inner
            .predict(ctx, &data, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("KMeans predict failed: {}", e)))
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
    km.fit(ctx, &data, n, d)
        .map_err(|e| PyRuntimeError::new_err(format!("KMeans fit failed: {}", e)))?;
    Ok((
        km.labels().to_vec(),
        km.centroids().to_vec(),
        km.n_iter(),
        km.inertia(),
    ))
}

#[pyfunction]
#[pyo3(signature = (data, n, d, n_clusters, max_iterations=100, tolerance=1e-4, seed=42))]
pub fn metal_kmeans_fit_bytes(
    data: &[u8],
    n: usize,
    d: usize,
    n_clusters: usize,
    max_iterations: usize,
    tolerance: f32,
    seed: u64,
) -> PyResult<(Vec<usize>, Vec<f32>, usize, f32)> {
    let data = bytes_to_vec_f32(data, "data")?;
    let ctx = get_context()?;
    let config = KMeansConfig {
        k: n_clusters,
        max_iterations,
        tolerance,
        seed,
        init_centroids: None,
    };
    let mut km = KMeans::new(config);
    km.fit(ctx, &data, n, d)
        .map_err(|e| PyRuntimeError::new_err(format!("KMeans fit failed: {}", e)))?;
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
        Self {
            inner: KNN::new(config),
        }
    }

    fn fit(&mut self, data: Vec<f32>, n: usize, d: usize) -> PyResult<()> {
        let ctx = get_context()?;
        self.inner
            .fit(ctx, &data, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("KNeighbors fit failed: {}", e)))
    }

    fn kneighbors(&self, queries: Vec<f32>, nq: usize) -> PyResult<(Vec<f32>, Vec<u32>)> {
        let ctx = get_context()?;
        self.inner
            .kneighbors(ctx, &queries, nq)
            .map_err(|e| PyRuntimeError::new_err(format!("KNeighbors kneighbors failed: {}", e)))
    }

    // ── Bytes-based variants (zero-copy for numpy inputs) ──

    fn fit_bytes(&mut self, data: &[u8], n: usize, d: usize) -> PyResult<()> {
        let data = bytes_to_vec_f32(data, "data")?;
        let ctx = get_context()?;
        self.inner
            .fit(ctx, &data, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("KNeighbors fit failed: {}", e)))
    }

    fn kneighbors_bytes(&self, queries: &[u8], nq: usize) -> PyResult<(Vec<f32>, Vec<u32>)> {
        let queries = bytes_to_vec_f32(queries, "queries")?;
        let ctx = get_context()?;
        self.inner
            .kneighbors(ctx, &queries, nq)
            .map_err(|e| PyRuntimeError::new_err(format!("KNeighbors kneighbors failed: {}", e)))
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
    knn.fit(ctx, &corpus, n_corpus, d)
        .map_err(|e| PyRuntimeError::new_err(format!("KNeighbors fit failed: {}", e)))?;
    knn.kneighbors(ctx, &queries, n_queries)
        .map_err(|e| PyRuntimeError::new_err(format!("KNeighbors kneighbors failed: {}", e)))
}

#[pyfunction]
#[pyo3(signature = (corpus, n_corpus, d, queries, n_queries, n_neighbors=5))]
pub fn metal_kneighbors_bytes(
    corpus: &[u8],
    n_corpus: usize,
    d: usize,
    queries: &[u8],
    n_queries: usize,
    n_neighbors: usize,
) -> PyResult<(Vec<f32>, Vec<u32>)> {
    let corpus = bytes_to_vec_f32(corpus, "corpus")?;
    let queries = bytes_to_vec_f32(queries, "queries")?;
    let ctx = get_context()?;
    let config = KNNConfig { k: n_neighbors };
    let mut knn = KNN::new(config);
    knn.fit(ctx, &corpus, n_corpus, d)
        .map_err(|e| PyRuntimeError::new_err(format!("KNeighbors fit failed: {}", e)))?;
    knn.kneighbors(ctx, &queries, n_queries)
        .map_err(|e| PyRuntimeError::new_err(format!("KNeighbors kneighbors failed: {}", e)))
}

// ── DBSCAN ─────────────────────────────────────────────────────

#[pyclass(name = "MetalDBSCAN")]
pub struct PyMetalDBSCAN {
    inner: DBSCAN,
}

#[pymethods]
impl PyMetalDBSCAN {
    #[new]
    #[pyo3(signature = (eps=0.5, min_samples=5))]
    fn new(eps: f32, min_samples: usize) -> Self {
        let config = DBSCANConfig { eps, min_samples };
        Self {
            inner: DBSCAN::new(config),
        }
    }

    fn fit(&mut self, data: Vec<f32>, n: usize, d: usize) -> PyResult<()> {
        let ctx = get_context()?;
        self.inner
            .fit(ctx, &data, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("DBSCAN fit failed: {}", e)))
    }

    fn fit_bytes(&mut self, data: &[u8], n: usize, d: usize) -> PyResult<()> {
        let data = bytes_to_vec_f32(data, "data")?;
        let ctx = get_context()?;
        self.inner
            .fit(ctx, &data, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("DBSCAN fit failed: {}", e)))
    }

    #[getter]
    fn labels(&self) -> Vec<isize> {
        self.inner.labels().to_vec()
    }

    #[getter]
    fn n_clusters(&self) -> usize {
        self.inner.n_clusters()
    }
}

#[pyfunction]
#[pyo3(signature = (data, n, d, eps=0.5, min_samples=5))]
pub fn metal_dbscan_fit(
    data: Vec<f32>,
    n: usize,
    d: usize,
    eps: f32,
    min_samples: usize,
) -> PyResult<(Vec<isize>, usize)> {
    let ctx = get_context()?;
    let config = DBSCANConfig { eps, min_samples };
    let mut db = DBSCAN::new(config);
    db.fit(ctx, &data, n, d)
        .map_err(|e| PyRuntimeError::new_err(format!("DBSCAN fit failed: {}", e)))?;
    Ok((db.labels().to_vec(), db.n_clusters()))
}

#[pyfunction]
#[pyo3(signature = (data, n, d, eps=0.5, min_samples=5))]
pub fn metal_dbscan_fit_bytes(
    data: &[u8],
    n: usize,
    d: usize,
    eps: f32,
    min_samples: usize,
) -> PyResult<(Vec<isize>, usize)> {
    let data = bytes_to_vec_f32(data, "data")?;
    let ctx = get_context()?;
    let config = DBSCANConfig { eps, min_samples };
    let mut db = DBSCAN::new(config);
    db.fit(ctx, &data, n, d)
        .map_err(|e| PyRuntimeError::new_err(format!("DBSCAN fit failed: {}", e)))?;
    Ok((db.labels().to_vec(), db.n_clusters()))
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
        Self {
            inner: PCA::new(config),
        }
    }

    fn fit(&mut self, data: Vec<f32>, n: usize, d: usize) -> PyResult<()> {
        let ctx = get_context()?;
        self.inner
            .fit(ctx, &data, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("PCA fit failed: {}", e)))
    }

    fn transform(&self, data: Vec<f32>, n: usize, d: usize) -> PyResult<Vec<f32>> {
        let ctx = get_context()?;
        self.inner
            .transform(ctx, &data, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("PCA transform failed: {}", e)))
    }

    fn fit_transform(&mut self, data: Vec<f32>, n: usize, d: usize) -> PyResult<Vec<f32>> {
        let ctx = get_context()?;
        self.inner
            .fit_transform(ctx, &data, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("PCA fit_transform failed: {}", e)))
    }

    // ── Bytes-based variants (zero-copy for numpy inputs) ──

    fn fit_bytes(&mut self, data: &[u8], n: usize, d: usize) -> PyResult<()> {
        let data = bytes_to_vec_f32(data, "data")?;
        let ctx = get_context()?;
        self.inner
            .fit(ctx, &data, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("PCA fit failed: {}", e)))
    }

    fn transform_bytes(&self, data: &[u8], n: usize, d: usize) -> PyResult<Vec<f32>> {
        let data = bytes_to_vec_f32(data, "data")?;
        let ctx = get_context()?;
        self.inner
            .transform(ctx, &data, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("PCA transform failed: {}", e)))
    }

    fn fit_transform_bytes(&mut self, data: &[u8], n: usize, d: usize) -> PyResult<Vec<f32>> {
        let data = bytes_to_vec_f32(data, "data")?;
        let ctx = get_context()?;
        self.inner
            .fit_transform(ctx, &data, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("PCA fit_transform failed: {}", e)))
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
    pca.fit(ctx, &data, n, d)
        .map_err(|e| PyRuntimeError::new_err(format!("PCA fit failed: {}", e)))?;
    Ok((
        pca.components().to_vec(),
        pca.explained_variance().to_vec(),
        pca.explained_variance_ratio().to_vec(),
    ))
}

#[pyfunction]
#[pyo3(signature = (data, n, d, n_components))]
pub fn metal_pca_fit_bytes(
    data: &[u8],
    n: usize,
    d: usize,
    n_components: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let data = bytes_to_vec_f32(data, "data")?;
    let ctx = get_context()?;
    let config = PCAConfig { n_components };
    let mut pca = PCA::new(config);
    pca.fit(ctx, &data, n, d)
        .map_err(|e| PyRuntimeError::new_err(format!("PCA fit failed: {}", e)))?;
    Ok((
        pca.components().to_vec(),
        pca.explained_variance().to_vec(),
        pca.explained_variance_ratio().to_vec(),
    ))
}

// ── NMF (Non-negative Matrix Factorization) ─────────────────────

#[pyclass(name = "MetalNMF")]
pub struct PyMetalNMF {
    inner: NMF,
}

#[pymethods]
impl PyMetalNMF {
    #[new]
    #[pyo3(signature = (n_components=2, max_iterations=200, tolerance=1e-4, seed=42))]
    fn new(n_components: usize, max_iterations: usize, tolerance: f32, seed: u64) -> Self {
        let config = NMFConfig {
            n_components,
            max_iterations,
            tolerance,
            seed,
            ..Default::default()
        };
        Self {
            inner: NMF::new(config),
        }
    }

    fn fit(&mut self, data: Vec<f32>, n: usize, d: usize) -> PyResult<()> {
        let ctx = get_context()?;
        self.inner
            .fit(ctx, &data, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("NMF fit failed: {}", e)))
    }

    fn fit_bytes(&mut self, data: &[u8], n: usize, d: usize) -> PyResult<()> {
        let data = bytes_to_vec_f32(data, "data")?;
        let ctx = get_context()?;
        self.inner
            .fit(ctx, &data, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("NMF fit failed: {}", e)))
    }

    fn transform(&mut self, data: Vec<f32>, n: usize, d: usize) -> PyResult<Vec<f32>> {
        let ctx = get_context()?;
        self.inner
            .transform(ctx, &data, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("NMF transform failed: {}", e)))
    }

    fn transform_bytes(&mut self, data: &[u8], n: usize, d: usize) -> PyResult<Vec<f32>> {
        let data = bytes_to_vec_f32(data, "data")?;
        let ctx = get_context()?;
        self.inner
            .transform(ctx, &data, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("NMF transform failed: {}", e)))
    }

    #[getter]
    fn components(&self) -> Vec<f32> {
        self.inner.components().to_vec()
    }

    #[getter]
    fn coeff(&self) -> Vec<f32> {
        self.inner.coeff().to_vec()
    }

    #[getter]
    fn reconstruction_error(&self) -> f32 {
        self.inner.reconstruction_error()
    }

    #[getter]
    fn n_iter(&self) -> usize {
        self.inner.n_iter()
    }
}

#[pyfunction]
#[pyo3(signature = (data, n, d, n_components, max_iterations=200, tolerance=1e-4, seed=42))]
pub fn metal_nmf_fit(
    data: Vec<f32>,
    n: usize,
    d: usize,
    n_components: usize,
    max_iterations: usize,
    tolerance: f32,
    seed: u64,
) -> PyResult<(Vec<f32>, Vec<f32>, f32, usize)> {
    let ctx = get_context()?;
    let config = NMFConfig {
        n_components,
        max_iterations,
        tolerance,
        seed,
        ..Default::default()
    };
    let mut nmf = NMF::new(config);
    nmf.fit(ctx, &data, n, d)
        .map_err(|e| PyRuntimeError::new_err(format!("NMF fit failed: {}", e)))?;
    Ok((
        nmf.components().to_vec(),
        nmf.coeff().to_vec(),
        nmf.reconstruction_error(),
        nmf.n_iter(),
    ))
}

#[pyfunction]
#[pyo3(signature = (data, n, d, n_components, max_iterations=200, tolerance=1e-4, seed=42))]
pub fn metal_nmf_fit_bytes(
    data: &[u8],
    n: usize,
    d: usize,
    n_components: usize,
    max_iterations: usize,
    tolerance: f32,
    seed: u64,
) -> PyResult<(Vec<f32>, Vec<f32>, f32, usize)> {
    let data = bytes_to_vec_f32(data, "data")?;
    let ctx = get_context()?;
    let config = NMFConfig {
        n_components,
        max_iterations,
        tolerance,
        seed,
        ..Default::default()
    };
    let mut nmf = NMF::new(config);
    nmf.fit(ctx, &data, n, d)
        .map_err(|e| PyRuntimeError::new_err(format!("NMF fit failed: {}", e)))?;
    Ok((
        nmf.components().to_vec(),
        nmf.coeff().to_vec(),
        nmf.reconstruction_error(),
        nmf.n_iter(),
    ))
}

// ── Linear Discriminant Analysis (LDA) ───────────────────────────

#[pyclass(name = "MetalLDA")]
pub struct PyMetalLDA {
    inner: LDA,
}

#[pymethods]
impl PyMetalLDA {
    #[new]
    fn new(n_components: usize) -> Self {
        let config = LDAConfig { n_components };
        Self {
            inner: LDA::new(config),
        }
    }

    fn fit(&mut self, data: Vec<f32>, labels: Vec<f32>, n: usize, d: usize) -> PyResult<()> {
        let ctx = get_context()?;
        self.inner
            .fit(ctx, &data, &labels, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("LDA fit failed: {}", e)))
    }

    fn transform(&self, data: Vec<f32>, n: usize, d: usize) -> PyResult<Vec<f32>> {
        let ctx = get_context()?;
        self.inner
            .transform(ctx, &data, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("LDA transform failed: {}", e)))
    }

    fn fit_transform(
        &mut self,
        data: Vec<f32>,
        labels: Vec<f32>,
        n: usize,
        d: usize,
    ) -> PyResult<Vec<f32>> {
        let ctx = get_context()?;
        self.inner
            .fit_transform(ctx, &data, &labels, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("LDA fit_transform failed: {}", e)))
    }

    fn predict(&self, data: Vec<f32>, n: usize, d: usize) -> PyResult<Vec<usize>> {
        let ctx = get_context()?;
        self.inner
            .predict(ctx, &data, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("LDA predict failed: {}", e)))
    }

    fn score(&self, data: Vec<f32>, labels: Vec<f32>, n: usize, d: usize) -> PyResult<f32> {
        let ctx = get_context()?;
        self.inner
            .score(ctx, &data, &labels, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("LDA score failed: {}", e)))
    }

    // ── Bytes-based variants (zero-copy for numpy inputs) ──

    fn fit_bytes(&mut self, data: &[u8], labels: &[u8], n: usize, d: usize) -> PyResult<()> {
        let data = bytes_to_vec_f32(data, "data")?;
        let labels = bytes_to_vec_f32(labels, "labels")?;
        let ctx = get_context()?;
        self.inner
            .fit(ctx, &data, &labels, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("LDA fit failed: {}", e)))
    }

    fn transform_bytes(&self, data: &[u8], n: usize, d: usize) -> PyResult<Vec<f32>> {
        let data = bytes_to_vec_f32(data, "data")?;
        let ctx = get_context()?;
        self.inner
            .transform(ctx, &data, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("LDA transform failed: {}", e)))
    }

    fn fit_transform_bytes(
        &mut self,
        data: &[u8],
        labels: &[u8],
        n: usize,
        d: usize,
    ) -> PyResult<Vec<f32>> {
        let data = bytes_to_vec_f32(data, "data")?;
        let labels = bytes_to_vec_f32(labels, "labels")?;
        let ctx = get_context()?;
        self.inner
            .fit_transform(ctx, &data, &labels, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("LDA fit_transform failed: {}", e)))
    }

    fn predict_bytes(&self, data: &[u8], n: usize, d: usize) -> PyResult<Vec<usize>> {
        let data = bytes_to_vec_f32(data, "data")?;
        let ctx = get_context()?;
        self.inner
            .predict(ctx, &data, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("LDA predict failed: {}", e)))
    }

    fn score_bytes(&self, data: &[u8], labels: &[u8], n: usize, d: usize) -> PyResult<f32> {
        let data = bytes_to_vec_f32(data, "data")?;
        let labels = bytes_to_vec_f32(labels, "labels")?;
        let ctx = get_context()?;
        self.inner
            .score(ctx, &data, &labels, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("LDA score failed: {}", e)))
    }

    #[getter]
    fn scalings(&self) -> Vec<f32> {
        self.inner.scalings().to_vec()
    }

    #[getter]
    fn coeffs(&self) -> Vec<f32> {
        self.inner.coeffs().to_vec()
    }

    #[getter]
    fn mean(&self) -> Vec<f32> {
        self.inner.mean().to_vec()
    }

    #[getter]
    fn eigenvalues(&self) -> Vec<f32> {
        self.inner.eigenvalues().to_vec()
    }

    #[getter]
    fn classes(&self) -> Vec<f32> {
        self.inner.classes().to_vec()
    }

    #[getter]
    fn class_counts(&self) -> Vec<f32> {
        self.inner.class_counts().to_vec()
    }

    #[getter]
    fn class_means(&self) -> Vec<f32> {
        self.inner.class_means().to_vec()
    }

    #[getter]
    fn n_components(&self) -> usize {
        self.inner.n_components()
    }
}

#[pyfunction]
#[pyo3(signature = (data, labels, n, d, n_components))]
pub fn metal_lda_fit(
    data: Vec<f32>,
    labels: Vec<f32>,
    n: usize,
    d: usize,
    n_components: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let ctx = get_context()?;
    let config = LDAConfig { n_components };
    let mut lda = LDA::new(config);
    lda.fit(ctx, &data, &labels, n, d)
        .map_err(|e| PyRuntimeError::new_err(format!("LDA fit failed: {}", e)))?;
    Ok((
        lda.scalings().to_vec(),
        lda.eigenvalues().to_vec(),
        lda.classes().to_vec(),
    ))
}

#[pyfunction]
#[pyo3(signature = (data, labels, n, d, n_components))]
pub fn metal_lda_fit_bytes(
    data: &[u8],
    labels: &[u8],
    n: usize,
    d: usize,
    n_components: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let data = bytes_to_vec_f32(data, "data")?;
    let labels = bytes_to_vec_f32(labels, "labels")?;
    let ctx = get_context()?;
    let config = LDAConfig { n_components };
    let mut lda = LDA::new(config);
    lda.fit(ctx, &data, &labels, n, d)
        .map_err(|e| PyRuntimeError::new_err(format!("LDA fit failed: {}", e)))?;
    Ok((
        lda.scalings().to_vec(),
        lda.eigenvalues().to_vec(),
        lda.classes().to_vec(),
    ))
}

// ── Logistic Regression ─────────────────────────────────────────

#[pyclass(name = "MetalLogisticRegression")]
pub struct PyMetalLogisticRegression {
    inner: LogisticRegression,
}

#[pymethods]
impl PyMetalLogisticRegression {
    #[new]
    #[pyo3(signature = (c=1.0, learning_rate=0.01, momentum=0.9, max_epochs=100, batch_size=256, tol=1e-4, seed=42))]
    fn new(
        c: f32,
        learning_rate: f32,
        momentum: f32,
        max_epochs: usize,
        batch_size: usize,
        tol: f32,
        seed: u64,
    ) -> Self {
        let config = LogisticRegressionConfig {
            c,
            learning_rate,
            momentum,
            max_epochs,
            batch_size,
            tol,
            seed,
        };
        Self {
            inner: LogisticRegression::new(config),
        }
    }

    fn fit(&mut self, data: Vec<f32>, y: Vec<f32>, n: usize, d: usize) -> PyResult<()> {
        let ctx = get_context()?;
        self.inner
            .fit(ctx, &data, &y, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("LogisticRegression fit failed: {}", e)))
    }

    fn predict(&self, data: Vec<f32>, n: usize, d: usize) -> PyResult<Vec<f32>> {
        let ctx = get_context()?;
        self.inner.predict(ctx, &data, n, d).map_err(|e| {
            PyRuntimeError::new_err(format!("LogisticRegression predict failed: {}", e))
        })
    }

    fn predict_proba(&self, data: Vec<f32>, n: usize, d: usize) -> PyResult<Vec<f32>> {
        let ctx = get_context()?;
        self.inner.predict_proba(ctx, &data, n, d).map_err(|e| {
            PyRuntimeError::new_err(format!("LogisticRegression predict_proba failed: {}", e))
        })
    }

    fn score(&self, data: Vec<f32>, y: Vec<f32>, n: usize, d: usize) -> PyResult<f32> {
        let ctx = get_context()?;
        self.inner
            .score(ctx, &data, &y, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("LogisticRegression score failed: {}", e)))
    }

    // ── Bytes-based variants (zero-copy for numpy inputs) ──
    // Accept the raw little-endian float32 bytes via the buffer protocol
    // (a contiguous numpy float32 array passes through without conversion),
    // avoiding the O(n) Python-list roundtrip that `Vec<f32>` arguments pay.

    fn fit_bytes(&mut self, data: &[u8], y: &[u8], n: usize, d: usize) -> PyResult<()> {
        let data = bytes_to_vec_f32(data, "data")?;
        let y = bytes_to_vec_f32(y, "y")?;
        let ctx = get_context()?;
        self.inner
            .fit(ctx, &data, &y, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("LogisticRegression fit failed: {}", e)))
    }

    fn predict_bytes(&self, data: &[u8], n: usize, d: usize) -> PyResult<Vec<f32>> {
        let data = bytes_to_vec_f32(data, "data")?;
        let ctx = get_context()?;
        self.inner.predict(ctx, &data, n, d).map_err(|e| {
            PyRuntimeError::new_err(format!("LogisticRegression predict failed: {}", e))
        })
    }

    fn predict_proba_bytes(&self, data: &[u8], n: usize, d: usize) -> PyResult<Vec<f32>> {
        let data = bytes_to_vec_f32(data, "data")?;
        let ctx = get_context()?;
        self.inner.predict_proba(ctx, &data, n, d).map_err(|e| {
            PyRuntimeError::new_err(format!("LogisticRegression predict_proba failed: {}", e))
        })
    }

    fn score_bytes(&self, data: &[u8], y: &[u8], n: usize, d: usize) -> PyResult<f32> {
        let data = bytes_to_vec_f32(data, "data")?;
        let y = bytes_to_vec_f32(y, "y")?;
        let ctx = get_context()?;
        self.inner
            .score(ctx, &data, &y, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("LogisticRegression score failed: {}", e)))
    }

    #[getter]
    fn weights(&self) -> Vec<f32> {
        self.inner.weights().to_vec()
    }

    #[getter]
    fn bias(&self) -> f32 {
        self.inner.bias()
    }

    #[getter]
    fn coef_(&self) -> Vec<f32> {
        self.inner.weights().to_vec()
    }

    #[getter]
    fn intercept_(&self) -> f32 {
        self.inner.bias()
    }

    #[getter]
    fn n_epochs(&self) -> usize {
        self.inner.n_epochs
    }

    #[getter]
    fn final_loss(&self) -> f32 {
        self.inner.final_loss
    }
}

#[pyfunction]
#[pyo3(signature = (data, y, n, d, c=1.0, learning_rate=0.01, momentum=0.9, max_epochs=100, batch_size=256, tol=1e-4, seed=42))]
pub fn metal_logistic_regression_fit(
    data: Vec<f32>,
    y: Vec<f32>,
    n: usize,
    d: usize,
    c: f32,
    learning_rate: f32,
    momentum: f32,
    max_epochs: usize,
    batch_size: usize,
    tol: f32,
    seed: u64,
) -> PyResult<(Vec<f32>, f32, usize, f32)> {
    let ctx = get_context()?;
    let config = LogisticRegressionConfig {
        c,
        learning_rate,
        momentum,
        max_epochs,
        batch_size,
        tol,
        seed,
    };
    let mut lr = LogisticRegression::new(config);
    lr.fit(ctx, &data, &y, n, d)
        .map_err(|e| PyRuntimeError::new_err(format!("LogisticRegression fit failed: {}", e)))?;
    Ok((lr.weights().to_vec(), lr.bias(), lr.n_epochs, lr.final_loss))
}

#[pyfunction]
#[pyo3(signature = (data, y, n, d, c=1.0, learning_rate=0.01, momentum=0.9, max_epochs=100, batch_size=256, tol=1e-4, seed=42))]
pub fn metal_logistic_regression_fit_bytes(
    data: &[u8],
    y: &[u8],
    n: usize,
    d: usize,
    c: f32,
    learning_rate: f32,
    momentum: f32,
    max_epochs: usize,
    batch_size: usize,
    tol: f32,
    seed: u64,
) -> PyResult<(Vec<f32>, f32, usize, f32)> {
    let data = bytes_to_vec_f32(data, "data")?;
    let y = bytes_to_vec_f32(y, "y")?;
    let ctx = get_context()?;
    let config = LogisticRegressionConfig {
        c,
        learning_rate,
        momentum,
        max_epochs,
        batch_size,
        tol,
        seed,
    };
    let mut lr = LogisticRegression::new(config);
    lr.fit(ctx, &data, &y, n, d)
        .map_err(|e| PyRuntimeError::new_err(format!("LogisticRegression fit failed: {}", e)))?;
    Ok((lr.weights().to_vec(), lr.bias(), lr.n_epochs, lr.final_loss))
}

// ── Linear Regression ─────────────────────────────────────────────

#[pyclass(name = "MetalLinearRegression")]
pub struct PyMetalLinearRegression {
    inner: LinearRegression,
}

#[pymethods]
impl PyMetalLinearRegression {
    #[new]
    #[pyo3(signature = (alpha=0.0, fit_intercept=true, max_iterations=100, tol=1e-4, seed=42))]
    fn new(alpha: f32, fit_intercept: bool, max_iterations: usize, tol: f32, seed: u64) -> Self {
        let config = LinearRegressionConfig {
            alpha,
            fit_intercept,
            max_iterations,
            tol,
            seed,
        };
        Self {
            inner: LinearRegression::new(config),
        }
    }

    fn fit(&mut self, data: Vec<f32>, y: Vec<f32>, n: usize, d: usize) -> PyResult<()> {
        let ctx = get_context()?;
        self.inner
            .fit(ctx, &data, &y, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("LinearRegression fit failed: {}", e)))
    }

    fn predict(&self, data: Vec<f32>, n: usize, d: usize) -> PyResult<Vec<f32>> {
        let ctx = get_context()?;
        self.inner
            .predict(ctx, &data, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("LinearRegression predict failed: {}", e)))
    }

    fn score(&self, data: Vec<f32>, y: Vec<f32>, n: usize, d: usize) -> PyResult<f32> {
        let ctx = get_context()?;
        self.inner
            .score(ctx, &data, &y, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("LinearRegression score failed: {}", e)))
    }

    // ── Bytes-based variants (zero-copy for numpy inputs) ──
    // Accept the raw little-endian float32 bytes via the buffer protocol
    // (a contiguous numpy float32 array passes through without conversion),
    // avoiding the O(n) Python-list roundtrip that `Vec<f32>` arguments pay.

    fn fit_bytes(&mut self, data: &[u8], y: &[u8], n: usize, d: usize) -> PyResult<()> {
        let data = bytes_to_vec_f32(data, "data")?;
        let y = bytes_to_vec_f32(y, "y")?;
        let ctx = get_context()?;
        self.inner
            .fit(ctx, &data, &y, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("LinearRegression fit failed: {}", e)))
    }

    fn predict_bytes(&self, data: &[u8], n: usize, d: usize) -> PyResult<Vec<f32>> {
        let data = bytes_to_vec_f32(data, "data")?;
        let ctx = get_context()?;
        self.inner
            .predict(ctx, &data, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("LinearRegression predict failed: {}", e)))
    }

    fn score_bytes(&self, data: &[u8], y: &[u8], n: usize, d: usize) -> PyResult<f32> {
        let data = bytes_to_vec_f32(data, "data")?;
        let y = bytes_to_vec_f32(y, "y")?;
        let ctx = get_context()?;
        self.inner
            .score(ctx, &data, &y, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("LinearRegression score failed: {}", e)))
    }

    #[getter]
    fn weights(&self) -> Vec<f32> {
        self.inner.weights().to_vec()
    }

    #[getter]
    fn bias(&self) -> f32 {
        self.inner.bias()
    }

    #[getter]
    fn coef_(&self) -> Vec<f32> {
        self.inner.weights().to_vec()
    }

    #[getter]
    fn intercept_(&self) -> f32 {
        self.inner.bias()
    }

    #[getter]
    fn n_iter(&self) -> usize {
        self.inner.n_iter
    }

    #[getter]
    fn final_loss(&self) -> f32 {
        self.inner.final_loss
    }
}

#[pyfunction]
#[pyo3(signature = (data, y, n, d, alpha=0.0, fit_intercept=true, max_iterations=100, tol=1e-4, seed=42))]
pub fn metal_linear_regression_fit(
    data: Vec<f32>,
    y: Vec<f32>,
    n: usize,
    d: usize,
    alpha: f32,
    fit_intercept: bool,
    max_iterations: usize,
    tol: f32,
    seed: u64,
) -> PyResult<(Vec<f32>, f32, usize, f32)> {
    let ctx = get_context()?;
    let config = LinearRegressionConfig {
        alpha,
        fit_intercept,
        max_iterations,
        tol,
        seed,
    };
    let mut lr = LinearRegression::new(config);
    lr.fit(ctx, &data, &y, n, d)
        .map_err(|e| PyRuntimeError::new_err(format!("LinearRegression fit failed: {}", e)))?;
    Ok((lr.weights().to_vec(), lr.bias(), lr.n_iter, lr.final_loss))
}

// ── Gaussian Naive Bayes ──────────────────────────────────────────────

#[pyclass(name = "MetalGaussianNB")]
pub struct PyMetalGaussianNB {
    inner: GaussianNB,
}

#[pymethods]
impl PyMetalGaussianNB {
    #[new]
    #[pyo3(signature = (var_smoothing=1e-9))]
    fn new(var_smoothing: f32) -> Self {
        let config = GaussianNBConfig { var_smoothing };
        Self {
            inner: GaussianNB::new(config),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        data: Vec<f32>,
        y: Vec<f32>,
        n: usize,
        d: usize,
        n_classes: usize,
    ) -> PyResult<()> {
        let ctx = get_context()?;
        self.inner
            .fit(ctx, &data, &y, n, d, n_classes)
            .map_err(|e| PyRuntimeError::new_err(format!("GaussianNB fit failed: {}", e)))
    }

    fn predict(&self, data: Vec<f32>, n: usize, d: usize) -> PyResult<Vec<f32>> {
        let ctx = get_context()?;
        self.inner
            .predict(ctx, &data, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("GaussianNB predict failed: {}", e)))
    }

    fn predict_log_proba(&self, data: Vec<f32>, n: usize, d: usize) -> PyResult<Vec<f32>> {
        let ctx = get_context()?;
        self.inner.predict_log_proba(ctx, &data, n, d).map_err(|e| {
            PyRuntimeError::new_err(format!("GaussianNB predict_log_proba failed: {}", e))
        })
    }

    fn predict_proba(&self, data: Vec<f32>, n: usize, d: usize) -> PyResult<Vec<f32>> {
        let ctx = get_context()?;
        self.inner
            .predict_proba(ctx, &data, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("GaussianNB predict_proba failed: {}", e)))
    }

    fn score(&self, data: Vec<f32>, y: Vec<f32>, n: usize, d: usize) -> PyResult<f32> {
        let ctx = get_context()?;
        self.inner
            .score(ctx, &data, &y, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("GaussianNB score failed: {}", e)))
    }

    #[allow(clippy::too_many_arguments)]
    fn fit_bytes(
        &mut self,
        data: &[u8],
        y: &[u8],
        n: usize,
        d: usize,
        n_classes: usize,
    ) -> PyResult<()> {
        let data = bytes_to_vec_f32(data, "data")?;
        let y = bytes_to_vec_f32(y, "y")?;
        let ctx = get_context()?;
        self.inner
            .fit(ctx, &data, &y, n, d, n_classes)
            .map_err(|e| PyRuntimeError::new_err(format!("GaussianNB fit failed: {}", e)))
    }

    fn predict_bytes(&self, data: &[u8], n: usize, d: usize) -> PyResult<Vec<f32>> {
        let data = bytes_to_vec_f32(data, "data")?;
        let ctx = get_context()?;
        self.inner
            .predict(ctx, &data, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("GaussianNB predict failed: {}", e)))
    }

    fn predict_log_proba_bytes(&self, data: &[u8], n: usize, d: usize) -> PyResult<Vec<f32>> {
        let data = bytes_to_vec_f32(data, "data")?;
        let ctx = get_context()?;
        self.inner.predict_log_proba(ctx, &data, n, d).map_err(|e| {
            PyRuntimeError::new_err(format!("GaussianNB predict_log_proba failed: {}", e))
        })
    }

    fn predict_proba_bytes(&self, data: &[u8], n: usize, d: usize) -> PyResult<Vec<f32>> {
        let data = bytes_to_vec_f32(data, "data")?;
        let ctx = get_context()?;
        self.inner
            .predict_proba(ctx, &data, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("GaussianNB predict_proba failed: {}", e)))
    }

    fn score_bytes(&self, data: &[u8], y: &[u8], n: usize, d: usize) -> PyResult<f32> {
        let data = bytes_to_vec_f32(data, "data")?;
        let y = bytes_to_vec_f32(y, "y")?;
        let ctx = get_context()?;
        self.inner
            .score(ctx, &data, &y, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("GaussianNB score failed: {}", e)))
    }

    #[getter]
    fn n_classes(&self) -> usize {
        self.inner.n_classes()
    }

    #[getter]
    fn n_features(&self) -> usize {
        self.inner.n_features()
    }

    #[getter]
    fn theta_(&self) -> Vec<f32> {
        self.inner.means().to_vec()
    }

    #[getter]
    fn var_(&self) -> Vec<f32> {
        self.inner.variances().to_vec()
    }

    #[getter]
    fn class_prior_(&self) -> Vec<f32> {
        self.inner.class_prior().to_vec()
    }
}

#[pyfunction]
#[pyo3(signature = (data, y, n, d, n_classes, var_smoothing=1e-9))]
pub fn metal_gaussian_nb_fit(
    data: Vec<f32>,
    y: Vec<f32>,
    n: usize,
    d: usize,
    n_classes: usize,
    var_smoothing: f32,
) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let ctx = get_context()?;
    let config = GaussianNBConfig { var_smoothing };
    let mut gnb = GaussianNB::new(config);
    gnb.fit(ctx, &data, &y, n, d, n_classes)
        .map_err(|e| PyRuntimeError::new_err(format!("GaussianNB fit failed: {}", e)))?;
    Ok((
        gnb.means().to_vec(),
        gnb.variances().to_vec(),
        gnb.priors().to_vec(),
    ))
}

#[pyfunction]
#[pyo3(signature = (data, y, n, d, n_classes, var_smoothing=1e-9))]
pub fn metal_gaussian_nb_fit_bytes(
    data: &[u8],
    y: &[u8],
    n: usize,
    d: usize,
    n_classes: usize,
    var_smoothing: f32,
) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let data = bytes_to_vec_f32(data, "data")?;
    let y = bytes_to_vec_f32(y, "y")?;
    let ctx = get_context()?;
    let config = GaussianNBConfig { var_smoothing };
    let mut gnb = GaussianNB::new(config);
    gnb.fit(ctx, &data, &y, n, d, n_classes)
        .map_err(|e| PyRuntimeError::new_err(format!("GaussianNB fit failed: {}", e)))?;
    Ok((
        gnb.means().to_vec(),
        gnb.variances().to_vec(),
        gnb.priors().to_vec(),
    ))
}

#[pyfunction]
#[pyo3(signature = (data, y, n, d, alpha=0.0, fit_intercept=true, max_iterations=100, tol=1e-4, seed=42))]
pub fn metal_linear_regression_fit_bytes(
    data: &[u8],
    y: &[u8],
    n: usize,
    d: usize,
    alpha: f32,
    fit_intercept: bool,
    max_iterations: usize,
    tol: f32,
    seed: u64,
) -> PyResult<(Vec<f32>, f32, usize, f32)> {
    let data = bytes_to_vec_f32(data, "data")?;
    let y = bytes_to_vec_f32(y, "y")?;
    let ctx = get_context()?;
    let config = LinearRegressionConfig {
        alpha,
        fit_intercept,
        max_iterations,
        tol,
        seed,
    };
    let mut lr = LinearRegression::new(config);
    lr.fit(ctx, &data, &y, n, d)
        .map_err(|e| PyRuntimeError::new_err(format!("LinearRegression fit failed: {}", e)))?;
    Ok((lr.weights().to_vec(), lr.bias(), lr.n_iter, lr.final_loss))
}

// ── t-Distributed Stochastic Neighbor Embedding (t-SNE) ───────────

#[pyclass(name = "MetalTSNE")]
pub struct PyMetalTSNE {
    inner: TSNE,
}

#[pymethods]
impl PyMetalTSNE {
    #[new]
    #[pyo3(signature = (n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000, early_exaggeration=12.0, exaggeration_iter=250, momentum=0.8, seed=42, min_grad_norm=1e-7))]
    fn new(
        n_components: usize,
        perplexity: f32,
        learning_rate: f32,
        n_iter: usize,
        early_exaggeration: f32,
        exaggeration_iter: usize,
        momentum: f32,
        seed: u64,
        min_grad_norm: f32,
    ) -> Self {
        let config = TSNEConfig {
            n_components,
            perplexity,
            learning_rate,
            n_iter,
            early_exaggeration,
            exaggeration_iter,
            momentum,
            seed,
            min_grad_norm,
        };
        Self {
            inner: TSNE::new(config),
        }
    }

    fn fit(&mut self, data: Vec<f32>, n: usize, d: usize) -> PyResult<()> {
        let ctx = get_context()?;
        self.inner
            .fit(ctx, &data, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("TSNE fit failed: {}", e)))
    }

    fn fit_bytes(&mut self, data: &[u8], n: usize, d: usize) -> PyResult<()> {
        let data = bytes_to_vec_f32(data, "data")?;
        let ctx = get_context()?;
        self.inner
            .fit(ctx, &data, n, d)
            .map_err(|e| PyRuntimeError::new_err(format!("TSNE fit failed: {}", e)))
    }

    #[getter]
    fn embedding(&self) -> Vec<f32> {
        self.inner.embedding().to_vec()
    }

    #[getter]
    fn n_iter(&self) -> usize {
        self.inner.n_iter()
    }

    #[getter]
    fn kl_divergence(&self) -> f32 {
        self.inner.kl_divergence()
    }
}

#[pyfunction]
#[pyo3(signature = (data, n, d, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000, early_exaggeration=12.0, exaggeration_iter=250, momentum=0.8, seed=42, min_grad_norm=1e-7))]
pub fn metal_tsne_fit(
    data: Vec<f32>,
    n: usize,
    d: usize,
    n_components: usize,
    perplexity: f32,
    learning_rate: f32,
    n_iter: usize,
    early_exaggeration: f32,
    exaggeration_iter: usize,
    momentum: f32,
    seed: u64,
    min_grad_norm: f32,
) -> PyResult<(Vec<f32>, usize, f32)> {
    let ctx = get_context()?;
    let config = TSNEConfig {
        n_components,
        perplexity,
        learning_rate,
        n_iter,
        early_exaggeration,
        exaggeration_iter,
        momentum,
        seed,
        min_grad_norm,
    };
    let mut tsne = TSNE::new(config);
    tsne.fit(ctx, &data, n, d)
        .map_err(|e| PyRuntimeError::new_err(format!("TSNE fit failed: {}", e)))?;
    Ok((
        tsne.embedding().to_vec(),
        tsne.n_iter(),
        tsne.kl_divergence(),
    ))
}

#[pyfunction]
#[pyo3(signature = (data, n, d, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000, early_exaggeration=12.0, exaggeration_iter=250, momentum=0.8, seed=42, min_grad_norm=1e-7))]
pub fn metal_tsne_fit_bytes(
    data: &[u8],
    n: usize,
    d: usize,
    n_components: usize,
    perplexity: f32,
    learning_rate: f32,
    n_iter: usize,
    early_exaggeration: f32,
    exaggeration_iter: usize,
    momentum: f32,
    seed: u64,
    min_grad_norm: f32,
) -> PyResult<(Vec<f32>, usize, f32)> {
    let data = bytes_to_vec_f32(data, "data")?;
    let ctx = get_context()?;
    let config = TSNEConfig {
        n_components,
        perplexity,
        learning_rate,
        n_iter,
        early_exaggeration,
        exaggeration_iter,
        momentum,
        seed,
        min_grad_norm,
    };
    let mut tsne = TSNE::new(config);
    tsne.fit(ctx, &data, n, d)
        .map_err(|e| PyRuntimeError::new_err(format!("TSNE fit failed: {}", e)))?;
    Ok((
        tsne.embedding().to_vec(),
        tsne.n_iter(),
        tsne.kl_divergence(),
    ))
}
