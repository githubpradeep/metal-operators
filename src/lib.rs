pub mod dbscan;
pub mod gmm;
pub mod kmeans;
pub mod knn;
pub mod lda;
pub mod linear_regression;
pub mod logistic_regression;
pub mod metal;
pub mod naive_bayes;
pub mod nmf;
pub mod pca;
pub mod svm;
pub mod svr;
pub mod tsne;

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
pub use python::metal_dbscan_fit;
#[cfg(feature = "python")]
pub use python::metal_dbscan_fit_bytes;
#[cfg(feature = "python")]
pub use python::metal_gaussian_nb_fit;
#[cfg(feature = "python")]
pub use python::metal_gmm_fit;
#[cfg(feature = "python")]
pub use python::metal_gmm_fit_bytes;
#[cfg(feature = "python")]
pub use python::metal_kmeans_fit;
#[cfg(feature = "python")]
pub use python::metal_kneighbors;
#[cfg(feature = "python")]
pub use python::metal_lda_fit;
#[cfg(feature = "python")]
pub use python::metal_lda_fit_bytes;
#[cfg(feature = "python")]
pub use python::metal_linear_regression_fit;
#[cfg(feature = "python")]
pub use python::metal_logistic_regression_fit;
#[cfg(feature = "python")]
pub use python::metal_nmf_fit;
#[cfg(feature = "python")]
pub use python::metal_nmf_fit_bytes;
#[cfg(feature = "python")]
pub use python::metal_pca_fit;
#[cfg(feature = "python")]
pub use python::metal_svc_fit;
#[cfg(feature = "python")]
pub use python::metal_svc_fit_bytes;
#[cfg(feature = "python")]
pub use python::metal_svr_fit;
#[cfg(feature = "python")]
pub use python::metal_svr_fit_bytes;
#[cfg(feature = "python")]
pub use python::metal_tsne_fit;
#[cfg(feature = "python")]
pub use python::metal_tsne_fit_bytes;
#[cfg(feature = "python")]
pub use python::PyMetalDBSCAN;
#[cfg(feature = "python")]
pub use python::PyMetalGMM;
#[cfg(feature = "python")]
pub use python::PyMetalGaussianNB;
#[cfg(feature = "python")]
pub use python::PyMetalKMeans;
#[cfg(feature = "python")]
pub use python::PyMetalKNeighbors;
#[cfg(feature = "python")]
pub use python::PyMetalLDA;
#[cfg(feature = "python")]
pub use python::PyMetalLinearRegression;
#[cfg(feature = "python")]
pub use python::PyMetalLogisticRegression;
#[cfg(feature = "python")]
pub use python::PyMetalNMF;
#[cfg(feature = "python")]
pub use python::PyMetalPCA;
#[cfg(feature = "python")]
pub use python::PyMetalSVC;
#[cfg(feature = "python")]
pub use python::PyMetalSVR;
#[cfg(feature = "python")]
pub use python::PyMetalTSNE;

#[cfg(feature = "python")]
mod py_bridge {
    use super::python;
    use pyo3::prelude::*;

    #[pymodule]
    pub fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<python::PyMetalDBSCAN>()?;
        m.add_function(wrap_pyfunction!(python::metal_dbscan_fit, m)?)?;
        m.add_function(wrap_pyfunction!(python::metal_dbscan_fit_bytes, m)?)?;
        m.add_class::<python::PyMetalKMeans>()?;
        m.add_function(wrap_pyfunction!(python::metal_kmeans_fit, m)?)?;
        m.add_function(wrap_pyfunction!(python::metal_kmeans_fit_bytes, m)?)?;
        m.add_class::<python::PyMetalKNeighbors>()?;
        m.add_function(wrap_pyfunction!(python::metal_kneighbors, m)?)?;
        m.add_function(wrap_pyfunction!(python::metal_kneighbors_bytes, m)?)?;
        m.add_class::<python::PyMetalLDA>()?;
        m.add_function(wrap_pyfunction!(python::metal_lda_fit, m)?)?;
        m.add_function(wrap_pyfunction!(python::metal_lda_fit_bytes, m)?)?;
        m.add_class::<python::PyMetalPCA>()?;
        m.add_function(wrap_pyfunction!(python::metal_pca_fit, m)?)?;
        m.add_function(wrap_pyfunction!(python::metal_pca_fit_bytes, m)?)?;
        m.add_class::<python::PyMetalSVC>()?;
        m.add_function(wrap_pyfunction!(python::metal_svc_fit, m)?)?;
        m.add_function(wrap_pyfunction!(python::metal_svc_fit_bytes, m)?)?;
        m.add_class::<python::PyMetalSVR>()?;
        m.add_function(wrap_pyfunction!(python::metal_svr_fit, m)?)?;
        m.add_function(wrap_pyfunction!(python::metal_svr_fit_bytes, m)?)?;
        m.add_class::<python::PyMetalLogisticRegression>()?;
        m.add_function(wrap_pyfunction!(python::metal_logistic_regression_fit, m)?)?;
        m.add_function(wrap_pyfunction!(
            python::metal_logistic_regression_fit_bytes,
            m
        )?)?;
        m.add_class::<python::PyMetalGaussianNB>()?;
        m.add_function(wrap_pyfunction!(python::metal_gaussian_nb_fit, m)?)?;
        m.add_function(wrap_pyfunction!(python::metal_gaussian_nb_fit_bytes, m)?)?;
        m.add_class::<python::PyMetalGMM>()?;
        m.add_function(wrap_pyfunction!(python::metal_gmm_fit, m)?)?;
        m.add_function(wrap_pyfunction!(python::metal_gmm_fit_bytes, m)?)?;
        m.add_class::<python::PyMetalLinearRegression>()?;
        m.add_function(wrap_pyfunction!(python::metal_linear_regression_fit, m)?)?;
        m.add_function(wrap_pyfunction!(
            python::metal_linear_regression_fit_bytes,
            m
        )?)?;
        m.add_class::<python::PyMetalNMF>()?;
        m.add_function(wrap_pyfunction!(python::metal_nmf_fit, m)?)?;
        m.add_function(wrap_pyfunction!(python::metal_nmf_fit_bytes, m)?)?;
        m.add_class::<python::PyMetalTSNE>()?;
        m.add_function(wrap_pyfunction!(python::metal_tsne_fit, m)?)?;
        m.add_function(wrap_pyfunction!(python::metal_tsne_fit_bytes, m)?)?;
        Ok(())
    }
}
