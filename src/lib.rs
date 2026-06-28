pub mod metal;
pub mod kmeans;
pub mod knn;

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
pub use python::PyMetalKMeans;
#[cfg(feature = "python")]
pub use python::metal_kmeans_fit;
#[cfg(feature = "python")]
pub use python::PyMetalKNeighbors;
#[cfg(feature = "python")]
pub use python::metal_kneighbors;

#[cfg(feature = "python")]
mod py_bridge {
    use pyo3::prelude::*;
    use super::python;

    #[pymodule]
    pub fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<python::PyMetalKMeans>()?;
        m.add_function(wrap_pyfunction!(python::metal_kmeans_fit, m)?)?;
        m.add_class::<python::PyMetalKNeighbors>()?;
        m.add_function(wrap_pyfunction!(python::metal_kneighbors, m)?)?;
        Ok(())
    }
}
