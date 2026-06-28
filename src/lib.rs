pub mod metal;
pub mod kmeans;

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
pub use python::PyMetalKMeans;
#[cfg(feature = "python")]
pub use python::metal_kmeans_fit;

#[cfg(feature = "python")]
mod py_bridge {
    use pyo3::prelude::*;
    use super::python;

    #[pymodule]
    pub fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<python::PyMetalKMeans>()?;
        m.add_function(wrap_pyfunction!(python::metal_kmeans_fit, m)?)?;
        Ok(())
    }
}
