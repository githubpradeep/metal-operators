//! Integration tests for the Support Vector Regressor (SVR).
//!
//! These run on real Metal (skipped when no GPU/device is present) and
//! exercise the shared `svm_kernel`/`svm_predict` shaders via the ε-SVR SMO.

use metal_operators::metal::MetalContext;
use metal_operators::svm::SVCKernel;
use metal_operators::svr::{SVRConfig, SVR};

#[test]
fn svr_fits_sinusoid() {
    let ctx = MetalContext::new().unwrap();
    let n = 150;
    let d = 1;
    let mut rng = fastrand::Rng::with_seed(11);
    let mut data = vec![0.0f32; n * d];
    let mut y = vec![0.0f32; n];
    for i in 0..n {
        let x = rng.f32() * 6.28f32;
        data[i] = x;
        y[i] = x.sin();
    }

    let mut m = SVR::new(SVRConfig {
        kernel: SVCKernel::Rbf,
        gamma: 0.6,
        c: 5.0,
        eps: 0.05,
        max_iter: 300,
        ..Default::default()
    });
    m.fit(&ctx, &data, &y, n, d).unwrap();

    let r2 = m.score(&ctx, &data, &y, n, d).unwrap();
    assert!(
        r2 > 0.9,
        "RBF SVR should fit a smooth sinusoid, got R²={}",
        r2
    );
    assert!(m.support_count() > 0 && m.support_count() <= n);

    // decision_function and predict are the same for SVR (single regressor).
    let preds = m.predict(&ctx, &data, n, d).unwrap();
    let dec = m.decision_function(&ctx, &data, n, d).unwrap();
    assert_eq!(preds, dec);
    assert_eq!(preds.len(), n);
}
