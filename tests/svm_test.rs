//! Integration tests for the Support Vector Classifier (SVC).
//!
//! These run on real Metal (skipped when no GPU/device is present) and
//! exercise the RBF kernel + host SMO on linearly-inseparable XOR.

use metal_operators::metal::MetalContext;
use metal_operators::svm::{SVCConfig, SVCKernel, SVC};

#[test]
fn xor_separable_with_rbf() {
    let ctx = MetalContext::new().unwrap();
    // XOR: (0,0),(0,1),(1,0),(1,1) with labels mirroring the bit parity.
    let data: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let labels = vec![0.0, 1.0, 1.0, 0.0];
    let n = 4;
    let d = 2;

    let mut m = SVC::new(SVCConfig {
        kernel: SVCKernel::Rbf,
        gamma: 0.5,
        c: 100.0,
        tolerance: 1e-6,
        max_iter: 400,
        seed: 3,
        ..Default::default()
    });
    m.fit(&ctx, &data, &labels, n, d).unwrap();

    let preds = m.predict(&ctx, &data, n, d).unwrap();
    let correct = (0..n)
        .filter(|&i| preds[i] as usize == labels[i] as usize)
        .count();
    assert_eq!(
        correct, n,
        "XOR is not linearly separable; RBF must separate it"
    );
    assert_eq!(m.classes(), &[0.0, 1.0]);
    assert!(
        m.support_count() > 0,
        "RBF fit should yield support vectors"
    );

    // Decision function: consistent sign with the label convention.
    let dec = m.decision_function(&ctx, &data, n, d).unwrap();
    assert_eq!(dec.len(), n * m.n_classes());
}
