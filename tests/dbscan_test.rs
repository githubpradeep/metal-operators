//! DBSCAN integration test: verifies the Metal-backed operator (exact
//! ε-neighborhood kernels, same L2 distance approach as KNN) against an
//! independent CPU reference implementation on well-separated blobs with
//! noise.

use metal_operators::dbscan::{DBSCANConfig, DBSCAN};
use metal_operators::metal::MetalContext;

/// Pure-CPU, full ε-neighborhood DBSCAN reference.
fn reference_dbscan(data: &[f32], n: usize, d: usize, eps: f32, min_samples: usize) -> Vec<isize> {
    let dist = |a: usize, b: usize| -> f32 {
        let mut s = 0.0f32;
        for k in 0..d {
            let diff = data[a * d + k] - data[b * d + k];
            s += diff * diff;
        }
        s.sqrt()
    };
    let mut core = vec![false; n];
    for p in 0..n {
        let cnt = (0..n).filter(|&q| dist(p, q) <= eps).count();
        core[p] = cnt >= min_samples;
    }
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut Vec<usize>, mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for p in 0..n {
        if !core[p] {
            continue;
        }
        for q in 0..n {
            if q != p && core[q] && dist(p, q) <= eps {
                let rp = find(&mut parent, p);
                let rq = find(&mut parent, q);
                if rp != rq {
                    parent[rp] = rq;
                }
            }
        }
    }
    let mut labels = vec![-1isize; n];
    let mut id_of_root: Vec<isize> = vec![-1; n];
    let mut next = 0isize;
    for p in 0..n {
        if core[p] {
            let r = find(&mut parent, p);
            if id_of_root[r] == -1 {
                id_of_root[r] = next;
                next += 1;
            }
            labels[p] = id_of_root[r];
        }
    }
    for p in 0..n {
        if core[p] {
            continue;
        }
        let mut assigned = -1;
        for q in 0..n {
            if q != p && core[q] && dist(p, q) <= eps && labels[q] >= 0 {
                assigned = labels[q];
                break;
            }
        }
        labels[p] = assigned;
    }
    labels
}

/// Compare two labelings up to cluster-id permutation; noise (-1) must match
/// exactly position-wise.
fn assert_same_clusters(a: &[isize], b: &[isize]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert_eq!(
            *x == -1,
            *y == -1,
            "noise assignment differs at position: {} vs {}",
            x,
            y
        );
    }
    // Build a permutation map from b -> a and verify it is consistent.
    let max_b = b.iter().copied().max().unwrap_or(-1);
    let mut perm: Vec<Option<isize>> = vec![None; (max_b + 1) as usize];
    for i in 0..a.len() {
        if b[i] == -1 {
            continue;
        }
        match perm[b[i] as usize] {
            None => perm[b[i] as usize] = Some(a[i]),
            Some(prev) => assert_eq!(prev, a[i], "label mismatch at {}", i),
        }
    }
    for i in 0..a.len() {
        if a[i] != -1 {
            assert_eq!(
                perm[b[i] as usize],
                Some(a[i]),
                "cluster membership mismatch at {}",
                i
            );
        }
    }
}

fn run_case(seed: u64, n_per_blob: usize, eps: f32, min_samples: usize) {
    let ctx = MetalContext::new().expect("No Metal device");
    let d = 2;
    let n = n_per_blob * 2 + 2; // two blobs + 2 outliers
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut data = Vec::with_capacity(n * d);
    for i in 0..n_per_blob {
        data.push((rng.f32() - 0.5) * 1.6); // blob A around (0,0)
        data.push((rng.f32() - 0.5) * 1.6);
    }
    for _ in 0..n_per_blob {
        data.push(8.0 + (rng.f32() - 0.5) * 1.6); // blob B around (8,8)
        data.push(8.0 + (rng.f32() - 0.5) * 1.6);
    }
    data.push(30.0); // outlier 1
    data.push(30.0);
    data.push(-25.0); // outlier 2
    data.push(-25.0);

    let mut db = DBSCAN::new(DBSCANConfig { eps, min_samples });
    db.fit(&ctx, &data, n, d).expect("fit failed");
    let labels = db.labels().to_vec();
    assert_eq!(labels.len(), n);
    assert_eq!(
        db.n_clusters(),
        2,
        "expected exactly 2 clusters, got {}",
        db.n_clusters()
    );

    let expected = reference_dbscan(&data, n, d, eps, min_samples);
    assert_same_clusters(&labels, &expected);
}

#[test]
fn test_blobs_against_reference() {
    for (seed, per_blob, eps, min_samples) in
        [(7u64, 25usize, 2.0, 4), (42, 40, 1.5, 3), (99, 15, 3.0, 5)]
    {
        run_case(seed, per_blob, eps, min_samples);
    }
}

#[test]
fn test_invalid_inputs_rejected() {
    let ctx = MetalContext::new().expect("No Metal device");
    let data = vec![0.0f32; 10];
    let mut db = DBSCAN::new(DBSCANConfig {
        eps: 0.5,
        min_samples: 2,
    });
    assert!(db.fit(&ctx, &data, 3, 2).is_err()); // length mismatch
    let mut db2 = DBSCAN::new(DBSCANConfig {
        eps: 0.0,
        min_samples: 2,
    });
    assert!(db2.fit(&ctx, &data, 5, 2).is_err()); // eps must be > 0
    let mut db3 = DBSCAN::new(DBSCANConfig {
        eps: 0.5,
        min_samples: 0,
    });
    assert!(db3.fit(&ctx, &data, 5, 2).is_err()); // min_samples must be >= 1
}
