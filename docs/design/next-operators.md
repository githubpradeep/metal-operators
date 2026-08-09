# Design: Next Set of Operators

Status: **design** · Bead: b10 · Author: Crew (Crew-1)

## 1. Motivation

`metal-operators` ships 13 GPU-accelerated classical-ML operators:

| Family | Operators |
|---|---|
| Clustering | KMeans, GMM, DBSCAN |
| Dimensionality reduction | PCA, LDA, NMF, t-SNE |
| Classification | KNN, LogisticRegression, GaussianNB, SVC |
| Regression | LinearRegression, SVR |

The coverage is biased toward **linear algebra–heavy** methods (Gram/scatter, matmul,
eigh) and **lazy/nearest-neighbour** families (KNN, DBSCAN). Three large gaps remain
before the library rivals a classical sklearn-style toolchain:

1. **Tree ensembles** (a dominant production family — RF / GBDT / ExtraTrees).
2. **Regularized + kernel linear methods** (Ridge–Lasso–ElasticNet — trivial on top
   of the existing `linreg` Gram path; Kernel-ridge as a natural SVC-sidekick).
3. **Nearest-graph / density clustering breadth** (OPTICS — already have KNN + DBSCAN
   primitives) and **spectral clustering** (KNN affinities + PCA eigh already exist).

This design identifies the **next set**, rooted in *maximal reuse of existing kernels and
host solve/iteration scaffolding* so each implement spawn stays small, verifiable, and
consistent with the established per-operator pattern.

## 2. Selection criteria

Priority was scored by:

- **Reuse** — how much of the existing GPU/host scaffolding is reused vs. built new.
- **User value** — how commonly the operator is used in real classical-ML workloads.
- **Kernel shape** — whether the inner loop maps to a Metal compute kernel the project
  already knows how to author (reductions, matmul, distance, scatter).
- **Verification** — whether sklearn-parity unit tests are straightforward.

## 3. Recommended next set (in priority order)

### P1 — Ensemble & regularized models

| # | Operator | Reuses | New GPU work | Comment |
|---|---|---|---|---|
| 1 | **Lasso** | `linreg_gram_xtx_tiled`, `linreg_gram_xty`, `linreg_reduce_gram`, host `solve_linear_system` | coordinate-descent host loop on the cached Gram (GPU builds Gram once) | Smallest lift; ephemeral coordinate descent is naturally on CPU over a small `(d+1)²` Gram |
| 2 | **ElasticNet** | same Gram path + host CD with L1+L2 | host coordinate-descent loop | mirrors sk learn `ElasticNet`; `l1_ratio` config |
| 3 | **Ridge** | same Gram path + host solve | ridge = augmented diagonal (λI) on host Gram before solve | tiny, highest value-per-line |
| 4 | **Gradient Boosting (GBR)** | `linreg_predict` (base learner apply) for weak learners; host residual CD | host boosting loop + weak-learner tree/CD fit | larger build; can be phased after Lasso/ElasticNet |

### 🧱 — Nearest-graph & spectral

| # | Operator | Reuses | New GPU work | Notes |
|---|---|---|---|---|
| 5 | **OPTICS** | KNN distance pipeline, `kmeans.compute_min_distances`-style D import | host reachability (seeded heap) | Dense distance reused from KNN; ordering strictly host |
| 6 | **SpectralClustering** | KNN affinities + PCA eigh (Jacobi / Accelerate `ssyevd`), KMeans-assign on embedding | host affinity construction + top-k eigh | rounds of sklearn `SpectralClustering` |
| 7 | **K-medoids (PAM)** | KMeans assign kernel + KNN distances | host medoid swap | sklearn `KMedoids` |

### 🧩 — cheap fills

| # | Operator | Reuses | Notes |
|---|---|---|---|
| 8 | **RidgeCV / LassoCV** | Lasso/Ridge host CD | cross-validation loop over `alphas`; small extra |
| 9 | **ExtraTrees / RandomForest (regression)** | shared with GBR tree path | grouped behind ensemble design once a tree learner exists |

## 4. Recommended first implement wave

Close this design, then spawn **implement** for the following (most reused, cheapest,
highest confidence — each is ~1.0 implement bead):

1. **Ridge** — build on `LinearRegressionConfig`-like struct;
   reuse `linreg_gram_xtx_tiled`, `linreg_gram_xty`, `linreg_reduce_gram`,
   host `solve_linear_system`; add `λI` on the host Gram before solve. New
   `src/ridge/mod.rs`, `shaders/ridge_shader` (may reuse `linear.metal`), PyO3
   bindings, `python/metal_ridge`, `tests/ridge_test.rs`, docs the README.

2. **Lasso** — same Gram path; host coordinate-descent loop over the cached
   `(d+1)×(d+1)` Gram, with `alpha`/`max_iter`/`tol` config.
   `tests/lasso_test.rs`, PyO3 `metal_lasso_fit`.

3. **ElasticNet** — Lasso + `l1_ratio`; shares the CD loop with a weighted penalty.
   Same Gram path.

4. **SpectralClustering** — KNN `kneighbors` for affinity, host spectral embedding,
   KMeans assign on embeddings. New `src/spectral/mod.rs`, `shaders/spectral.metal`
   or reuse `pca.matmul`/`kmeans`.

> Implement the immutable Gram/Xᵀy build **once** as shared helper so Ridge/Lasso/
> ElasticNet reuse it via `pub(crate)`; keep `linreg` helper internal.

## 5. Non-goals (this wave)

- **Deep/neural** operators (MLP, Conv) — different kernel family, larger scope.
- **t-SNE/UMAP variants** — already heavy; revisit only on request.
- **Full AutoML / search** — out of scope for a single operator library.
- **Tree-based full parity with sklearn** — retain greedy/single-pass splits first.

## 6. Reuse map (shared scaffolding)

| Shared piece | Location | Reused by |
|---|---|---|
| Gram XᵀX tiled+naive kernels | `shaders/linear.metal: linreg_gram_xtx_*`, `linreg_gram_xty`, `linreg_reduce_gram` | Ridge, Lasso, ElasticNet, RidgeCV |
| host solve (Gaussian elim) | `src/linear_regression/mod.rs: solve_linear_system` | Ridge, Lasso closed-form |
| KNN distance pipeline | `shaders/knn.metal` + `src/knn` | OPTICS, SpectralClustering, K-medoids |
| PCA eigh (Jacobi + Accelerate) | `src/pca` `pca_eigh_*` | SpectralClustering |
| kmeans-guard (KMeans++ assign) | `src/kmeans` + `shaders/kmeans.metal` | SpectralClustering embedding label |

## 7. Acceptance criteria (per added operator)

- `cargo check` (default + `--features python`); no new warnings.
- Unit tests pass on real Metal (matches sklearn reference on synthetic data; minimum
  tolerance for f32 FMA ordering vs reference).
- New Python package installs via maturin; functional + sklearn-style API both work
- README + `docs/python_api.md` / `docs/rust_api.md` sections added.
- Reuse is `pub(crate)` shared, not copy-; no dead code.