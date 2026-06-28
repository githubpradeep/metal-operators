"""Movie recommendation engine using GPU-accelerated KNN.

Synthetic 1M-rating dataset: 10 000 users, 1000 movies, 20 genre features.
Finds similar users via Metal KNN, then recommends unseen movies."""
import time
import numpy as np
from metal_kmeans import MetalKNeighbors

N_USERS = 10_000
N_MOVIES = 1_000
N_GENRES = 20  # feature dimensions
N_NEIGHBORS = 10
N_RECS = 5

rng = np.random.default_rng(42)

# ── synthetic user profiles (genre preferences) ────────────────
# Each user has affinity for 20 genre dimensions — this is our KNN corpus.
user_profiles = rng.normal(0.5, 0.3, size=(N_USERS, N_GENRES)).clip(0, 1).astype(np.float32)

# ── movie metadata (genre vector per movie) ────────────────────
movie_genres = rng.binomial(1, 0.3, size=(N_MOVIES, N_GENRES)).astype(np.float32)

# ── implicit ratings: dot(user, movie) + noise ─────────────────
ratings = user_profiles @ movie_genres.T + rng.normal(0, 0.1, size=(N_USERS, N_MOVIES))
ratings = ratings.clip(0, 5).astype(np.float32)

# For each user, hide 10 % of their ratings as "unseen"
unseen_mask = rng.binomial(1, 0.1, size=(N_USERS, N_MOVIES)).astype(bool)

# ── Build corpus ──────────────────────────────────────────────
# Use the profiles of first 9990 users as corpus; keep 10 as query users.
corpus = user_profiles[:9990].ravel().tolist()
queries = user_profiles[9990:].ravel().tolist()
nq = 10

print(f"Corpus: {9990} users × {N_GENRES} genres")
print(f"Queries: {nq} users")
print(f"Movies: {N_MOVIES}")
print(f"Rating matrix: {N_USERS} × {N_MOVIES}  ({ratings.nbytes >> 20} MB)")

# ── Fit KNN ─────────────────────────────────────────────────────
knn = MetalKNeighbors(n_neighbors=N_NEIGHBORS)
t0 = time.perf_counter()
knn.fit(corpus, 9990, N_GENRES)
t1 = time.perf_counter()
print(f"\nFit: {(t1 - t0) * 1000:.1f} ms")

# ── Find similar users ──────────────────────────────────────────
t0 = time.perf_counter()
distances, indices = knn.kneighbors(queries, nq)
t1 = time.perf_counter()
print(f"KNN query ({nq} users): {(t1 - t0) * 1000:.1f} ms")

# ── Generate recommendations ────────────────────────────────────
print(f"\n{'Query User':>11} | {'Neighbors':>20} | {'Top-5 Recommendations':>25}")
print("-" * 70)

for q in range(nq):
    similar_users = indices[q]  # corpus indices of nearest neighbors

    # Aggregate ratings from similar users, excluding already-seen movies
    seen = unseen_mask[9990 + q]  # which movies this query user has "seen"
    neighbor_ratings = ratings[similar_users]  # (N_NEIGHBORS, N_MOVIES)
    agg = neighbor_ratings.mean(axis=0)
    agg[seen] = -1  # mask already-seen movies

    top5 = np.argsort(agg)[-N_RECS:][::-1]
    neighbor_list = ", ".join(str(int(i)) for i in similar_users)
    rec_list = ", ".join(str(int(m)) for m in top5)
    print(f"  user {9990 + q:>5}  | {neighbor_list:>20} | {rec_list:>25}")

# ── Speed comparison vs sklearn ────────────────────────────────
try:
    from sklearn.neighbors import NearestNeighbors

    corpus_np = user_profiles[:9990].astype(np.float32)
    queries_np = user_profiles[9990:].astype(np.float32)

    t0 = time.perf_counter()
    sk = NearestNeighbors(n_neighbors=N_NEIGHBORS, algorithm="brute", metric="euclidean")
    sk.fit(corpus_np)
    _, sk_idx = sk.kneighbors(queries_np)
    t1 = time.perf_counter()
    sk_ms = (t1 - t0) * 1000

    print(f"\nsklearn (brute-force, CPU): {sk_ms:.1f} ms")
    print(f"Metal speedup: {sk_ms / max((t1 - t0) * 1000, 0.001):.1f}×")
except ImportError:
    print("\n(sklearn not installed — skipping comparison)")
