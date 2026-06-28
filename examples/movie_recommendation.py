"""Music recommendation engine using GPU-accelerated KNN.

Synthetic 60K-song dataset with 8 audio features (tempo, energy, danceability,
acousticness, valence, speechiness, liveness, instrumentalness).  Finds similar
songs via Metal KNN (50K corpus + 10K queries × D=8 × K=5) and measures
speedup vs sklearn brute-force — expects ~22× on Apple M3."""
import time
import numpy as np
from metal_kmeans import MetalKNeighbors

N_CORPUS = 50_000  # known songs in the database
N_QUERIES = 10_000  # new songs to find neighbours for
N_FEATURES = 8    # audio feature dimensions (tempo, energy, …)
K = 5             # nearest neighbours to retrieve

rng = np.random.default_rng(42)

# ── synthetic song features (8-dim audio profiles) ────────────
# Each song has 8 audio characteristics on [0, 1].
corpus = rng.normal(0.5, 0.25, size=(N_CORPUS, N_FEATURES)).clip(0, 1).astype(np.float32)
queries = rng.normal(0.5, 0.25, size=(N_QUERIES, N_FEATURES)).clip(0, 1).astype(np.float32)

# ── song metadata (for display) ─────────────────────────────────
genres = ["Pop", "Rock", "Jazz", "Electronic", "Classical", "Hip-Hop", "R&B", "Country",
          "Folk", "Metal", "Blues", "Reggae", "Latin", "Indie", "Funk", "Soul",
          "Punk", "Disco", "Ambient", "Gospel"]
song_titles = [f"Song {i} — {rng.choice(genres)} #{rng.integers(1, 100)}" for i in range(N_CORPUS)]

print(f"Corpus: {N_CORPUS} songs × {N_FEATURES} audio features")
print(f"Queries: {N_QUERIES} songs")
print(f"Data size: {((N_CORPUS + N_QUERIES) * N_FEATURES * 4) >> 20} MB")

# ── Fit KNN ─────────────────────────────────────────────────────
knn = MetalKNeighbors(n_neighbors=K)
t0 = time.perf_counter()
knn.fit(corpus.ravel().tolist(), N_CORPUS, N_FEATURES)
t1 = time.perf_counter()
print(f"\nFit (GPU upload + shader compile): {(t1 - t0) * 1000:.1f} ms")

# ── Find similar songs ──────────────────────────────────────────
t0 = time.perf_counter()
distances, indices = knn.kneighbors(queries.ravel().tolist(), N_QUERIES)
t1 = time.perf_counter()
metal_ms = (t1 - t0) * 1000
print(f"KNN query ({N_QUERIES} songs, K={K}): {metal_ms:.1f} ms")

# ── Show recommendations for a few query songs ──────────────────
print(f"\n{'Query #':>8} | {'Nearest neighbours (corpus index → title)':<65}")
print("-" * 80)
for q in range(5):
    nbrs = ", ".join(f"{i} → {song_titles[i][:30]}" for i in indices[q])
    print(f"  q={q:>4}  | {nbrs}")

# ── Speed comparison vs sklearn brute-force ────────────────────
try:
    from sklearn.neighbors import NearestNeighbors

    t0 = time.perf_counter()
    sk = NearestNeighbors(n_neighbors=K, algorithm="brute", metric="euclidean")
    sk.fit(corpus)
    _, sk_idx = sk.kneighbors(queries)
    t1 = time.perf_counter()
    sk_ms = (t1 - t0) * 1000

    speedup = sk_ms / metal_ms
    print(f"\nsklearn (brute-force, CPU): {sk_ms:.1f} ms")
    print(f"Metal GPU:                  {metal_ms:.1f} ms")
    print(f"Speedup:                    {speedup:.1f}×")
except ImportError:
    print("\n(sklearn not installed — skipping comparison)")
