import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "output"
OUT_DIR = os.path.join(OUTPUT_DIR, "clustering")
FIG_DIR = os.path.join(OUT_DIR, "figures")
MATRIX_PATH = os.path.join(OUTPUT_DIR, "combined_matrix.csv")

# k=4 is the number of cancer types. the other values are for the silhouette curve
KS = [2, 3, 4, 5, 6, 7, 8]
# distances are mostly near 1, so eps has to come from the close pairs
DBSCAN_PERCENTILES = [0.1, 0.5, 1, 2, 5, 10]
DBSCAN_MIN_SAMPLES = [5, 10]
DISTANCE_PERCENTILES = [0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 90, 99]
KMEANS_N_INIT = 10
KMEANS_MAX_ITER = 100
KMEDOIDS_MAX_ITER = 100
RANDOM_STATE = 0

K_METHODS = [
    "agglomerative_average",
    "agglomerative_complete",
    "kmedoids",
    "kmeans",
]


def load_matrix():
    # samples are rows, genes are columns. cancer type is not loaded
    if not os.path.exists(MATRIX_PATH):
        print(f"Matrix not found: {MATRIX_PATH}")
        print("Run python mutation_matrices.py first.")
        return None
    X = pd.read_csv(MATRIX_PATH, index_col=0)
    if X.index.duplicated().any():
        print("some samples are duplicated???")
    return X

def jaccard_distance(X):
    # pairwise jaccard distance for a 0/1 sample-by-gene matrix.
    # same shared/union formula as jaccard_matrix in compute_similarity.py,
    # then distance = 1 - similarity. diagonal is 0.
    # two tumors with no retained mutations have distance 0.
    # an empty tumor and a non-empty tumor have distance 1.
    # compute_similarity.py leaves those empty pairs as NaN, so this does not call it.
    # returns a square float64 array in the same row order as X
    A = np.asarray(X, dtype=np.float64)
    if A.ndim != 2 or not np.isin(A, [0.0, 1.0]).all():
        raise ValueError("jaccard_distance expects a 0/1 sample-by-gene matrix")

    shared = A @ A.T
    counts = A.sum(axis=1)
    union = counts[:, None] + counts[None, :] - shared
    both_empty = (counts[:, None] == 0) & (counts[None, :] == 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = shared / union
    similarity[both_empty] = 1.0
    np.fill_diagonal(similarity, 1.0)
    dist = 1.0 - similarity
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    np.clip(dist, 0.0, 1.0, out=dist)
    return dist

def distance_summary(dist):
    # min, mean, max, and percentiles of the off-diagonal distances.
    # the dbscan eps values are the low percentiles of this same list
    off = dist[np.triu_indices(dist.shape[0], k=1)]
    rows = [
        {"statistic": "min", "value": float(off.min())},
        {"statistic": "mean", "value": float(off.mean())},
    ]
    for percentile in DISTANCE_PERCENTILES:
        rows.append({
            "statistic": f"percentile_{percentile}",
            "value": float(np.percentile(off, percentile)),
        })
    rows.append({"statistic": "max", "value": float(off.max())})
    return pd.DataFrame(rows, columns=["statistic", "value"])

def agglomerative(dist, linkage):
    # average or complete linkage on a square distance matrix.
    # each step merges the closest remaining pair and updates that row in place.
    # average uses the size-weighted mean. complete uses the max.
    # returns merges with one row per step and columns
    # survivor index, absorbed index, distance, new size.
    # those indices are positions in dist, not new cluster ids
    if linkage not in ("average", "complete"):
        raise ValueError("linkage must be average or complete")

    n = dist.shape[0]
    d = np.array(dist, dtype=np.float64, copy=True)
    np.fill_diagonal(d, np.inf)
    sizes = np.ones(n, dtype=np.float64)
    merges = np.zeros((n - 1, 4), dtype=np.float64)

    for step in range(n - 1):
        if step % 500 == 0:
            print(f"  {linkage} merge {step + 1}/{n - 1}")
        flat = int(np.argmin(d))
        i, j = divmod(flat, n)
        if i == j or not np.isfinite(d[i, j]):
            raise ValueError("no finite distance left to merge")
        if i > j:
            i, j = j, i
        height = float(d[i, j])
        ni = sizes[i]
        nj = sizes[j]
        if linkage == "average":
            updated = (ni * d[i] + nj * d[j]) / (ni + nj)
        else:
            updated = np.maximum(d[i], d[j])
        d[i, :] = updated
        d[:, i] = updated
        d[j, :] = np.inf
        d[:, j] = np.inf
        d[i, i] = np.inf
        sizes[i] = ni + nj
        sizes[j] = 0
        merges[step] = (i, j, height, sizes[i])
    return merges

def cut_tree(merges, n, k):
    # cluster labels 0..k-1 from the first n-k merges.
    # merges is the array returned by agglomerative
    if k < 2 or k >= n:
        raise ValueError(f"k must be between 2 and {n - 1}")
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for step in range(n - k):
        survivor = int(merges[step, 0])
        absorbed = int(merges[step, 1])
        parent[find(absorbed)] = find(survivor)
    roots = np.array([find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels

def kmedoids(dist, k, random_state=RANDOM_STATE):
    # lloyd-style k-medoids on a square distance matrix.
    # assign every tumor to the nearest medoid, then replace the medoid
    # with the member whose distances to the other members sum to the least.
    # init is k-medoids++. this is not the full PAM swap.
    # returns labels 0..k-1 in the same order as dist
    if k < 2 or k >= dist.shape[0]:
        raise ValueError(f"k must be between 2 and {dist.shape[0] - 1}")
    rng = np.random.default_rng(random_state)
    medoids = init_medoids(dist, k, rng)
    for _ in range(KMEDOIDS_MAX_ITER):
        labels = np.argmin(dist[:, medoids], axis=1)
        updated = medoids.copy()
        for j in range(k):
            members = np.flatnonzero(labels == j)
            if len(members) == 0:
                updated[j] = reseed_medoid(dist, updated)
                continue
            costs = dist[np.ix_(members, members)].sum(axis=1)
            updated[j] = members[int(np.argmin(costs))]
        if np.array_equal(updated, medoids):
            break
        medoids = updated
    return np.argmin(dist[:, medoids], axis=1)

def init_medoids(dist, k, rng):
    # k-medoids++. first medoid is random, later ones are picked with
    # probability proportional to squared distance from the nearest medoid
    n = dist.shape[0]
    medoids = np.empty(k, dtype=int)
    medoids[0] = int(rng.integers(0, n))
    closest = dist[medoids[0]].copy()
    chosen = {int(medoids[0])}
    for j in range(1, k):
        weights = closest ** 2
        for index in chosen:
            weights[index] = 0.0
        remaining = np.flatnonzero(weights > 0)
        if len(remaining) == 0:
            open_indices = [i for i in range(n) if i not in chosen]
            pick = int(open_indices[int(rng.integers(0, len(open_indices)))])
        else:
            probs = weights[remaining]
            probs = probs / probs.sum()
            pick = int(rng.choice(remaining, p=probs))
        medoids[j] = pick
        chosen.add(pick)
        np.minimum(closest, dist[pick], out=closest)
    return medoids

def reseed_medoid(dist, medoids):
    # an empty cluster gets the tumor farthest from its current medoid
    taken = set(int(medoid) for medoid in medoids)
    available = [i for i in range(dist.shape[0]) if i not in taken]
    if not available:
        return int(medoids[0])
    nearest = dist[np.ix_(available, medoids)].min(axis=1)
    return int(available[int(np.argmax(nearest))])

def kmeans(X, k, n_init=KMEANS_N_INIT, random_state=RANDOM_STATE):
    # euclidean k-means on a numeric sample-by-gene matrix.
    # n_init restarts, keep the one with the smallest within-cluster sum of squares.
    # returns labels 0..k-1
    A = np.asarray(X, dtype=np.float64)
    n = A.shape[0]
    if k < 2 or k >= n:
        raise ValueError(f"k must be between 2 and {n - 1}")
    rng = np.random.default_rng(random_state)
    best_labels = None
    best_inertia = np.inf
    for _ in range(n_init):
        centroids = init_centroids(A, k, rng)
        labels = np.zeros(n, dtype=int)
        for _ in range(KMEANS_MAX_ITER):
            d2 = squared_distances(A, centroids)
            labels = np.argmin(d2, axis=1)
            new_centroids = centroids.copy()
            for j in range(k):
                members = np.flatnonzero(labels == j)
                if len(members) == 0:
                    own = d2[np.arange(n), labels]
                    new_centroids[j] = A[int(np.argmax(own))]
                else:
                    new_centroids[j] = A[members].mean(axis=0)
            if np.allclose(new_centroids, centroids):
                centroids = new_centroids
                break
            centroids = new_centroids
        d2 = squared_distances(A, centroids)
        labels = np.argmin(d2, axis=1)
        inertia = float(d2[np.arange(n), labels].sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
    return best_labels

def init_centroids(A, k, rng):
    # k-means++. same idea as init_medoids, but on squared euclidean distance
    n = A.shape[0]
    pick = int(rng.integers(0, n))
    chosen = [pick]
    centroids = [A[pick]]
    closest = np.sum((A - A[pick]) ** 2, axis=1)
    for _ in range(1, k):
        weights = closest.copy()
        weights[chosen] = 0.0
        total = float(weights.sum())
        if total <= 0:
            open_indices = [i for i in range(n) if i not in set(chosen)]
            pick = int(open_indices[int(rng.integers(0, len(open_indices)))])
        else:
            pick = int(rng.choice(n, p=weights / total))
        chosen.append(pick)
        centroids.append(A[pick])
        closest = np.minimum(closest, np.sum((A - A[pick]) ** 2, axis=1))
    return np.vstack(centroids)

def squared_distances(A, centroids):
    # rows of A against each centroid. ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x·c
    a2 = np.sum(A * A, axis=1)[:, None]
    c2 = np.sum(centroids * centroids, axis=1)[None, :]
    d2 = a2 + c2 - 2.0 * (A @ centroids.T)
    np.maximum(d2, 0.0, out=d2)
    return d2

def dbscan(dist, eps, min_samples):
    # dbscan on a square distance matrix. noise is labeled -1.
    # a point is a core point when at least min_samples tumors (counting itself)
    # sit at distance <= eps
    n = dist.shape[0]
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster = 0
    for i in range(n):
        if visited[i]:
            continue
        neighbors = np.flatnonzero(dist[i] <= eps)
        if len(neighbors) < min_samples:
            visited[i] = True
            continue
        labels[i] = cluster
        visited[i] = True
        seeds = list(neighbors)
        t = 0
        while t < len(seeds):
            q = seeds[t]
            t += 1
            if not visited[q]:
                visited[q] = True
                q_neighbors = np.flatnonzero(dist[q] <= eps)
                if len(q_neighbors) >= min_samples:
                    seeds.extend(int(index) for index in q_neighbors)
            if labels[q] < 0:
                labels[q] = cluster
        cluster += 1
    return labels

def silhouette(dist, labels):
    # mean silhouette from a square distance matrix.
    # noise (label < 0) is left out. a cluster of one tumor scores 0.
    # returns nan when fewer than 2 clusters are left, or every tumor is its own cluster
    labels = np.asarray(labels)
    keep = labels >= 0
    labels = labels[keep]
    dist = dist[np.ix_(keep, keep)]
    clusters = np.unique(labels)
    n_clusters = len(clusters)
    n = len(labels)
    if n_clusters < 2 or n_clusters >= n:
        return np.nan

    scores = np.zeros(n, dtype=np.float64)
    for c in clusters:
        members = np.flatnonzero(labels == c)
        if len(members) == 1:
            scores[members[0]] = 0.0
            continue
        a = dist[np.ix_(members, members)].sum(axis=1) / (len(members) - 1)
        b = np.full(len(members), np.inf)
        for other in clusters:
            if other == c:
                continue
            others = np.flatnonzero(labels == other)
            b = np.minimum(b, dist[np.ix_(members, others)].mean(axis=1))
        denom = np.maximum(a, b)
        value = np.zeros(len(members), dtype=np.float64)
        nonzero = denom > 0
        value[nonzero] = (b[nonzero] - a[nonzero]) / denom[nonzero]
        scores[members] = value
    return float(scores.mean())

def euclidean_distance(X):
    # pairwise euclidean distance. used for the k-means silhouette only
    A = np.asarray(X, dtype=np.float64)
    squared_norm = np.sum(A * A, axis=1)
    squared = squared_norm[:, None] + squared_norm[None, :] - 2.0 * (A @ A.T)
    np.maximum(squared, 0.0, out=squared)
    dist = np.sqrt(squared)
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    return dist

def clustering_tables(sample_ids, method, params, labels, jaccard_dist, native_dist=None):
    # one clustering run, as three dataframes: assignments, one silhouette row, cluster sizes.
    # native_dist is set for k-means so silhouette_native is euclidean.
    # for the jaccard methods the two silhouette columns match
    labels = np.asarray(labels, dtype=int)
    n_noise = int(np.sum(labels < 0))
    n_clusters = int(len(np.unique(labels[labels >= 0])))
    sil_j = silhouette(jaccard_dist, labels)
    if native_dist is None:
        sil_n = sil_j
    else:
        sil_n = silhouette(native_dist, labels)
    assignments = pd.DataFrame({
        "sample": sample_ids,
        "method": method,
        "params": params,
        "cluster": labels,
    })
    scores = pd.DataFrame([{
        "method": method,
        "params": params,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "silhouette_jaccard": sil_j,
        "silhouette_native": sil_n,
    }])
    values, counts = np.unique(labels, return_counts=True)
    sizes = pd.DataFrame({
        "method": method,
        "params": params,
        "cluster": values.astype(int),
        "n_samples": counts.astype(int),
    })
    return assignments, scores, sizes

def plot_silhouette(summary):
    # jaccard silhouette against k. dbscan has no k, so it stays in the csv only
    plt.figure(figsize=(6, 4))
    k_rows = summary[summary["params"].str.startswith("k=")].copy()
    k_rows["k"] = k_rows["params"].str.slice(2).astype(int)
    labels = {
        "agglomerative_average": "agglomerative average",
        "agglomerative_complete": "agglomerative complete",
        "kmedoids": "k-medoids",
        "kmeans": "k-means",
    }
    for method in K_METHODS:
        rows = k_rows[k_rows["method"] == method].sort_values("k")
        rows = rows[np.isfinite(rows["silhouette_jaccard"])]
        if rows.empty:
            continue
        plt.plot(rows["k"], rows["silhouette_jaccard"], marker="o", label=labels[method])
    plt.xlabel("k")
    plt.ylabel("Jaccard silhouette")
    plt.title("Silhouette vs k")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "silhouette_vs_k.png"), dpi=300)
    plt.close()

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    X = load_matrix()
    if X is None:
        return

    print(f"Loaded {X.shape[0]} samples and {X.shape[1]} genes")
    print("Computing Jaccard distance")
    dist = jaccard_distance(X)
    empty = int((np.asarray(X).sum(axis=1) == 0).sum())
    print(f"Tumors with no retained mutations: {empty}")

    np.save(os.path.join(OUT_DIR, "jaccard_distance.npy"), dist)
    pd.DataFrame({"sample": X.index}).to_csv(os.path.join(OUT_DIR, "samples.csv"), index=False)
    distance_summary(dist).to_csv(os.path.join(OUT_DIR, "distance_summary.csv"), index=False)

    assignments = []
    scores = []
    sizes = []

    def keep(method, params, labels, native_dist=None):
        tables = clustering_tables(X.index, method, params, labels, dist, native_dist)
        assignments.append(tables[0])
        scores.append(tables[1])
        sizes.append(tables[2])

    for linkage_name in ("average", "complete"):
        print(f"agglomerative {linkage_name}")
        merges = agglomerative(dist, linkage_name)
        np.save(os.path.join(OUT_DIR, f"linkage_{linkage_name}.npy"), merges)
        for k in KS:
            labels = cut_tree(merges, len(X), k)
            keep(f"agglomerative_{linkage_name}", f"k={k}", labels)

    for k in KS:
        print(f"kmedoids k={k}")
        keep("kmedoids", f"k={k}", kmedoids(dist, k))

    print("k-means")
    euc = euclidean_distance(X)
    for k in KS:
        print(f"kmeans k={k}")
        keep("kmeans", f"k={k}", kmeans(X, k), native_dist=euc)

    print("dbscan")
    off = dist[np.triu_indices(len(dist), k=1)]
    seen_eps = set()
    for percentile in DBSCAN_PERCENTILES:
        eps = round(float(np.percentile(off, percentile)), 6)
        if eps in seen_eps:
            continue
        seen_eps.add(eps)
        for min_samples in DBSCAN_MIN_SAMPLES:
            params = f"eps={eps:.6f},min_samples={min_samples}"
            print(f"dbscan {params}")
            keep("dbscan", params, dbscan(dist, eps, min_samples))

    assignments = pd.concat(assignments, ignore_index=True)
    scores = pd.concat(scores, ignore_index=True)
    sizes = pd.concat(sizes, ignore_index=True)
    assignments.to_csv(os.path.join(OUT_DIR, "assignments.csv"), index=False)
    scores.to_csv(os.path.join(OUT_DIR, "silhouette.csv"), index=False)
    sizes.to_csv(os.path.join(OUT_DIR, "cluster_sizes.csv"), index=False)
    plot_silhouette(scores)

    print("\nSilhouette:\n", scores.to_string(index=False))
    print(f"\nResults saved to {OUT_DIR}/")

if __name__ == "__main__":
    main()
