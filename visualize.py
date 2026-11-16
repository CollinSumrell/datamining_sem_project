#!/usr/bin/env python3
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from compute_similarity import (
    DEFAULT_MATRIX,
    OUTPUT_DIR,
    jaccard_matrix,
    length_weights,
)

CLUSTER_DIR = os.path.join(OUTPUT_DIR, "clustering")
METHODS = ["pca", "mds", "tsne", "umap"]
DIMS = [2, 3]
SEED = 0


def feature_set_name(matrix_path, weighted):
    if matrix_path == DEFAULT_MATRIX:
        name = "all_genes"
    else:
        name = os.path.basename(matrix_path).replace("_matrix.csv", "")
    if weighted:
        name += "_length_weighted"
    return name


def output_dirs(matrix_path, weighted):
    if matrix_path == DEFAULT_MATRIX and not weighted:
        results_dir = OUTPUT_DIR
        fig_dir = os.path.join(OUTPUT_DIR, "figures")
    else:
        name = feature_set_name(matrix_path, weighted)
        results_dir = os.path.join(OUTPUT_DIR, "results", name)
        fig_dir = os.path.join(results_dir, "figures")
    return results_dir, fig_dir


def load_data(matrix_path):
    X = pd.read_csv(matrix_path, index_col=0)
    labels = pd.read_csv(os.path.join(OUTPUT_DIR, "sample_labels.csv"), index_col="sample")
    cohorts = labels.loc[X.index, "cohort"]
    return X, cohorts


def load_assignments(name):
    path = os.path.join(CLUSTER_DIR, name, "assignments.csv")
    if not os.path.exists(path):
        print(f"  no assignments at {path}, skipping cluster colorings")
        return None
    return pd.read_csv(path, dtype={"sample": str, "method": str, "params": str})


def cached_distance(name, index):
    d_path = os.path.join(CLUSTER_DIR, name, "jaccard_distance.npy")
    s_path = os.path.join(CLUSTER_DIR, name, "samples.csv")
    if not (os.path.exists(d_path) and os.path.exists(s_path)):
        return None

    samples = pd.read_csv(s_path)["sample"].astype(str).tolist()
    wanted = index.astype(str)
    if set(samples) != set(wanted):
        print("  cached distance matrix doesn't match these samples, recomputing")
        return None

    D = np.load(d_path)
    if D.shape != (len(samples), len(samples)):
        print("  cached distance matrix has the wrong shape, recomputing")
        return None

    order = pd.Series(range(len(samples)), index=samples).reindex(wanted).to_numpy()
    print(f"  using {d_path}")
    return D[np.ix_(order, order)]


def distance_matrix(X, weights):
    sim = jaccard_matrix(X, weights)
    D = np.nan_to_num(1.0 - sim.astype(np.float64), nan=0.0)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0
    return np.clip(D, 0.0, None)


def fit_one(method, n_dims, X, D, weights):
    if method == "pca":
        from sklearn.decomposition import PCA
        A = X.to_numpy(dtype=np.float64)
        if weights is not None:
            A = A * weights
        return PCA(n_components=n_dims, random_state=SEED).fit_transform(A)

    if method == "mds":
        from sklearn.manifold import MDS
        try:
            return MDS(n_components=n_dims, metric="precomputed", init="random",
                       random_state=SEED, n_init=1).fit_transform(D)
        except TypeError:
            return MDS(n_components=n_dims, dissimilarity="precomputed",
                       random_state=SEED, n_init=1,
                       normalized_stress="auto").fit_transform(D)

    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=n_dims, metric="precomputed", init="random",
                    perplexity=30, random_state=SEED).fit_transform(D)

    if method == "umap":
        import umap
        return umap.UMAP(n_components=n_dims, metric="precomputed",
                         random_state=SEED).fit_transform(D)

    raise ValueError(f"unknown method {method}")


def compute_embeddings(X, D, weights, methods, results_dir, force=False):
    cache = os.path.join(results_dir, "embeddings.csv")

    if os.path.exists(cache) and not force:
        cached = pd.read_csv(cache, dtype={"sample": str})
        have = set(zip(cached["method"], cached["n_dims"]))
    else:
        cached = None
        have = set()

    rows = [cached] if cached is not None else []
    added = False

    for method in methods:
        for n_dims in DIMS:
            if (method, n_dims) in have:
                print(f"  {method} {n_dims}d: cached")
                continue
            print(f"  {method} {n_dims}d: fitting...", flush=True)
            coords = fit_one(method, n_dims, X, D, weights)
            rows.append(pd.DataFrame({
                "sample": X.index,
                "method": method,
                "n_dims": n_dims,
                "dim1": coords[:, 0],
                "dim2": coords[:, 1],
                "dim3": coords[:, 2] if n_dims == 3 else np.nan,
            }))
            added = True

    out = pd.concat(rows, ignore_index=True)
    if added:
        out.to_csv(cache, index=False)
        print(f"  embeddings cached to {cache}")
    return out


def burden_series(X):
    raw = X.sum(axis=1)
    w = length_weights(X.columns)
    norm = pd.Series(X.to_numpy(dtype=np.float64) @ w, index=X.index)
    return {"burden": raw, "burden_normalized": norm}


def cluster_colorings(assignments, index, only_methods=None, only_params=None):
    if assignments is None:
        return {}

    out = {}
    for (method, params), group in assignments.groupby(["method", "params"], sort=True):
        if only_methods and method not in only_methods:
            continue
        if only_params and params not in only_params:
            continue
        series = group.set_index("sample")["cluster"].reindex(index)
        tag = "cluster_{}_{}".format(
            method,
            str(params).replace("=", "").replace(" ", "").replace(",", "_"),
        )
        out[tag] = series
    return out


def plot_embedding(coords, values, title, path, categorical, n_dims):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d") if n_dims == 3 else fig.add_subplot()
    xyz = [coords[:, i] for i in range(n_dims)]

    if categorical:
        keys = pd.Series(values).dropna().unique()
        keys = sorted(keys, key=lambda v: (isinstance(v, str), v))
        cmap = plt.get_cmap("tab20" if len(keys) > 10 else "tab10")
        for i, key in enumerate(keys):
            mask = (values == key).to_numpy()
            ax.scatter(*[a[mask] for a in xyz], s=6, alpha=0.7,
                       color=cmap(i % cmap.N), label=str(key), linewidths=0)
        if len(keys) <= 20:
            ax.legend(markerscale=2, fontsize=8, loc="best", framealpha=0.9)
    else:
        v = pd.Series(values).to_numpy(dtype=float)
        positive = v[v > 0]
        norm = LogNorm(vmin=positive.min(), vmax=v.max()) if positive.size else None
        sc = ax.scatter(*xyz, c=np.where(v > 0, v, np.nan), s=6, alpha=0.7,
                        cmap="viridis", norm=norm, linewidths=0)
        fig.colorbar(sc, ax=ax, shrink=0.75, label=title.split(" - ")[-1])

    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    if n_dims == 3:
        ax.set_zlabel("dim 3")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", nargs="?", default=DEFAULT_MATRIX)
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--cluster-methods", nargs="+", default=None)
    parser.add_argument("--cluster-params", nargs="+", default=["k=4"])
    parser.add_argument("--all-cluster-configs", action="store_true")
    parser.add_argument("--no-clusters", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    name = feature_set_name(args.matrix, args.weighted)
    results_dir, fig_dir = output_dirs(args.matrix, args.weighted)
    os.makedirs(fig_dir, exist_ok=True)

    X, cohorts = load_data(args.matrix)
    print(f"Loaded {X.shape[0]} samples and {X.shape[1]} genes ({name})")

    weights = length_weights(X.columns) if args.weighted else None

    colorings = {"cohort": (cohorts, True)}
    for key, series in burden_series(X).items():
        colorings[key] = (series, False)

    if not args.no_clusters:
        only_params = None if args.all_cluster_configs else args.cluster_params
        found = cluster_colorings(load_assignments(name), X.index,
                                  args.cluster_methods, only_params)
        for key, series in found.items():
            colorings[key] = (series, True)

    n_figs = len(args.methods) * len(DIMS) * len(colorings)
    print(f"{len(colorings)} colorings x {len(args.methods)} methods x {len(DIMS)} dims "
          f"= {n_figs} figures into {fig_dir}/")
    if args.dry_run:
        for key in colorings:
            print(f"  {key}")
        return

    print("Embeddings:")
    if set(args.methods) - {"pca"}:
        D = cached_distance(name, X.index)
        if D is None:
            D = distance_matrix(X, weights)
    else:
        D = None
    embeddings = compute_embeddings(X, D, weights, args.methods, results_dir, args.force)

    print("Figures:")
    for method in args.methods:
        for n_dims in DIMS:
            sub = embeddings[(embeddings["method"] == method)
                             & (embeddings["n_dims"] == n_dims)]
            sub = sub.set_index("sample").reindex(X.index)
            coords = sub[["dim1", "dim2", "dim3"][:n_dims]].to_numpy(dtype=float)

            for key, (values, categorical) in colorings.items():
                path = os.path.join(fig_dir, f"{method}_{n_dims}d_{key}.png")
                plot_embedding(coords, values, f"{method.upper()} {n_dims}D - {key}",
                               path, categorical, n_dims)
            print(f"  {method} {n_dims}d: {len(colorings)} figures")

    print(f"\nfigs saved to {fig_dir}/")


if __name__ == "__main__":
    main()