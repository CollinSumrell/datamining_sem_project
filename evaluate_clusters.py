"""Evaluate cluster assignments against the actual cancer types.

Run after cluster_tumors.py:
    python evaluate_clusters.py
    python evaluate_clusters.py output/clustering/no_hypermutated_length_weighted

Reads assignments.csv and silhouette.csv from the selected clustering folder.
Uses sample_labels.csv for cancer types and combined_matrix.csv for raw mutation
burden (number of mutated genes per tumor).

Calculates purity, ARI, NMI, cluster-by-cancer counts, and burden summaries.
Reuses silhouette scores already calculated by cluster_tumors.py.

DBSCAN noise is excluded from agreement metrics but retained in cluster summaries.
Results are written to an evaluation/ subfolder of the selected clustering folder.

Requires numpy, pandas, and scikit-learn.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

DEFAULT_INPUT = Path("output/clustering/all_genes")
LABELS_PATH = Path("output/sample_labels.csv")
MATRIX_PATH = Path("output/combined_matrix.csv")


def load_data(input_dir):
    """Join assignments with cancer labels and raw mutation burdens by sample ID."""
    assignments = pd.read_csv(input_dir / "assignments.csv", dtype={"sample": str})
    labels = pd.read_csv(LABELS_PATH, dtype={"sample": str})
    matrix = pd.read_csv(MATRIX_PATH, index_col=0)
    matrix.index = matrix.index.astype(str)
    burden = matrix.sum(axis=1).rename("mutation_burden")
    burden.index.name = "sample"

    if assignments.duplicated(["sample", "method", "params"]).any():
        raise ValueError("A sample has duplicate assignments within a clustering run.")

    data = assignments.merge(labels[["sample", "cohort"]], on="sample", how="left", validate="many_to_one")
    data = data.merge(burden.reset_index(), on="sample", how="left", validate="many_to_one")

    if data[["cohort", "cluster", "mutation_burden"]].isna().any().any():
        raise ValueError("Some samples have missing labels, assignments, or mutation burdens.")

    scores = pd.read_csv(input_dir / "silhouette.csv")
    return data, scores


def evaluate_runs(data):
    """Calculate agreement metrics for each method and configuration."""
    rows = []

    for (method, params), group in data.groupby(["method", "params"], sort=False):
        assigned = group[group["cluster"] >= 0]
        n_noise = int((group["cluster"] < 0).sum())
        purity = ari = nmi = np.nan

        if len(assigned) > 0:
            counts = pd.crosstab(assigned["cluster"], assigned["cohort"])
            purity = counts.max(axis=1).sum() / len(assigned)
            ari = adjusted_rand_score(assigned["cohort"], assigned["cluster"])
            nmi = normalized_mutual_info_score(assigned["cohort"], assigned["cluster"])

        rows.append({
            "method": method,
            "params": params,
            "n_samples": len(group),
            "n_evaluated": len(assigned),
            "n_noise": n_noise,
            "n_clusters": assigned["cluster"].nunique(),
            "purity": purity,
            "ARI": ari,
            "NMI": nmi,
        })

    return pd.DataFrame(rows)


def summarize_clusters(data):
    """Count cancer types and summarize raw mutation burden within each cluster."""
    keys = ["method", "params", "cluster"]
    counts = data.groupby(keys + ["cohort"]).size().unstack("cohort", fill_value=0)
    counts.columns.name = None

    burdens = data.groupby(keys)["mutation_burden"].agg(
        n_samples="size", mean_mutated_genes="mean", median_mutated_genes="median",
        min_mutated_genes="min", max_mutated_genes="max")

    return counts.reset_index(), burdens.reset_index()


def main():
    parser = argparse.ArgumentParser(description="Evaluate clusters against cancer types.")
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT, help="Clustering output folder.")
    args = parser.parse_args()

    data, scores = load_data(args.input)
    metrics = evaluate_runs(data)

    # Reuse the scores calculated with the distances used for this clustering run.
    score_columns = ["method", "params", "silhouette_jaccard", "silhouette_native"]
    metrics = metrics.merge(scores[score_columns], on=["method", "params"], how="left", validate="one_to_one")

    counts, burdens = summarize_clusters(data)
    output_dir = args.input / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(output_dir / "metrics.csv", index=False)
    counts.to_csv(output_dir / "cluster_by_cancer.csv", index=False)
    burdens.to_csv(output_dir / "mutation_burden_by_cluster.csv", index=False)

    # Agreement metrics for methods explicitly run with k=4.
    k4_metrics = metrics[metrics["params"] == "k=4"]
    k4_metrics.to_csv(output_dir / "k4_metrics.csv", index=False)

    # Cancer-type counts in the k-medoids clusters at k=4.
    kmedoids_counts = counts[(counts["method"] == "kmedoids") & (counts["params"] == "k=4")]
    kmedoids_counts = kmedoids_counts.drop(columns=["method", "params"])
    kmedoids_counts.to_csv(output_dir / "kmedoids_k4_counts.csv", index=False)

    print("\nk=4 evaluation:")
    print(k4_metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nk-medoids k=4 cancer counts:")
    print(kmedoids_counts.to_string(index=False))
    print(f"\nSaved evaluation files to {output_dir.resolve()}")

if __name__ == "__main__":
    main()