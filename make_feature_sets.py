import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from compute_similarity import load_data, OUTPUT_DIR

FEATURE_DIR = os.path.join(OUTPUT_DIR, "features")

MIN_FRACTION = 0.02
HYPERMUTATED_CUTOFF = 2000
CANDIDATE_CUTOFFS = [250, 500, 1000, 2000]

def frequent_genes(X, cohorts):
    keep = pd.Series(False, index=X.columns)
    for c in cohorts.unique():
        frac = X[cohorts == c].mean(axis=0)
        keep = keep | (frac > MIN_FRACTION)
    return X.loc[:, keep]

def remove_hypermutated(X, burden):
    kept = X[burden < HYPERMUTATED_CUTOFF]
    return kept.loc[:, kept.sum(axis=0) > 0]

def print_cutoff_counts(burden, cohorts):
    rows = []
    for cutoff in CANDIDATE_CUTOFFS:
        row = {"cutoff": cutoff}
        for c in cohorts.unique():
            row[c] = int((burden[cohorts == c] > cutoff).sum())
        row["total"] = int((burden > cutoff).sum())
        rows.append(row)
    print("\nTumors flagged as hypermutated at each candidate cutoff:")
    print(pd.DataFrame(rows).to_string(index=False))

def plot_burden_histogram(burden):
    bins = np.logspace(0, np.log10(burden.max() + 1), 60)
    plt.figure(figsize=(7, 4))
    plt.hist(burden + 1, bins=bins, color="gray")
    plt.xscale("log")
    plt.axvline(HYPERMUTATED_CUTOFF, color="red", linestyle="--", label=f"cutoff = {HYPERMUTATED_CUTOFF} genes")
    plt.xlabel("Mutated genes per tumor (log scale)")
    plt.ylabel("Number of tumors")
    plt.title("Mutation burden distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FEATURE_DIR, "burden_histogram.png"), dpi=300)
    plt.close()

def save(matrix, name):
    path = os.path.join(FEATURE_DIR, f"{name}_matrix.csv")
    matrix.to_csv(path)
    print(f"{name}: {matrix.shape[0]} tumors x {matrix.shape[1]} genes saved to {path}")

def main():
    os.makedirs(FEATURE_DIR, exist_ok=True)

    X, cohorts = load_data()
    burden = X.sum(axis=1)
    print(f"Loaded {X.shape[0]} tumors x {X.shape[1]} genes")

    print_cutoff_counts(burden, cohorts)
    plot_burden_histogram(burden)
    flagged = burden > HYPERMUTATED_CUTOFF
    hyper = pd.DataFrame({"cohort": cohorts[flagged], "genes_mutated": burden[flagged]})
    hyper.sort_values("genes_mutated", ascending=False).to_csv(os.path.join(FEATURE_DIR, "hypermutated_tumors.csv"))
    print(f"\nUsing cutoff of {HYPERMUTATED_CUTOFF} genes, {flagged.sum()} tumors flagged as hypermutated")

    save(frequent_genes(X, cohorts), "frequent_genes")

    no_hyper = remove_hypermutated(X, burden)
    save(no_hyper, "no_hypermutated")

    save(frequent_genes(no_hyper, cohorts[no_hyper.index]), "frequent_genes_no_hypermutated")

if __name__ == "__main__":
    main()