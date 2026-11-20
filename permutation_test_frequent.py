"""Permutation tests with frequent-gene selection repeated for each label shuffle.

Input must be the mutation matrix BEFORE frequent-gene filtering:
    python permutation_test_frequent.py output/combined_matrix.csv
    python permutation_test_frequent.py output/features/no_hypermutated_matrix.csv

Uses the statistics and Holm adjustment from permutation_test.py, and the gene
selection and Jaccard similarity functions from the project's existing scripts.

Tests whether within-type similarity exceeds between-type similarity using
shuffled cancer labels. Cohort sizes are preserved. Gene selection, similarities,
and valid-pair counts are recalculated for every shuffle.

Defaults to 499 permutations. Use --permutations to override the count.
Results are saved under output/results/<version>/permutation_results.csv.
Use --output to override the destination. Existing output files are replaced.
"""

import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd

from compute_similarity import load_data, jaccard_matrix
from make_feature_sets import frequent_genes
from permutation_test import calculate_statistics, holm_adjust, RANDOM_SEED

N_PERMUTATIONS = 499


def statistics_for_labels(X, labels, n_groups):
    """Select genes with these labels, calculate similarities, and summarize pairs."""
    cohorts = pd.Series(labels, index=X.index)
    selected = frequent_genes(X, cohorts)

    if selected.shape[1] == 0:
        raise ValueError("No genes passed frequent-gene selection.")

    sim = jaccard_matrix(selected).astype(np.float64)
    valid = np.isfinite(sim)
    sim = np.where(valid, sim, 0.0)
    row_sums = sim.sum(axis=1)
    row_counts = valid.sum(axis=1)

    return calculate_statistics(sim, valid, row_sums, row_counts, labels, n_groups)


def main():
    parser = argparse.ArgumentParser(description="Permutation test with frequent-gene selection repeated for each shuffle.")
    parser.add_argument("input", type=Path, help="Mutation CSV before frequent-gene filtering.")
    parser.add_argument("--output", type=Path, help="Optional results CSV destination.")
    parser.add_argument("--permutations", type=int, default=N_PERMUTATIONS, help="Number of shuffles.")
    args = parser.parse_args()

    if args.permutations < 1:
        parser.error("--permutations must be positive.")

    input_name = args.input.stem.removesuffix("_matrix")
    version = "frequent_genes" if input_name == "combined" else f"frequent_genes_{input_name}"
    output_path = args.output if args.output is not None else Path("output/results")/version/"permutation_results.csv"

    start = perf_counter()
    X, cohorts = load_data(str(args.input))

    if X.empty or X.index.duplicated().any():
        raise ValueError("The matrix needs unique sample IDs and nonempty dimensions.")
    if cohorts.isna().any() or (cohorts == "").any():
        raise ValueError("Cancer labels must not be missing or empty.")
    if not np.isin(X.to_numpy(), [0, 1]).all():
        raise ValueError("The mutation matrix must contain only 0 and 1.")

    names = list(pd.unique(cohorts))
    labels = pd.Categorical(cohorts, categories=names).codes

    if len(names) < 2 or "ALL" in names or np.any(np.bincount(labels) < 2):
        raise ValueError("Need at least two cohorts with two tumors each; ALL is reserved.")

    print(f"Input: {args.input}", flush=True)
    print(f"Loaded {len(X)} tumors and {X.shape[1]} genes before filtering.", flush=True)

    within, between, n_within, n_between = statistics_for_labels(X, labels, len(names))
    observed_gaps = within - between

    rng = np.random.default_rng(RANDOM_SEED)
    extreme_counts = np.zeros(len(names) + 1, dtype=int)
    gap_sum = np.zeros(len(names) + 1)
    gap_squared_sum = np.zeros(len(names) + 1)

    for iteration in range(1, args.permutations + 1):
        # Repeat gene selection using the shuffled labels.
        shuffled = rng.permutation(labels)
        shuffled_within, shuffled_between, _, _ = statistics_for_labels(X, shuffled, len(names))
        gaps = shuffled_within - shuffled_between

        # Count shuffled gaps at least as large as observed, including ties.
        extreme_counts += gaps >= observed_gaps - 1e-12
        gap_sum += gaps
        gap_squared_sum += gaps**2

        if iteration == 1 or iteration % 50 == 0 or iteration == args.permutations:
            print(f"Completed {iteration}/{args.permutations} permutations", flush=True)

    # The +1 correction prevents reporting a Monte Carlo p-value of zero.
    p_values = (extreme_counts + 1)/(args.permutations + 1)
    null_mean = gap_sum/args.permutations

    # Population SD of shuffled gaps, not an error bar for the observed gap.
    null_sd = np.sqrt(np.maximum(0.0, gap_squared_sum/args.permutations - null_mean**2))

    results = pd.DataFrame({
        "cohort": names + ["ALL"],
        "mean_within": within,
        "mean_between": between,
        "within_minus_between": observed_gaps,
        "n_within_pairs": n_within,
        "n_between_pairs": n_between,
        "null_mean_gap": null_mean,
        "null_sd_gap": null_sd,
        "p_value": p_values,
        "p_val_adjusted_holm": np.r_[holm_adjust(p_values[:-1]), np.nan],
        "n_permutations": args.permutations,
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(results.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nSaved to {output_path.resolve()}")
    print(f"Total time: {(perf_counter() - start)/60:.1f} minutes.")


if __name__ == "__main__":
    main()