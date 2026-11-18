"""Permutation test for project's within-versus-between similarity hypothesis.

Run the following after compute_similarity.py has saved pairwise_similarity.npz(shows with or without path options):
    python permutation_test.py
    python permutation_test.py output/results/all_genes_length_weighted/pairwise_similarity.npz

The default input path is output/pairwise_similarity.npz. 

The script tests if mean within-cancer-type similarity exceeds the mean between-cancer similarity more than expected with labeling shuffled.
Shuffles preserve the cohort sizes.
The individual cancer-type tests receive a Holm-Bonferroni multiple-testing correction shown as an adjusted p-value.
The overall test compares the average similarity across all same-cancer pairs with the average across all different-cancer pairs.

Results are saved as permutation_results.csv in the same directory as the input file.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_INPUT = Path("output/pairwise_similarity.npz")
N_PERMUTATIONS = 9999 # 1 less than 10,000 so we get clean numbers after the +1 in p value calculation
RANDOM_SEED = 23

def load_data(input_path):
    """Load similarities and matching labels saved by compute_similarity.py."""
    with np.load(input_path, allow_pickle=False) as data:
        sim=data["similarity"].astype(np.float64)
        samples=data["samples"]
        cohorts=data["cohorts"]

    if samples.ndim != 1 or cohorts.ndim != 1:
        raise ValueError("Sample IDs and cancer labels must be one-dimensional.")
    
    if sim.shape != (len(samples), len(samples)) or len(cohorts) != len(samples): 
        raise ValueError("Similarity dimensions do not match the sample labels.")
    
    if len(np.unique(samples)) != len(samples):
        raise ValueError("Sample IDs must be unique.")
    
    if np.any(cohorts == ""): 
        raise ValueError("Cancer labels must not be empty.") 
    
    if np.isinf(sim).any(): 
        raise ValueError("Similarities must not contain infinite values.")

    names = list(pd.unique(cohorts))
    codes = pd.Categorical(cohorts, categories=names).codes

    if len(names) < 2 or "ALL" in names or np.any(np.bincount(codes) < 2):
        raise ValueError("Need at least two cohorts with two tumors each; ALL is reserved.")

    # Don't include tumors' similarities to themselves
    np.fill_diagonal(sim, np.nan)  
    return sim, codes, names  
 
def calculate_statistics(sim, valid, row_sums, row_counts, labels, n_groups):
    """Calculate within/between means and unique-pair counts, including ALL test."""
    within_sum = np.zeros(n_groups+1) 
    between_sum = np.zeros(n_groups+1)  
    within_count = np.zeros(n_groups+1, dtype=int)
    between_count = np.zeros(n_groups+1, dtype=int) 

    for group in range(n_groups):
        idx = np.flatnonzero(labels == group)
        block = np.ix_(idx, idx)

        # Divide by 2 to account for pairs occuring two times in matrix at (i,j) and (j,i)
        within_sum[group] = sim[block].sum()/2
        within_count[group] = valid[block].sum()//2

        # These rows contain within pairs two times and between pairs only once.
        between_sum[group] = row_sums[idx].sum() - 2*within_sum[group]
        between_count[group] = row_counts[idx].sum() - 2*within_count[group]

    # ALL pools unique pairs rather than simply averaging the means of the cohorts.
    within_sum[-1] = within_sum[:-1].sum()
    within_count[-1] = within_count[:-1].sum()
    between_sum[-1] = row_sums.sum()/2 - within_sum[-1]
    between_count[-1] = row_counts.sum()//2 - within_count[-1]

    if np.any(within_count == 0) or np.any(between_count == 0): 
        raise ValueError("A group has no valid within/between pairs.")

    return within_sum/within_count, between_sum/between_count, within_count, between_count



def holm_adjust(p_values): 
    """Adjust the cancer-level p-values using holm-bonferroni method."""
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values) 
    multipliers = np.arange(len(p_values), 0, -1) 
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(p_values[order]*multipliers))
    return adjusted


def main():  
    parser = argparse.ArgumentParser(description="Test within-versus-between similarity by shuffling cancer labels.")
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT, help="Saved pairwise_similarity.npz file.")
    args = parser.parse_args()

    if N_PERMUTATIONS< 1:
        raise ValueError("N_PERMUTATIONS must be positive.")

    sim, labels, names = load_data(args.input)
    output_path = args.input.parent/"permutation_results.csv"

    print(f"Input: {args.input}", flush=True)
    print(f"Loaded {len(labels)} tumors from {len(names)} cohorts.", flush=True) 

    # Undefined pairs don't contribute to sums or counts
    valid = np.isfinite(sim)
    sim = np.where(valid, sim, 0.0)
    row_sums = sim.sum(axis=1)
    row_counts = valid.sum(axis=1)

    within, between, n_within, n_between = calculate_statistics(sim, valid, row_sums, row_counts, labels, len(names))
    observed_gaps = within - between

    rng = np.random.default_rng(RANDOM_SEED)
    extreme_counts = np.zeros(len(names) + 1, dtype=int)
    gap_sum = np.zeros(len(names) + 1)
    gap_squared_sum = np.zeros(len(names) + 1 )
  
    for iteration in range(1, N_PERMUTATIONS + 1):
        # Shuffle tumors labels
        shuffled = rng.permutation(labels) 
        shuffled_within, shuffled_between, _, _ = calculate_statistics(sim, valid, row_sums, row_counts, shuffled, len(names))
        gaps = shuffled_within - shuffled_between 
 
        # Include ties in count, allowing small rounding differences
        extreme_counts += gaps >= observed_gaps - 1e-12
        gap_sum += gaps
        gap_squared_sum += gaps**2


        if iteration == 1 or iteration % 500 == 0 or iteration == N_PERMUTATIONS:
            print(f"Completed {iteration}/{N_PERMUTATIONS} permutations", flush=True)

    # The +1 on both numerator and denominator prevents reporting p-value of zero
    p_values =(extreme_counts + 1)/(N_PERMUTATIONS + 1)
    null_mean = gap_sum/N_PERMUTATIONS

    # Population SD of shuffled gaps, not an error bar for the observed gap 
    null_sd = np.sqrt(np.maximum(0.0, gap_squared_sum / N_PERMUTATIONS - null_mean**2)) 


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
        "n_permutations": N_PERMUTATIONS,
    })
 
    results.to_csv(output_path, index=False) 
    print(results.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nSaved to {output_path.resolve()}") 


if __name__ == "__main__":
    main()