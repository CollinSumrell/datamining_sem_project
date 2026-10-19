import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "output"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
TOP_N_GENES = 10

def load_data():
    # for loading the data from csv into a df
    X = pd.read_csv(os.path.join(OUTPUT_DIR, "combined_matrix.csv"), index_col=0)
    labels = pd.read_csv(os.path.join(OUTPUT_DIR, "sample_labels.csv"), index_col="sample")
    cohorts = labels.loc[X.index, "cohort"]
    return X, cohorts

def summarize_cohorts(X, cohorts, order):
    # basic counts (mutation burden)
    burden = X.sum(axis=1)
    rows = []
    for c in order:
        in_cohort = cohorts == c
        b = burden[in_cohort]
        rows.append({
            "cohort": c,
            "tumors": len(b),
            "genes_mutated_in_any_tumor": int((X[in_cohort].sum(axis=0) > 0).sum()),
            "median_genes_per_tumor": b.median(),
            "mean_genes_per_tumor": round(b.mean(), 1),
            "max_genes_per_tumor": b.max(),
            "tumors_with_no_mutations": int((b ==0).sum()),
        })
    return pd.DataFrame(rows), burden

def top_genes(X, cohorts, order):
    # TOP_N_GENES in each cohort. takes in X (pandas dataframe with rows as samples
    # and columns as genes), cohorts (pandas series with one entry per tumor; basically
    # a dict of which sample belongs to COAD/LUAD/PRAD/UCEC in the order of the samples in X),
    # and order which is a list of the 4 cancer types that tells which order to process in

    # returns a dataframe with 40 rows (4 cohorts x 10 genes) and 4 columns (cohort e.g. COAD, rank 1-10 within
    # that cohort, gene name, and fraction of tumors mutated (share of that cohort's tumros with that gene mutated 
    # as a decimal 0-1))
    raise NotImplementedError

def jaccard_matrix(X):
    # jaccard similarity matrix for every pair of tumors
    A = X.to_numpy(dtype=np.float32)
    shared = A @ A.T
    counts = A.sum(axis=1)
    union = counts[:, None] + counts[None, :] - shared

    with np.errstate(divide="ignore", invalid="ignore"):
        sim = shared / union
    np.fill_diagonal(sim, np.nan)
    return sim

def similarity_by_cohort(sim, labels, order):
    # mean similarity for every pair of cohorts
    table = pd.DataFrame(index=order, columns=order, dtype=float)
    for a in order:
        for b in order:
            block = sim[np.ix_(labels == a, labels == b)]
            table.loc[a, b] = np.nanmean(block)
    return table

def within_between_values(sim, labels, c):
    # takes in sim, the similarity 2D matrix from jaccard_matrix, labels (a 1D
    # numpy array of 2098 strings giving each cancer type as a string in the same
    # order as sim), and c (once cancer type as a string, e.g. "COAD")

    # returns two 1D numpy arrays:
    # within, whihc holds the similarity of every pair of two different tumors that
    # are both type c. counts each pair once (include (i,j) but not (j,i)), leave out
    # each tumor compared with itself, drop any NaN.
    # between holds the similarity of every pair where one tumor is type c and the other
    # is not. 
    # order of values doesnt matter, just a helper for within_vs_between and plot_within_between
    raise NotImplementedError

def within_vs_between(sim, labels, order):
    # sim, labels, and order defined as above.
    # outputs a DataFrame with 5 rows, once per cancer type in order and then an ALL row
    # four cols: cohort, which will notate COAD, LUAD, PRAD, UCEC, or ALL
    # mean_within, an average of that cohort's within vals (so the mean of the within array
    # returned by within_between_values(sim, labels, c))
    # mean_between, same as mean_within but for between
    # within_minus_between, which is just mean_within minus mean_between for that cohort
    raise NotImplementedError

def plot_burden(burden, cohorts, order):
    raise NotImplementedError

def plot_heatmap(table):
    plt.figure(figsize=(5,4))
    plt.imshow(table.to_numpy(), cmap="Blues")
    plt.colorbar(label="Mean Jaccard similarity")
    plt.xticks(range(len(table)), table.columns)
    plt.yticks(range(len(table)), table.index)
    halfway = table.to_numpy().max() / 2
    for i in range(len(table)):
        for j in range(len(table)):
            value = table.iloc[i, j]
            color = "white" if value > halfway else "black"
            plt.text(j, i, f"{value:.4f}", ha="center", va="center", fontsize=9, color=color)
    plt.title("Mean similarity between cancer types")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "similarity_heatmap.png"), dpi=300)
    plt.close()

def plot_within_between(sim, labels, order):
    raise NotImplementedError

def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    X, cohorts = load_data()
    order = list(cohorts.unique())
    labels = cohorts.to_numpy()
    print(f"Loaded {X.shape[0]} samples and {X.shape[1]} genes from {len(order)} cohorts")

    stats, burden = summarize_cohorts(X, cohorts, order)
    stats.to_csv(os.path.join(OUTPUT_DIR, "cohort_stats.csv"), index=False)
    print("\nDescriptive stats:\n", stats.to_string(index=False))

    # top_genes(X, cohorts, order).to_csv(os.path.join(OUTPUT_DIR, "top_genes.csv"), index=False)

    sim = jaccard_matrix(X)

    table = similarity_by_cohort(sim, labels, order)
    table.round(4).to_csv(os.path.join(OUTPUT_DIR, "similarity_by_cohort.csv"))
    print("\nMean similarty by cohort pair:\n", table.round(4).to_string())

    # summary = within_vs_between(sim, labels, order)
    # summary.to_csv(os.path.join(OUTPUT_DIR, "within_vs_between.csv"), index=False)
    # print("\nWithin vs between:\n", summary.to_string(index=False))

    # plot_burden(burden, cohorts, order)
    plot_heatmap(table)
    # plot_within_between(sim, labels, order)
    # print(f"\nfigs saved to {FIG_DIR}/")

if __name__ == "__main__":
    main()

