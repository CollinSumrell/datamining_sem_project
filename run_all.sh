#!/usr/bin/env bash

# first time: chmod +x run_all.sh

# you need these in data/
# gencode.v50.pc_transcripts.fa.gz (https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_50/gencode.v50.pc_transcripts.fa.gz)
# hgnc_complete_set.txt (https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt)
# all four cbioportal study folders (get from google doc links)

set -e
cd "$(dirname "$0")"

if [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi

# Feature sets to run every analysis on. "" means the full combined matrix.
FEATURE_SETS=(
    ""
    "output/features/frequent_genes_matrix.csv"
    "output/features/no_hypermutated_matrix.csv"
    "output/features/frequent_genes_no_hypermutated_matrix.csv"
)
# Feature sets to also run with gene-length weighting
WEIGHTED_SETS=(
    ""
    "output/features/no_hypermutated_matrix.csv"
)

step() {
    echo
    echo "=================================================="
    echo "  $1"
    echo "=================================================="
}

# --- Check the input data is there before spending time on anything ---
step "Checking input data"
for study in coadread luad prad ucec; do
    file="data/${study}_tcga_pan_can_atlas_2018/data_mutations.txt"
    if [ ! -f "$file" ]; then
        echo "Missing $file"
        exit 1
    fi
done
if ! ls data/gencode.*.pc_transcripts.fa.gz > /dev/null 2>&1; then
    echo "Missing data/gencode.vNN.pc_transcripts.fa.gz (see make_gene_lengths.py)"
    exit 1
fi
if [ ! -f data/hgnc_complete_set.txt ]; then
    echo "Missing data/hgnc_complete_set.txt (see make_gene_lengths.py)"
    exit 1
fi
echo "All input files found"

# --- 1. Build the 0/1 matrices ---
step "1. Building mutation matrices"
python mutation_matrices.py

# --- 2. Alternative feature sets and gene lengths ---
step "2. Building feature sets"
python make_feature_sets.py

step "3. Getting gene lengths"
python make_gene_lengths.py

# --- 3. Similarity analysis on every version ---
for matrix in "${FEATURE_SETS[@]}"; do
    step "4. Similarity: ${matrix:-all genes}"
    python compute_similarity.py $matrix
done
for matrix in "${WEIGHTED_SETS[@]}"; do
    step "4. Similarity: ${matrix:-all genes} (length weighted)"
    python compute_similarity.py $matrix --weighted
done

# --- 4. Clustering on the agreed versions ---
step "5. Clustering: all genes"
python cluster_tumors.py
step "5. Clustering: no hypermutated (length weighted)"
python cluster_tumors.py output/features/no_hypermutated_matrix.csv --weighted
step "5. Clustering: frequent genes, no hypermutated"
python cluster_tumors.py output/features/frequent_genes_no_hypermutated_matrix.csv


# --- 5. Permutation tests on all six versions ---
# These four versions reuse saved similarity scores and use 9,999 permutations
step "6. Permutation test: all genes"
python permutation_test.py output/pairwise_similarity.npz

step "6. Permutation test: all genes, length weighted"
python permutation_test.py output/results/all_genes_length_weighted/pairwise_similarity.npz

step "6. Permutation test: no hypermutated"
python permutation_test.py output/results/no_hypermutated/pairwise_similarity.npz

step "6. Permutation test: no hypermutated, length weighted"
python permutation_test.py output/results/no_hypermutated_length_weighted/pairwise_similarity.npz

# Frequent-gene selection uses cancer labels, so repeat gene selection and similarity calculation for each shuffle. Use 499 permutations to limit runtime.
# (Minimum p-value is 0.002, compared with 0.0001 for the other 4 versions)
step "6. Permutation test: frequent genes"
python permutation_test_frequent.py output/combined_matrix.csv --permutations 499

step "6. Permutation test: frequent genes, no hypermutated"
python permutation_test_frequent.py output/features/no_hypermutated_matrix.csv --permutations 499

# --- 6. Evaluate clustering on the three agreed versions ---
step "7. Evaluation: all genes"
python evaluate_clusters.py output/clustering/all_genes

step "7. Evaluation:  no hypermutated (length weighted)"
python evaluate_clusters.py output/clustering/no_hypermutated_length_weighted

step "7. Evaluation: frequent genes, no hypermutated"
python evaluate_clusters.py output/clustering/frequent_genes_no_hypermutated

# Add visualization scripts here once they exist, e.g.
# python visualize.py

step "Done"
echo "Similarity results: output/ (all genes) and output/results/<feature set>/"
echo "Clustering results: output/clustering/<feature set>/"