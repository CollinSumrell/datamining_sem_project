import os
import gzip
import pandas as pd
from compute_similarity import OUTPUT_DIR
from make_feature_sets import FEATURE_DIR

GENCODE_FILE = os.path.join("data", "gencode.v50.pc_transcripts.fa.gz")
HGNC_FILE = os.path.join("data", "hgnc_complete_set.txt")

def read_cds_lengths(path):
    lengths = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            if not line.startswith(">"):
                continue

            fields = line[1:].strip().split("|")
            gene = fields[5]
            cds = [x for x in fields if x.startswith("CDS:")]
            if not cds:
                continue
            start, end = cds[0][4:].split("-")
            length = int(end) - int(start) + 1
            lengths[gene] = max(length, lengths.get(gene, 0))
    return pd.Series(lengths, name="cds_length")

def read_renames(path):
    hgnc = pd.read_csv(path, sep="\t", usecols=["symbol", "prev_symbol"], dtype=str)
    renames = {}
    for current, previous in zip(hgnc["symbol"], hgnc["prev_symbol"]):
        if pd.isna(previous):
            continue
        for old in previous.split("|"):
            renames[old] = current

    return renames

def main():
    os.makedirs(FEATURE_DIR, exist_ok=True)

    lengths = read_cds_lengths(GENCODE_FILE)
    print(f"Read CDS lengths for {len(lengths)} genes from {GENCODE_FILE}")

    matrix_genes = pd.read_csv(os.path.join(OUTPUT_DIR, "combined_matrix.csv"), index_col=0, nrows=0).columns
    renames = read_renames(HGNC_FILE)
    rows = []
    n_direct, n_renamed = 0, 0
    for gene in matrix_genes:
        if gene in lengths.index:
            rows.append({"gene": gene, "cds_length": lengths[gene], "current_name": gene})
            n_direct += 1
        elif renames.get(gene) in lengths.index:
            rows.append({"gene": gene, "cds_length": lengths[renames[gene]], "current_name": renames[gene]})
            n_renamed += 1
    out = pd.DataFrame(rows)
    missing = matrix_genes[~matrix_genes.isin(out["gene"])]

    print(f"Matched {n_direct} matrix genes directly and {n_renamed} more through HGNC renames")
    print(f"{len(missing)} of {len(matrix_genes)} matrix genes still have no length")
    print("Examples still not found:", list(missing[:15]))

    out.to_csv(os.path.join(FEATURE_DIR, "gene_lengths.csv"), index=False)
    pd.Series(missing, name="gene").to_csv(os.path.join(FEATURE_DIR, "genes_without_lengths.csv"), index=False)
    print(f"Saved {len(out)} gene lengths to {FEATURE_DIR}/gene_lengths.csv")
    print("CDS length summary (bases):")
    print(out["cds_length"].describe().round().to_string())

if __name__ == "__main__":
    main()