import os
import pandas as pd

DATA_DIR = "data"
OUTPUT_DIR = "output"

COHORTS = {
    "COAD": "coadread_tcga_pan_can_atlas_2018",
    "LUAD": "luad_tcga_pan_can_atlas_2018",
    "PRAD": "prad_tcga_pan_can_atlas_2018",
    "UCEC": "ucec_tcga_pan_can_atlas_2018",
}

# pick types of mutations we want to keep
# (silent and non-coding mutations 
NONSYNONYMOUS = [
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "In_Frame_Del",
    "In_Frame_Ins",
    "Splice_Site",
    "Translation_Start_Site",
    "Nonstop_Mutation"
]

def load_mutations(study):
    # load mutation data for a given study
    path = os.path.join(DATA_DIR, study, "data_mutations.txt")

    df = pd.read_csv(path, sep="\t", usecols=["Hugo_Symbol", "Variant_Classification", "Tumor_Sample_Barcode"], dtype=str)
    return df

def make_binary_matrix(muts):
    # create a binary mutation matrix from df
    # samples are rows, genes are columns, 1 if mutated, 0 if not
    all_samples = sorted(muts["Tumor_Sample_Barcode"].unique())

    kept = muts[muts["Variant_Classification"].isin(NONSYNONYMOUS)]
    kept = kept[kept["Hugo_Symbol"] != "Unknown"]

    matrix = pd.crosstab(kept["Tumor_Sample_Barcode"], kept["Hugo_Symbol"])
    matrix = (matrix > 0).astype(int)

    matrix = matrix.reindex(index=all_samples, fill_value=0)
    matrix.index.name = "Tumor_Sample_Barcode"
    matrix.columns.name = None

    return matrix

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # create and save matrix for each
    matrices = {}
    for label, study in COHORTS.items():
        muts = load_mutations(study)
        matrix = make_binary_matrix(muts)
        matrices[label] = matrix

        out_path = os.path.join(OUTPUT_DIR, f"{label}_matrix.csv")
        matrix.to_csv(out_path)
        print(f"{label} matrix saved to {out_path} with {matrix.shape[0]} samples and {matrix.shape[1]} genes")

    # save combined matrix 
    combined = pd.concat(matrices.values(), axis=0).fillna(0).astype(int)
    combined = combined[sorted(combined.columns)]
    combined.index.name = "Tumor_Sample_Barcode"

    if combined.index.duplicated().any():
        print("some samples are duplicated???")

    combined_path = os.path.join(OUTPUT_DIR, "combined_matrix.csv")
    combined.to_csv(combined_path)
    print(f"Combined matrix saved to {combined_path} with {combined.shape[0]} samples and {combined.shape[1]} genes")

    # save pairs of sample and cohort labels
    labels = []
    for label, study in matrices.items():
        for sample in matrix.index:
            labels.append({"sample": sample, "cohort": label})
    labels_path = os.path.join(OUTPUT_DIR, "sample_labels.csv")
    pd.DataFrame(labels).to_csv(labels_path, index=False)
    print(f"Sample labels saved to {labels_path} with {len(labels)} samples")

if __name__ == "__main__":
    main()