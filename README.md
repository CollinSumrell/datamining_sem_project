# Mutation Patterns Across Four Cancers

## Setup

1. Create the virtual env and install the packages:

```bash
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Download the data into `data/`. Get the cbioportal stuff from the links in the proposal document, and gencode and hgnc files from the discord

## Running everything

```bash
chmod +x run_all.sh
./run_all.sh
```

## Feature sets

| Name | Tumors | Genes |
|---|---|---|
| `all_genes` | all | all |
| `frequent_genes` | all | mutated in >= 2% of tumors in at least one cohort |
| `no_hypermutated` | hypermutated tumors removed | all |
| `frequent_genes_no_hypermutated` | hypermutated tumors removed | frequent genes, recomputed without them |
| `..._length_weighted` (`--weighted`) | as above | each gene weighted by 1 / coding length |
