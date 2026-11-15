# Mutation Patterns Across Four Cancers

## Checkpoint 2 Clustering (Preston)

Cancer-type labels are not used. Paths and k are set at the top of `cluster_tumors.py`.

```bash
pip install -r requirements.txt
python mutation_matrices.py
python cluster_tumors.py
```

Writes `output/clustering/`. Join `assignments.csv` to `output/sample_labels.csv` on `sample`.

For another 0/1 matrix, change `MATRIX_PATH` and `OUT_DIR`.
