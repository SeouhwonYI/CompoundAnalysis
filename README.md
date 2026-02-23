# CompoundAnalysis

## Cuisine-only cluster analysis (Python runner)

To avoid large notebook diffs, `analysis/cuisine_only_cluster_analysis.py` is the canonical runner.

### Input
- `result/graph/<cuisine>/000_recipe_molecule_edges.csv`

### Output
- `result/analysis/<cuisine>/000_cluster_summary.csv`
- `result/analysis/<cuisine>/001_molecule_cluster_membership.csv`
- `result/analysis/<cuisine>/002_recipe_cluster_mass.csv`
- `result/analysis/<cuisine>/checkpoints/*.json`
- `result/plots/<cuisine>/000_cluster_overview.png`

### Run (default Thai/Korean)
```bash
python analysis/cuisine_only_cluster_analysis.py
```

### Run all cuisines
```bash
python analysis/cuisine_only_cluster_analysis.py --all-cuisines
```
