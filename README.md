# CompoundAnalysis

## Pipeline Overview
Recommended flow:
1. `generate/generate.py` (or `generate/generate.ipynb`)
2. `analysis/cuisine_only_cluster_analysis.py`

`generate` builds cuisine-wise molecule graph inputs.
`analysis` consumes those outputs and produces summary plots + cluster tables.

## 1) Generate Stage
### Script
- `generate/generate.py`

### Main outputs
- `result/recipes_long_normalized.csv`
- `result/graph/recipe_molecule_edges.csv`
- `result/graph/recipe_unk_mass.csv`
- `result/graph/<cuisine>/000_recipe_molecule_edges.csv`
- `result/graph/<cuisine>/001_molecule_weight.csv`
- `result/graph/<cuisine>/002_molecule_recipe_edges.csv`
- `result/graph/ALL/*` (when ALL export is enabled)
- `result/analysis/analysis_summary.csv`

### Cuisine behavior
- If `--cuisines` is omitted: exports all cuisines + `ALL` (full dataset aggregate)
- If `--cuisines` is provided: exports only those cuisines
- Add `--include-all` to also export `ALL` with selected cuisines

### Run examples
All cuisines + ALL:
```bash
python generate/generate.py
```

Selected cuisines only:
```bash
python generate/generate.py --cuisines Korean Thai
```

Selected cuisines + ALL:
```bash
python generate/generate.py --cuisines Korean Thai --include-all
```

Optional cluster overview PNGs:
```bash
python generate/generate.py --export-cluster-overview-plots
```

## 2) Analysis Stage
### Script
- `analysis/cuisine_only_cluster_analysis.py`

### Input priority
1. `saved_state/*` (if present)
2. `result/graph/<cuisine>/000_recipe_molecule_edges.csv`
3. Ingredient fallback from `result/recipes_long_normalized.csv`

### Main outputs
- `result/plots/<cuisine>/<cuisine>_summary.png`
- `result/plots/<cuisine>/<cuisine>_moleculespace_graph.png`
- `result/plots/<cuisine>/<cuisine>_moleculespace_graph.html`
- `result/analysis/<cuisine>/000_cluster_summary.csv`

### Run examples
Auto-detect cuisines from `result/graph`:
```bash
python analysis/cuisine_only_cluster_analysis.py
```

Specific cuisines:
```bash
python analysis/cuisine_only_cluster_analysis.py --cuisines Korean Thai
```

All cuisines:
```bash
python analysis/cuisine_only_cluster_analysis.py --all-cuisines
```
