# Notebooks

Interactive tutorials (Jupyter) for this repo. From the repository root:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

| Notebook | Content |
|----------|---------|
| `01_pyod_quickstart.ipynb` | Synthetic data + kNN scores |
| `02_model_comparison.ipynb` | Multi-model ranking table |
| `03_odds_benchmark.ipynb` | ODDS real data (e.g. cardio) |

Scripts under the repo root (`compare_models.py`, `benchmark_odds.py`) are the
non-interactive counterparts used for CSV exports.
