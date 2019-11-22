# ODDS Datasets

Real-world outlier detection benchmarks from
[ODDS](http://odds.cs.stonybrook.edu/) (Outlier Detection DataSets),
distributed here as MATLAB `.mat` files (keys: `X`, `y`) — the same format
used by early PyOD tutorials (2019).

## Bundled files

| File | Samples | Features | Outlier ratio (approx.) | Domain |
|------|--------:|---------:|------------------------:|--------|
| `cardio.mat` | 1831 | 21 | ~9.6% | Cardiotocography |
| `ionosphere.mat` | 351 | 33 | ~35.9% | Radar returns |
| `arrhythmia.mat` | 452 | 274 | ~14.6% | ECG / arrhythmia |
| `pima.mat` | 768 | 8 | ~34.9% | Diabetes screening |

Labels: `0` = inlier, `1` = outlier.

## Usage

```python
from utils.data_loading import load_odds_mat, describe_dataset

X, y, meta = load_odds_mat('cardio')
describe_dataset('cardio')
```

Or run the full benchmark:

```bash
python benchmark_odds.py
python benchmark_odds.py --datasets cardio ionosphere
```

## Source / license note

Datasets originate from ODDS aggregations of public UCI / medical / sensor
collections. Please cite the original dataset papers and the ODDS project when
using these benchmarks in publications.
