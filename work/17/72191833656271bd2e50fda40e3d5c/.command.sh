#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

files = list(Path('.').glob('*.metrics.csv'))
dfs = [pd.read_csv(f) for f in files if f.stat().st_size > 0]

if dfs:
    combined = pd.concat(dfs, ignore_index=True)
else:
    combined = pd.DataFrame()

combined.to_csv('aggregate_metrics.csv', index=False)
