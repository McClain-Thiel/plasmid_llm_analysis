#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

# Matching reference qc_oriv_arg2.py: combine per-sequence filtered CSVs
# Filtering + overlap resolution already done in BLAST_ORI process

files = list(Path('.').glob('*.ori_hits.tsv'))
dfs = []

for f in files:
    if f.stat().st_size > 0:
        df = pd.read_csv(f)  # Already CSV format from BLAST_ORI post-processing
        if not df.empty:
            dfs.append(df)

if dfs:
    combined = pd.concat(dfs, ignore_index=True)
else:
    combined = pd.DataFrame(columns=['sequence', 'ori_type', 'pct_identity', 'pct_cov_subject', 'q_start', 'q_end'])

combined.to_csv('aggregate_ori_calls.csv', index=False)
