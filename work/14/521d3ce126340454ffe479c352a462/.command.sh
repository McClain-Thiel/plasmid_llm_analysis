#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

files = list(Path('.').glob('*.amr_hits.tsv'))
dfs = []

for f in files:
    if f.stat().st_size > 0:
        df = pd.read_csv(f, sep='\t')
        if len(df) > 0:
            # Use actual contig/sequence ID from AMR output
            if 'Contig id' in df.columns:
                df['sequence'] = df['Contig id']
            else:
                sample_id = f.stem.replace('.amr_hits', '')
                df['sequence'] = sample_id
            dfs.append(df)

if dfs:
    combined = pd.concat(dfs, ignore_index=True)
    # Rename columns to match expected format
    combined = combined.rename(columns={
        'Element symbol': 'symbol',
        'Element name': 'name',
        '% Identity to reference': 'pct_identity',
        '% Coverage of reference': 'pct_cov'
    })
else:
    combined = pd.DataFrame(columns=['sequence', 'symbol', 'name', 'pct_identity', 'pct_cov'])

combined.to_csv('aggregate_amr_calls.csv', index=False)
