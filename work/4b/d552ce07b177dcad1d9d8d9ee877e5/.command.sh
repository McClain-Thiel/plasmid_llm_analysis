#!/bin/bash -euo pipefail
# Run BLAST with 12-field outfmt (matching reference qc_oriv_arg2.py)
    echo -e "qseqid\tsseqid\tpident\tlength\tevalue\tbitscore\tqstart\tqend\tqlen\tsstart\tsend\tslen" > McClain_PlasmidGPT_RL_ATG.ori_hits.tsv

    blastn \
        -query sequences.fasta \
        -db ori_db \
        -outfmt "6 qseqid sseqid pident length evalue bitscore qstart qend qlen sstart send slen" \
        -evalue 0.00001 \
        -max_target_seqs 2000 \
        -num_threads 4 \
        -task dc-megablast \
        -soft_masking true \
        -dust yes >> McClain_PlasmidGPT_RL_ATG.ori_hits.tsv

    # Post-process: filter by thresholds and resolve overlaps per sequence
    # Matches reference qc_oriv_arg2.py filter_ori_hits + choose_non_overlapping_highest_identity
    python3 << 'PYEOF'
import pandas as pd
import numpy as np

df = pd.read_csv("McClain_PlasmidGPT_RL_ATG.ori_hits.tsv", sep='\t')

if not df.empty:
    # Calculate qcov, scovs (matching qc_oriv_arg2.py lines 55-60)
    df['qcov'] = 100.0 * df['length'] / df['qlen'].replace(0, np.nan)
    df['scovs'] = 100.0 * df['length'] / df['slen'].replace(0, np.nan)
    df['q_from'] = df[['qstart', 'qend']].min(axis=1).astype('Int64')
    df['q_to'] = df[['qstart', 'qend']].max(axis=1).astype('Int64')
    df['strand'] = np.where(df['sstart'] <= df['send'], '+', '-')

    # Filter by thresholds (matching qc_oriv_arg2.py filter_ori_hits)
    df = df[
        (df['pident'] >= 85) &
        (df['scovs'] >= 80) &
        (df['length'] >= 100)
    ]

    if not df.empty:
        # Sort best-first (matching qc_oriv_arg2.py line 85)
        df = df.sort_values(by=['pident', 'bitscore', 'scovs', 'length'], 
                           ascending=[False, False, False, False])

        # Resolve overlaps: keep highest identity (matching qc_oriv_arg2.py choose_non_overlapping_highest_identity)
        chosen = []
        intervals = []
        for _, r in df.iterrows():
            s, e = int(r['q_from']), int(r['q_to'])
            # Check overlap with already chosen intervals
            overlaps = any(not (e < cs or s > ce) for cs, ce in intervals)
            if not overlaps:
                chosen.append(r)
                intervals.append((s, e))

        df = pd.DataFrame(chosen) if chosen else pd.DataFrame()

    # Process each unique sequence ID separately (per-sequence overlap resolution)
    if not df.empty:
        df = df.rename(columns={'sseqid': 'ori_type', 'pident': 'pct_identity', 'scovs': 'pct_cov_subject', 'q_from': 'q_start', 'q_to': 'q_end'})
        
        # Group by qseqid and resolve overlaps per sequence
        all_results = []
        for seq_id in df['qseqid'].unique():
            seq_df = df[df['qseqid'] == seq_id].copy()
            
            # Sort best-first
            seq_df = seq_df.sort_values(by=['pct_identity', 'bitscore', 'pct_cov_subject', 'length'], 
                                        ascending=[False, False, False, False])
            
            # Resolve overlaps for this sequence
            chosen = []
            intervals = []
            for _, r in seq_df.iterrows():
                s, e = int(r['q_start']), int(r['q_end'])
                overlaps = any(not (e < cs or s > ce) for cs, ce in intervals)
                if not overlaps:
                    chosen.append(r)
                    intervals.append((s, e))
            
            if chosen:
                result_df = pd.DataFrame(chosen)
                result_df['sequence'] = seq_id  # Use actual sequence ID from FASTA
                all_results.append(result_df)
        
        if all_results:
            df = pd.concat(all_results, ignore_index=True)
            out_cols = ['sequence', 'ori_type', 'pct_identity', 'pct_cov_subject', 'q_start', 'q_end', 'strand', 'qlen', 'sstart', 'send', 'slen', 'length', 'bitscore', 'evalue']
            df = df[[c for c in out_cols if c in df.columns]]
        else:
            df = pd.DataFrame(columns=['sequence', 'ori_type', 'pct_identity', 'pct_cov_subject', 'q_start', 'q_end'])
    else:
        df = pd.DataFrame(columns=['sequence', 'ori_type', 'pct_identity', 'pct_cov_subject', 'q_start', 'q_end'])
else:
    df = pd.DataFrame(columns=['sequence', 'ori_type', 'pct_identity', 'pct_cov_subject', 'q_start', 'q_end'])

df.to_csv("McClain_PlasmidGPT_RL_ATG.ori_calls.csv", index=False)
PYEOF

    # Replace raw TSV with filtered CSV for aggregation
    mv McClain_PlasmidGPT_RL_ATG.ori_calls.csv McClain_PlasmidGPT_RL_ATG.ori_hits.tsv
