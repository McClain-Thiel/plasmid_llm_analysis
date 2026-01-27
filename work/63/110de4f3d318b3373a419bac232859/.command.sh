#!/usr/bin/env python3
from plasmid_analytics.eval.kmer import build_reference_distribution
from plasmid_analytics.generate.sampler import read_fasta
import pandas as pd
from pathlib import Path

has_ref = False

# Load reference sequences if available
ref_dist = {}
if has_ref and Path('NO_FILE').exists():
    ref_seqs = [r['sequence'] for r in read_fasta('NO_FILE')]
    if ref_seqs:
        ref_dist = build_reference_distribution(ref_seqs, k=3)

# Load query sequence
query_seqs = list(read_fasta('sequences.fasta'))

results = []
for record in query_seqs:
    seq = record['sequence']
    js = 0.0
    if ref_dist:
        from plasmid_analytics.eval.kmer import js_divergence_from_reference
        js = js_divergence_from_reference(seq, ref_dist, k=3)

    results.append({
        'plasmid_id': record['id'],
        'js_divergence_3mer': js
    })

pd.DataFrame(results).to_csv('UCL_CSSB_PlasmidGPT_ATG.kmer.csv', index=False)
