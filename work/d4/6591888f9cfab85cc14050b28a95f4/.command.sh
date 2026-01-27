#!/usr/bin/env python3
from plasmid_analytics.eval.mfe import mfe_from_fasta

mfe_from_fasta(
    'sequences.fasta',
    output_csv='McClain_PlasmidGPT_RL_GFP_cassette.mfe.csv',
    circular=True,
    include_structure=False
)
