#!/usr/bin/env python3
from plasmid_analytics.eval.mfe import mfe_from_fasta

mfe_from_fasta(
    'sequences.fasta',
    output_csv='UCL_CSSB_PlasmidGPT_ATG.mfe.csv',
    circular=True,
    include_structure=False
)
