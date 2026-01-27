#!/bin/bash -euo pipefail
mob_typer_patch.py \
    --infile sequences.fasta \
    --out_file McClain_PlasmidGPT_RL_ATG_mobtyper_results.txt \
     \
    --num_threads 4

# Create contig report if multiple contigs
if [ -f mobtyper_contig_report.txt ]; then
    mv mobtyper_contig_report.txt McClain_PlasmidGPT_RL_ATG_mobtyper_contig_report.txt
fi
