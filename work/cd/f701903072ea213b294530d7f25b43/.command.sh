#!/bin/bash -euo pipefail
plasmid-repeats \
    sequences.fasta \
    --out UCL_CSSB_PlasmidGPT_ATG.repeats.csv \
    --circular \
    --min-len 2
