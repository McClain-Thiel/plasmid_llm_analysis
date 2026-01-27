#!/bin/bash -euo pipefail
plasmid-repeats \
    sequences.fasta \
    --out McClain_PlasmidGPT_RL_ATG.repeats.csv \
    --circular \
    --min-len 2
