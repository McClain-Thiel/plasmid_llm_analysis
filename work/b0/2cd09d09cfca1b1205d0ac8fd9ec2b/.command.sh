#!/bin/bash -euo pipefail
plasmid-repeats \
    sequences.fasta \
    --out UCL_CSSB_PlasmidGPT_GFP_cassette.repeats.csv \
    --circular \
    --min-len 2
