#!/bin/bash -euo pipefail
# Check sequence length
SEQ_LEN=$(grep -v "^>" sequences.fasta | tr -d '\n' | wc -c)
if [ $SEQ_LEN -ge 20000 ]; then
    MODE="-p single"
else
    MODE="-p meta"
fi

prodigal \
    -i sequences.fasta \
    -a McClain_PlasmidGPT_RL_GFP_cassette.proteins.faa \
    -d McClain_PlasmidGPT_RL_GFP_cassette.genes.fna \
    -o McClain_PlasmidGPT_RL_GFP_cassette.genes.gff \
    -f gff \
    $MODE
