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
    -a UCL_CSSB_PlasmidGPT_ATG.proteins.faa \
    -d UCL_CSSB_PlasmidGPT_ATG.genes.fna \
    -o UCL_CSSB_PlasmidGPT_ATG.genes.gff \
    -f gff \
    $MODE
