#!/bin/bash -euo pipefail
amrfinder \
    --nucleotide sequences.fasta \
    --output McClain_PlasmidGPT_RL_GFP_cassette.amr_hits.tsv \
    --threads 4 \
    --name McClain_PlasmidGPT_RL_GFP_cassette

# Ensure file exists even if no hits
if [ ! -s McClain_PlasmidGPT_RL_GFP_cassette.amr_hits.tsv ]; then
    echo "Name	Protein identifier	Contig id	Start	Stop	Strand	Gene symbol	Sequence name	Scope	Element type	Element subtype	Class	Subclass	Method	Target length	Reference sequence length	% Coverage of reference sequence	% Identity to reference sequence	Alignment length	Accession of closest sequence	Name of closest sequence	HMM id	HMM description" > McClain_PlasmidGPT_RL_GFP_cassette.amr_hits.tsv
fi
