# PlasmidGPT Evaluation Pipeline

This repository contains a Snakemake workflow for generating, characterizing, and benchmarking plasmid sequences produced by Large Language Models (LLMs). The pipeline assesses biological viability through strict quality control (QC) checks involving Origin of Replication detection, Antimicrobial Resistance (AMR) gene identification, and structural repeat analysis.

## Overview

The workflow proceeds through four main stages:

1.  **Generation**: Sequences are generated using vLLM from multiple model checkpoints (Base, SFT, RL/GRPO).
2.  **Quality Control (QC)**:
    *   **Origins**: Detection using NCBI BLAST+ against a curated OriDB.
    *   **AMR Genes**: Identification using AMRFinderPlus in nucleotide mode.
    *   **Gene Prediction**: Open Reading Frame (ORF) prediction using Prodigal.
    *   **Repeat Analysis**: Detection of long exact repeats (direct and inverted) using suffix arrays.
3.  **Benchmarking**: Calculation of perplexity and completion metrics against a held-out set of real reference plasmids.
4.  **Analysis**: Comprehensive statistical comparison of models based on biological metrics (GC content, ORF length, MFE density, k-mer divergence).

## Acknowledgements

The Quality Control (QC) pipeline logic (`src/qc/`) was originally developed by **Angus Cunningham** (University College London).
Original Repository: [plasmidbackbonedesign](https://github.com/angusgcunningham/plasmidbackbonedesign/tree/main)

Key tools utilized:
*   **NCBI BLAST+**: Camacho et al., *BMC Bioinformatics* 2009.
*   **AMRFinderPlus**: Feldgarden et al., *Scientific Reports* 2021.
*   **Prodigal**: Hyatt et al., *BMC Bioinformatics* 2010.
*   **ViennaRNA**: Lorenz et al., *Algorithms for Molecular Biology* 2011.

## Requirements

*   **Hardware**: Linux workstation with NVIDIA GPU (CUDA support required for vLLM generation).
*   **Software**:
    *   Python 3.10+
    *   Conda / Mamba
    *   NCBI BLAST+
    *   AMRFinderPlus
    *   Prodigal

## Installation

Create the environment using the provided YAML file:

```bash
mamba env create -f environment.yml
mamba activate plasmidgpt
```

Ensure the AMRFinderPlus database is initialized:

```bash
amrfinder -u
```

## Data Setup

1.  **Reference Plasmids**: Place FASTA files of real plasmids (e.g., pUC19, pBR322) in `assets/annotations/`.
2.  **Origin Database**: Ensure `assets/oriV_refs.fasta` contains the reference Origin of Replication sequences.

## Configuration

Adjust experiment parameters in `config.yaml`:

```yaml
models:
  Base: "UCL-CSSB/PlasmidGPT"
  SFT: "UCL-CSSB/PlasmidGPT-SFT"
  RL: "UCL-CSSB/PlasmidGPT-GRPO"
  SFT_GRPO: "McClain/PlasmidGPT-RL"

qc:
  ori_strict_id: 99.0  # Minimum identity for valid Origin
  amr_strict_id: 100.0 # Minimum identity for valid ARG
```

## Execution

Run the full pipeline using Snakemake:

```bash
snakemake --cores 8 --resources gpu=1
```

## Output

Results are stored in the `results/` directory:

*   `results/analysis/model_comparison_summary.csv`: Pass rates and summary statistics.
*   `results/analysis/metrics_plots.png`: Comparative boxplots of biological metrics (Length, GC, MFE, etc.).
*   `results/qc/{model}/passed.csv`: List of sequences passing all QC filters for each model.
