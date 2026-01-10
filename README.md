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
*   `results/analysis/report.html`: **Comprehensive HTML Report** containing:
    *   Experiment Configuration (Prompts, Sampling Params).
    *   Pass Rates & Diversity Scores (Mash Distance).
    *   Biological Metrics (Length, GC, ORF, MFE, etc.).
    *   Benchmarking Results (Completion Confidence, Surprisal Gap).
    *   Similarity Classification against NCBI RefSeq Plasmids.
*   `results/qc/{model}/passed.csv`: List of sequences passing all QC filters for each model.

## Remote Execution (GPU Machine)

For large-scale generation and analysis, execute on a GPU instance (e.g., `g6-big`).

**Prerequisites:**
1.  **Disk Space**: Ensure at least **50GB** free space (RefSeq Plasmid DB is ~15GB). Use NVMe storage if available (e.g., `/opt/dlami/nvme`).
2.  **Hugging Face Token**: Required for gated models (e.g., `UCL-CSSB/PlasmidGPT-SFT`). Set `HF_TOKEN`.

**Execution Command:**

```bash
# 1. Setup Directories on Large Drive
export WORKDIR=/opt/dlami/nvme/plasmid_analysis
mkdir -p $WORKDIR
cd $WORKDIR

# 2. Clone Repository
git clone https://github.com/McClain-Thiel/plasmid_llm_analysis.git
cd plasmid_llm_analysis

# 3. Create Environment (Redirect cache to avoid root partition overflow)
export TMPDIR=$WORKDIR/tmp
export PIP_CACHE_DIR=$WORKDIR/pip_cache
export HF_HOME=$WORKDIR/hf_cache
export XDG_CACHE_HOME=$WORKDIR/xdg_cache
mkdir -p $TMPDIR $PIP_CACHE_DIR $HF_HOME $XDG_CACHE_HOME

conda env create -f environment.yml -p ./env

# 4. Initialize AMR Database
conda run -p ./env amrfinder -u

# 5. Run Pipeline (Background)
export HF_TOKEN="your_token_here"
nohup conda run -p ./env snakemake --cores 32 --resources gpu=1 --rerun-incomplete > run.log 2>&1 &

# 6. Monitor
tail -f run.log
```

**Retrieving Results:**

```bash
scp g6-big:/opt/dlami/nvme/plasmid_analysis/plasmid_llm_analysis/results/analysis/report.html ~/Downloads/
```

## Analysis Details

*   **Diversity**: Calculated using `sourmash` (MinHash) as `1 - mean_pairwise_jaccard_similarity`.
*   **Similarity**: Sequences are BLASTed against the **NCBI RefSeq Plasmid** database (downloaded locally during analysis) to classify them as "Exact Match" (>99% ID), "Similar" (>80% ID), or "Novel".
*   **Benchmarking**:
    *   **Completion**: Log-probability of the next 100bp given a 400bp prefix from a held-out test set.
    *   **Surprisal**: Log-probability gap (Model - Base) on specific Promoter→CDS transitions.
