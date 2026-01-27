#!/usr/bin/env python3
"""
Generate novelty classification charts based on NCBI BLAST results.

Categories:
- Exists: identity >= 99% AND coverage >= 95%
- Similar: identity >= 95% AND coverage >= 80%
- Novel: anything else

This script is optional and requires NCBI BLAST results to be present.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup paths - use relative paths from script location or environment
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Allow override via environment variable or default to results/
BASE_DIR = Path(os.environ.get('RESULTS_DIR', PROJECT_ROOT / 'results'))
PUB_DIR = BASE_DIR / 'publication'
BLAST_DIR = BASE_DIR / 'ncbi_blast'

PUB_DIR.mkdir(parents=True, exist_ok=True)

# Model order (3 models)
MODEL_ORDER = ['Base', 'SFT', 'RL']
MODEL_FILES = {
    'Base': 'Base_blast_results.tsv',
    'SFT': 'SFT_blast_results.tsv',
    'RL': 'RL_blast_results.tsv',
}

MODEL_COLORS = {'Base': '#2E4057', 'SFT': '#8B5CF6', 'RL': '#E11D48'}

# Novelty category colors
CATEGORY_COLORS = {'Exists': '#EF4444', 'Similar': '#F59E0B', 'Novel': '#22C55E'}

# Figure settings
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
sns.set_theme(style="white")


def classify_sequence(identity, coverage):
    """
    Classify a sequence based on best BLAST hit.

    Thresholds:
    - Exists: identity >= 99% AND coverage >= 95%
    - Similar: identity >= 95% AND coverage >= 80%
    - Novel: everything else
    """
    if identity >= 99 and coverage >= 95:
        return 'Exists'
    elif identity >= 95 and coverage >= 80:
        return 'Similar'
    else:
        return 'Novel'


def process_blast_results(filepath):
    """Process BLAST results and get best hit per query."""
    df = pd.read_csv(filepath, sep='\t')

    # Get best hit per query (highest identity * coverage product)
    df['score'] = df['pct_identity'] * df['query_coverage']
    best_hits = df.loc[df.groupby('query_id')['score'].idxmax()]

    # Classify each sequence
    classifications = []
    for _, row in best_hits.iterrows():
        cat = classify_sequence(row['pct_identity'], row['query_coverage'])
        classifications.append({
            'query_id': row['query_id'],
            'pct_identity': row['pct_identity'],
            'query_coverage': row['query_coverage'],
            'category': cat
        })

    return pd.DataFrame(classifications)


def save_figure(filename_base):
    """Save figure in both PNG and PDF formats."""
    plt.tight_layout()
    plt.savefig(PUB_DIR / f'{filename_base}.png', dpi=300, bbox_inches='tight')
    plt.savefig(PUB_DIR / f'{filename_base}.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename_base}")


def main():
    print("=" * 60)
    print("Generating Novelty Classification Charts")
    print("=" * 60)
    print(f"BLAST results directory: {BLAST_DIR}")
    print(f"Publication output: {PUB_DIR}")

    if not BLAST_DIR.exists():
        print(f"\nError: BLAST results directory not found: {BLAST_DIR}")
        print("This script requires NCBI BLAST results to be present.")
        print("Run NCBI BLAST first, then re-run this script.")
        return

    print("\nProcessing BLAST results...")
    all_results = []

    for model, filename in MODEL_FILES.items():
        filepath = BLAST_DIR / filename
        if filepath.exists():
            results = process_blast_results(filepath)
            results['model'] = model
            all_results.append(results)
            print(f"  {model}: {len(results)} sequences")
        else:
            print(f"  {model}: File not found ({filepath})")

    if not all_results:
        print("\nNo BLAST results found. Exiting.")
        return

    df = pd.concat(all_results, ignore_index=True)

    # Count by model and category
    counts = df.groupby(['model', 'category']).size().unstack(fill_value=0)

    # Ensure all categories exist
    for cat in ['Exists', 'Similar', 'Novel']:
        if cat not in counts.columns:
            counts[cat] = 0

    counts = counts[['Exists', 'Similar', 'Novel']]  # Order columns
    counts = counts.reindex([m for m in MODEL_ORDER if m in counts.index])  # Order rows

    print("\nCounts by model and category:")
    print(counts)

    # Calculate percentages
    totals = counts.sum(axis=1)
    pcts = counts.div(totals, axis=0) * 100

    print("\nPercentages:")
    print(pcts.round(1))

    # Save data
    counts.to_csv(PUB_DIR / 'novelty_counts.csv')
    pcts.to_csv(PUB_DIR / 'novelty_percentages.csv')
    print(f"\nSaved novelty_counts.csv and novelty_percentages.csv")

    # Get actual models present in data
    models_present = [m for m in MODEL_ORDER if m in counts.index]

    # ===== FIGURE 1: Stacked bar chart (counts) =====
    print("\nGenerating figures...")
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models_present))
    width = 0.6

    bottom = np.zeros(len(models_present))
    for cat in ['Exists', 'Similar', 'Novel']:
        values = counts.loc[models_present, cat].values
        bars = ax.bar(x, values, width, bottom=bottom, label=cat,
                      color=CATEGORY_COLORS[cat], edgecolor='white', linewidth=1)

        # Add count labels on bars
        for i, (val, bot) in enumerate(zip(values, bottom)):
            if val > 0:
                ax.text(x[i], bot + val/2, str(int(val)),
                        ha='center', va='center', fontsize=10, fontweight='bold',
                        color='white' if cat != 'Novel' else 'black')

        bottom += values

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Number of Sequences', fontsize=12)
    ax.set_title('Sequence Novelty Classification (NCBI BLAST)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models_present)
    ax.legend(loc='upper right', frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_figure('fig_novelty_stacked')

    # ===== FIGURE 2: Stacked bar chart (percentages) =====
    fig, ax = plt.subplots(figsize=(10, 6))

    bottom = np.zeros(len(models_present))
    for cat in ['Exists', 'Similar', 'Novel']:
        values = pcts.loc[models_present, cat].values
        bars = ax.bar(x, values, width, bottom=bottom, label=cat,
                      color=CATEGORY_COLORS[cat], edgecolor='white', linewidth=1)

        # Add percentage labels
        for i, (val, bot) in enumerate(zip(values, bottom)):
            if val > 5:  # Only label if > 5%
                ax.text(x[i], bot + val/2, f'{val:.0f}%',
                        ha='center', va='center', fontsize=10, fontweight='bold',
                        color='white' if cat != 'Novel' else 'black')

        bottom += values

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Percentage of Sequences', fontsize=12)
    ax.set_title('Sequence Novelty Classification (NCBI BLAST)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models_present)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_figure('fig_novelty_percentage')

    # ===== FIGURE 3: Grouped bar chart =====
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models_present))
    width = 0.25

    for i, cat in enumerate(['Exists', 'Similar', 'Novel']):
        offset = (i - 1) * width
        values = pcts.loc[models_present, cat].values
        bars = ax.bar(x + offset, values, width, label=cat, color=CATEGORY_COLORS[cat],
                      edgecolor='black', linewidth=1)

        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.0f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Percentage of Sequences', fontsize=12)
    ax.set_title('Sequence Novelty by Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models_present)
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_figure('fig_novelty_grouped')

    # Print summary
    print("\n" + "=" * 50)
    print("Summary Table:")
    print("=" * 50)
    for model in models_present:
        row = pcts.loc[model]
        print(f"{model:10s}: Exists={row['Exists']:5.1f}%, Similar={row['Similar']:5.1f}%, Novel={row['Novel']:5.1f}%")

    print("\n" + "=" * 60)
    print("All novelty charts saved to:", PUB_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()
