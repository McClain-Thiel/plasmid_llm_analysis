#!/usr/bin/env python3
"""
Publication figure generation for plasmid language model comparison.
ICML-ready figures with seaborn white theme.

Generates publication-quality figures in PNG and PDF format.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Setup paths - use relative paths from script location or environment
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Allow override via environment variable or default to results/
BASE_DIR = Path(os.environ.get('RESULTS_DIR', PROJECT_ROOT / 'results'))
PUB_DIR = BASE_DIR / 'publication'
PUB_DIR.mkdir(parents=True, exist_ok=True)

# Model order for all plots (3 models per user preference)
MODEL_ORDER = ['Base', 'SFT', 'RL']

# Use seaborn deep palette
sns.set_palette("deep")
_deep = sns.color_palette("deep")
MODEL_COLORS = {'Base': _deep[0], 'SFT': _deep[1], 'RL': _deep[2]}

# Mapping from directory names to display names
DIR_TO_NAME = {'Base': 'Base', 'SFT': 'SFT', 'RL': 'RL'}
NAME_TO_DIR = {v: k for k, v in DIR_TO_NAME.items()}

# Mapping for summary CSV (GRPO -> RL)
SUMMARY_NAME_MAP = {'GRPO': 'RL'}

# Figure settings
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.figsize'] = (8, 6)

sns.set_theme(style="white")


def load_all_data():
    """Load and combine all data from all models."""
    all_data = []

    for model_dir in ['Base', 'SFT', 'RL']:
        model_name = DIR_TO_NAME[model_dir]

        # Load outputs (sequences and prompts)
        outputs_path = BASE_DIR / 'generations' / model_dir / 'outputs.csv'
        if not outputs_path.exists():
            print(f"Warning: {outputs_path} not found, skipping {model_dir}")
            continue

        outputs = pd.read_csv(outputs_path)
        outputs['model'] = model_name
        outputs['prompt_type'] = outputs['prompt'].apply(
            lambda x: 'GFP' if len(str(x)) > 10 else 'ATG'
        )

        # Load repeats (has sequence length)
        repeats_path = BASE_DIR / 'qc' / model_dir / 'repeats.csv'
        if repeats_path.exists():
            repeats = pd.read_csv(repeats_path)
            repeats = repeats.rename(columns={'plasmid_id': 'id'})
        else:
            repeats = pd.DataFrame(columns=['id', 'seq_length', 'circular', 'longest_len', 'longest_fraction'])

        # Load QC summary (ORI and AMR counts)
        qc_path = BASE_DIR / 'qc' / model_dir / 'qc_summary.csv'
        if qc_path.exists():
            qc = pd.read_csv(qc_path)
            qc = qc.rename(columns={'sample': 'id'})
        else:
            qc = pd.DataFrame(columns=['id', 'n_ori_kept', 'n_amr'])

        # Load passed sequences
        passed_path = BASE_DIR / 'qc' / model_dir / 'passed.csv'
        if passed_path.exists():
            passed = pd.read_csv(passed_path)
            passed_ids = set(passed['Plasmid_ID'].values) if 'Plasmid_ID' in passed.columns else set()
        else:
            passed_ids = set()

        # Merge data
        merged = outputs.copy()
        if len(repeats) > 0:
            merged = merged.merge(
                repeats[['id', 'seq_length', 'circular', 'longest_len', 'longest_fraction']],
                on='id', how='left'
            )
        else:
            merged['seq_length'] = merged['full'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)

        if len(qc) > 0:
            merged = merged.merge(qc[['id', 'n_ori_kept', 'n_amr']], on='id', how='left')

        merged['passed'] = merged['id'].isin(passed_ids)

        all_data.append(merged)

    if not all_data:
        raise ValueError("No data found. Check that results directory contains model outputs.")

    return pd.concat(all_data, ignore_index=True)


def compute_gc_content(seq):
    """Compute GC content of a sequence."""
    if pd.isna(seq) or len(str(seq)) == 0:
        return np.nan
    seq = str(seq).upper()
    gc = sum(1 for c in seq if c in 'GC')
    return gc / len(seq) * 100


def add_sequence_metrics(df):
    """Add computed metrics to dataframe."""
    print("Computing GC content...")
    df['gc_content'] = df['full'].apply(compute_gc_content)

    print("Computing log10 length...")
    if 'seq_length' not in df.columns:
        df['seq_length'] = df['full'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    df['log_length'] = np.log10(df['seq_length'].replace(0, np.nan))

    return df


def save_figure(filename_base):
    """Save figure in both PNG and PDF formats."""
    plt.tight_layout()
    plt.savefig(PUB_DIR / f'{filename_base}.png', dpi=300, bbox_inches='tight')
    plt.savefig(PUB_DIR / f'{filename_base}.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename_base}")


# ============== FIGURE GENERATION FUNCTIONS ==============

def plot_pass_rate_overall(df):
    """Bar chart of pass rates by model (both prompts combined)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    pass_rates = df.groupby('model')['passed'].mean() * 100
    pass_rates = pass_rates.reindex(MODEL_ORDER)

    bars = ax.bar(MODEL_ORDER, pass_rates.values,
                  color=[MODEL_COLORS[m] for m in MODEL_ORDER],
                  edgecolor='black', linewidth=1)

    # Add value labels on bars
    for bar, val in zip(bars, pass_rates.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('QC Pass Rate (%)', fontsize=12)
    #ax.set_title removed - no titles
    ax.set_ylim(0, 105)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_figure('fig01_pass_rate_overall')


def plot_pass_rate_by_prompt(df):
    """Bar chart of pass rates by model and prompt type."""
    fig, ax = plt.subplots(figsize=(10, 6))

    pass_rates = df.groupby(['model', 'prompt_type'])['passed'].mean() * 100
    pass_rates = pass_rates.unstack()
    pass_rates = pass_rates.reindex(MODEL_ORDER)

    x = np.arange(len(MODEL_ORDER))
    width = 0.35

    bars1 = ax.bar(x - width/2, pass_rates['ATG'], width, label='ATG Prompt',
                   color='#3B82F6', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, pass_rates['GFP'], width, label='GFP Prompt',
                   color='#F59E0B', edgecolor='black', linewidth=1)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                        f'{height:.0f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('QC Pass Rate (%)', fontsize=12)
    #ax.set_title removed - no titles
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylim(0, 110)
    ax.legend(loc='upper left', frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_figure('fig02_pass_rate_by_prompt')


def plot_diversity_overall(df):
    """Bar chart of diversity by model."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Load diversity from summary
    summary_path = BASE_DIR / 'analysis' / 'model_comparison_summary.csv'
    if not summary_path.exists():
        print(f"Warning: {summary_path} not found, skipping diversity plot")
        plt.close()
        return

    summary = pd.read_csv(summary_path)
    summary['Model'] = summary['Model'].replace(SUMMARY_NAME_MAP)
    summary = summary[summary['Model'].isin(MODEL_ORDER)]
    summary = summary.set_index('Model').reindex(MODEL_ORDER)

    bars = ax.bar(MODEL_ORDER, summary['Diversity'].values,
                  color=[MODEL_COLORS[m] for m in MODEL_ORDER],
                  edgecolor='black', linewidth=1)

    # Add value labels
    for bar, val in zip(bars, summary['Diversity'].values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Self-Diversity (Mash Distance)', fontsize=12)
    #ax.set_title removed - no titles
    ax.set_ylim(0, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_figure('fig03_diversity_overall')


def plot_pass_rate_vs_diversity(df):
    """Scatter plot showing pass rate vs diversity trade-off."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Load summary data
    summary_path = BASE_DIR / 'analysis' / 'model_comparison_summary.csv'
    if not summary_path.exists():
        print(f"Warning: {summary_path} not found, skipping pass vs diversity plot")
        plt.close()
        return

    summary = pd.read_csv(summary_path)
    summary['Model'] = summary['Model'].replace(SUMMARY_NAME_MAP)
    summary = summary[summary['Model'].isin(MODEL_ORDER)]

    for _, row in summary.iterrows():
        model = row['Model']
        if model not in MODEL_COLORS:
            continue
        ax.scatter(row['Diversity'], row['PassRate'],
                   s=200, c=MODEL_COLORS[model], edgecolors='black', linewidth=2,
                   label=model, zorder=5)
        ax.annotate(model, (row['Diversity'], row['PassRate']),
                    xytext=(10, 5), textcoords='offset points', fontsize=11, fontweight='bold')

    ax.set_xlabel('Self-Diversity (Mash Distance)', fontsize=12)
    ax.set_ylabel('QC Pass Rate (%)', fontsize=12)
    #ax.set_title removed - no titles
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0, 105)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

    save_figure('fig04_pass_vs_diversity')


def plot_length_distribution(df):
    """Violin/box plot of sequence lengths by model."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter to reasonable lengths (> 100 bp)
    df_filt = df[df['seq_length'] > 100].copy()
    df_filt['log_length'] = np.log10(df_filt['seq_length'])

    # Create ordered categorical
    df_filt['model'] = pd.Categorical(df_filt['model'], categories=MODEL_ORDER, ordered=True)

    palette = [MODEL_COLORS[m] for m in MODEL_ORDER]
    sns.violinplot(data=df_filt, x='model', y='log_length', palette=palette,
                   order=MODEL_ORDER, ax=ax, inner='box')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Sequence Length (log10 bp)', fontsize=12)
    #ax.set_title removed - no titles

    # Add reference lines for typical plasmid sizes
    ax.axhline(y=np.log10(3000), color='gray', linestyle='--', alpha=0.5, label='3 kb')
    ax.axhline(y=np.log10(10000), color='gray', linestyle=':', alpha=0.5, label='10 kb')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_figure('fig05_length_distribution')


def plot_gc_distribution(df):
    """Distribution of GC content by model."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter to sequences with valid GC
    df_filt = df[df['gc_content'].notna() & (df['seq_length'] > 100)].copy()
    df_filt['model'] = pd.Categorical(df_filt['model'], categories=MODEL_ORDER, ordered=True)

    palette = [MODEL_COLORS[m] for m in MODEL_ORDER]
    sns.violinplot(data=df_filt, x='model', y='gc_content', palette=palette,
                   order=MODEL_ORDER, ax=ax, inner='box')

    # Add reference line for typical plasmid GC (around 50%)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% GC')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('GC Content (%)', fontsize=12)
    #ax.set_title removed - no titles
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_figure('fig06_gc_distribution')


def plot_repeat_fraction(df):
    """Distribution of longest repeat fraction by model."""
    if 'longest_fraction' not in df.columns:
        print("Warning: longest_fraction not in data, skipping repeat fraction plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    df_filt = df[df['longest_fraction'].notna() & (df['seq_length'] > 100)].copy()
    df_filt['model'] = pd.Categorical(df_filt['model'], categories=MODEL_ORDER, ordered=True)

    palette = [MODEL_COLORS[m] for m in MODEL_ORDER]
    sns.boxplot(data=df_filt, x='model', y='longest_fraction', palette=palette,
                order=MODEL_ORDER, ax=ax)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Longest Repeat Fraction', fontsize=12)
    #ax.set_title removed - no titles
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_figure('fig07_repeat_fraction')


def plot_completion_benchmark(df):
    """Plot completion benchmark results."""
    completion_path = BASE_DIR / 'analysis' / 'completion_benchmark.csv'
    if not completion_path.exists():
        print(f"Warning: {completion_path} not found, skipping completion benchmark plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    completion = pd.read_csv(completion_path)
    completion['Model'] = completion['Model'].replace(SUMMARY_NAME_MAP)
    completion = completion[completion['Model'].isin(MODEL_ORDER)]

    # Aggregate by model
    model_means = completion.groupby('Model')['AvgLogProb'].mean()
    model_stds = completion.groupby('Model')['AvgLogProb'].std()
    model_means = model_means.reindex(MODEL_ORDER)
    model_stds = model_stds.reindex(MODEL_ORDER)

    bars = ax.bar(MODEL_ORDER, model_means.values,
                  yerr=model_stds.values, capsize=5,
                  color=[MODEL_COLORS[m] for m in MODEL_ORDER],
                  edgecolor='black', linewidth=1)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Mean Log Probability', fontsize=12)
    #ax.set_title removed - no titles
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Note: higher (less negative) is better
    ax.text(0.02, 0.98, 'Higher = Better', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', style='italic')

    save_figure('fig08_completion_benchmark')


def plot_surprisal_benchmark():
    """Plot surprisal benchmark results."""
    surprisal_path = BASE_DIR / 'analysis' / 'surprisal_benchmark.csv'
    if not surprisal_path.exists():
        print(f"Warning: {surprisal_path} not found, skipping surprisal benchmark plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    surprisal = pd.read_csv(surprisal_path)
    surprisal['Model'] = surprisal['Model'].replace(SUMMARY_NAME_MAP)
    surprisal = surprisal[surprisal['Model'].isin(MODEL_ORDER)]

    # Aggregate by model
    model_means = surprisal.groupby('Model')['MeanLogProb'].mean()
    model_stds = surprisal.groupby('Model')['MeanLogProb'].std()
    model_means = model_means.reindex(MODEL_ORDER)
    model_stds = model_stds.reindex(MODEL_ORDER)

    bars = ax.bar(MODEL_ORDER, model_means.values,
                  yerr=model_stds.values, capsize=5,
                  color=[MODEL_COLORS[m] for m in MODEL_ORDER],
                  edgecolor='black', linewidth=1)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Mean Log Probability at Functional Sites', fontsize=12)
    #ax.set_title removed - no titles
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.text(0.02, 0.98, 'Higher = Better', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', style='italic')

    save_figure('fig09_surprisal_benchmark')


def plot_completion_by_plasmid():
    """Plot completion benchmark broken down by reference plasmid."""
    completion_path = BASE_DIR / 'analysis' / 'completion_benchmark.csv'
    if not completion_path.exists():
        print(f"Warning: {completion_path} not found, skipping completion by plasmid plot")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    completion = pd.read_csv(completion_path)
    completion['Model'] = completion['Model'].replace(SUMMARY_NAME_MAP)
    completion = completion[completion['Model'].isin(MODEL_ORDER)]
    completion['Plasmid'] = completion['Plasmid'].str.replace('.fasta', '')

    # Aggregate by model and plasmid
    pivot = completion.groupby(['Plasmid', 'Model'])['AvgLogProb'].mean().unstack()
    pivot = pivot[[m for m in MODEL_ORDER if m in pivot.columns]]

    x = np.arange(len(pivot.index))
    width = 0.25

    for i, model in enumerate(MODEL_ORDER):
        if model not in pivot.columns:
            continue
        offset = (i - 1) * width
        ax.bar(x + offset, pivot[model], width, label=model,
               color=MODEL_COLORS[model], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Reference Plasmid', fontsize=12)
    ax.set_ylabel('Mean Log Probability', fontsize=12)
    #ax.set_title removed - no titles
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_figure('fig11_completion_by_plasmid')


def plot_ori_amr_counts(df):
    """Plot ORI and AMR gene counts by model."""
    if 'n_ori_kept' not in df.columns or 'n_amr' not in df.columns:
        print("Warning: ORI/AMR data not available, skipping ORI/AMR plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    df_filt = df[df['seq_length'] > 100].copy()
    df_filt['model'] = pd.Categorical(df_filt['model'], categories=MODEL_ORDER, ordered=True)

    # ORI counts
    ax = axes[0]
    palette = [MODEL_COLORS[m] for m in MODEL_ORDER]
    sns.boxplot(data=df_filt, x='model', y='n_ori_kept', palette=palette,
                order=MODEL_ORDER, ax=ax)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Number of ORIs', fontsize=12)
    #ax.set_title removed
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.7, label='Target: 1')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # AMR counts
    ax = axes[1]
    sns.boxplot(data=df_filt, x='model', y='n_amr', palette=palette,
                order=MODEL_ORDER, ax=ax)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Number of AMR Genes', fontsize=12)
    #ax.set_title removed
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.7, label='Target: 1')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_figure('fig12_ori_amr_counts')


def plot_combined_summary():
    """Create a combined 2x2 summary figure."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Load summary data
    summary_path = BASE_DIR / 'analysis' / 'model_comparison_summary.csv'
    if not summary_path.exists():
        print(f"Warning: {summary_path} not found, skipping combined summary plot")
        plt.close()
        return

    summary = pd.read_csv(summary_path)
    summary['Model'] = summary['Model'].replace(SUMMARY_NAME_MAP)
    summary = summary[summary['Model'].isin(MODEL_ORDER)]
    summary = summary.set_index('Model').reindex(MODEL_ORDER)

    # Panel A: Pass Rate
    ax = axes[0, 0]
    bars = ax.bar(MODEL_ORDER, summary['PassRate'].values,
                  color=[MODEL_COLORS[m] for m in MODEL_ORDER],
                  edgecolor='black', linewidth=1)
    for bar, val in zip(bars, summary['PassRate'].values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('QC Pass Rate (%)', fontsize=11)
    #ax.set_title removed
    ax.set_ylim(0, 105)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: Diversity
    ax = axes[0, 1]
    bars = ax.bar(MODEL_ORDER, summary['Diversity'].values,
                  color=[MODEL_COLORS[m] for m in MODEL_ORDER],
                  edgecolor='black', linewidth=1)
    for bar, val in zip(bars, summary['Diversity'].values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('Self-Diversity', fontsize=11)
    #ax.set_title removed
    ax.set_ylim(0, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel C: Trade-off scatter
    ax = axes[1, 0]
    for model in MODEL_ORDER:
        if model not in summary.index:
            continue
        row = summary.loc[model]
        ax.scatter(row['Diversity'], row['PassRate'],
                   s=200, c=MODEL_COLORS[model], edgecolors='black', linewidth=2,
                   label=model, zorder=5)
        ax.annotate(model, (row['Diversity'], row['PassRate']),
                    xytext=(8, 3), textcoords='offset points', fontsize=10)
    ax.set_xlabel('Self-Diversity', fontsize=11)
    ax.set_ylabel('QC Pass Rate (%)', fontsize=11)
    #ax.set_title removed
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel D: Placeholder for NCBI Diversity (optional)
    ax = axes[1, 1]
    ncbi_path = BASE_DIR / 'ncbi_blast' / 'ncbi_diversity_summary.csv'
    if ncbi_path.exists():
        ncbi = pd.read_csv(ncbi_path)
        ncbi['model'] = ncbi['model'].replace(SUMMARY_NAME_MAP)
        ncbi = ncbi[ncbi['model'].isin(MODEL_ORDER)]
        ncbi = ncbi.set_index('model').reindex(MODEL_ORDER)
        bars = ax.bar(MODEL_ORDER, ncbi['diversity_ratio'].values,
                      color=[MODEL_COLORS[m] for m in MODEL_ORDER],
                      edgecolor='black', linewidth=1)
        for bar, val in zip(bars, ncbi['diversity_ratio'].values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_ylabel('NCBI Diversity Ratio', fontsize=11)
        #ax.set_title removed
        ax.set_ylim(0, 1.1)
    else:
        ax.text(0.5, 0.5, 'NCBI Diversity Data\nNot Available', ha='center', va='center',
                fontsize=12, transform=ax.transAxes)
        #ax.set_title removed

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_figure('fig00_summary_panel')


# ============== MAIN ==============

def main():
    print("=" * 60)
    print("Generating Publication Figures")
    print("=" * 60)
    print(f"Results directory: {BASE_DIR}")
    print(f"Publication output: {PUB_DIR}")

    # Load all data
    print("\nLoading data...")
    try:
        df = load_all_data()
        print(f"Loaded {len(df)} sequences across {df['model'].nunique()} models")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure results directory contains model outputs.")
        return

    # Add computed metrics
    print("\nComputing sequence metrics...")
    df = add_sequence_metrics(df)

    # Save combined data
    df.to_csv(PUB_DIR / 'combined_data.csv', index=False)
    print(f"Saved combined data to {PUB_DIR / 'combined_data.csv'}")

    # Generate figures
    print("\nGenerating figures...")
    print("-" * 40)

    plot_combined_summary()
    plot_pass_rate_overall(df)
    plot_pass_rate_by_prompt(df)
    plot_diversity_overall(df)
    plot_pass_rate_vs_diversity(df)
    plot_length_distribution(df)
    plot_gc_distribution(df)
    plot_repeat_fraction(df)
    plot_completion_benchmark(df)
    plot_surprisal_benchmark()
    plot_completion_by_plasmid()
    plot_ori_amr_counts(df)

    print("\n" + "=" * 60)
    print("All figures saved to:", PUB_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()
