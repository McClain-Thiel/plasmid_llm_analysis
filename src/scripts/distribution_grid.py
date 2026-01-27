#!/usr/bin/env python3
"""
Create distribution grid with 4 panels showing:
- Sequence Length
- JS-3mer divergence vs real plasmids
- MFE density (thermodynamic stability)
- GC content

Includes "Real" plasmids as reference alongside generated models.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from scipy.spatial.distance import jensenshannon
import warnings
warnings.filterwarnings('ignore')

# Try to import ViennaRNA for MFE calculation
try:
    import RNA
    HAS_RNA = True
except ImportError:
    HAS_RNA = False
    print("Warning: ViennaRNA (RNA) not installed. MFE panel will be skipped.")

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Allow override via environment variable
BASE_DIR = Path(os.environ.get('RESULTS_DIR', PROJECT_ROOT / 'results'))
PUB_DIR = BASE_DIR / 'publication'
REAL_DIR = Path(os.environ.get('REAL_PLASMIDS_DIR', PROJECT_ROOT / 'assets' / 'annotations'))

PUB_DIR.mkdir(parents=True, exist_ok=True)

# Model order including Real plasmids
MODEL_ORDER = ['Real', 'Base', 'SFT', 'RL']
MODEL_COLORS = {
    'Real': '#22C55E',
    'Base': '#2E4057',
    'SFT': '#8B5CF6',
    'RL': '#E11D48'
}

DIR_TO_NAME = {'Base': 'Base', 'SFT': 'SFT', 'RL': 'RL'}

# Figure settings
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
sns.set_theme(style="white")


def compute_gc(seq):
    """Compute GC content of a sequence."""
    if pd.isna(seq) or len(seq) == 0:
        return np.nan
    seq = seq.upper()
    return (seq.count('G') + seq.count('C')) / len(seq)


def compute_mfe_density(seq):
    """Compute MFE density (MFE / length) for circular DNA."""
    if not HAS_RNA:
        return np.nan
    if pd.isna(seq) or len(seq) < 100:
        return np.nan
    try:
        seq = seq.upper().replace('N', 'A')  # Replace N with A
        md = RNA.md()
        md.circ = 1  # Circular
        fc = RNA.fold_compound(seq, md)
        _, mfe = fc.mfe()
        return mfe / len(seq)
    except:
        return np.nan


def kmer_distribution(seq, k=3):
    """Compute k-mer frequency distribution."""
    if pd.isna(seq) or len(seq) < k:
        return {}
    seq = seq.upper()
    kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
    counts = Counter(kmers)
    total = sum(counts.values())
    return {kmer: count/total for kmer, count in counts.items()}


def js_divergence_to_ref(seq, ref_dist, k=3):
    """Compute Jensen-Shannon divergence between sequence and reference distribution."""
    seq_dist = kmer_distribution(seq, k)
    if not seq_dist:
        return np.nan
    all_keys = set(seq_dist.keys()) | set(ref_dist.keys())
    p = np.array([seq_dist.get(key, 0) for key in all_keys])
    q = np.array([ref_dist.get(key, 0) for key in all_keys])
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    return float(jensenshannon(p, q, base=2.0))


def load_data():
    """Load all sequences from real plasmids and generated models."""
    all_data = []

    # Load real plasmids
    if REAL_DIR.exists():
        print(f"Loading real plasmids from: {REAL_DIR}")
        for f in REAL_DIR.glob('*.fasta'):
            # Skip cassette files if present
            if 'cassette' in f.name.lower():
                continue
            with open(f) as fh:
                seq = ''.join(line.strip() for line in fh if not line.startswith('>'))
                if seq and len(seq) > 1000:
                    all_data.append({
                        'model': 'Real',
                        'full': seq.upper(),
                        'seq_length': len(seq)
                    })
        print(f"  Loaded {sum(1 for d in all_data if d['model'] == 'Real')} real plasmids")
    else:
        print(f"Warning: Real plasmids directory not found: {REAL_DIR}")

    # Load generated sequences
    for model_dir in ['Base', 'SFT', 'RL']:
        model_name = DIR_TO_NAME[model_dir]
        outputs_path = BASE_DIR / 'generations' / model_dir / 'outputs.csv'
        if outputs_path.exists():
            df = pd.read_csv(outputs_path)
            count = 0
            for _, row in df.iterrows():
                if pd.notna(row['full']) and len(str(row['full'])) > 500:
                    all_data.append({
                        'model': model_name,
                        'full': str(row['full']).upper(),
                        'seq_length': len(str(row['full']))
                    })
                    count += 1
            print(f"  Loaded {count} sequences from {model_name}")
        else:
            print(f"  Warning: {outputs_path} not found, skipping {model_dir}")

    return pd.DataFrame(all_data)


def save_figure(filename_base):
    """Save figure in both PNG and PDF formats."""
    plt.tight_layout()
    plt.savefig(PUB_DIR / f'{filename_base}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(PUB_DIR / f'{filename_base}.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filename_base}")


def main():
    print("=" * 60)
    print("Creating Distribution Grid")
    print("=" * 60)
    print(f"Results directory: {BASE_DIR}")
    print(f"Real plasmids directory: {REAL_DIR}")
    print(f"Publication output: {PUB_DIR}")

    print("\nLoading data...")
    df = load_data()

    if len(df) == 0:
        print("Error: No data loaded. Check paths and try again.")
        return

    print(f"\nLoaded {len(df)} total sequences")
    print(f"By model: {df['model'].value_counts().to_dict()}")

    # Compute GC content
    print("\nComputing GC content...")
    df['gc'] = df['full'].apply(compute_gc)

    # Compute MFE density (can be slow)
    if HAS_RNA:
        print("Computing MFE density (this may take a while)...")
        total = len(df)
        mfe_values = []
        for i, seq in enumerate(df['full']):
            if i % 20 == 0:
                print(f"  MFE progress: {i}/{total} ({100*i/total:.0f}%)", flush=True)
            mfe_values.append(compute_mfe_density(seq))
        df['mfe_density'] = mfe_values
        print(f"  MFE progress: {total}/{total} (100%)")
    else:
        df['mfe_density'] = np.nan

    # Compute 3-mer JS divergence
    print("Computing 3-mer JS divergence...")
    real_seqs = df[df['model'] == 'Real']['full'].dropna()
    if len(real_seqs) > 0:
        ref_concat = ''.join(real_seqs.values)
    else:
        # Fall back to all sequences if no real plasmids
        ref_concat = ''.join(df['full'].dropna().values)
    ref_dist = kmer_distribution(ref_concat, k=3)
    df['js_3mer'] = df['full'].apply(lambda x: js_divergence_to_ref(x, ref_dist, k=3))

    # Prepare for plotting
    df['log_length'] = np.log10(df['seq_length'])
    df['model'] = pd.Categorical(df['model'], categories=MODEL_ORDER, ordered=True)

    # Save computed metrics
    metrics_path = PUB_DIR / 'distribution_grid_metrics.csv'
    df.to_csv(metrics_path, index=False)
    print(f"Saved computed metrics to {metrics_path}")

    # Create figure
    print("\nCreating figure...")

    # Determine which panels to include based on data availability
    has_mfe = df['mfe_density'].notna().sum() > 0

    if has_mfe:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        print("Note: MFE data not available, panel C will show placeholder")

    order = [m for m in MODEL_ORDER if m in df['model'].unique()]
    palette = [MODEL_COLORS[m] for m in order]

    # A) Sequence Length
    ax = axes[0, 0]
    sns.violinplot(data=df, x='model', y='seq_length', order=order, palette=palette,
                   ax=ax, inner='box', cut=0)
    ax.set_xlabel('Model')
    ax.set_ylabel('Length (bp)')
    ax.set_title('A) Sequence Length', fontweight='bold', loc='left')
    ax.axhline(y=3000, color='gray', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # B) 3-mer JS Divergence
    ax = axes[0, 1]
    sns.violinplot(data=df, x='model', y='js_3mer', order=order, palette=palette,
                   ax=ax, inner='box', cut=0)
    ax.set_xlabel('Model')
    ax.set_ylabel('JS Divergence (3-mer)')
    ax.set_title('B) 3-mer Compositional Divergence', fontweight='bold', loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # C) MFE Density (Thermodynamic Stability)
    ax = axes[1, 0]
    if has_mfe:
        sns.violinplot(data=df, x='model', y='mfe_density', order=order, palette=palette,
                       ax=ax, inner='box', cut=0)
        ax.set_xlabel('Model')
        ax.set_ylabel('MFE Density (kcal/mol/nt)')
        ax.set_title('C) Thermodynamic Stability', fontweight='bold', loc='left')
    else:
        ax.text(0.5, 0.5, 'MFE Data\nNot Available\n(ViennaRNA required)',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_title('C) Thermodynamic Stability', fontweight='bold', loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # D) GC Content
    ax = axes[1, 1]
    sns.violinplot(data=df, x='model', y='gc', order=order, palette=palette,
                   ax=ax, inner='box', cut=0)
    ax.set_xlabel('Model')
    ax.set_ylabel('GC Content')
    ax.set_title('D) GC Content', fontweight='bold', loc='left')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_figure('fig_distribution_grid')

    print("\n" + "=" * 60)
    print("Distribution grid saved to:", PUB_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()
