#!/usr/bin/env python3
"""
Generate comprehensive statistics for the paper.
Outputs a markdown file with all key metrics.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.special import rel_entr
from collections import Counter

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
BASE_DIR = Path(os.environ.get('RESULTS_DIR', PROJECT_ROOT / 'results'))

MODEL_ORDER = ['Base', 'SFT', 'RL']


def compute_kl_divergence(p_counts, q_counts, k=3):
    """Compute KL divergence D(P||Q) between two k-mer distributions."""
    # Get all possible k-mers
    all_kmers = set(p_counts.keys()) | set(q_counts.keys())

    # Convert to probability distributions with smoothing
    total_p = sum(p_counts.values()) + len(all_kmers)  # Add-1 smoothing
    total_q = sum(q_counts.values()) + len(all_kmers)

    p = np.array([(p_counts.get(kmer, 0) + 1) / total_p for kmer in all_kmers])
    q = np.array([(q_counts.get(kmer, 0) + 1) / total_q for kmer in all_kmers])

    # KL divergence
    kl = np.sum(rel_entr(p, q))
    return kl


def get_kmer_counts(sequences, k=3):
    """Get k-mer counts from a list of sequences."""
    counts = Counter()
    for seq in sequences:
        if pd.isna(seq):
            continue
        seq = seq.upper()
        for i in range(len(seq) - k + 1):
            counts[seq[i:i+k]] += 1
    return counts


def main():
    print("=" * 60)
    print("Generating Paper Statistics")
    print("=" * 60)
    print(f"Results directory: {BASE_DIR}")

    stats_output = []
    stats_output.append("# Paper Statistics\n")
    stats_output.append(f"**Results Directory:** `{BASE_DIR}`\n")
    stats_output.append("---\n")

    # =========================================================================
    # 1. QC Pass Rates
    # =========================================================================
    stats_output.append("## 1. QC Pass Rates\n")

    pass_rates = {}
    for model in MODEL_ORDER:
        gen_csv = BASE_DIR / 'generations' / model / 'outputs.csv'
        pass_csv = BASE_DIR / 'qc' / model / 'passed.csv'

        if gen_csv.exists() and pass_csv.exists():
            total = len(pd.read_csv(gen_csv))
            passed = len(pd.read_csv(pass_csv))
            rate = 100 * passed / total if total > 0 else 0
            pass_rates[model] = {'passed': passed, 'total': total, 'rate': rate}
            stats_output.append(f"- **{model}:** {passed}/{total} ({rate:.1f}%)\n")

    stats_output.append("\n")

    # =========================================================================
    # 2. Completion Benchmark (Log-Probability)
    # =========================================================================
    stats_output.append("## 2. Held-Out Continuation Task (Log-Probability)\n")

    completion_csv = BASE_DIR / 'analysis' / 'completion_benchmark.csv'
    if completion_csv.exists():
        df_comp = pd.read_csv(completion_csv)

        stats_output.append("| Model | Mean Log-Prob | Std | N |\n")
        stats_output.append("|-------|---------------|-----|---|\n")

        for model in MODEL_ORDER:
            subset = df_comp[df_comp['Model'] == model]['AvgLogProb']
            if len(subset) > 0:
                mean_lp = subset.mean()
                std_lp = subset.std()
                n = len(subset)
                stats_output.append(f"| {model} | {mean_lp:.4f} | {std_lp:.4f} | {n} |\n")

        stats_output.append("\n")

        # Pairwise t-tests
        stats_output.append("### Pairwise T-Tests (Completion)\n")
        for i, m1 in enumerate(MODEL_ORDER):
            for m2 in MODEL_ORDER[i+1:]:
                s1 = df_comp[df_comp['Model'] == m1]['AvgLogProb'].dropna()
                s2 = df_comp[df_comp['Model'] == m2]['AvgLogProb'].dropna()
                if len(s1) > 0 and len(s2) > 0:
                    t_stat, p_val = stats.ttest_ind(s1, s2)
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    stats_output.append(f"- {m1} vs {m2}: t={t_stat:.3f}, p={p_val:.2e} {sig}\n")

        stats_output.append("\n")

    # =========================================================================
    # 3. Surprisal Benchmark
    # =========================================================================
    stats_output.append("## 3. Surprisal Benchmark\n")

    surprisal_csv = BASE_DIR / 'analysis' / 'surprisal_benchmark.csv'
    if surprisal_csv.exists():
        df_surp = pd.read_csv(surprisal_csv)

        stats_output.append("| Model | Mean Log-Prob | Std | N |\n")
        stats_output.append("|-------|---------------|-----|---|\n")

        for model in MODEL_ORDER:
            subset = df_surp[df_surp['Model'] == model]['MeanLogProb']
            if len(subset) > 0:
                mean_lp = subset.mean()
                std_lp = subset.std()
                n = len(subset)
                stats_output.append(f"| {model} | {mean_lp:.4f} | {std_lp:.4f} | {n} |\n")

        stats_output.append("\n")

        # Pairwise t-tests
        stats_output.append("### Pairwise T-Tests (Surprisal)\n")
        for i, m1 in enumerate(MODEL_ORDER):
            for m2 in MODEL_ORDER[i+1:]:
                s1 = df_surp[df_surp['Model'] == m1]['MeanLogProb'].dropna()
                s2 = df_surp[df_surp['Model'] == m2]['MeanLogProb'].dropna()
                if len(s1) > 0 and len(s2) > 0:
                    t_stat, p_val = stats.ttest_ind(s1, s2)
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    stats_output.append(f"- {m1} vs {m2}: t={t_stat:.3f}, p={p_val:.2e} {sig}\n")

        stats_output.append("\n")

    # =========================================================================
    # 4. Distribution Metrics (KL Divergence, etc.)
    # =========================================================================
    stats_output.append("## 4. Distribution Metrics\n")

    dist_csv = BASE_DIR / 'publication' / 'distribution_grid_metrics.csv'
    if dist_csv.exists():
        df_dist = pd.read_csv(dist_csv)

        # Get real plasmid sequences
        real_seqs = df_dist[df_dist['model'] == 'Real']['full'].dropna().tolist()
        real_kmer_counts = get_kmer_counts(real_seqs, k=3)

        stats_output.append("### 3-mer KL Divergence from Real Plasmids\n")
        stats_output.append("| Model | KL(Model||Real) | Mean JS Divergence |\n")
        stats_output.append("|-------|-----------------|--------------------|\n")

        for model in MODEL_ORDER:
            model_seqs = df_dist[df_dist['model'] == model]['full'].dropna().tolist()
            if model_seqs:
                model_kmer_counts = get_kmer_counts(model_seqs, k=3)
                kl_div = compute_kl_divergence(model_kmer_counts, real_kmer_counts, k=3)
                mean_js = df_dist[df_dist['model'] == model]['js_3mer'].mean()
                stats_output.append(f"| {model} | {kl_div:.4f} | {mean_js:.4f} |\n")

        stats_output.append("\n")

        # GC Content
        stats_output.append("### GC Content\n")
        stats_output.append("| Model | Mean GC | Std |\n")
        stats_output.append("|-------|---------|-----|\n")

        for model in ['Real'] + MODEL_ORDER:
            subset = df_dist[df_dist['model'] == model]['gc'].dropna()
            if len(subset) > 0:
                stats_output.append(f"| {model} | {subset.mean():.4f} | {subset.std():.4f} |\n")

        stats_output.append("\n")

        # Sequence Length
        stats_output.append("### Sequence Length\n")
        stats_output.append("| Model | Mean Length | Std | Median |\n")
        stats_output.append("|-------|-------------|-----|--------|\n")

        for model in ['Real'] + MODEL_ORDER:
            subset = df_dist[df_dist['model'] == model]['seq_length'].dropna()
            if len(subset) > 0:
                stats_output.append(f"| {model} | {subset.mean():.0f} | {subset.std():.0f} | {subset.median():.0f} |\n")

        stats_output.append("\n")

        # MFE Density
        stats_output.append("### MFE Density (Thermodynamic Stability)\n")
        stats_output.append("| Model | Mean MFE/nt | Std |\n")
        stats_output.append("|-------|-------------|-----|\n")

        for model in ['Real'] + MODEL_ORDER:
            subset = df_dist[df_dist['model'] == model]['mfe_density'].dropna()
            if len(subset) > 0:
                stats_output.append(f"| {model} | {subset.mean():.4f} | {subset.std():.4f} |\n")

        stats_output.append("\n")

    # =========================================================================
    # 5. Diversity Metrics
    # =========================================================================
    stats_output.append("## 5. Diversity Metrics\n")

    summary_csv = BASE_DIR / 'analysis' / 'model_comparison_summary.csv'
    if summary_csv.exists():
        df_sum = pd.read_csv(summary_csv)

        stats_output.append("| Model | Pass Rate (%) | Diversity Score |\n")
        stats_output.append("|-------|---------------|----------------|\n")

        for _, row in df_sum.iterrows():
            stats_output.append(f"| {row['Model']} | {row['PassRate']:.1f} | {row['Diversity']:.4f} |\n")

        stats_output.append("\n")

    # =========================================================================
    # 6. Sample Counts
    # =========================================================================
    stats_output.append("## 6. Sample Counts\n")

    if dist_csv.exists():
        df_dist = pd.read_csv(dist_csv)
        counts = df_dist['model'].value_counts()

        stats_output.append("| Category | Count |\n")
        stats_output.append("|----------|-------|\n")

        for model in ['Real'] + MODEL_ORDER:
            if model in counts:
                stats_output.append(f"| {model} | {counts[model]} |\n")

        stats_output.append(f"| **Total** | {len(df_dist)} |\n")
        stats_output.append("\n")

    # =========================================================================
    # Write output
    # =========================================================================
    output_path = BASE_DIR / 'publication' / 'paper_statistics.md'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(''.join(stats_output))

    print(f"\nStatistics written to: {output_path}")
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(''.join(stats_output))


if __name__ == '__main__':
    main()
