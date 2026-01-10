import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
import multiprocessing
from scipy.spatial.distance import jensenshannon
from collections import Counter
from Bio.Seq import Seq
from tqdm import tqdm

try:
    import RNA
except ImportError:
    pass

# --- Helper Functions ---

def gc_content(seq: str) -> float:
    seq = seq.upper()
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)

def get_circular_mfe(seq_str: str) -> tuple[float, float]:
    if not seq_str: return 0.0, 0.0
    try:
        md = RNA.md()
        md.circ = 1
        fc = RNA.fold_compound(str(seq_str), md)
        (structure, mfe) = fc.mfe()
        return mfe, mfe / len(seq_str)
    except NameError: # ViennaRNA not installed
        return 0.0, 0.0

def fast_longest_orf_atg(seq_str: str, min_aa: int = 0) -> int:
    seq = Seq(seq_str)
    max_len = 0
    for frame in range(3):
        try:
            prot_seq = str(seq[frame:].translate(to_stop=False))
            matches = re.findall(r'M[^*]*\*', prot_seq)
            if matches:
                longest_in_frame = max(len(m) - 1 for m in matches)
                if longest_in_frame >= min_aa:
                    max_len = max(max_len, longest_in_frame)
        except:
            pass
    return int(max_len)

def get_orfs_both_strands_fast(seq_str: str) -> tuple[int, int, int]:
    fwd = fast_longest_orf_atg(seq_str)
    rev_seq = str(Seq(seq_str).reverse_complement())
    rev = fast_longest_orf_atg(rev_seq)
    return fwd, rev, max(fwd, rev)

def count_orfs_above(seq_str: str, min_aa: int = 100) -> int:
    seq = Seq(seq_str)
    count = 0
    for frame in range(3):
        try:
            prots = seq[frame:].translate().split("*")
            count += sum(1 for p in prots if len(p) >= min_aa)
        except: pass
    return int(count)

def kmer_distribution(seq: str, k: int = 3):
    seq = seq.upper()
    kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
    counts = Counter(kmers)
    total = sum(counts.values())
    if total == 0: return {}
    return {kmer: c / total for kmer, c in counts.items()}

def js_divergence_kmers(seq_a: str, seq_b_concat: str, k: int = 3) -> float:
    # Approximate JS by comparing seq to global distribution
    # Pre-computing dist_b is better but this is 'per seq' worker
    # We assume seq_b_concat is passed.
    # Note: re-computing dist_b every time is slow. 
    # Optimization: Calculate dist_b ONCE outside.
    # Here we assume seq_b_concat is the raw string.
    # For speed in worker, we will just count kmers for seq_a
    # and require dist_b (dict) to be passed in args if possible.
    # But since we pass strings, let's just do it.
    
    dist_a = kmer_distribution(seq_a, k)
    # We need a reference distribution. 
    # Calculating on huge string every time is bad.
    # Let's assume 'ref_dist' is passed instead of 'ref_concat' in args if possible.
    # See process_single_plasmid below.
    return 0.0 # Placeholder, handled differently

def js_divergence_from_dist(dist_a, dist_ref):
    all_keys = sorted(set(dist_a.keys()) | set(dist_ref.keys()))
    if not all_keys: return 0.0
    p = np.array([dist_a.get(k, 0.0) for k in all_keys])
    q = np.array([dist_ref.get(k, 0.0) for k in all_keys])
    return float(jensenshannon(p, q, base=2.0))

# --- Worker ---
def process_single_plasmid(args):
    seq_str, ref_dist, model_tag, prompt_tag, name_tag = args
    
    # Metrics
    atg_fwd, atg_rev, atg_both = get_orfs_both_strands_fast(seq_str)
    mfe_total, mfe_density = get_circular_mfe(seq_str)
    gc = gc_content(seq_str)
    n_orfs = count_orfs_above(seq_str, min_aa=100)
    
    # JS
    dist_a = kmer_distribution(seq_str, k=3)
    js3 = js_divergence_from_dist(dist_a, ref_dist)
    
    return {
        "Model": model_tag,
        "Prompt": prompt_tag,
        "Name": name_tag,
        "Length": len(seq_str),
        "GC": gc,
        "Num_ORFs_>=100AA": n_orfs,
        "JS_3mer_vs_real": js3,
        "Longest_ORF_ATG_both": atg_both,
        "MFE_Density": mfe_density
    }

def main(sm):
    # 1. Load Real Plasmids
    real_files = [f for f in os.listdir(sm.input.real_data) if f.endswith(".fasta")]
    real_seqs = []
    for rf in real_files:
        path = os.path.join(sm.input.real_data, rf)
        with open(path) as f:
            # Read single record simple fasta
            lines = [l.strip() for l in f if not l.startswith(">")]
            seq = "".join(lines).upper()
            if seq:
                real_seqs.append({"Name": rf, "Sequence": seq, "Model": "Real", "Prompt": "Real"})
    
    # Global Ref Dist
    real_concat = "".join([x["Sequence"] for x in real_seqs])
    ref_dist = kmer_distribution(real_concat, k=3)
    
    # 2. Load Generated Plasmids
    gen_tasks = []
    for model in sm.params.models:
        csv_file = [f for f in sm.input.generations if f"/{model}/" in f][0]
        try:
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                # row keys: id, prompt, full
                # The 'full' sequence is the plasmid
                prompt_label = "ATG" if len(row['prompt']) < 10 else "GFP"
                gen_tasks.append((row['full'], ref_dist, model, prompt_label, row['id']))
        except Exception as e:
            print(f"Skipping {model}: {e}")

    # Add Real to tasks
    all_tasks = gen_tasks + [(x["Sequence"], ref_dist, "Real", "Real", x["Name"]) for x in real_seqs]
    
    # 3. Parallel Compute
    print(f"Computing metrics for {len(all_tasks)} sequences...")
    results = []
    with multiprocessing.Pool(processes=min(16, multiprocessing.cpu_count())) as pool:
        for res in tqdm(pool.imap(process_single_plasmid, all_tasks), total=len(all_tasks)):
            results.append(res)
            
    metrics_df = pd.DataFrame(results)
    
    # 4. Generate Pass Rate Summary (Legacy logic integrated)
    summary_rows = []
    for model in sm.params.models:
        qc_file = [f for f in sm.input.qc_pass_csvs if f"/{model}/" in f][0]
        gen_file = [f for f in sm.input.generations if f"/{model}/" in f][0]
        try:
            passed = len(pd.read_csv(qc_file))
            total = len(pd.read_csv(gen_file))
            rate = (passed / total * 100) if total else 0
        except:
            rate = 0
        summary_rows.append({"Model": model, "PassRate": rate})
    
    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(sm.output.summary, index=False)
    
    # 5. Plotting
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # A. Pass Rate
    plt.figure(figsize=(6, 4))
    sns.barplot(data=sum_df, x="Model", y="PassRate")
    plt.title("Pass Rate by Model")
    plt.savefig(sm.output.plot)
    plt.close()
    
    # B. Metrics Panel
    model_order = ["Real"] + sm.params.models
    metrics_df["Model"] = pd.Categorical(metrics_df["Model"], categories=model_order, ordered=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    plot_params = [
        ("Length", "Length (bp)", True),
        ("GC", "GC Content", False),
        ("Longest_ORF_ATG_both", "Longest ORF (aa)", True),
        ("Num_ORFs_>=100AA", "# ORFs >= 100aa", True),
        ("JS_3mer_vs_real", "3-mer Divergence (JS)", False),
        ("MFE_Density", "MFE Density", False)
    ]
    
    for i, (col, title, log_scale) in enumerate(plot_params):
        ax = axes[i]
        data = metrics_df.copy()
        if log_scale:
            data[col] = np.log10(data[col].clip(lower=1))
            title += " (log10)"
            
        sns.boxplot(data=data, x="Model", y=col, hue="Prompt", showfliers=False, ax=ax)
        ax.set_title(title)
        ax.legend([],[], frameon=False) # Hide legend per plot to save space
        
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    plt.savefig(sm.output.metrics_plot)
    plt.close()

if __name__ == "__main__":
    main(snakemake)