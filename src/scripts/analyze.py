import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from scipy.spatial.distance import jensenshannon
from collections import Counter
from Bio.Seq import Seq
from tqdm import tqdm
import re
import subprocess
import sourmash
import tempfile
import shutil
import time

try:
    import RNA
except ImportError:
    pass

# --- Configuration ---
MODEL_MAP = {
    'Base': 'Base',
    'SFT': 'SFT',
    'RL': 'GRPO',
    'SFT_GRPO': 'SFT+GRPO'
}
ORDER = ['Base', 'SFT', 'GRPO', 'SFT+GRPO']
ORDER_WITH_REAL = ['Real'] + ORDER

# --- Helper Functions (Metrics) ---
def gc_content(seq: str) -> float:
    seq = seq.upper()
    if not seq: return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)

def get_circular_mfe(seq_str: str) -> tuple[float, float]:
    if not seq_str: return 0.0, 0.0
    try:
        md = RNA.md()
        md.circ = 1
        fc = RNA.fold_compound(str(seq_str), md)
        (structure, mfe) = fc.mfe()
        return mfe, mfe / len(seq_str)
    except NameError: return 0.0, 0.0

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
        except: pass
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

def js_divergence_from_dist(dist_a, dist_ref):
    all_keys = sorted(set(dist_a.keys()) | set(dist_ref.keys()))
    if not all_keys: return 0.0
    p = np.array([dist_a.get(k, 0.0) for k in all_keys])
    q = np.array([dist_ref.get(k, 0.0) for k in all_keys])
    return float(jensenshannon(p, q, base=2.0))

# --- New: Diversity (Self-Mash) ---
def calculate_diversity_mash(seqs, k=21, n=1000):
    hashes = []
    for s in seqs:
        mh = sourmash.MinHash(n=n, ksize=k)
        mh.add_sequence(s)
        hashes.append(mh)
    
    if len(hashes) < 2: return 0.0
    
    sims = []
    for i in range(len(hashes)):
        for j in range(i+1, len(hashes)):
            sims.append(hashes[i].jaccard(hashes[j]))
    
    # Return diversity = 1 - average_similarity
    return 1.0 - np.mean(sims) if sims else 0.0

# --- New: Local RefSeq Plasmid DB ---
def setup_local_plasmid_db(output_dir):
    db_dir = os.path.join(output_dir, "refseq_data")
    db_out = os.path.join(db_dir, "refseq_plasmids")
    
    if os.path.exists(db_out + ".nsq"):
        return db_out
        
    os.makedirs(db_dir, exist_ok=True)
    print("Setting up Local RefSeq Plasmid Database (this may take time)...")
    
    # 1. Download (if empty)
    if not any(f.endswith(".fna.gz") for f in os.listdir(db_dir)):
        print("Downloading from NCBI FTP...")
        cmd = f"wget -q -P {db_dir} 'ftp://ftp.ncbi.nlm.nih.gov/refseq/release/plasmid/*.genomic.fna.gz'"
        try:
            subprocess.run(cmd, shell=True, check=True)
        except:
            print("wget failed, trying curl loop...")
            # Fallback hardcoded loop if glob fails
            base_url = "ftp://ftp.ncbi.nlm.nih.gov/refseq/release/plasmid"
            files = [f"plasmid.{i}.{j}.genomic.fna.gz" for i in range(1, 10) for j in range(1, 3)]
            for f in files:
                subprocess.run(f"curl -s -o {db_dir}/{f} {base_url}/{f}", shell=True)

    # 2. Build DB
    print("Building BLAST DB...")
    # Clean up any partials
    cmd = f"zcat {db_dir}/*.genomic.fna.gz | makeblastdb -in - -dbtype nucl -out {db_out} -title 'RefSeq Plasmids' -parse_seqids"
    subprocess.run(cmd, shell=True, check=True)
    
    return db_out

# --- BLAST Batch ---
def run_blast_batch(seq_dict, db_path):
    """
    seq_dict: {id: sequence}
    Returns: {id: classification}
    """
    if not seq_dict: return {}
    
    # Write all to one file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp:
        tmp_name = tmp.name
        for sid, seq in seq_dict.items():
            tmp.write(f">{sid}\n{seq}\n")
    
    print(f"Running BLAST against {db_path} for {len(seq_dict)} sequences...")
    results_map = {sid: "Novel" for sid in seq_dict}
    
    cmd = [
        "blastn", "-query", tmp_name, "-db", db_path,
        "-outfmt", "6 qseqid pident length qlen slen", 
        "-max_target_seqs", "1", "-task", "megablast", "-num_threads", "16"
    ]
    
    try:
        res = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        for line in res.split('\n'):
            if not line: continue
            parts = line.split('\t')
            if len(parts) < 4: continue
            
            qid = parts[0]
            pident = float(parts[1])
            length = float(parts[2])
            qlen = float(parts[3])
            
            cov = length / qlen
            
            if pident > 95 and cov > 0.90:
                cls = "Exact Match"
            elif pident > 80:
                cls = "Similar"
            else:
                cls = "Novel"
            
            results_map[qid] = cls
            
    except Exception as e:
        print(f"BLAST failed: {e}")
        
    os.remove(tmp_name)
    return results_map

# --- Worker ---
def process_single_plasmid(args):
    seq_str, ref_dist, model_tag, prompt_tag, name_tag = args
    
    atg_fwd, atg_rev, atg_both = get_orfs_both_strands_fast(seq_str)
    mfe_total, mfe_density = get_circular_mfe(seq_str)
    gc = gc_content(seq_str)
    n_orfs = count_orfs_above(seq_str, min_aa=100)
    dist_a = kmer_distribution(seq_str, k=3)
    js3 = js_divergence_from_dist(dist_a, ref_dist)
    
    return {
        "Model": model_tag, "Prompt": prompt_tag, "Name": name_tag,
        "Length": len(seq_str), "GC": gc, "Num_ORFs_>=100AA": n_orfs,
        "JS_3mer_vs_real": js3, "Longest_ORF_ATG_both": atg_both,
        "MFE_Density": mfe_density
    }

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def main(sm):
    out_dir = os.path.dirname(sm.output.summary)
    # data_dir for DB download
    data_dir = os.path.join(os.path.dirname(sm.output.summary), "../../data") # results/analysis/../../data -> data/
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

    # --- 0. Prepare Local RefSeq DB ---
    try:
        refseq_db = setup_local_plasmid_db(data_dir)
    except Exception as e:
        print(f"Failed to setup local DB: {e}. Skipping BLAST.")
        refseq_db = None

    # --- 1. Load Data ---
    ref_dir = sm.input.real_data
    ref_files = [os.path.join(ref_dir, f) for f in os.listdir(ref_dir) if f.endswith(".fasta")]
    
    real_seqs = []
    for rf in ref_files:
        with open(rf) as f:
            lines = [l.strip() for l in f if not l.startswith(">")]
            seq = "".join(lines).upper()
            if seq: real_seqs.append({"Name": os.path.basename(rf), "Sequence": seq, "Model": "Real", "Prompt": "Real"})
    
    real_concat = "".join([x["Sequence"] for x in real_seqs])
    ref_dist = kmer_distribution(real_concat, k=3)
    
    # Load Generated
    gen_data = [] # List of (seq, model, prompt, id)
    model_sequences = {m: [] for m in sm.params.models} # For Diversity
    
    # Collect IDs for BLAST
    blast_candidates = {} # {unique_id: seq}
    
    for model in sm.params.models:
        csv_file = [f for f in sm.input.generations if f"/{model}/" in f][0]
        try:
            df = pd.read_csv(csv_file)
            # Store full set for diversity
            full_seqs = df['full'].tolist()
            model_sequences[model] = full_seqs
            
            # Subset 10 per prompt for heavy metrics & BLAST
            subset = df.groupby('prompt').apply(lambda x: x.sample(n=min(len(x), 10), random_state=42)).reset_index(drop=True)
            
            for _, row in subset.iterrows():
                mod_name = MODEL_MAP.get(model, model)
                prompt_label = "ATG" if len(row['prompt']) < 10 else "GFP"
                unique_id = f"{mod_name}_{row['id']}"
                gen_data.append((row['full'], mod_name, prompt_label, unique_id))
                blast_candidates[unique_id] = row['full']
        except: pass

    # --- 2. Run BLAST (Batch) ---
    if refseq_db:
        blast_results = run_blast_batch(blast_candidates, refseq_db)
    else:
        blast_results = {}
    
    # --- 3. Compute Metrics (Parallel) ---
    all_tasks = [(s, ref_dist, m, p, i) for s, m, p, i in gen_data] + \
                [(x["Sequence"], ref_dist, "Real", "Real", x["Name"]) for x in real_seqs]
    
    with multiprocessing.Pool(processes=min(16, multiprocessing.cpu_count())) as pool:
        results = list(tqdm(pool.imap(process_single_plasmid, all_tasks), total=len(all_tasks), desc="Metrics"))
    
    # Merge BLAST results into metrics
    for res in results:
        if res["Model"] == "Real":
            res["Similarity"] = "Reference"
        else:
            res["Similarity"] = blast_results.get(res["Name"], "Novel")
            
    metrics_df = pd.DataFrame(results)
    
    # --- 4. Pass Rates & Diversity ---
    summary_rows = []
    for model in sm.params.models:
        qc_file = [f for f in sm.input.qc_pass_csvs if f"/{model}/" in f][0]
        gen_file = [f for f in sm.input.generations if f"/{model}/" in f][0]
        try:
            passed = len(pd.read_csv(qc_file))
            total = len(pd.read_csv(gen_file))
            rate = (passed / total * 100) if total else 0
        except: rate = 0
        
        # Diversity
        div = calculate_diversity_mash(model_sequences.get(model, []))
        summary_rows.append({"Model": MODEL_MAP.get(model, model), "PassRate": rate, "Diversity (Mash)": div})
    
    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(sm.output.summary, index=False)
    
    # Plot Pass Rate
    plt.figure(figsize=(8, 6))
    sns.barplot(data=sum_df, x="Model", y="PassRate", order=ORDER, palette="viridis")
    plt.title("Pass Rate by Model")
    plt.ylabel("Pass Rate (%)")
    save_plot(f"{out_dir}/fig1_pass_rate.png")
    # Legacy save
    sns.barplot(data=sum_df, x="Model", y="PassRate", order=ORDER, palette="viridis")
    save_plot(sm.output.plot)
    
    # Plot Diversity
    plt.figure(figsize=(8, 6))
    sns.barplot(data=div_df, x="Model", y="Diversity (Mash)", order=ORDER, palette="magma")
    plt.title("Sequence Diversity (1 - Avg Mash Similarity)")
    plt.ylabel("Diversity Score")
    save_plot(f"{out_dir}/fig10_diversity.png")

    # --- 5. Plot Metrics & Similarity ---
    metrics = [
        ("Length", "Length (bp)", True),
        ("GC", "GC Content", False),
        ("Longest_ORF_ATG_both", "Longest ORF (aa)", True),
        ("Num_ORFs_>=100AA", "# ORFs >= 100aa", True),
        ("JS_3mer_vs_real", "3-mer Divergence (JS)", False),
        ("MFE_Density", "MFE Density", False)
    ]
    
    for i, (col, title, log_scale) in enumerate(metrics):
        plt.figure(figsize=(8, 6))
        data = metrics_df.copy()
        if log_scale:
            data[col] = np.log10(data[col].clip(lower=1))
            title += " (log10)"
        
        valid_order = [m for m in ORDER_WITH_REAL if m in data['Model'].unique()]
        sns.boxplot(data=data, x="Model", y=col, hue="Prompt", order=valid_order, showfliers=False)
        plt.title(title)
        save_plot(f"{out_dir}/fig{i+2}_{col.lower()}.png")

    # Plot Similarity
    plt.figure(figsize=(8, 6))
    sim_counts = metrics_df[metrics_df['Model'] != 'Real'].groupby(['Model', 'Similarity']).size().reset_index(name='Count')
    # Calculate pct
    model_totals = metrics_df[metrics_df['Model'] != 'Real'].groupby('Model').size().reset_index(name='Total')
    sim_counts = sim_counts.merge(model_totals, on='Model')
    sim_counts['Percent'] = sim_counts['Count'] / sim_counts['Total'] * 100
    
    sns.barplot(data=sim_counts, x="Model", y="Percent", hue="Similarity", order=ORDER)
    plt.title("Similarity to NCBI Plasmid DB (Subset)")
    plt.ylabel("Percent of Sequences")
    save_plot(f"{out_dir}/fig11_similarity.png")

    # Combined metrics plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, (col, title, log_scale) in enumerate(metrics):
        ax = axes[i]
        data = metrics_df.copy()
        if log_scale: data[col] = np.log10(data[col].clip(lower=1))
        valid_order = [m for m in ORDER_WITH_REAL if m in data['Model'].unique()]
        sns.boxplot(data=data, x="Model", y=col, order=valid_order, ax=ax, showfliers=False)
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(sm.output.metrics_plot)
    plt.close()

    # --- 6. Completion & Surprisal ---
    try:
        comp_df = pd.read_csv(sm.input.bench_comp)
        comp_df['Model'] = comp_df['Model'].map(MODEL_MAP)
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=comp_df, x="Model", y="AvgLogProb", order=ORDER, showfliers=False)
        plt.title("Held-out Completion Confidence")
        plt.ylabel("Avg LogProb (Next 100bp)")
        save_plot(f"{out_dir}/fig8_completion.png")
    except Exception as e: print(f"Completion plot failed: {e}")

    try:
        surp_df = pd.read_csv(sm.input.bench_surp)
        surp_df['Model'] = surp_df['Model'].map(MODEL_MAP)
        pivoted = surp_df.pivot_table(index=['Plasmid', 'PromoterStart'], columns='Model', values='MeanLogProb')
        
        if 'Base' in pivoted.columns:
            gap_data = []
            for model in pivoted.columns:
                if model == 'Base': continue
                diff = pivoted[model] - pivoted['Base']
                for val in diff.dropna():
                    gap_data.append({"Model": model, "Gap": val})
            
            gap_df = pd.DataFrame(gap_data)
            plt.figure(figsize=(8, 6))
            gap_order = [m for m in ORDER if m != 'Base']
            sns.stripplot(data=gap_df, x="Model", y="Gap", order=gap_order, jitter=True, alpha=0.6)
            plt.axhline(0, color='black', linestyle='--')
            plt.title("Surprisal Gap (Model - Base)")
            plt.ylabel("LogProb Difference (Positive = Better)")
            save_plot(f"{out_dir}/fig9_surprisal.png")
    except Exception as e: print(f"Surprisal plot failed: {e}")
    
    # Cleanup temp db
    # shutil.rmtree(tmp_dir) # Don't cleanup if we want to reuse? But here we use 'data/refseq_data' which is persistent.
    # The 'tmp_dir' in this function was for blast query temp file.
    # The 'tmp_dir' in the previous version was for remote db.
    # Here I don't use 'tmp_dir' for DB. I use 'data/refseq_data'. So I don't need to clean it up.

if __name__ == "__main__":
    main(snakemake)