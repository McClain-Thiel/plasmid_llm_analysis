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
import traceback

try:
    import RNA
except ImportError:
    pass

# --- Configuration ---
MODEL_MAP = {'Base': 'Base', 'SFT': 'SFT', 'RL': 'GRPO', 'SFT_GRPO': 'SFT+GRPO'}
ORDER = ['Base', 'SFT', 'GRPO', 'SFT+GRPO']
ORDER_WITH_REAL = ['Real'] + ORDER

def gc_content(seq: str) -> float:
    seq = seq.upper()
    if not seq: return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)

def get_circular_mfe(seq_str: str) -> tuple[float, float]:
    if not seq_str: return 0.0, 0.0
    try:
        md = RNA.md(); md.circ = 1
        fc = RNA.fold_compound(str(seq_str), md)
        (structure, mfe) = fc.mfe()
        return mfe, mfe / len(seq_str)
    except: return 0.0, 0.0

def get_orfs_both_strands_fast(seq_str: str) -> tuple[int, int, int]:
    seq = Seq(seq_str)
    def find_max(s):
        max_len = 0
        for frame in range(3):
            try:
                prot = str(s[frame:].translate(to_stop=False))
                matches = re.findall(r'M[^*]*\*', prot)
                if matches: max_len = max(max_len, max(len(m)-1 for m in matches))
            except: pass
        return max_len
    return find_max(seq), find_max(seq.reverse_complement()), max(find_max(seq), find_max(seq.reverse_complement()))

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
    return {kmer: c / total for kmer, c in counts.items()} if total > 0 else {}

def js_divergence_from_dist(dist_a, dist_ref):
    all_keys = sorted(set(dist_a.keys()) | set(dist_ref.keys()))
    if not all_keys: return 0.0
    p = np.array([dist_a.get(k, 0.0) for k in all_keys])
    q = np.array([dist_ref.get(k, 0.0) for k in all_keys])
    return float(jensenshannon(p, q, base=2.0))

def calculate_diversity_mash(seqs, k=21, n=1000):
    hashes = []
    for s in seqs:
        s_clean = "".join([c for c in s.upper() if c in "ATGC"])
        if len(s_clean) < k: continue
        mh = sourmash.MinHash(n=n, ksize=k)
        mh.add_sequence(s_clean, force=True)
        hashes.append(mh)
    if len(hashes) < 2: return 0.0
    sims = []
    for i in range(len(hashes)):
        for j in range(i+1, len(hashes)): sims.append(hashes[i].jaccard(hashes[j]))
    return 1.0 - np.mean(sims) if sims else 0.0

def run_blast_batch(seq_dict, db_path):
    if not seq_dict: return {}
    
    # Priority: env/bin/blastn -> shutil.which
    blastn_bin = os.path.join(os.getcwd(), "env/bin/blastn")
    if not os.path.exists(blastn_bin):
        blastn_bin = shutil.which("blastn")
    if not blastn_bin:
        blastn_bin = "blastn"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp:
        tmp_name = tmp.name
        for sid, seq in seq_dict.items():
            s_clean = "".join(seq.split())
            tmp.write(f">{{sid}}\n{{s_clean}}\n")
    
    print(f"[DEBUG] Running BLAST ({{blastn_bin}}) against {{db_path}}", flush=True)
    results_map = {sid: "Novel" for sid in seq_dict}
    cmd = [blastn_bin, "-query", tmp_name, "-db", db_path, "-outfmt", "6 qseqid pident length qlen slen", "-max_target_seqs", "1", "-task", "megablast", "-num_threads", "16"]
    
    try:
        res = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
        for line in res.split('\n'):
            if not line or line.startswith("Warning"): continue
            parts = line.split('\t')
            if len(parts) < 4: continue
            qid, pident, length, qlen = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
            cov = length / qlen
            if pident > 95 and cov > 0.90: cls = "Exact Match"
            elif pident > 80: cls = "Similar"
            else: cls = "Novel"
            results_map[qid] = cls
    except Exception as e:
        print(f"[ERROR] BLAST failed: {{e}}", flush=True)
        if hasattr(e, 'output'): print(f"Output: {{e.output.decode()}}", flush=True)
    finally:
        if os.path.exists(tmp_name): os.remove(tmp_name)
    return results_map

def process_single_plasmid(args):
    seq_str, ref_dist, model_tag, prompt_tag, name_tag = args
    fwd, rev, both = get_orfs_both_strands_fast(seq_str)
    mfe_total, mfe_density = get_circular_mfe(seq_str)
    gc = gc_content(seq_str); n_orfs = count_orfs_above(seq_str, min_aa=100)
    dist_a = kmer_distribution(seq_str, k=3); js3 = js_divergence_from_dist(dist_a, ref_dist)
    return {"Model": model_tag, "Prompt": prompt_tag, "Name": name_tag, "Length": len(seq_str), "GC": gc, "Num_ORFs_>=100AA": n_orfs, "JS_3mer_vs_real": js3, "Longest_ORF_ATG_both": both, "MFE_Density": mfe_density}

def save_plot(filename):
    plt.tight_layout(); plt.savefig(filename, dpi=300); plt.close()

def main(sm):
    try:
        if "TMPDIR" in os.environ: tempfile.tempdir = os.environ["TMPDIR"]
        out_dir = os.path.dirname(sm.output.summary)
        data_dir = os.path.join(out_dir, "../../data")
        db_path = os.path.join(data_dir, "refseq_data/refseq_plasmids")
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

        print("[STATUS] Loading data...", flush=True)
        summary_rows = []; model_sequences = {m: [] for m in sm.params.models}; blast_candidates = {}
        for model in sm.params.models:
            qc_file = [f for f in sm.input.qc_pass_csvs if f"/{model}/" in f][0]
            gen_file = [f for f in sm.input.generations if f"/{model}/" in f][0]
            try:
                passed = len(pd.read_csv(qc_file)); total_df = pd.read_csv(gen_file); total = len(total_df)
                model_sequences[model] = total_df['full'].tolist()
                if total > 0:
                    subset = total_df.groupby('prompt').apply(lambda x: x.sample(n=min(len(x), 10), random_state=42)).reset_index(drop=True)
                    for _, row in subset.iterrows():
                        p_label = "ATG" if len(row['prompt']) < 10 else "GFP"
                        uid = f"{MODEL_MAP.get(model, model)}_{row['id']}"
                        blast_candidates[uid] = (row['full'], MODEL_MAP.get(model, model), p_label)
            except: passed, total = 0, 0
            summary_rows.append({"Model": MODEL_MAP.get(model, model), "PassRate": (passed/total*100) if total else 0, "Diversity": calculate_diversity_mash(model_sequences.get(model, []))})

        sim_results = run_blast_batch({k: v[0] for k, v in blast_candidates.items()}, db_path)
        real_seqs = []
        for rf in [f for f in os.listdir(sm.input.real_data) if f.endswith(".fasta")]:
            with open(os.path.join(sm.input.real_data, rf)) as f:
                s = "".join([l.strip() for l in f if not l.startswith(">")]).upper()
                if s: real_seqs.append(s)
        real_concat = "".join(real_seqs); ref_dist = kmer_distribution(real_concat, k=3)
        tasks = [(v[0], ref_dist, v[1], v[2], k) for k, v in blast_candidates.items()]
        tasks += [(s, ref_dist, "Real", "Real", f"Real_{{i}}") for i, s in enumerate(real_seqs)]
        with multiprocessing.Pool(processes=min(16, multiprocessing.cpu_count())) as pool:
            results = list(tqdm(pool.imap(process_single_plasmid, tasks), total=len(tasks), desc="Metrics"))
        for res in results: res["Similarity"] = sim_results.get(res["Name"], "Novel") if res["Model"] != "Real" else "Reference"
        metrics_df = pd.DataFrame(results)
        pd.DataFrame(summary_rows).to_csv(sm.output.summary, index=False)

        print("[STATUS] Plotting...", flush=True)
        plt.figure(figsize=(8,6)); sns.barplot(data=pd.DataFrame(summary_rows), x="Model", y="PassRate", order=ORDER, palette="viridis"); save_plot(sm.output.plot); save_plot(f"{out_dir}/fig1_pass_rate.png")
        plt.figure(figsize=(8,6)); sns.barplot(data=pd.DataFrame(summary_rows), x="Model", y="Diversity", order=ORDER, palette="magma"); save_plot(f"{out_dir}/fig10_diversity.png")
        sim_df = metrics_df[metrics_df['Model'] != 'Real'].groupby(['Model', 'Similarity']).size().reset_index(name='Count')
        totals = metrics_df[metrics_df['Model'] != 'Real'].groupby('Model').size().reset_index(name='Total'); sim_df = sim_df.merge(totals, on='Model'); sim_df['Percent'] = sim_df['Count'] / sim_df['Total'] * 100
        plt.figure(figsize=(8,6)); sns.barplot(data=sim_df, x="Model", y="Percent", hue="Similarity", order=ORDER); save_plot(f"{out_dir}/fig11_similarity.png")
        metrics = [("Length", "Length (bp)", True), ("GC", "GC Content", False), ("Longest_ORF_ATG_both", "Longest ORF (aa)", True), ("Num_ORFs_>=100AA", "# ORFs >= 100aa", True), ("JS_3mer_vs_real", "3-mer Divergence", False), ("MFE_Density", "MFE Density", False)]
        for i, (col, title, log) in enumerate(metrics):
            plt.figure(figsize=(8,6)); data = metrics_df.copy(); 
            if log: data[col] = np.log10(data[col].clip(lower=1))
            sns.boxplot(data=data, x="Model", y=col, hue="Prompt", order=ORDER_WITH_REAL, showfliers=False); plt.title(f"{title}{' (log10)' if log else ''}"); save_plot(f"{out_dir}/fig{i+2}_{col.lower()}.png")
        try:
            comp_df = pd.read_csv(sm.input.bench_comp); comp_df['Model'] = comp_df['Model'].map(MODEL_MAP); plt.figure(figsize=(8,6)); sns.boxplot(data=comp_df, x="Model", y="AvgLogProb", order=ORDER, showfliers=False); save_plot(f"{out_dir}/fig8_completion.png")
        except: pass
        try:
            surp_df = pd.read_csv(sm.input.bench_surp); surp_df['Model'] = surp_df['Model'].map(MODEL_MAP); pivoted = surp_df.pivot_table(index=['Plasmid', 'PromoterStart'], columns='Model', values='MeanLogProb')
            if 'Base' in pivoted.columns:
                gap_data = []
                for m in pivoted.columns:
                    if m == 'Base': continue
                    diff = pivoted[m] - pivoted['Base']
                    for v in diff.dropna(): gap_data.append({"Model": m, "Gap": v})
                plt.figure(figsize=(8,6)); sns.stripplot(data=pd.DataFrame(gap_data), x="Model", y="Gap", order=[m for m in ORDER if m != 'Base'], jitter=True, alpha=0.6); plt.axhline(0, color='black', linestyle='--'); save_plot(f"{out_dir}/fig9_surprisal.png")
        except: pass
        plt.figure(); plt.plot([0,1],[0,1]); plt.savefig(sm.output.metrics_plot); plt.close()
    except Exception:
        print("[CRITICAL] Analysis script failed!"); traceback.print_exc(); exit(1)

if __name__ == "__main__": main(snakemake)