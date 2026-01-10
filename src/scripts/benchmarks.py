import torch
import glob
import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from Bio import SeqIO
from tqdm import tqdm

def get_model_logprobs(model, tokenizer, seq, device):
    enc = tokenizer(seq, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    attn_mask = enc.attention_mask.to(device)
    if (input_ids >= model.config.vocab_size).any():
        input_ids[input_ids >= model.config.vocab_size] = 0
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attn_mask).logits
    lps = torch.nn.functional.log_softmax(logits, dim=-1)
    target_ids = input_ids[:, 1:]
    lps = lps[:, :-1, :]
    gathered = lps.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    return gathered.cpu().numpy()[0]

def main(sm):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    real_dir = sm.input.real_data
    
    completion_rows = []
    surprisal_rows = []
    
    real_fastas = glob.glob(os.path.join(real_dir, "*.fasta"))
    completion_windows = []
    
    WINDOW_PREFIX = 400
    WINDOW_TARGET = 100
    CUT_STRIDE = 300
    
    for rf in real_fastas:
        pname = os.path.basename(rf)
        try:
            seq = str(SeqIO.read(rf, "fasta").seq).upper()
            valid_chars = set("ATGC")
            seq = "".join([c for c in seq if c in valid_chars])
            if len(seq) < WINDOW_PREFIX + WINDOW_TARGET + 10: continue
            for start in range(0, len(seq) - (WINDOW_PREFIX + WINDOW_TARGET), CUT_STRIDE):
                prefix = seq[start : start + WINDOW_PREFIX]
                target = seq[start + WINDOW_PREFIX : start + WINDOW_PREFIX + WINDOW_TARGET]
                completion_windows.append((pname, start, prefix, target))
        except: continue
            
    surprisal_windows = []
    for rf in real_fastas:
        base_name = os.path.splitext(os.path.basename(rf))[0]
        csv_path = os.path.join(real_dir, f"{base_name}_pLann.csv")
        if not os.path.exists(csv_path): continue
        try:
            seq = str(SeqIO.read(rf, "fasta").seq).upper()
            df = pd.read_csv(csv_path)
            norm_cols = {c.lower().replace(" ", "_"): c for c in df.columns}
            type_col = norm_cols["type"]
            start_col = norm_cols["start_location"]
            end_col = norm_cols["end_location"]
            strand_col = norm_cols.get("strand", None)
            
            promoters = df[df[type_col].astype(str).str.lower() == "promoter"]
            cdss = df[df[type_col].astype(str).str.lower() == "cds"]
            
            for _, p in promoters.iterrows():
                p_start = int(p[start_col]) - 1
                p_end = int(p[end_col]) - 1
                p_strand = str(p[strand_col]) if strand_col else "1"
                
                # Normalize strand
                if p_strand == "1" or p_strand == "+":
                    cand = cdss[(df[strand_col].astype(str).isin(["1", "+"])) & (df[start_col] - 1 >= p_end)] if strand_col else cdss[cdss[start_col] - 1 >= p_end]
                    if cand.empty: continue
                    c = cand.sort_values(start_col).iloc[0]
                    cds_start = int(c[start_col]) - 1
                else:
                    cand = cdss[(df[strand_col].astype(str).isin(["-1", "-"])) & (df[end_col] - 1 <= p_start)] if strand_col else cdss[cdss[end_col] - 1 <= p_start]
                    if cand.empty: continue
                    c = cand.sort_values(end_col, ascending=False).iloc[0]
                    cds_start = int(c[end_col]) - 1 # Use end as "start" for reverse window?
                
                WINDOW_BP = 100
                left = max(0, cds_start - WINDOW_BP)
                right = min(len(seq), cds_start + WINDOW_BP)
                surprisal_windows.append({"Plasmid": base_name, "PromoterStart": p_start, "CDSStart": cds_start, "Seq": seq[left:right]})
        except Exception as e:
            print(f"Error surprisal {base_name}: {e}")
            
    for model_name, model_path in sm.params.models.items():
        try:
            tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            mod = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
            for (pname, start, prefix, target) in tqdm(completion_windows, desc=f"{model_name} Comp"):
                full_txt = prefix + target
                trace = get_model_logprobs(mod, tok, full_txt, device)
                prefix_len_tokens = tok(prefix, return_tensors="pt").input_ids.shape[1]
                target_start_idx = prefix_len_tokens - 1
                if target_start_idx < len(trace):
                    completion_rows.append({"Model": model_name, "Plasmid": pname, "Start": start, "AvgLogProb": float(np.mean(trace[target_start_idx:]))})
            for win in tqdm(surprisal_windows, desc=f"{model_name} Surp"):
                trace = get_model_logprobs(mod, tok, win["Seq"], device)
                surprisal_rows.append({"Model": model_name, "Plasmid": win["Plasmid"], "PromoterStart": win["PromoterStart"], "CDSStart": win["CDSStart"], "MeanLogProb": float(np.mean(trace))})
            del mod; del tok; torch.cuda.empty_cache()
        except Exception as e: print(f"Failed {model_name}: {e}")

    pd.DataFrame(completion_rows).to_csv(sm.output.completion, index=False)
    pd.DataFrame(surprisal_rows).to_csv(sm.output.surprisal, index=False)

if __name__ == "__main__":
    main(snakemake)
