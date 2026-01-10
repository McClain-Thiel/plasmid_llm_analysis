import torch
import glob
import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from Bio import SeqIO
from tqdm import tqdm

def get_model_logprobs(model, tokenizer, seq, device):
    """
    Returns a numpy array of log-probs for each token in seq (except the first one).
    Matches 'logprob_trace' from reference.
    """
    enc = tokenizer(seq, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    attn_mask = enc.attention_mask.to(device)
    
    # Safety clamp
    if (input_ids >= model.config.vocab_size).any():
        input_ids[input_ids >= model.config.vocab_size] = 0

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attn_mask).logits
    
    # log_softmax over vocabulary
    lps = torch.nn.functional.log_softmax(logits, dim=-1)
    
    # Shift so that lps[i] predicts input_ids[i+1]
    # input_ids: [Batch, T]
    # logits:    [Batch, T, Vocab] -> we want logits[t] to predict input_ids[t+1]
    # Standard CLM training: logits[t] predicts input_ids[t+1]
    
    target_ids = input_ids[:, 1:]
    lps = lps[:, :-1, :]
    
    # Gather the log-prob of the true token
    gathered = lps.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    return gathered.cpu().numpy()[0] # [T-1]

def main(sm):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    real_dir = sm.input.real_data
    models_config = sm.params.models # Dict: {name: path}
    
    # 1. Load Models (One by one to save VRAM? Or all at once?)
    # Benchmarking is sequential per model usually.
    
    # Data containers
    completion_rows = []
    surprisal_rows = []
    
    # Pre-load Real Sequences for Completion
    real_fastas = glob.glob(os.path.join(real_dir, "*.fasta"))
    completion_windows = [] # (PlasmidName, Start, Prefix, Target)
    
    WINDOW_PREFIX = 400
    WINDOW_TARGET = 100
    CUT_STRIDE = 300
    
    for rf in real_fastas:
        pname = os.path.basename(rf)
        try:
            seq = str(SeqIO.read(rf, "fasta").seq).upper()
            valid_chars = set("ATGC")
            seq = "".join([c for c in seq if c in valid_chars])
            
            if len(seq) < WINDOW_PREFIX + WINDOW_TARGET + 10:
                continue
                
            for start in range(0, len(seq) - (WINDOW_PREFIX + WINDOW_TARGET), CUT_STRIDE):
                prefix = seq[start : start + WINDOW_PREFIX]
                target = seq[start + WINDOW_PREFIX : start + WINDOW_PREFIX + WINDOW_TARGET]
                completion_windows.append((pname, start, prefix, target))
        except:
            continue
            
    # Pre-load Annotation Windows for Surprisal
    surprisal_windows = [] # (PlasmidName, PromoterStart, WindowSeq, PromoterEnd, CDSStart)
    
    for rf in real_fastas:
        base_name = os.path.splitext(os.path.basename(rf))[0]
        csv_path = os.path.join(real_dir, f"{base_name}_pLann.csv")
        
        if not os.path.exists(csv_path):
            continue
            
        try:
            seq = str(SeqIO.read(rf, "fasta").seq).upper()
            df = pd.read_csv(csv_path)
            
            # Normalize cols
            norm_cols = {c.lower().replace(" ", "_"): c for c in df.columns}
            if not all(k in norm_cols for k in ["type", "start_location", "end_location"]):
                continue
                
            type_col = norm_cols["type"]
            start_col = norm_cols["start_location"]
            end_col = norm_cols["end_location"]
            strand_col = norm_cols.get("strand", None)
            
            promoters = df[df[type_col].astype(str).str.lower() == "promoter"]
            cdss = df[df[type_col].astype(str).str.lower() == "cds"]
            
            if promoters.empty or cdss.empty:
                continue
                
            for _, p in promoters.iterrows():
                p_start = int(p[start_col]) - 1
                p_end = int(p[end_col]) - 1
                p_strand = str(p[strand_col]) if strand_col else "+"
                
                # Find nearest downstream CDS
                if p_strand == "+":
                    cand = cdss[(cdss[strand_col].astype(str) == "+") & (cdss[start_col] - 1 >= p_end)] if strand_col else cdss[cdss[start_col] - 1 >= p_end]
                    if cand.empty: continue
                    c = cand.sort_values(start_col).iloc[0]
                    cds_start = int(c[start_col]) - 1
                else:
                    # Reverse strand logic (simplified for now, strictly downstream)
                    # p_end is actually start of promoter in 5'->3' on minus strand?
                    # Let's stick to simple forward logic from reference or skip minus if risky
                    continue 

                # Extract window around CDS start (similar to reference)
                WINDOW_BP = 100
                left = max(0, cds_start - WINDOW_BP)
                right = min(len(seq), cds_start + WINDOW_BP)
                window_seq = seq[left:right]
                
                if len(window_seq) > 50:
                    surprisal_windows.append({
                        "Plasmid": base_name,
                        "PromoterStart": p_start,
                        "CDSStart": cds_start,
                        "Seq": window_seq
                    })
        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            
    print(f"Found {len(completion_windows)} completion windows.")
    print(f"Found {len(surprisal_windows)} surprisal windows.")

    # 3. Iterate Models
    for model_name, model_path in sm.params.models.items():
        print(f"Evaluating {model_name}...")
        try:
            tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            mod = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
            
            # --- Exp 2: Completion ---
            for (pname, start, prefix, target) in tqdm(completion_windows, desc="Completion"):
                # We need logprob of TARGET given PREFIX
                # get_model_logprobs returns all token LPs
                # We tokenize 'prefix + target'
                full_txt = prefix + target
                full_ids = tok(full_txt, return_tensors="pt").input_ids
                prefix_len_tokens = tok(prefix, return_tensors="pt").input_ids.shape[1]
                
                if full_ids.shape[1] > mod.config.max_position_embeddings:
                    continue # Skip too long
                
                trace = get_model_logprobs(mod, tok, full_txt, device)
                
                # Trace is length L-1. 
                # trace[i] is log P(token[i+1] | 0...i)
                # We want tokens corresponding to target.
                # Target starts at prefix_len_tokens (roughly)
                # Let's be precise:
                # The target tokens start at index 'prefix_len_tokens' in input_ids
                # So we want trace indices starting at 'prefix_len_tokens - 1'
                
                # Adjust for potential tokenizer oddities (merges):
                # Using the slice method from reference is safer:
                
                # Re-do specific slicing logic for this window
                # Reference: 
                # target_slice = slice(prefix_len - 1, full_len - 1)
                # relevant_log_probs = log_probs[:, target_slice, :]
                
                # My trace function returns 1D array of all transition probs
                # trace[i] -> prob of token i+1
                # Target tokens are at indices [prefix_len_tokens, prefix_len_tokens+1, ...] in input_ids
                # So we want trace at indices [prefix_len_tokens-1, ...]
                
                target_start_idx = prefix_len_tokens - 1
                if target_start_idx < len(trace):
                    target_lps = trace[target_start_idx:]
                    avg_lp = float(np.mean(target_lps))
                    completion_rows.append({
                        "Model": model_name,
                        "Plasmid": pname,
                        "Start": start,
                        "AvgLogProb": avg_lp
                    })

            # --- Exp 3: Surprisal ---
            for win in tqdm(surprisal_windows, desc="Surprisal"):
                # Just compute average logprob over the whole window
                # Or specific region? Reference used 'trace' and took mean difference.
                # "Compute logprob traces for Base and RL... diff = RL - Base ... mean"
                # So here we just store the mean logprob of the window
                
                seq = win["Seq"]
                trace = get_model_logprobs(mod, tok, seq, device)
                mean_lp = float(np.mean(trace))
                
                surprisal_rows.append({
                    "Model": model_name,
                    "Plasmid": win["Plasmid"],
                    "PromoterStart": win["PromoterStart"],
                    "CDSStart": win["CDSStart"],
                    "MeanLogProb": mean_lp
                })
                
            del mod
            del tok
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Failed to benchmark {model_name}: {e}")

    # Save Results
    pd.DataFrame(completion_rows).to_csv(sm.output.completion, index=False)
    pd.DataFrame(surprisal_rows).to_csv(sm.output.surprisal, index=False)

if __name__ == "__main__":
    main(snakemake)