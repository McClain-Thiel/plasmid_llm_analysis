import torch, glob, os, pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from Bio import SeqIO
from tqdm import tqdm

def get_logprob(model, tokenizer, prefix, target, device):
    full = prefix + target
    enc = tokenizer(full, return_tensors="pt").to(device)
    prefix_len = tokenizer(prefix, return_tensors="pt").input_ids.shape[1]
    
    with torch.no_grad():
        logits = model(**enc).logits
    
    lps = torch.nn.functional.log_softmax(logits, dim=-1)
    target_lps = []
    for i in range(prefix_len - 1, enc.input_ids.shape[1] - 1):
        target_lps.append(lps[0, i, enc.input_ids[0, i+1]].item())
    return sum(target_lps) / len(target_lps) if target_lps else 0.0

def main(sm):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    real_files = glob.glob(f"{sm.input.real_data}/*.fasta")
    comp_res, surp_res = [], []
    
    for name, path in tqdm(sm.params.models.items(), desc="Benchmarking Models"):
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        mod = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True).to(device).eval()
        
        # Exp 2: Completion
        for rf in tqdm(real_files, desc=f"Exp2 {name}", leave=False):
            seq = str(SeqIO.read(rf, "fasta").seq).upper()
            if len(seq) > 500:
                for i in range(0, len(seq)-500, 500):
                    lp = get_logprob(mod, tok, seq[i:i+400], seq[i+400:i+500], device)
                    comp_res.append({"Model": name, "Plasmid": os.path.basename(rf), "AvgLogProb": lp})

        # Exp 3: Surprisal
        for rf in tqdm(real_files, desc=f"Exp3 {name}", leave=False):
            base = os.path.splitext(rf)[0]
            if os.path.exists(f"{base}_pLann.csv"):
                # Insert logic to extract promoter->CDS window here
                surp_res.append({"Model": name, "Gap": 0.0})

    pd.DataFrame(comp_res).to_csv(sm.output.completion, index=False)
    pd.DataFrame(surp_res).to_csv(sm.output.surprisal, index=False)

if __name__ == "__main__":
    main(snakemake)
