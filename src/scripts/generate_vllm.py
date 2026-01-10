import argparse, os, pandas as pd
from vllm import LLM, SamplingParams
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu_util", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--samples", type=int, default=50)
    return parser.parse_args()

def main():
    args = parse_args()
    try: raw_prompts = eval(args.prompts)
    except: raw_prompts = [args.prompts]
    
    input_prompts = [p.upper() for p in raw_prompts * args.samples]
    print(f"Generating {len(input_prompts)} sequences...")
    
    llm = LLM(model=args.model_path, gpu_memory_utilization=args.gpu_util)
    sampling_params = SamplingParams(
        max_tokens=256, temperature=args.temperature, top_p=0.90, stop_token_ids=[2]
    )
    outputs = llm.generate(input_prompts, sampling_params)
    
    records = []
    os.makedirs(args.output_dir, exist_ok=True)
    
    for i, output in enumerate(tqdm(outputs, desc="Saving FASTAs")):
        prompt = output.prompt.replace(" ", "").replace("\n", "").replace("\r", "")
        raw_comp = output.outputs[0].text
        # Sanitize: keep only ATGC, upper case
        comp = "".join([c for c in raw_comp.upper() if c in "ATGC"])
        
        full = prompt + comp
        records.append({"id": f"seq_{i}", "prompt": prompt, "full": full})
        
        with open(os.path.join(args.output_dir, f"seq_{i}.fasta"), "w") as f:
            f.write(f">seq_{i}\n{full}\n")

    pd.DataFrame(records).to_csv(os.path.join(args.output_dir, "outputs.csv"), index=False)

if __name__ == "__main__":
    main()