import yaml
import os
import subprocess
import shutil

def main():
    if not os.path.exists("config.yaml.orig"):
        shutil.copy("config.yaml", "config.yaml.orig")

    with open("config.yaml.orig") as f:
        base_config = yaml.safe_load(f)

    # Expanded sweep with 10+ combinations
    experiments = [
        # Previous 4 (Skipped if results exist and snakemake detects them, but useful to keep for reference)
        {"name": "Exp1_Baseline",   "temp": 0.95, "rep_pen": 1.0},
        {"name": "Exp2_HighDiv",    "temp": 1.1,  "rep_pen": 1.0},
        {"name": "Exp3_StrictSFT",  "temp": 0.8,  "rep_pen": 1.2},
        {"name": "Exp4_Balanced",   "temp": 0.9,  "rep_pen": 1.1},
        
        # New 10 Combinations
        {"name": "Exp5_LowTemp_HighRep",   "temp": 0.7,  "rep_pen": 1.3}, # Focus on rigid structure, preventing loops
        {"name": "Exp6_LowTemp_NoRep",     "temp": 0.7,  "rep_pen": 1.0}, # Pure likelihood maximization
        {"name": "Exp7_HighTemp_HighRep",  "temp": 1.2,  "rep_pen": 1.3}, # Forced novelty + diversity
        {"name": "Exp8_MedTemp_MedRep",    "temp": 0.85, "rep_pen": 1.1}, # Conservative SFT adjustment
        {"name": "Exp9_MedTemp_HighRep",   "temp": 0.85, "rep_pen": 1.5}, # Strong anti-repetition constraint
        {"name": "Exp10_HighTemp_MedRep",  "temp": 1.0,  "rep_pen": 1.2}, # Balanced diversity push
        {"name": "Exp11_VeryHighTemp",     "temp": 1.3,  "rep_pen": 1.05}, # Chaos/Creativity check
        {"name": "Exp12_Strict_SFT_v2",    "temp": 0.6,  "rep_pen": 1.2}, # Near-deterministic
        {"name": "Exp13_Standard_v2",      "temp": 0.9,  "rep_pen": 1.05}, # Light penalty
        {"name": "Exp14_Aggressive",       "temp": 1.15, "rep_pen": 1.25}, # High temp + High penalty
    ]

    for exp in experiments:
        results_dir = f"results_{exp['name']}"
        report_path = os.path.join(results_dir, "analysis", "report.html")
        
        # Simple skip logic: if report exists, assume done. 
        # Snakemake would handle this too, but this saves the overhead of starting snakemake.
        if os.path.exists(report_path):
            print(f"Skipping {exp['name']} (Report exists)")
            continue

        print(f"\n========================================")
        print(f"=== Starting {exp['name']} ===")
        print(f"=== Temp: {exp['temp']}, RepPen: {exp['rep_pen']} ===")
        print(f"========================================\n")
        
        config = base_config.copy()
        config["output_dir"] = results_dir
        config["generation"]["temperature"] = exp["temp"]
        config["generation"]["repetition_penalty"] = exp["rep_pen"]
        
        with open("config.yaml", "w") as f:
            yaml.dump(config, f)
        
        # Unlock
        subprocess.run("conda run -p ./env snakemake --unlock", shell=True)
        
        # Run - forcing checking of changed params via generated files might be needed?
        # No, output_dir changes, so it's a fresh run for each exp.
        cmd = "conda run -p ./env snakemake --cores 32 --resources gpu=1 --rerun-incomplete"
        subprocess.run(cmd, shell=True, check=False) 

    # Restore
    shutil.copy("config.yaml.orig", "config.yaml")

if __name__ == "__main__":
    main()
