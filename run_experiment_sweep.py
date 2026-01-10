import yaml
import os
import subprocess
import shutil

def main():
    if not os.path.exists("config.yaml.orig"):
        shutil.copy("config.yaml", "config.yaml.orig")

    with open("config.yaml.orig") as f:
        base_config = yaml.safe_load(f)

    experiments = [
        {"name": "Exp1_Baseline", "temp": 0.95, "rep_pen": 1.0},
        {"name": "Exp2_HighDiv", "temp": 1.1, "rep_pen": 1.0},
        {"name": "Exp3_StrictSFT", "temp": 0.8, "rep_pen": 1.2},
        {"name": "Exp4_Balanced", "temp": 0.9, "rep_pen": 1.1},
    ]

    for exp in experiments:
        print(f"\n========================================")
        print(f"=== Starting {exp['name']} ===")
        print(f"=== Temp: {exp['temp']}, RepPen: {exp['rep_pen']} ===")
        print(f"========================================\n")
        
        config = base_config.copy()
        config["output_dir"] = f"results_{exp['name']}"
        config["generation"]["temperature"] = exp["temp"]
        config["generation"]["repetition_penalty"] = exp["rep_pen"]
        
        with open("config.yaml", "w") as f:
            yaml.dump(config, f)
        
        # Unlock
        subprocess.run("conda run -p ./env snakemake --unlock", shell=True)
        
        # Run
        cmd = "conda run -p ./env snakemake --cores 32 --resources gpu=1 --rerun-incomplete"
        subprocess.run(cmd, shell=True, check=False) # Don't crash entire sweep if one fails

    # Restore
    shutil.copy("config.yaml.orig", "config.yaml")

if __name__ == "__main__":
    main()