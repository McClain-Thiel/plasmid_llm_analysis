import os
import subprocess
from pathlib import Path

plasmid_dir = Path("/Users/mcclainthiel/Downloads/plasmids")
fasta_files = list(plasmid_dir.glob("*.fasta"))

for fasta_file in fasta_files:
    csv_file = fasta_file.with_name(f"{fasta_file.stem}_pLann.csv")
    
    if csv_file.exists():
        print(f"Skipping {fasta_file.name} (already annotated)")
        continue
    
    with open(fasta_file, 'r') as f:
        content = f.read().strip()
    
    if not content.startswith('>'):
        cleaned_content = f">{fasta_file.stem}\n{content}\n"
        
        temp_file = fasta_file.with_suffix('.tmp.fasta')
        with open(temp_file, 'w') as f:
            f.write(cleaned_content)
        
        print(f"Annotating {fasta_file.name}...")
        try:
            subprocess.run([
                f"{os.path.expanduser('~')}/miniconda3/envs/plannotate/bin/plannotate",
                "batch",
                "-i", str(temp_file),
                "-l",
                "-c",
                "--no_gbk",
                "-o", str(plasmid_dir)
            ], check=True, capture_output=True)
            
            temp_csv = plasmid_dir / f"{temp_file.stem}_pLann.csv"
            if temp_csv.exists():
                temp_csv.rename(csv_file)
            
            print(f"  ✓ Created {csv_file.name}")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Error: {e.stderr.decode()[:200]}")
        finally:
            if temp_file.exists():
                temp_file.unlink()
    else:
        print(f"Annotating {fasta_file.name}...")
        try:
            subprocess.run([
                f"{os.path.expanduser('~')}/miniconda3/envs/plannotate/bin/plannotate",
                "batch",
                "-i", str(fasta_file),
                "-l",
                "-c",
                "--no_gbk",
                "-o", str(plasmid_dir)
            ], check=True, capture_output=True)
            print(f"  ✓ Created {csv_file.name}")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Error: {e.stderr.decode()[:200]}")

print("\nDone!")
