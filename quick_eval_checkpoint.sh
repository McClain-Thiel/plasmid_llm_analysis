#!/bin/bash
# Quick checkpoint evaluation script
# Usage: ./quick_eval_checkpoint.sh <checkpoint_path> <output_name>

CHECKPOINT_PATH=$1
OUTPUT_NAME=$2

if [ -z "$CHECKPOINT_PATH" ] || [ -z "$OUTPUT_NAME" ]; then
    echo "Usage: ./quick_eval_checkpoint.sh <checkpoint_path> <output_name>"
    exit 1
fi

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate /opt/dlami/nvme/mcclain_analysis/plasmid_llm_analysis/env

OUTPUT_DIR="results_${OUTPUT_NAME}"
GEN_DIR="${OUTPUT_DIR}/generations/${OUTPUT_NAME}"
QC_DIR="${OUTPUT_DIR}/qc/${OUTPUT_NAME}"

mkdir -p "$GEN_DIR" "$QC_DIR"

echo "=== Evaluating checkpoint: $CHECKPOINT_PATH ==="
echo "Output: $OUTPUT_DIR"

# Step 1: Generate sequences
echo "[1/3] Generating sequences..."
python src/scripts/generate_vllm.py \
    --model_path "$CHECKPOINT_PATH" \
    --output_dir "$GEN_DIR" \
    --gpu_util 0.8 \
    --temperature 0.95 \
    --prompts "ATG" "tttacggctagctcagtcctaggtatagtgctagcTACTagagaaagaggagaaatactaAATGatgcgtaaaggagaagaacttttcactggagttgtcccaattcttgttgaattagatggtgatgttaatgggcacaaattttctgtcagtggagagggtgaaggtgatgcaacatacggaaaacttacccttaaatttatttgcactactggaaaactacctgttccatggccaacacttgtcactactttcggttatggtgttcaatgctttgcgagatacccagatcatatgaaacagcatgactttttcaagagtgccatgcccgaaggttatgtacaggaaagaactatatttttcaaagatgacgggaactacaagacacgtgctgaagtcaagtttgaaggtgatacccttgttaatagaatcgagttaaaaggtattgattttaaagaagatggaaacattcttggacacaaattggaatacaactataactcacacaatgtatacatcatggcagacaaacaaaagaatggaatcaaagttaacttcaaattagacacaacattgaagatggaagcgttcaactagcagaccattatcaacaaaatactccaattggcgatggccctgtccttttaccagacaaccattacctgtccacacaatctgccctttcgaaagatcccaacgaaaagagagatcacatggtccttcttgagtttgtaacagctgctgggattacacatggcatggatgaactatacaaataataaAGGTccaggcatcaaataaaacgaaaggctcagtcgaaagactgggcctttcgttttatctgttgtttgtcggtgaacgctctctactagagtcacactggctcaccttcgggtgggcctttctgcgtttata" \
    --samples 50

# Step 2: Run QC
echo "[2/3] Running QC pipeline..."
python src/qc/qc_oriv_arg2.py \
    --in "$GEN_DIR" \
    --outdir "$QC_DIR" \
    --oridb_prefix data/oridb_nucl \
    --oridb_ref assets/oriV_refs.fasta \
    --threads 8

# Step 3: Run repeats analysis
echo "[3/3] Running repeats analysis..."
python src/qc/repeats2.py "$GEN_DIR" \
    --circular \
    --out "$QC_DIR/repeats.csv"

# Step 4: Run filtering
python src/qc/filter_qc_two_stage2.py \
    --qc_out "$QC_DIR" \
    --repeats_csv "$QC_DIR/repeats.csv" \
    --out_pass "$QC_DIR/passed.csv" \
    --out_fail "$QC_DIR/failed.csv" \
    --ori_strict_identity 99.0 \
    --amr_strict_identity 100.0 \
    --repeat_max_len 50 \
    --repeat_ge

# Step 5: Calculate stats
echo ""
echo "=== RESULTS FOR $OUTPUT_NAME ==="
TOTAL=$(ls "$GEN_DIR"/*.fasta 2>/dev/null | wc -l)
PASSED=$(tail -n +2 "$QC_DIR/passed.csv" 2>/dev/null | wc -l)
UNIQUE_ORIS=$(cut -d',' -f3 "$QC_DIR/aggregate_ori_calls.csv" 2>/dev/null | tail -n +2 | sort -u | wc -l)
UNIQUE_AMRS=$(cut -d',' -f2 "$QC_DIR/aggregate_amr_calls.csv" 2>/dev/null | tail -n +2 | sort -u | wc -l)
ORI_TYPES=$(cut -d',' -f3 "$QC_DIR/aggregate_ori_calls.csv" 2>/dev/null | tail -n +2 | sort -u | tr '\n' ';' | sed 's/;$//')
AMR_TYPES=$(cut -d',' -f2 "$QC_DIR/aggregate_amr_calls.csv" 2>/dev/null | tail -n +2 | sort -u | tr '\n' ';' | sed 's/;$//')

echo "Total generated: $TOTAL"
echo "QC passed: $PASSED"
echo "Pass rate: $(echo "scale=1; $PASSED * 100 / $TOTAL" | bc)%"
echo "Unique ORIs: $UNIQUE_ORIS ($ORI_TYPES)"
echo "Unique AMRs: $UNIQUE_AMRS ($AMR_TYPES)"
