#!/usr/bin/env python3
"""
BLAST generated plasmid sequences against NCBI nt database and calculate diversity metrics.

Outputs:
  - Per-sequence BLAST results (top hit similarity, coverage)
  - Aggregate diversity metrics per model
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from Bio import SeqIO, Entrez
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Blast import NCBIWWW, NCBIXML

# NCBI credentials - set via environment or command line
# With API key: 10 requests/second; without: 3 requests/second
NCBI_EMAIL = os.environ.get('NCBI_EMAIL', None)
NCBI_API_KEY = os.environ.get('NCBI_API_KEY', None)


def extract_sequences_to_fasta(csv_path: Path, fasta_path: Path, model_name: str) -> int:
    """Extract sequences from outputs.csv to FASTA format."""
    records = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_id = f"{model_name}_{row['id']}"
            sequence = row['full']
            record = SeqRecord(Seq(sequence), id=seq_id, description="")
            records.append(record)

    with open(fasta_path, 'w') as f:
        SeqIO.write(records, f, 'fasta')

    return len(records)


def blast_sequence(seq_record, db="nt", program="blastn", hitlist_size=5, timeout=120):
    """Run BLAST for a single sequence against NCBI."""
    try:
        result_handle = NCBIWWW.qblast(
            program,
            db,
            seq_record.format("fasta"),
            hitlist_size=hitlist_size,
            megablast=True,  # faster
        )
        blast_records = NCBIXML.parse(result_handle)
        record = next(blast_records)

        results = []
        for alignment in record.alignments:
            for hsp in alignment.hsps:
                identity = (hsp.identities / hsp.align_length) * 100
                query_coverage = (hsp.align_length / record.query_length) * 100
                subject_coverage = (hsp.align_length / alignment.length) * 100 if alignment.length > 0 else 0

                results.append({
                    'query_id': seq_record.id,
                    'query_length': record.query_length,
                    'subject_id': alignment.hit_id,
                    'subject_title': alignment.title[:100],
                    'subject_length': alignment.length,
                    'align_length': hsp.align_length,
                    'identities': hsp.identities,
                    'pct_identity': round(identity, 2),
                    'query_coverage': round(query_coverage, 2),
                    'subject_coverage': round(subject_coverage, 2),
                    'evalue': hsp.expect,
                    'bitscore': hsp.bits,
                })
        return results
    except Exception as e:
        print(f"Error blasting {seq_record.id}: {e}", file=sys.stderr)
        return []


def blast_all_sequences(fasta_path: Path, output_tsv: Path, has_api_key: bool = False):
    """BLAST all sequences in a FASTA file."""
    records = list(SeqIO.parse(fasta_path, 'fasta'))
    print(f"BLASTing {len(records)} sequences against NCBI nt...")

    all_results = []
    completed = 0

    # Rate limit: 10 req/sec with API key, 3 req/sec without
    sleep_time = 0.15 if has_api_key else 0.5

    for i, record in enumerate(records):
        print(f"  [{i+1}/{len(records)}] {record.id} ({len(record.seq)} bp)...", flush=True)
        results = blast_sequence(record)
        all_results.extend(results)
        completed += 1

        # Rate limiting - wait between requests
        if i < len(records) - 1:
            time.sleep(sleep_time)

    # Write results
    if all_results:
        fieldnames = ['query_id', 'query_length', 'subject_id', 'subject_title',
                      'subject_length', 'align_length', 'identities', 'pct_identity',
                      'query_coverage', 'subject_coverage', 'evalue', 'bitscore']
        with open(output_tsv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            writer.writerows(all_results)

    return all_results


def calculate_diversity_metrics(blast_results: list) -> dict:
    """Calculate diversity metrics from BLAST results."""
    if not blast_results:
        return {'n_sequences': 0, 'n_with_hits': 0}

    # Group by query
    by_query = {}
    for r in blast_results:
        qid = r['query_id']
        if qid not in by_query:
            by_query[qid] = []
        by_query[qid].append(r)

    # Get best hit per query
    best_hits = []
    for qid, hits in by_query.items():
        # Best by bitscore
        best = max(hits, key=lambda x: x['bitscore'])
        best_hits.append(best)

    # Unique subjects hit
    unique_subjects = set(r['subject_id'] for r in best_hits)

    # Average similarity/coverage
    avg_identity = sum(r['pct_identity'] for r in best_hits) / len(best_hits) if best_hits else 0
    avg_coverage = sum(r['query_coverage'] for r in best_hits) / len(best_hits) if best_hits else 0

    return {
        'n_sequences': len(by_query),
        'n_with_hits': len(best_hits),
        'n_unique_subjects': len(unique_subjects),
        'avg_pct_identity': round(avg_identity, 2),
        'avg_query_coverage': round(avg_coverage, 2),
        'diversity_ratio': round(len(unique_subjects) / len(by_query), 3) if by_query else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="BLAST plasmid sequences against NCBI and compute diversity")
    parser.add_argument("--exp-dir", required=True, help="Experiment directory (e.g., results_Exp14_Aggressive)")
    parser.add_argument("--models", nargs="+", default=["Base", "SFT", "RL", "SFT_GRPO"], help="Models to process")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: exp-dir/ncbi_blast)")
    parser.add_argument("--sample", type=int, default=None, help="Only process first N sequences per model")
    parser.add_argument("--email", default=NCBI_EMAIL, help="NCBI email for rate limiting (or set NCBI_EMAIL env var)")
    parser.add_argument("--api-key", default=NCBI_API_KEY, help="NCBI API key for higher rate limits (or set NCBI_API_KEY env var)")
    args = parser.parse_args()

    # Configure NCBI credentials
    if args.email:
        Entrez.email = args.email
        print(f"Using NCBI email: {args.email}")
    else:
        print("WARNING: No NCBI email set. Rate limits may apply. Use --email or set NCBI_EMAIL env var.")

    if args.api_key:
        Entrez.api_key = args.api_key
        print("Using NCBI API key (10 req/sec limit)")
    else:
        print("No API key set (3 req/sec limit). Get one at: https://www.ncbi.nlm.nih.gov/account/settings/")

    exp_dir = Path(args.exp_dir)
    output_dir = Path(args.output_dir) if args.output_dir else exp_dir / "ncbi_blast"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for model in args.models:
        print(f"\n{'='*60}")
        print(f"Processing model: {model}")
        print(f"{'='*60}")

        csv_path = exp_dir / "generations" / model / "outputs.csv"
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found, skipping")
            continue

        # Extract to FASTA
        fasta_path = output_dir / f"{model}.fasta"
        n_seqs = extract_sequences_to_fasta(csv_path, fasta_path, model)
        print(f"  Extracted {n_seqs} sequences to {fasta_path}")

        # Optionally sample
        if args.sample and args.sample < n_seqs:
            print(f"  Sampling first {args.sample} sequences")
            records = list(SeqIO.parse(fasta_path, 'fasta'))[:args.sample]
            with open(fasta_path, 'w') as f:
                SeqIO.write(records, f, 'fasta')
            n_seqs = args.sample

        # BLAST (skip if results already exist)
        blast_tsv = output_dir / f"{model}_blast_results.tsv"
        if blast_tsv.exists():
            print(f"  BLAST results already exist at {blast_tsv}, loading...")
            results = []
            with open(blast_tsv, 'r') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    # Convert numeric fields back to proper types
                    row['pct_identity'] = float(row['pct_identity'])
                    row['query_coverage'] = float(row['query_coverage'])
                    row['bitscore'] = float(row['bitscore'])
                    results.append(row)
            print(f"  Loaded {len(results)} existing results")
        else:
            results = blast_all_sequences(fasta_path, blast_tsv, has_api_key=bool(args.api_key))
            print(f"  BLAST results written to {blast_tsv}")

        # Calculate diversity
        metrics = calculate_diversity_metrics(results)
        metrics['model'] = model
        summary_rows.append(metrics)

        print(f"  Metrics: {metrics}")

    # Write summary
    if summary_rows:
        summary_path = output_dir / "ncbi_diversity_summary.csv"
        fieldnames = ['model', 'n_sequences', 'n_with_hits', 'n_unique_subjects',
                      'avg_pct_identity', 'avg_query_coverage', 'diversity_ratio']
        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
