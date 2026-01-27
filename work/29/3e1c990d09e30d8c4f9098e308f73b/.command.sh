#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
from plasmid_analytics.eval.compare import compare_models, pass_rate_by_model

# Load data
metrics = pd.read_csv('aggregate_metrics.csv')
passed = pd.read_csv('passed.csv')
failed = pd.read_csv('failed.csv')

# Add QC status to metrics
passed_ids = set(passed['Plasmid_ID'].astype(str))
failed_ids = set(failed['Plasmid_ID'].astype(str))

metrics['qc_status'] = metrics['plasmid_id'].astype(str).apply(
    lambda x: 'pass' if x in passed_ids else ('fail' if x in failed_ids else 'unknown')
)

# Extract model from plasmid_id if present
# Format: ModelName_PromptType_NNNN (e.g., UCL_CSSB_PlasmidGPT_ATG_0001)
if 'model' not in metrics.columns:
    # Extract everything before the last two underscore-separated segments (prompt_type_id)
    def extract_model(pid):
        parts = str(pid).split('_')
        if len(parts) >= 3:
            # Last part is sequence number, second-to-last is prompt type
            return '_'.join(parts[:-2])
        return parts[0] if parts else 'unknown'
    metrics['model'] = metrics['plasmid_id'].apply(extract_model)

# Run comparison
results = compare_models(
    df=metrics,
    model_column='model',
    qc_column='qc_status',
    metric_columns=['gc_content', 'longest_orf_atg_both', 'num_orfs_100aa'],
    output_dir='comparison_results'
)

# Save pass rates
results['pass_rates'].to_csv('pass_rates.csv', index=False)

# Summary stats per model
summary = metrics.groupby('model').agg({
    'length': ['mean', 'std', 'min', 'max'],
    'gc_content': ['mean', 'std'],
    'longest_orf_atg_both': ['mean', 'max'],
    'num_orfs_100aa': ['mean', 'max']
}).round(3)
summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
summary.reset_index().to_csv('summary_stats.csv', index=False)
