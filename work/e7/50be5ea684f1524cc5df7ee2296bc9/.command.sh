#!/bin/bash -euo pipefail
# Create symlinks BEFORE running filter (files may have different names)
ln -sf input_ori.csv aggregate_ori_calls.csv
ln -sf input_amr.csv aggregate_amr_calls.csv

plasmid-filter \
    --qc-out . \
    --out-pass passed.csv \
    --out-fail failed.csv \
    --ori-min-identity 85 \
    --ori-min-cov 80 \
    --amr-min-identity 85 \
    --amr-min-cov 80 \
    --ori-count-min 1 \
    --ori-count-max 1 \
    --amr-count-min 1 \
    --amr-count-max 1 \
    --repeat-max-len 50 \
    --two-stage \
    --repeats-csv aggregate_repeats.csv
