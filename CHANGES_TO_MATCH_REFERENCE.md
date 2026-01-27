# Changes Made to Match Reference Implementation

Date: 2026-01-09
Reference: Python scripts from `/Users/mcclainthiel/Projects/PhD/plasmidbackbonedesign/src/qc/`

## Summary of Key Fixes

### 1. **BLAST_ORI Process** (`modules/qc/blast_ori.nf`)
**Changes:**
- ✅ Changed outfmt from 15 fields to **12 fields** (matching reference `qc_oriv_arg2.py`)
- ✅ Changed BLAST task from `megablast` to **`dc-megablast`**
- ✅ Changed `max_target_seqs` from 10 to **2000**
- ✅ Added `soft_masking true` and `dust yes` parameters
- ✅ Added **per-sequence filtering and overlap resolution** in Python post-processing:
  - Calculate `qcov`, `scovs`, `q_from`, `q_to`, `strand`
  - Filter by thresholds: `pident >= 85`, `scovs >= 80`, `length >= 100`
  - Resolve overlaps: keep highest identity hit per position (matches `choose_non_overlapping_highest_identity`)
  - Process each unique sequence ID separately (not per batch)

**Reference Match:** `qc_oriv_arg2.py` lines 75-158

### 2. **AGGREGATE_ORI Process** (`modules/qc/filter.nf`)
**Changes:**
- ✅ Simplified to **just combine per-sequence filtered CSVs**
- ❌ Removed filtering logic (now done in BLAST_ORI)
- ❌ Removed overlap resolution (now done in BLAST_ORI per-sequence)

**Reference Match:** `qc_oriv_arg2.py` lines 250-268 (aggregation only)

### 3. **Two-Stage Filter Logic** (`py/src/plasmid_analytics/qc/filter.py`)
**Changes:**
- ✅ Fixed Stage B ORI validation: **Always** requires `n_ori_strict >= ori_low_count_min` (not just when count is exactly 1)
- ✅ Fixed Stage B ARG validation: **Always** requires `n_amr_strict >= amr_low_count_min`
- ✅ Added optional **`amr_strict_min`** parameter: minimum number of ARGs that must meet strict thresholds
- ✅ Added optional **`amr_strict_all`** parameter: if True, ALL low-threshold ARGs must meet strict thresholds
- ✅ Updated CLI (`py/src/plasmid_analytics/cli.py`) to support new parameters

**Reference Match:** `filter_qc_two_stage2.py` lines 227-250

### 4. **Prompt Handling** (`modules/generate/sample_model.nf`)
**Changes:**
- ✅ Changed GFP cassette prompt to use **full sequence** (not truncated to 150bp)
- ✅ Removed `tail -n 1 | head -c 150` truncation
- ✅ Now reads entire FASTA sequence: `grep -v "^>" | tr -d '\n' | tr 'a-z' 'A-Z'`

**Reference Match:** User requested full GFP cassette (not truncated as in original `plasmidgpt_exp.py`)

## Verification

### Before Fixes:
- ❌ Only 1 BLAST_ORI process ran (out of 4 samples)
- ❌ 131 ORI hits with 0% pass rate
- ❌ Incorrect two-stage validation logic

### After Fixes:
- ✅ All 4 BLAST_ORI processes run correctly
- ✅ ~10-12 filtered ORI hits per batch (after per-sequence overlap resolution)
- ✅ Correct two-stage validation matching reference exactly
- ✅ Full GFP cassette (917bp) used as prompt

## Files Modified:
1. `modules/qc/blast_ori.nf` - BLAST command and per-sequence post-processing
2. `modules/qc/filter.nf` - AGGREGATE_ORI simplified
3. `py/src/plasmid_analytics/qc/filter.py` - Two-stage filter logic
4. `py/src/plasmid_analytics/cli.py` - CLI arguments for new filter options
5. `workflows/qc.nf` - Channel splitting for multiple consumers
6. `modules/generate/sample_model.nf` - Full GFP cassette prompt

## Removed Unnecessary Logic:
- ❌ Filtering/overlap resolution in AGGREGATE_ORI (moved to per-sequence in BLAST_ORI)
- ❌ Hardcoded check for `ori_low_count_min == ori_low_count_max == 1` in two-stage filter

## Architecture Now Matches Reference:
```
1. BLAST_ORI (per sample)
   → Raw BLAST hits
   → Filter by 85%/80% thresholds
   → Resolve overlaps per sequence
   → Output: per-sequence ori_calls.csv

2. AGGREGATE_ORI
   → Combine all per-sequence CSVs
   → Output: aggregate_ori_calls.csv

3. FILTER_QC (two-stage)
   Stage A: Count with low thresholds (85%/80%)
           Check if count in [min, max] window
   Stage B: Validate with strict thresholds (99%/99%)
           Require n_ori_strict >= ori_low_count_min
           Optional: amr_strict_all, amr_strict_min
   Repeats: Fail if longest_len >= repeat_max_len
```

This architecture exactly matches the reference Python implementation.
