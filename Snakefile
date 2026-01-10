configfile: "config.yaml"

OUT = config["output_dir"]
MODELS = list(config["models"].keys())

rule all:
    input:
        f"{OUT}/analysis/model_comparison_summary.csv",
        f"{OUT}/analysis/completion_benchmark.csv",
        f"{OUT}/analysis/surprisal_benchmark.csv",
        f"{OUT}/analysis/pass_rate.png",
        f"{OUT}/analysis/metrics_plots.png",
        f"{OUT}/analysis/report.html"

# ... (existing rules) ...

# --- 5. Reporting ---
rule report:
    input:
        summary = f"{OUT}/analysis/model_comparison_summary.csv",
        completion = f"{OUT}/analysis/completion_benchmark.csv",
        surprisal = f"{OUT}/analysis/surprisal_benchmark.csv",
        pass_plot = f"{OUT}/analysis/pass_rate.png",
        metrics_plot = f"{OUT}/analysis/metrics_plots.png"
    output:
        html = f"{OUT}/analysis/report.html"
    benchmark:
        f"{OUT}/benchmarks/report_rule.tsv"
    script:
        "src/scripts/generate_report.py"

# --- 1. Generation ---
rule generate_vllm:
    output:
        outdir = directory(f"{OUT}/generations/{{model}}"),
        csv = f"{OUT}/generations/{{model}}/outputs.csv"
    params:
        model_path = lambda w: config["models"][w.model],
        prompts = config["generation"]["prompts"],
        samples = config["generation"]["samples_per_prompt"],
        gpu_util = config["generation"]["gpu_utilization"],
        temp = config["generation"]["temperature"]
    resources:
        gpu = 1
    shell:
        """
        python src/scripts/generate_vllm.py \
            --model_path {params.model_path} \
            --output_dir {output.outdir} \
            --gpu_util {params.gpu_util} \
            --temperature {params.temp} \
            --prompts \"{params.prompts}\" \
            --samples {params.samples}
        """

# --- 2. QC (EXACT Implementation) ---
rule make_blast_db:
    input:
        config["oridb_ref"]
    output:
        multiext(config["oridb_prefix"], ".nhr", ".nin", ".nsq")
    shell:
        "makeblastdb -in {input} -dbtype nucl -out {config[oridb_prefix]}"

rule run_qc:
    input:
        gen_dir = f"{OUT}/generations/{{model}}",
        ref = config["oridb_ref"],
        db = multiext(config["oridb_prefix"], ".nhr", ".nin", ".nsq")
    output:
        qc_dir = directory(f"{OUT}/qc/{{model}}"),
        pass_csv = f"{OUT}/qc/{{model}}/passed.csv"
    params:
        prefix = config["oridb_prefix"],
        threads = config["qc"]["threads"],
        # Filter Params
        ori_strict = config["qc"]["ori_strict_id"],
        amr_strict = config["qc"]["amr_strict_id"],
        # We pass the default "low" thresholds from your script implicitly, 
        # or you can override them here if you want to change the "Stage A" logic.
    shell:
        """
        # 1. Run the BLAST/AMR/Prodigal pipeline
        # Note: qc_oriv_arg2.py runs AMRFinder in nucleotide mode (-n)
        python src/qc/qc_oriv_arg2.py \
            --in {input.gen_dir} \
            --outdir {output.qc_dir} \
            --oridb_prefix {params.prefix} \
            --oridb_ref {input.ref} \
            --threads {params.threads}
        
        # 2. Run Repeats Finder
        python src/qc/repeats2.py {input.gen_dir} \
            --circular \
            --out {output.qc_dir}/repeats.csv

        # 3. Run Two-Stage Filtering
        # This takes the outputs from 1 & 2 and produces the final passed.csv
        python src/qc/filter_qc_two_stage2.py \
            --qc_out {output.qc_dir} \
            --repeats_csv {output.qc_dir}/repeats.csv \
            --out_pass {output.pass_csv} \
            --out_fail {output.qc_dir}/failed.csv \
            --ori_strict_identity {params.ori_strict} \
            --amr_strict_identity {params.amr_strict} \
            --repeat_max_len 50 \
            --repeat_ge
        """

# --- 3. Benchmarking ---
rule benchmarks:
    input:
        real_data = config["real_plasmids_dir"]
    output:
        completion = f"{OUT}/analysis/completion_benchmark.csv",
        surprisal = f"{OUT}/analysis/surprisal_benchmark.csv"
    params:
        models = config["models"]
    resources:
        gpu = 1
    benchmark:
        f"{OUT}/benchmarks/benchmarks_rule.tsv"
    script:
        "src/scripts/benchmarks.py"

# --- 4. Analysis ---
rule analyze:
    input:
        qc_pass_csvs = expand(f"{OUT}/qc/{{model}}/passed.csv", model=MODELS),
        generations = expand(f"{OUT}/generations/{{model}}/outputs.csv", model=MODELS),
        real_data = config["real_plasmids_dir"],
        bench_comp = f"{OUT}/analysis/completion_benchmark.csv",
        bench_surp = f"{OUT}/analysis/surprisal_benchmark.csv"
    output:
        summary = f"{OUT}/analysis/model_comparison_summary.csv",
        plot = f"{OUT}/analysis/pass_rate.png",
        metrics_plot = f"{OUT}/analysis/metrics_plots.png"
    params:
        models = MODELS
    benchmark:
        f"{OUT}/benchmarks/analyze_rule.tsv"
    script:
        "src/scripts/analyze.py"