import pandas as pd
import base64
import os
import glob
from datetime import datetime

def img_to_base64(img_path):
    if not os.path.exists(img_path):
        return ""
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def main(sm):
    analysis_dir = os.path.dirname(sm.input.summary)
    
    # Load Data
    summary_df = pd.read_csv(sm.input.summary)
    
    # Images
    images = {}
    for i in range(1, 12):
        # Find files matching fig{i}_*.png
        matches = glob.glob(os.path.join(analysis_dir, f"fig{i}_*.png"))
        if matches:
            images[f"fig{i}"] = img_to_base64(matches[0])
        else:
            images[f"fig{i}"] = ""

    # Config
    gen_config = sm.params.gen_config
    config_html = "<ul>"
    for k, v in gen_config.items():
        if k == "prompts": continue # Skip raw prompts if too long, or summarize
        config_html += f"<li><strong>{k}:</strong> {v}</li>"
    config_html += "</ul>"

    # HTML Template
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PlasmidGPT Analysis Report</title>
        <style>
            body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; margin: 40px; background-color: #f4f4f9; }}
            h1, h2 {{ color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }}
            .container {{ background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #2c3e50; color: white; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .img-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
            .img-box {{ text-align: center; border: 1px solid #ddd; padding: 10px; background: white; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>PlasmidGPT Experiment Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <div class="container">
            <h2>Experiment Configuration</h2>
            {config_html}
        </div>

        <div class="container">
            <h2>1. Pass Rates & Diversity</h2>
            {summary_df.to_html(index=False, classes='table')}
            <div class="img-grid">
                <div class="img-box">
                    <h3>Pass Rate</h3>
                    <img src="data:image/png;base64,{images['fig1']}" />
                </div>
                <div class="img-box">
                    <h3>Diversity (Mash)</h3>
                    <p>Higher is better (more diverse sequences).</p>
                    <img src="data:image/png;base64,{images['fig10']}" />
                </div>
            </div>
        </div>

        <div class="container">
            <h2>2. Similarity to Reference</h2>
            <div class="img-box">
                <img src="data:image/png;base64,{images['fig11']}" />
                <p>Similarity classification against reference plasmid set (subset of 10 sequences/model).</p>
            </div>
        </div>

        <div class="container">
            <h2>3. Biological Metrics</h2>
            <div class="img-grid">
                <div class="img-box"><h3>Length</h3><img src="data:image/png;base64,{images['fig2']}" /></div>
                <div class="img-box"><h3>GC Content</h3><img src="data:image/png;base64,{images['fig3']}" /></div>
                <div class="img-box"><h3>Longest ORF</h3><img src="data:image/png;base64,{images['fig4']}" /></div>
                <div class="img-box"><h3>Gene Count</h3><img src="data:image/png;base64,{images['fig5']}" /></div>
                <div class="img-box"><h3>3-mer Divergence (JS)</h3><img src="data:image/png;base64,{images['fig6']}" /></div>
                <div class="img-box"><h3>MFE Density</h3><img src="data:image/png;base64,{images['fig7']}" /></div>
            </div>
        </div>

        <div class="container">
            <h2>4. Benchmarking</h2>
            <div class="img-grid">
                <div class="img-box">
                    <h3>Completion Confidence</h3>
                    <p>Log-prob of next 100bp given 400bp prefix.</p>
                    <img src="data:image/png;base64,{images['fig8']}" />
                </div>
                <div class="img-box">
                    <h3>Surprisal Gap</h3>
                    <p>Model Confidence - Base Confidence on Promoter->CDS transitions.</p>
                    <img src="data:image/png;base64,{images['fig9']}" />
                </div>
            </div>
        </div>

    </body>
    </html>
    """

    with open(sm.output.html, "w") as f:
        f.write(html)

if __name__ == "__main__":
    main(snakemake)