#!/usr/bin/env python3
import torch.utils._pytree as _pytree
if not hasattr(_pytree, "register_pytree_node") and hasattr(_pytree, "_register_pytree_node"):
    def _register_pytree_node_wrapper(cls, flatten_fn, unflatten_fn, serialized_type_name=None):
        return _pytree._register_pytree_node(cls, flatten_fn, unflatten_fn)
    _pytree.register_pytree_node = _register_pytree_node_wrapper

from plasmid_analytics.generate.sampler import sample_sequences, load_model, write_fasta
import pandas as pd

# Load model (device auto-detects CUDA if available)
model, tokenizer = load_model("UCL-CSSB/PlasmidGPT")

# Generate sequences (matching vLLM script parameters)
sequences = sample_sequences(
    model=model,
    tokenizer=tokenizer,
    prompt="ATG",
    num_samples=50,
    max_new_tokens=256,
    temperature=0.95,
    top_p=0.90,
    top_k=0,
    repetition_penalty=1.0,
    batch_size=32,
)

# Write FASTA
write_fasta(sequences, "sequences.fasta", prefix="UCL_CSSB_PlasmidGPT_ATG")

# Write CSV with metadata
records = []
for i, seq in enumerate(sequences):
    records.append({
        "model": "UCL_CSSB_PlasmidGPT",
        "prompt": "ATG",
        "sequence_id": f"UCL_CSSB_PlasmidGPT_ATG_{i:04d}",
        "sequence": seq,
        "length": len(seq),
    })
pd.DataFrame(records).to_csv("sequences.csv", index=False)
