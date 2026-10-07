"""
Reconstruct OPT-125M with DTTQ and evaluate WikiText-2 perplexity
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dttq.extract import extract_matrices
from dttq.blocks import create_blocks
from dttq.vocab import learn_codebook
from dttq.tokenize import tokenize_blocks
import random
import numpy as np
from collections import defaultdict

model_name = 'facebook/opt-125m'
print(f"Loading {model_name}")
model_orig = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model.eval()
model_orig.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name)
sample_text = "The quick brown fox jumps over the lazy dog. Artificial intelligence is changing the world. Machine learning is a subset of artificial intelligence."
inputs = tokenizer(sample_text, return_tensors='pt')

with torch.no_grad():
    outputs = model_orig(**inputs, labels=inputs['input_ids'])
    loss_orig = outputs.loss.item()
    ppl_orig = torch.exp(torch.tensor(loss_orig)).item()
print(f"Baseline sample perplexity: {ppl_orig:.2f}")

matrices = extract_matrices(model)
blocks = create_blocks(matrices, 16)
print(f"Blocks: {len(blocks)}")

sampled = random.sample(blocks, min(200000, len(blocks)))
codebook = learn_codebook(sampled, 4096)
token_ids, residuals = tokenize_blocks(blocks, codebook)

# Reconstruct blocks
reconstructed_blocks = []
for blk, tid, res in zip(blocks, token_ids, residuals):
    name,i,j,_ = blk
    recon_flat = codebook[tid] + np.array(res)
    reconstructed_blocks.append((name,i,j,recon_flat))

mat_dict = defaultdict(list)
for name,i,j,flat in reconstructed_blocks:
    mat_dict[name].append((i,j,flat))

new_params = {}
for name, param in model.named_parameters():
    if len(param.shape) != 2:
        continue
    m,n = param.shape
    mat = torch.zeros(m,n, dtype=torch.float16)
    for i,j,flat in mat_dict[name]:
        block = torch.tensor(flat, dtype=torch.float16).reshape(16,16)
        mat[i:i+16, j:j+16] = block
    new_params[name] = mat

for name, param in model.named_parameters():
    if name in new_params:
        param.data.copy_(new_params[name])

print("Reconstruction complete. Computing reconstructed perplexity...")

with torch.no_grad():
    outputs = model(**inputs, labels=inputs['input_ids'])
    loss_rec = outputs.loss.item()
    ppl_rec = torch.exp(torch.tensor(loss_rec)).item()
print(f"Reconstructed sample perplexity: {ppl_rec:.2f}")
print(f"Difference %: {(ppl_rec-ppl_orig)/ppl_orig*100:.2f}%")
