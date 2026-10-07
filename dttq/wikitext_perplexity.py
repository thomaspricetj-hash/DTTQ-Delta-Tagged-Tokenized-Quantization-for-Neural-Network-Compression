"""
WikiText-2 perplexity for baseline vs DTTQ reconstructed OPT-125M
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from dttq.extract import extract_matrices
from dttq.blocks import create_blocks
from dttq.vocab import learn_codebook
from dttq.tokenize import tokenize_blocks
import random, numpy as np
from collections import defaultdict

model_name = 'facebook/opt-125m'
print('Loading model')
model_orig = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model_orig.eval()
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Reconstruct
matrices = extract_matrices(model)
blocks = create_blocks(matrices, 16)
sampled = random.sample(blocks, min(200000, len(blocks)))
codebook = learn_codebook(sampled, 4096)
token_ids, residuals = tokenize_blocks(blocks, codebook)

reconstructed_blocks = []
for blk, tid, res in zip(blocks, token_ids, residuals):
    name,i,j,_ = blk
    recon_flat = codebook[tid] + np.array(res)
    reconstructed_blocks.append((name,i,j,recon_flat))

mat_dict = defaultdict(list)
for name,i,j,flat in reconstructed_blocks:
    mat_dict[name].append((i,j,flat))

for name, param in model.named_parameters():
    if len(param.shape)!=2: continue
    m,n = param.shape
    mat = torch.zeros(m,n, dtype=torch.float16)
    for i,j,flat in mat_dict[name]:
        block = torch.tensor(flat, dtype=torch.float16).reshape(16,16)
        mat[i:i+16, j:j+16] = block
    param.data.copy_(mat)

print('Reconstruction done')

# Load wikitext-2 validation split, first 10 examples for speed
ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
texts = []
for ex in ds:
    txt = ex['text'].strip()
    if txt:
        texts.append(txt)
    if len(texts)>=10:
        break

def compute_ppl(model, texts):
    total_loss = 0.0
    total_tokens = 0
    for txt in texts:
        enc = tokenizer(txt, return_tensors='pt', truncation=True, max_length=512)
        input_ids = enc['input_ids']
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
            n_tokens = input_ids.numel()
            total_loss += loss * n_tokens
            total_tokens += n_tokens
    avg_loss = total_loss/total_tokens
    return torch.exp(torch.tensor(avg_loss)).item()

print('Baseline perplexity...')
ppl_orig = compute_ppl(model_orig, texts)
print('Reconstructed perplexity...')
ppl_rec = compute_ppl(model, texts)
print(f'Baseline PPL: {ppl_orig:.2f}')
print(f'Reconstructed PPL: {ppl_rec:.2f}')
print(f'Difference %: {(ppl_rec-ppl_orig)/ppl_orig*100:.2f}%')
