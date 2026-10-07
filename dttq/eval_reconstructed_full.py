"""
Reconstruct OPT-125M with DTTQ, save to temp dir, evaluate with lm-eval
"""
import torch
from transformers import AutoModelForCausalLM
from dttq.extract import extract_matrices
from dttq.blocks import create_blocks
from dttq.vocab import learn_codebook
from dttq.tokenize import tokenize_blocks
import random, numpy as np
from collections import defaultdict
import os, shutil
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

model_name = 'facebook/opt-125m'
print('Loading model for reconstruction')
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model.eval()

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

tmp_dir = 'dttq_reconstructed_opt125m'
if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir)
model.save_pretrained(tmp_dir)

print('Reconstruction saved, evaluating with lm-eval')
hflm = HFLM(pretrained=tmp_dir, device="cpu")
tasks = ["piqa","hellaswag","arc_easy","arc_challenge","winogrande"]
results = evaluator.simple_evaluate(
    model=hflm,
    tasks=tasks,
    limit=10,
    batch_size=1,
    log_samples=False
)

for task, metrics in results.get('results', {}).items():
    print(task, metrics)
