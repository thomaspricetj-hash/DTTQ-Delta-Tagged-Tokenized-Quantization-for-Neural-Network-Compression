"""
Quick demo using tiny model for fast iteration
"""
import torch
from transformers import AutoModelForCausalLM
from dttq.extract import extract_matrices
from dttq.blocks import create_blocks
from dttq.vocab import learn_codebook
from dttq.tokenize import tokenize_blocks
from dttq.delta import tag_residuals
from dttq.compress import compute_stats
import random

model_name = 'distilgpt2'
print(f"Loading {model_name}")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model.eval()

matrices = extract_matrices(model)
blocks = create_blocks(matrices, block_size=16)
print(f"Blocks: {len(blocks)}")

sampled = random.sample(blocks, min(50000, len(blocks)))
codebook = learn_codebook(sampled, vocab_size=1024)
token_ids, residuals = tokenize_blocks(blocks, codebook)
tags = tag_residuals(residuals)
stats = compute_stats(matrices, token_ids, residuals, tags, blocks)
print(stats)
