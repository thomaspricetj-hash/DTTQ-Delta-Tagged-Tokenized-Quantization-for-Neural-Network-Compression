import torch
from transformers import AutoModelForCausalLM
from dttq.extract import extract_matrices
from dttq.blocks import create_blocks
from dttq.vocab import learn_codebook
from dttq.tokenize import tokenize_blocks
from dttq.delta import tag_residuals
import numpy as np
from collections import Counter
import math

model_name = 'facebook/opt-125m'
block_size = 16
vocab_size = 4096
sample_blocks = 200000

print(f"Loading {model_name}")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model.eval()

matrices = extract_matrices(model)
blocks = create_blocks(matrices, block_size)
print(f"Blocks: {len(blocks)}")

import random
sampled = random.sample(blocks, min(sample_blocks, len(blocks)))
codebook = learn_codebook(sampled, vocab_size)

token_ids, residuals = tokenize_blocks(blocks, codebook)
tags = tag_residuals(residuals)

# Token entropy
counter = Counter(token_ids)
total = len(token_ids)
probs = np.array(list(counter.values())) / total
entropy = -np.sum(probs * np.log2(probs))
print(f"Token entropy: {entropy:.3f} bits/token")
print(f"Vocabulary used: {len(counter)}/{vocab_size}")

# Top tokens
top10 = counter.most_common(10)
print("Top 10 tokens:")
for tid, cnt in top10:
    pct = cnt/total*100
    print(f"  Token {tid} = {pct:.2f}%")

# Coverage
top100_count = sum(cnt for _,cnt in counter.most_common(100))
coverage100 = top100_count/total*100
print(f"Top 100 tokens cover {coverage100:.2f}% of blocks")

# Residual entropy
# Flatten residuals
all_res = np.concatenate([r.flatten() for r in residuals])
# Quantize residuals to estimate entropy
# Use histogram
hist, _ = np.histogram(all_res, bins=256)
hist = hist[hist>0]
p = hist / hist.sum()
res_entropy = -np.sum(p * np.log2(p))
print(f"Residual entropy: {res_entropy:.3f} bits/value")

# Compression stats
num_blocks = len(token_ids)
block_size_vals = 256
original_bytes = num_blocks * block_size_vals * 2
residual_bytes = 0
for tag, r in zip(tags, residuals):
    if tag == 0:
        residual_bytes += 0
    elif tag == 1:
        residual_bytes += block_size_vals * 0.5
    elif tag == 2:
        residual_bytes += block_size_vals * 1
    else:
        residual_bytes += block_size_vals * 2
compressed_bytes = num_blocks * 2 + num_blocks * 1 + residual_bytes
ratio = original_bytes / compressed_bytes
print(f"Compression ratio: {ratio:.2f}x")
