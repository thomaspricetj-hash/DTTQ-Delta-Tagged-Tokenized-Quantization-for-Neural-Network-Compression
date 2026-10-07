import torch
from transformers import AutoModelForCausalLM
from dttq.extract import extract_matrices
from dttq.blocks import create_blocks
from dttq.vocab import learn_codebook
from dttq.tokenize import tokenize_blocks
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
import random
sampled = random.sample(blocks, min(sample_blocks, len(blocks)))
codebook = learn_codebook(sampled, vocab_size)
token_ids, _ = tokenize_blocks(blocks, codebook)

seq = np.array(token_ids)
# Mutual information between consecutive tokens
# Compute joint and marginal distributions
pairs = list(zip(seq[:-1], seq[1:]))
pair_counter = Counter(pairs)
single_counter = Counter(seq)

total = len(pairs)
# Entropies
def entropy(counter, total):
    probs = np.array(list(counter.values()))/total
    return -np.sum(probs*np.log2(probs+1e-12))

H_X = entropy(single_counter, len(seq))
H_Y = entropy(single_counter, len(seq))  # same marginal approx
# Joint entropy
H_XY = entropy(pair_counter, total)

I = H_X + H_Y - H_XY
print(f"H(token) = {H_X:.4f} bits")
print(f"H(token, next) = {H_XY:.4f} bits")
print(f"Mutual information I(token; next) = {I:.4f} bits")

# Conditional entropy
H_Y_given_X = H_XY - H_X
print(f"H(next|current) = {H_Y_given_X:.4f} bits")
print(f"Reduction vs unconditional: {(H_Y - H_Y_given_X)/H_Y*100:.2f}%")

# Top transitions
print("Top 10 transitions:")
for (a,b),c in pair_counter.most_common(10):
    print(f"  {a} -> {b}: {c}")
