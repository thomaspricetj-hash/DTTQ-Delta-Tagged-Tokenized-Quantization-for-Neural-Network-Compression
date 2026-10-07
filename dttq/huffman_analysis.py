import torch
from transformers import AutoModelForCausalLM
from dttq.extract import extract_matrices
from dttq.blocks import create_blocks
from dttq.vocab import learn_codebook
from dttq.tokenize import tokenize_blocks
from dttq.delta import tag_residuals
import numpy as np
from collections import Counter
import heapq

class Node:
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman(freq_dict):
    heap = [Node(sym, f) for sym, f in freq_dict.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)
        merged = Node(freq=n1.freq + n2.freq, left=n1, right=n2)
        heapq.heappush(heap, merged)
    return heap[0]

def get_codes(node, prefix='', codes=None):
    if codes is None:
        codes = {}
    if node.symbol is not None:
        codes[node.symbol] = prefix or '0'
    else:
        get_codes(node.left, prefix+'0', codes)
        get_codes(node.right, prefix+'1', codes)
    return codes

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

counter = Counter(token_ids)
total = len(token_ids)
probs = np.array(list(counter.values())) / total
entropy = -np.sum(probs * np.log2(probs))
print(f"Token entropy: {entropy:.3f} bits/token")

# Huffman
freq_dict = dict(counter)
root = build_huffman(freq_dict)
codes = get_codes(root)
avg_bits = sum(counter[sym]*len(codes[sym]) for sym in counter)/total
print(f"Huffman average bits/token: {avg_bits:.3f}")
print(f"Uniform 12-bit baseline: 12.000")
print(f"Savings vs uniform: {(12-avg_bits)/12*100:.1f}%")

# Top tokens
top10 = counter.most_common(10)
print("Top 10 tokens:")
for tid, cnt in top10:
    pct = cnt/total*100
    print(f"  Token {tid} = {pct:.2f}%")

top100_count = sum(cnt for _,cnt in counter.most_common(100))
coverage100 = top100_count/total*100
print(f"Top 100 tokens cover {coverage100:.2f}% of blocks")
