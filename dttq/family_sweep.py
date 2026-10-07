"""
Family sweep 4096 -> 256 -> 64 -> 16 -> 8
"""
import torch
from transformers import AutoModelForCausalLM
from dttq.extract import extract_matrices
from dttq.blocks import create_blocks
from dttq.vocab import learn_codebook
from dttq.tokenize import tokenize_blocks
import random
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from collections import Counter

model_name = 'facebook/opt-125m'
print(f"Loading {model_name}")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model.eval()

matrices = extract_matrices(model)
blocks = create_blocks(matrices, 16)
print(f"Blocks: {len(blocks)}")

sampled = random.sample(blocks, min(200000, len(blocks)))
codebook = learn_codebook(sampled, 4096)
token_ids, residuals = tokenize_blocks(blocks, codebook)

data = np.stack([b[3] for b in blocks])
reconstructed = np.stack([codebook[tid] + res for tid,res in zip(token_ids, residuals)])
mse_full = float(np.mean((data - reconstructed)**2))
print(f"Full codebook MSE: {mse_full:.6f}")

family_sizes = [256,64,16,8]
results = []
for k in family_sizes:
    print(f"\nClustering into {k} families...")
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=256)
    family_ids = kmeans.fit_predict(codebook)
    token_to_family = {i:int(family_ids[i]) for i in range(len(family_ids))}
    family_tokens = [token_to_family[tid] for tid in token_ids]
    # Reconstruct using family centroids
    family_centroids = kmeans.cluster_centers_
    recon_family = np.stack([family_centroids[token_to_family[tid]] + res for tid,res in zip(token_ids, residuals)])
    mse = float(np.mean((data - recon_family)**2))
    # Entropy
    counter = Counter(family_tokens)
    total = len(family_tokens)
    probs = np.array(list(counter.values()))/total
    entropy = -np.sum(probs*np.log2(probs+1e-12))
    print(f"Families {k}: MSE {mse:.6f}, entropy {entropy:.3f} bits/token")
    results.append((k, entropy, mse))

print("\nSweep results:")
for k, e, m in results:
    print(f"{k}\t{e:.3f}\t{m:.6f}")
