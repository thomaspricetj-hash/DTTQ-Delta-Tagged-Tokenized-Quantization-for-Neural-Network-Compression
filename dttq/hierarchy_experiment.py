"""
Hierarchical Motif Family Chunking
"""
import torch
from transformers import AutoModelForCausalLM
from dttq.extract import extract_matrices
from dttq.blocks import create_blocks
from dttq.vocab import learn_codebook
from dttq.tokenize import tokenize_blocks
import random, numpy as np
from sklearn.cluster import MiniBatchKMeans
from collections import Counter
import csv, os

model_name = 'facebook/opt-125m'
print('Loading model')
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model.eval()

matrices = extract_matrices(model)
blocks = create_blocks(matrices, 16)
sampled = random.sample(blocks, min(200000, len(blocks)))
codebook = learn_codebook(sampled, 4096)
token_ids, residuals = tokenize_blocks(blocks, codebook)

data = np.stack([b[3] for b in blocks])
recon_full = np.stack([codebook[tid] + res for tid,res in zip(token_ids, residuals)])
mse_full = float(np.mean((data - recon_full)**2))
print(f'Full codebook MSE: {mse_full}')

# First level: motifs -> 64 families
kmeans_64 = MiniBatchKMeans(n_clusters=64, random_state=0, batch_size=256)
family_ids = kmeans_64.fit_predict(codebook)
family_centroids = kmeans_64.cluster_centers_
token_to_family = {i:int(family_ids[i]) for i in range(len(family_ids))}
family_tokens = [token_to_family[tid] for tid in token_ids]

# Helper to reconstruct using family mapping
def reconstruct_with_family_map(token_to_family, family_centroids):
    recon = []
    for tid, res in zip(token_ids, residuals):
        f = token_to_family[tid]
        recon.append(family_centroids[f] + res)
    return np.stack(recon)

results = []
# 64 families baseline
entropy_64 = -np.sum(np.array(list(Counter(family_tokens).values()))/len(family_tokens) * np.log2(np.array(list(Counter(family_tokens).values()))/len(family_tokens)))
recon_64 = reconstruct_with_family_map(token_to_family, family_centroids)
mse_64 = float(np.mean((data - recon_64)**2))
# cosine similarity between motifs and their family centroids
cos_sims = []
for tid in range(len(codebook)):
    f = token_to_family[tid]
    a = codebook[tid]
    b = family_centroids[f]
    cos = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)
    cos_sims.append(cos)
cos_sim_64 = float(np.mean(cos_sims))
results.append([64, entropy_64, mse_64, cos_sim_64])
print(f'64 families: entropy {entropy_64:.3f}, MSE {mse_64:.6f}, cos_sim {cos_sim_64:.3f}')

levels = [32,16,8,4,2]
for k in levels:
    kmeans_super = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=256)
    super_ids_map = kmeans_super.fit_predict(family_centroids)
    super_centroids = kmeans_super.cluster_centers_
    # map family -> super family centroid
    family_to_super_centroid = {}
    for f_idx in range(64):
        sf = super_ids_map[f_idx]
        family_to_super_centroid[f_idx] = super_centroids[sf]
    # reconstruct using super family centroids
    recon = []
    for tid, res in zip(token_ids, residuals):
        f = token_to_family[tid]
        recon.append(family_to_super_centroid[f] + res)
    recon = np.stack(recon)
    mse = float(np.mean((data - recon)**2))
    # entropy of super family stream
    super_family_tokens = [int(super_ids_map[token_to_family[tid]]) for tid in token_ids]
    cnt = Counter(super_family_tokens)
    total = len(super_family_tokens)
    probs = np.array(list(cnt.values()))/total
    entropy = -np.sum(probs*np.log2(probs+1e-12))
    # cosine similarity between family centroids and their super centroids
    cos_sims = []
    for f_idx in range(64):
        a = family_centroids[f_idx]
        b = super_centroids[super_ids_map[f_idx]]
        cos = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)
        cos_sims.append(cos)
    cos_sim = float(np.mean(cos_sims))
    results.append([k, entropy, mse, cos_sim])
    print(f'Level {k}: entropy {entropy:.3f}, MSE {mse:.6f}, cos_sim {cos_sim:.3f}')

os.makedirs('hierarchy_results', exist_ok=True)
with open('hierarchy_results/hierarchy_results.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Families','Entropy','MSE','CosSim'])
    writer.writerows(results)

print('Saved hierarchy_results/hierarchy_results.csv')
