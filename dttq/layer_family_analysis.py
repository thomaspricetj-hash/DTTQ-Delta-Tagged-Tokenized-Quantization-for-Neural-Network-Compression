"""
Layer Family Analysis
"""
import torch
from transformers import AutoModelForCausalLM
from dttq.extract import extract_matrices
from dttq.blocks import create_blocks
from dttq.vocab import learn_codebook
from dttq.tokenize import tokenize_blocks
import random, numpy as np
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict, Counter
import csv

model_name = 'facebook/opt-125m'
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model.eval()
matrices = extract_matrices(model)
blocks = create_blocks(matrices, 16)

sampled = random.sample(blocks, min(200000, len(blocks)))
codebook = learn_codebook(sampled, 4096)
token_ids, residuals = tokenize_blocks(blocks, codebook)

kmeans_64 = MiniBatchKMeans(n_clusters=64, random_state=0, batch_size=256)
family_ids = kmeans_64.fit_predict(codebook)
token_to_family = {i:int(family_ids[i]) for i in range(len(family_ids))}

# Map block to layer type
layer_counts = defaultdict(lambda: Counter())
for blk, tid in zip(blocks, token_ids):
    name, i, j, flat = blk
    fam = token_to_family[tid]
    # classify layer
    if 'embed' in name.lower():
        layer = 'Embeddings'
    elif 'attn' in name.lower() or 'attention' in name.lower():
        if 'q' in name.lower():
            layer = 'Attention Q'
        elif 'k' in name.lower():
            layer = 'Attention K'
        elif 'v' in name.lower():
            layer = 'Attention V'
        else:
            layer = 'Attention'
    elif 'mlp' in name.lower() or 'fc' in name.lower():
        layer = 'MLP'
    else:
        layer = 'Other'
    layer_counts[layer][fam] += 1

# Save distribution
with open('hierarchy_results/layer_family_distribution.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Layer','Family','Count'])
    for layer, cnt in layer_counts.items():
        for fam, c in cnt.most_common(20):
            writer.writerow([layer, fam, c])

print('Layer family distribution saved')
# Summary
print('Top families per layer:')
for layer, cnt in layer_counts.items():
    top = cnt.most_common(3)
    print(layer, top)
