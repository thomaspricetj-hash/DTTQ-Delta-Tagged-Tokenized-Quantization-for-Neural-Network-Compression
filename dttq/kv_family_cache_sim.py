"""
KV Family Cache Simulation
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from collections import Counter
import csv, os

model_name = 'facebook/opt-125m'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model.eval()

def collect_kv(model, input_ids):
    past = None
    all_k = []
    with torch.no_grad():
        for i in range(input_ids.shape[1]):
            out = model(input_ids[:, i:i+1], past_key_values=past, use_cache=True)
            past = out.past_key_values
            layer = past.layers[-1]
            k = layer.keys.cpu().numpy()
            all_k.append(k)
    return np.concatenate(all_k, axis=2)

text = "Artificial intelligence is changing the world. Machine learning is a subset of artificial intelligence. Deep learning models are trained on large datasets. Transformers use self attention mechanisms. The future of AI depends on better algorithms and more data. " * 10
input_ids = tokenizer(text, return_tensors='pt')['input_ids']
keys = collect_kv(model, input_ids)

def block_matrix(mat, block_size):
    if mat.ndim == 4:
        mat = mat.reshape(-1, mat.shape[-2], mat.shape[-1])
    blocks = []
    for i in range(0, mat.shape[1], block_size):
        for j in range(0, mat.shape[2], block_size):
            b = mat[:, i:i+block_size, j:j+block_size]
            if b.shape[1]==block_size and b.shape[2]==block_size:
                blocks.append(b.reshape(b.shape[0], -1))
    return np.concatenate(blocks, axis=0)

block_size = 16
key_blocks = block_matrix(keys, block_size)
n_clusters = 512
sample = key_blocks[:20000]
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=256)
kmeans.fit(sample)
codebook = kmeans.cluster_centers_
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=1)
nn.fit(codebook)
dist, idx = nn.kneighbors(key_blocks)
token_ids = idx.flatten()

kmeans_fam = MiniBatchKMeans(n_clusters=64, random_state=0, batch_size=256)
fam_ids = kmeans_fam.fit_predict(codebook)
token_to_family = {i:int(fam_ids[i]) for i in range(len(fam_ids))}
family_stream = [token_to_family[tid] for tid in token_ids]

# Step 1: frequency
cnt = Counter(family_stream)
total = len(family_stream)
os.makedirs('kv_cache_results', exist_ok=True)
with open('kv_cache_results/kv_family_frequency.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Family_ID','Count','Percentage'])
    for fam, c in cnt.most_common():
        writer.writerow([fam, c, c/total*100])

# Step 2: cumulative coverage
sorted_counts = [c for _,c in cnt.most_common()]
cum = np.cumsum(sorted_counts)
coverage = {n: cum[n-1]/total*100 for n in [1,5,10,25,50]}
print('Cumulative coverage:', coverage)

with open('kv_cache_results/kv_cache_simulation.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Top_N','Coverage_Pct'])
    for n in [1,5,10,25,50]:
        writer.writerow([n, coverage[n]])

# Step 3: cache simulation
for N in [1,5,10,25,50]:
    top_fams = set([fam for fam,_ in cnt.most_common(N)])
    hits = sum(1 for f in family_stream if f in top_fams)
    hit_rate = hits/total*100
    print(f'Top {N} families hit rate: {hit_rate:.2f}%')

# Step 5: chunk cache
def chunk_stats(stream, k):
    chunks = [tuple(stream[i:i+k]) for i in range(len(stream)-k+1)]
    c = Counter(chunks)
    total_c = len(chunks)
    top_cov = sum(v for _,v in c.most_common(10))/total_c*100
    return top_cov

for k in [2,3,4]:
    top10 = chunk_stats(family_stream, k)
    print(f'Chunk size {k} top10 coverage: {top10:.2f}%')

print('KV Family Cache Simulation complete')
