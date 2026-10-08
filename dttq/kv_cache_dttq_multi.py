"""
DTTQ for KV Cache multi block sizes
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from collections import Counter

model_name = 'facebook/opt-125m'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model.eval()

def collect_kv(model, input_ids):
    past = None
    all_k = []
    all_v = []
    with torch.no_grad():
        for i in range(input_ids.shape[1]):
            out = model(input_ids[:, i:i+1], past_key_values=past, use_cache=True)
            past = out.past_key_values
            layer = past.layers[-1]
            k = layer.keys.cpu().numpy()
            v = layer.values.cpu().numpy()
            all_k.append(k)
            all_v.append(v)
    return np.concatenate(all_k, axis=2), np.concatenate(all_v, axis=2)

text = "Artificial intelligence is changing the world. Machine learning is a subset of artificial intelligence. Deep learning models are trained on large datasets. Transformers use self attention mechanisms. The future of AI depends on better algorithms and more data. " * 10
input_ids = tokenizer(text, return_tensors='pt')['input_ids']
keys, values = collect_kv(model, input_ids)
print('KV shapes:', keys.shape)

def block_matrix(mat, block_size):
    if mat.ndim == 4:
        mat = mat.reshape(-1, mat.shape[-2], mat.shape[-1])
    blocks = []
    for i in range(0, mat.shape[1], block_size):
        for j in range(0, mat.shape[2], block_size):
            b = mat[:, i:i+block_size, j:j+block_size]
            if b.shape[1]==block_size and b.shape[2]==block_size:
                blocks.append(b.reshape(b.shape[0], -1))
    if not blocks:
        return np.empty((0, block_size*block_size))
    return np.concatenate(blocks, axis=0)

def entropy(counter, total):
    probs = np.array(list(counter.values()))/total
    return -np.sum(probs*np.log2(probs+1e-12))

for block_size in [8,16,32]:
    key_blocks = block_matrix(keys, block_size)
    print(f'\nBlock size {block_size}: {key_blocks.shape[0]} blocks')
    n_clusters = 512
    sample = key_blocks[:min(20000, len(key_blocks))]
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=256)
    kmeans.fit(sample)
    codebook = kmeans.cluster_centers_
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(codebook)
    dist, idx = nn.kneighbors(key_blocks)
    token_ids = idx.flatten()
    cnt = Counter(token_ids)
    total = len(token_ids)
    ent = entropy(cnt, total)
    top100_cov = sum(c for _,c in cnt.most_common(100))/total
    kmeans_fam = MiniBatchKMeans(n_clusters=64, random_state=0, batch_size=256)
    fam_ids = kmeans_fam.fit_predict(codebook)
    token_to_family = {i:int(fam_ids[i]) for i in range(len(fam_ids))}
    family_stream = [token_to_family[tid] for tid in token_ids]
    fam_cnt = Counter(family_stream)
    fam_ent = entropy(fam_cnt, len(family_stream))
    print(f'Entropy: {ent:.3f} bits/token')
    print(f'Top100 coverage: {top100_cov*100:.2f}%')
    print(f'Family entropy: {fam_ent:.3f} bits/token')
