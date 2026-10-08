"""
BPE and FamilyGPT-like PPL for KV Cache
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

# BPE
def bpe_len(seq, merges):
    seq_list = list(seq)
    for _ in range(merges):
        pairs = Counter(zip(seq_list, seq_list[1:]))
        if not pairs: break
        pair,_ = pairs.most_common(1)[0]
        new_token = max(seq_list)+1
        new_seq = []
        i=0
        while i < len(seq_list):
            if i < len(seq_list)-1 and seq_list[i]==pair[0] and seq_list[i+1]==pair[1]:
                new_seq.append(new_token); i+=2
            else:
                new_seq.append(seq_list[i]); i+=1
        seq_list = new_seq
    return len(seq_list)

orig_len = len(family_stream)
merged_len = bpe_len(family_stream, 100)
reduction = (1 - merged_len/orig_len)*100
print(f'BPE 100 merges on family stream: orig {orig_len}, merged {merged_len}, reduction {reduction:.2f}%')

# Simple PPL estimate via entropy
def entropy(counter, total):
    probs = np.array(list(counter.values()))/total
    return -np.sum(probs*np.log2(probs+1e-12))

fam_cnt = Counter(family_stream)
fam_ent = entropy(fam_cnt, len(family_stream))
print(f'Family entropy: {fam_ent:.3f} bits/token')
print(f'Estimated PPL ~ 2^{fam_ent:.3f} = {2**fam_ent:.2f}')
