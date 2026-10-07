"""
Steps 3 and 4: Chunking and BPE on family stream
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
family_stream = [token_to_family[tid] for tid in token_ids]

# Step 3: Chunking
def chunk_stats(stream, chunk_size):
    chunks = [tuple(stream[i:i+chunk_size]) for i in range(0, len(stream)-chunk_size+1, chunk_size)]
    cnt = Counter(chunks)
    total = len(chunks)
    probs = np.array(list(cnt.values()))/total
    entropy = -np.sum(probs*np.log2(probs+1e-12))
    coverage = sum(v for k,v in cnt.most_common(100))/total
    return entropy, coverage, len(cnt)

os.makedirs('hierarchy_results', exist_ok=True)
with open('hierarchy_results/family_chunk_stats.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['chunk_size','entropy','coverage','unique_chunks'])
    for cs in [2,3,4]:
        e,c,u = chunk_stats(family_stream, cs)
        writer.writerow([cs,e,c,u])
        print(f'Chunk size {cs}: entropy {e:.3f}, coverage top100 {c:.3f}, unique {u}')

# Step 4: BPE on family stream
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

with open('hierarchy_results/family_bpe.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['merges','orig_len','merged_len','reduction_pct'])
    writer.writerow([100,orig_len,merged_len,reduction])
