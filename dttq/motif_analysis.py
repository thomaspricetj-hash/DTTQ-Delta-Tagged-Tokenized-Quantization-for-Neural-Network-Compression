"""
motif_analysis.py
Top 100 motifs analysis with stats and PCA visualization
"""
import argparse, torch
from transformers import AutoModelForCausalLM
from dttq.extract import extract_matrices
from dttq.blocks import create_blocks
from dttq.vocab import learn_codebook
from dttq.tokenize import tokenize_blocks
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def motif_stats(codebook, token_ids, top_ids):
    stats = []
    for rank, tid in enumerate(top_ids, 1):
        motif = codebook[tid]
        freq = sum(1 for t in token_ids if t == tid)
        mean = float(np.mean(motif))
        var = float(np.var(motif))
        sparsity = float(np.mean(np.abs(motif) < 1e-4))
        # Low-rank score: ratio of top singular value to sum
        s = np.linalg.svd(motif.reshape(16,16), compute_uv=False)
        low_rank_score = float(s[0] / (np.sum(s) + 1e-12))
        stats.append({
            'rank': rank,
            'token_id': int(tid),
            'frequency': int(freq),
            'mean': mean,
            'variance': var,
            'sparsity': sparsity,
            'low_rank_score': low_rank_score,
            'motif': motif
        })
    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='facebook/opt-125m')
    args = parser.parse_args()
    
    print(f"Loading {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    matrices = extract_matrices(model)
    blocks = create_blocks(matrices, 16)
    import random
    sampled = random.sample(blocks, min(200000, len(blocks)))
    codebook = learn_codebook(sampled, 4096)
    token_ids, _ = tokenize_blocks(blocks, codebook)
    
    counter = Counter(token_ids)
    top_ids = [tid for tid,_ in counter.most_common(100)]
    stats = motif_stats(codebook, token_ids, top_ids)
    
    # Print table
    print("\nTop 100 motifs")
    print("Rank TokenID Freq Mean Var Sparsity LowRank")
    for s in stats:
        print(f"{s['rank']:3d} {s['token_id']:5d} {s['frequency']:7d} {s['mean']:7.4f} {s['variance']:7.4f} {s['sparsity']:7.3f} {s['low_rank_score']:7.3f}")
    
    # PCA embedding of top motifs
    motifs_arr = np.stack([s['motif'] for s in stats])
    pca = PCA(n_components=2)
    emb = pca.fit_transform(motifs_arr)
    
    plt.figure(figsize=(8,6))
    plt.scatter(emb[:,0], emb[:,1], s=30)
    for i,s in enumerate(stats[:20]):
        plt.annotate(f"{s['token_id']}", (emb[i,0], emb[i,1]))
    plt.title("PCA of Top 100 Motifs")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig("motif_pca.png")
    print("\nSaved motif_pca.png")
    
    # Save stats
    import json
    out = []
    for s in stats:
        out.append({k:v for k,v in s.items() if k!='motif'})
    with open("motif_stats.json","w") as f:
        json.dump(out, f)
    print("Saved motif_stats.json")

if __name__ == "__main__":
    main()
