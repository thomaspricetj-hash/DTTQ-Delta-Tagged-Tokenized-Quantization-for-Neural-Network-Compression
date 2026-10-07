"""
motif_visualize.py
Visualize top 100 motifs as 16x16 heatmaps with layer distribution
"""
import argparse, torch
from transformers import AutoModelForCausalLM
from dttq.extract import extract_matrices
from dttq.blocks import create_blocks
from dttq.vocab import learn_codebook
from dttq.tokenize import tokenize_blocks
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import json

def layer_name_from_param(name):
    # Map parameter name to layer type
    if 'embed' in name.lower():
        return 'Embeddings'
    if 'attention' in name.lower() or 'attn' in name.lower():
        if 'q' in name.lower():
            return 'Attention Q'
        if 'k' in name.lower():
            return 'Attention K'
        if 'v' in name.lower():
            return 'Attention V'
        return 'Attention'
    if 'mlp' in name.lower() or 'fc' in name.lower() or 'intermediate' in name.lower():
        return 'MLP'
    return 'Other'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='facebook/opt-125m')
    args = parser.parse_args()
    
    print(f"Loading {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    matrices = extract_matrices(model)
    blocks = create_blocks(matrices, 16)
    # Need name per block
    # blocks is list of (name,i,j,flat)
    import random
    sampled = random.sample(blocks, min(200000, len(blocks)))
    codebook = learn_codebook(sampled, 4096)
    token_ids, _ = tokenize_blocks(blocks, codebook)
    
    # Map block index to name
    block_names = [b[0] for b in blocks]
    # Count frequency and layer distribution per motif
    counter = Counter(token_ids)
    top_ids = [tid for tid,_ in counter.most_common(100)]
    
    layer_dist = {tid: defaultdict(int) for tid in top_ids}
    for tid, name in zip(token_ids, block_names):
        if tid in layer_dist:
            layer = layer_name_from_param(name)
            layer_dist[tid][layer] += 1
    
    # Build stats
    stats = []
    for rank, tid in enumerate(top_ids, 1):
        motif = codebook[tid]
        freq = counter[tid]
        # Effective rank via SVD
        U,S,Vt = np.linalg.svd(motif.reshape(16,16), full_matrices=False)
        eff_rank = float(np.sum(S > 0.01*S[0]))
        # Layer distribution
        ld = layer_dist[tid]
        total = sum(ld.values())
        layer_pct = {k: v/total*100 for k,v in ld.items()}
        stats.append({
            'rank': rank,
            'token_id': int(tid),
            'frequency': int(freq),
            'effective_rank': eff_rank,
            'layer_distribution': layer_pct
        })
    
    # Save stats
    with open('motif_layer_stats.json','w') as f:
        json.dump(stats, f, indent=2)
    
    # Plot heatmaps grid 10x10
    fig, axes = plt.subplots(10,10, figsize=(20,20))
    for idx, tid in enumerate(top_ids):
        r, c = divmod(idx,10)
        ax = axes[r,c]
        motif = codebook[tid]
        im = ax.imshow(motif.reshape(16,16), cmap='coolwarm')
        ax.set_title(f"{tid}\n{counter[tid]}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('motif_heatmaps.png', dpi=150)
    print("Saved motif_heatmaps.png and motif_layer_stats.json")
    
    # Print layer summary for top 5
    print("\nTop 5 motifs layer distribution:")
    for s in stats[:5]:
        print(f"Rank {s['rank']} Token {s['token_id']} Freq {s['frequency']}")
        for k,v in s['layer_distribution'].items():
            print(f"  {k}: {v:.1f}%")
        print(f"  Effective rank: {s['effective_rank']:.2f}")

if __name__ == "__main__":
    main()
