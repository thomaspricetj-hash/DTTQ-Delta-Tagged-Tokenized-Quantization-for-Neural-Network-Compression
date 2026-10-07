"""
Design Upgrade #1: Motif Dictionary Model
Top 100 motifs cached, every block = Motif ID + Residual ID
"""
import numpy as np

class MotifDictionaryModel:
    def __init__(self, top_k=100):
        self.top_k = top_k
        self.motifs = None
        self.residual_codebook = None
        
    def fit(self, codebook, token_ids, blocks):
        # Select top K most frequent motifs
        from collections import Counter
        cnt = Counter(token_ids)
        top_ids = [tid for tid,_ in cnt.most_common(self.top_k)]
        self.motifs = codebook[top_ids]
        # Remaining blocks use residual coding
        # For simplicity, store mapping token_id -> motif index or residual
        self.token_map = {tid:i for i,tid in enumerate(top_ids)}
        return self.motifs
