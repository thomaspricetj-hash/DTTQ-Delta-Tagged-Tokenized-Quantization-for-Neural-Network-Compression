import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import List, Tuple

def tokenize_blocks(blocks: List[Tuple], codebook: np.ndarray):
    data = np.stack([b[3] for b in blocks])
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
    nn.fit(codebook)
    distances, indices = nn.kneighbors(data)
    token_ids = indices.flatten().tolist()
    residuals = []
    for flat, tid in zip(data, token_ids):
        token_vec = codebook[tid]
        residual = flat - token_vec
        residuals.append(residual)
    return token_ids, residuals
