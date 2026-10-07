import numpy as np
from typing import List

def compute_residuals(blocks, token_ids, codebook):
    # Already computed in tokenize, placeholder
    return []

def tag_residuals(residuals: List[np.ndarray]):
    tags = []
    for r in residuals:
        max_abs = float(np.max(np.abs(r)))
        if max_abs < 0.01:
            tag = 0
        elif max_abs < 0.125:
            tag = 1
        elif max_abs < 1.0:
            tag = 2
        else:
            tag = 3
        tags.append(tag)
    return tags
