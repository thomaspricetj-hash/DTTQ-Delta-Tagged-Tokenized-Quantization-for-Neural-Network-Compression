import numpy as np
from typing import List

def compute_stats(matrices, token_ids, residuals, tags, blocks):
    num_blocks = len(token_ids)
    block_size = 256
    original_bytes = num_blocks * block_size * 2
    residual_bytes = 0
    mse_sum = 0.0
    for tag, r in zip(tags, residuals):
        if tag == 0:
            residual_bytes += 0
        elif tag == 1:
            residual_bytes += block_size * 0.5
        elif tag == 2:
            residual_bytes += block_size * 1
        else:
            residual_bytes += block_size * 2
        mse_sum += np.mean(r**2)
    compressed_bytes = num_blocks * 2 + num_blocks * 1 + residual_bytes
    ratio = original_bytes / max(compressed_bytes, 1)
    mse = mse_sum / num_blocks if num_blocks else 0.0
    return {'ratio': ratio, 'mse': mse, 'original_bytes': original_bytes, 'compressed_bytes': compressed_bytes}
