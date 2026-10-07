import torch
import numpy as np
from typing import List, Tuple

def create_blocks(matrices: List[Tuple[str, torch.Tensor]], block_size: int):
    blocks = []
    for name, mat in matrices:
        m, n = mat.shape
        for i in range(0, m, block_size):
            for j in range(0, n, block_size):
                block = mat[i:i+block_size, j:j+block_size]
                if block.shape == (block_size, block_size):
                    flat = block.reshape(-1).numpy()
                    blocks.append((name, i, j, flat))
    return blocks
