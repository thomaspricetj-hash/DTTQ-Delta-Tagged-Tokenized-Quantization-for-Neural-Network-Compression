"""
Design 1: Motif Dictionary Network
Store model as Dictionary + Token Assignments
"""
import numpy as np
from dttq.extract import extract_matrices
from dttq.blocks import create_blocks
from dttq.vocab import learn_codebook
from dttq.tokenize import tokenize_blocks

class MotifDictionaryNetwork:
    def __init__(self, vocab_size=4096, block_size=16):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.codebook = None
        self.assignments = {}  # name -> token_ids array

    def compress(self, model):
        matrices = extract_matrices(model)
        blocks = create_blocks(matrices, self.block_size)
        # Build codebook from sampled blocks
        import random
        sampled = random.sample(blocks, min(200000, len(blocks)))
        self.codebook = learn_codebook(sampled, self.vocab_size)
        # Tokenize all blocks
        token_ids, _ = tokenize_blocks(blocks, self.codebook)
        # Reconstruct assignments per matrix
        # Simplified: store flat token list
        self.assignments['tokens'] = np.array(token_ids)
        return {
            'codebook_shape': self.codebook.shape,
            'num_tokens': len(token_ids),
            'num_blocks': len(blocks)
        }

    def reconstruct_block(self, token_id):
        return self.codebook[token_id]
