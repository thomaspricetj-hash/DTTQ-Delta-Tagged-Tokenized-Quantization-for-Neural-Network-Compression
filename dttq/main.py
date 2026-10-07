import argparse
import torch
from transformers import AutoModelForCausalLM
from dttq.extract import extract_matrices
from dttq.blocks import create_blocks
from dttq.vocab import learn_codebook
from dttq.tokenize import tokenize_blocks
from dttq.delta import tag_residuals
from dttq.compress import compute_stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='facebook/opt-125m')
    parser.add_argument('--block_size', type=int, default=16)
    parser.add_argument('--vocab_size', type=int, default=4096)
    parser.add_argument('--sample_blocks', type=int, default=200000)
    args = parser.parse_args()

    print(f"Loading model {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    model.eval()

    matrices = extract_matrices(model)
    print(f"Extracted {len(matrices)} matrices")

    blocks = create_blocks(matrices, args.block_size)
    print(f"Created {len(blocks)} blocks")

    import random
    sampled = random.sample(blocks, min(args.sample_blocks, len(blocks)))
    codebook = learn_codebook(sampled, args.vocab_size)
    print(f"Codebook shape: {codebook.shape}")

    token_ids, residuals = tokenize_blocks(blocks, codebook)
    tags = tag_residuals(residuals)

    stats = compute_stats(matrices, token_ids, residuals, tags, blocks)
    print(f"Compression ratio: {stats['ratio']:.2f}x")
    print(f"Reconstruction MSE: {stats['mse']:.6f}")
    print(f"Original bytes: {stats['original_bytes']}")
    print(f"Compressed bytes: {stats['compressed_bytes']}")

if __name__ == '__main__':
    main()
