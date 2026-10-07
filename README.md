# DTTQ: Delta-Tagged Tokenized Quantization
 
## Reusable Motif Vocabularies in Transformer Weight Spaces
 
DTTQ (Delta-Tagged Tokenized Quantization) is an experimental neural network compression and weight-structure research framework that represents transformer weights as reusable motif tokens, residual tokens, and tags rather than independent floating-point values.
 
The project investigates a simple question:
 
> Do large transformer models contain reusable computational motifs that can be represented as a symbolic vocabulary?
 
Initial experiments on OPT-125M and GPT-2 suggest the answer may be yes.
 
---
 
## Core Idea
 
Traditional quantization treats weights as individual numbers:
 
```text
weight → FP16
weight → FP8
weight → FP4
weight → INT4
```
 
DTTQ instead treats model weights as recurring structures:
 
```text
Weight Matrix
↓
Block Extraction
↓
Motif Vocabulary
↓
Weight Tokens
↓
Residual Tokens
↓
Tags
```
 
Result:
 
```text
WeightToken + ResidualToken + Tag
```
 
instead of storing every weight independently.
 
---
 
## Motivation
 
Modern transformer models contain hundreds of millions or billions of parameters.
 
DTTQ explores whether these parameters are better described as:
 
```text
Motif Dictionary
+
Motif Placement
+
Residual Corrections
```
 
rather than:
 
```text
Billions of independent weights
```
 
The framework combines:
 
- Weight tokenization
- Learned motif dictionaries
- Residual coding
- Predictive motif coding
- BPE-style motif sequence compression
- Weight-language modeling (WeightGPT)
 
---
 
## Experimental Results
 
### OPT-125M
 
| Metric | Result |
|----------|----------|
| Blocks | 488,736 |
| Vocabulary Size | 4096 |
| Compression Ratio | 2.64× |
| Reconstruction MSE | 0.001654 |
| Token Entropy | 5.47 bits |
| Huffman Average | 5.49 bits |
| Top 100 Coverage | 96.34% |
| BPE Reduction | 38.28% |
| Mutual Information | 1.72 bits |
| WeightGPT Perplexity | 12.63 |
 
### GPT-2
 
| Metric | Result |
|----------|----------|
| Blocks | 485,616 |
| Token Entropy | 7.02 bits |
| Huffman Average | 7.05 bits |
| Top 100 Coverage | 89.56% |
| BPE Reduction | 28.83% |
 
---
 
## Key Findings
 
### Motif Reuse
 
A very small subset of motifs dominates model structure.
 
For OPT-125M:
 
```text
Top 100 motifs
=
96.34% of all blocks
```
 
### Information Compression
 
Expected entropy for a 4096-token vocabulary:
 
```text
12 bits/token
```
 
Observed:
 
```text
5.47 bits/token
```
 
indicating strong motif reuse.
 
### Predictable Sequences
 
Neighboring motifs contain predictive information:
 
```text
Mutual Information
=
1.72 bits
```
 
Knowing the current motif reduces uncertainty about the next motif by:
 
```text
31.4%
```
 
### Motif Language Structure
 
Applying only 100 BPE merges:
 
```text
38.28%
```
 
sequence-length reduction.
 
This suggests recurring motif phrases rather than isolated motif reuse.
 
### WeightGPT
 
A small transformer trained on motif streams achieved:
 
```text
Perplexity = 12.63
```
 
demonstrating that motif sequences are substantially predictable.
 
---
 
## Repository Structure
 
```text
dttq/
│
├── motif_analysis.py
├── motif_visualize.py
├── huffman_analysis.py
├── mutual_info.py
├── bpe_weight_tokens.py
│
├── upgrades/
│ ├── motif_dictionary_model.py
│ ├── predict_motifs.py
│ └── weight_gpt.py
│
├── advanced/
│ ├── residual_motifs.py
│ └── motif_transformer.py
│
├── outputs/
│ ├── motif_stats.json
│ ├── motif_layer_stats.json
│ ├── motif_pca.png
│ └── motif_heatmaps.png
│
└── paper/
```
 
---
 
## Design Upgrades
 
### 1. Motif Dictionary Model
 
Represent model blocks as:
 
```text
Motif ID
+
Residual ID
```
 
rather than raw weights.
 
---
 
### 2. Predictive Motif Coding
 
Use motif transition probabilities:
 
```text
P(next motif | current motif)
```
 
to encode prediction errors instead of full motif IDs.
 
---
 
### 3. WeightGPT
 
Train a language model directly on motif sequences:
 
```text
3066 3881 3561 ...
```
 
and encode prediction residuals.
 
---
 
### 4. Motif Transformers
 
Train neural networks directly using motif dictionaries instead of dense weight tensors.
 
---
 
### 5. Residual Languages
 
Residuals become:
 
```text
ResidualToken
+
Tag
```
 
making the entire model recursively tokenizable.
 
---
 
## Research Hypothesis
 
DTTQ investigates the hypothesis that:
 
> Transformer weight spaces contain reusable motif vocabularies with measurable entropy reduction, sequence predictability, and hierarchical structure.
 
The goal is not only improved compression but also a deeper understanding of how large neural networks organize their parameter space.
 
---
 
## Current Status
 
### Completed
 
- Weight tokenization
- Motif dictionary generation
- Entropy analysis
- Huffman coding analysis
- Mutual information analysis
- BPE motif compression
- Motif visualization
- Motif family clustering
- WeightGPT prototype
 
### In Progress
 
- Cross-model validation
- Llama
- Gemma
- Mistral
- Qwen
 
- Reconstruction perplexity evaluation
- Residual vocabulary learning
- Hierarchical motif family compression
 
---
 
## Installation
 
```bash
git clone https://github.com/thomaspricetj-hash/DTTQ-Delta-Tagged-Tokenized-Quantization-for-Neural-Network-Compression.git
 
cd DTTQ-Delta-Tagged-Tokenized-Quantization-for-Neural-Network-Compression
 
pip install -r requirements.txt
```
 
---
 
## Example Usage
 
Run the main DTTQ experiment:
 
```bash
python main.py \
--model facebook/opt-125m \
--block_size 16 \
--vocab_size 4096 \
--sample_blocks 200000
```
 
Run motif analysis:
 
```bash
python motif_analysis.py
```
 
Run motif visualization:
 
```bash
python motif_visualize.py
```
 
Run Huffman analysis:
 
```bash
python huffman_analysis.py
```
 
Run WeightGPT:
 
```bash
python upgrades/weight_gpt.py
```
 
Run all upgrades:
 
```bash
python upgrades/demo_upgrades.py \
--model facebook/opt-125m
```
 
---
 
## Citation
 
If you use this repository in research, please cite:
 
```bibtex
@misc{price2026dttq,
title={DTTQ: Delta-Tagged Tokenized Quantization and Reusable Motif Vocabularies in Transformer Weight Spaces},
author={Thomas Price},
year={2026},
note={Research Prototype}
}
```
 
---
 
## License
 
Released under the MIT License.
 
See LICENSE.md for details.
 
---
 
## Disclaimer
 
This repository is an experimental research project exploring the structure of transformer weight spaces. Results should be considered preliminary until validated across additional architectures and downstream evaluation benchmarks.DTTQ: Delta-Tagged Tokenized Quantization
Reusable Motif Vocabularies in Transformer Weight Spaces
DTTQ (Delta-Tagged Tokenized Quantization) is an experimental neural network compression and weight-structure research framework that represents transformer weights as reusable motif tokens, residual tokens, and tags rather than independent floating-point values.

The project investigates a simple question:

Do large transformer models contain reusable computational motifs that can be represented as a symbolic vocabulary?

Initial experiments on OPT-125M and GPT-2 suggest the answer may be yes.

Core Idea
Traditional quantization treats weights as individual numbers:

weight → FP16
weight → FP8
weight → FP4
weight → INT4
DTTQ instead treats model weights as recurring structures:

Weight Matrix
      ↓
Block Extraction
      ↓
Motif Vocabulary
      ↓
Weight Tokens
      ↓
Residual Tokens
      ↓
Tags
Result:

WeightToken + ResidualToken + Tag
instead of storing every weight independently.

Motivation
Modern transformer models contain hundreds of millions or billions of parameters.

DTTQ explores whether these parameters are better described as:

Motif Dictionary
+
Motif Placement
+
Residual Corrections
rather than:

Billions of independent weights
The framework combines:

Weight tokenization
Learned motif dictionaries
Residual coding
Predictive motif coding
BPE-style motif sequence compression
Weight-language modeling (WeightGPT)
Experimental Results
OPT-125M
Metric	Result
Blocks	488,736
Vocabulary Size	4096
Compression Ratio	2.64×
Reconstruction MSE	0.001654
Token Entropy	5.47 bits
Huffman Average	5.49 bits
Top 100 Coverage	96.34%
BPE Reduction	38.28%
Mutual Information	1.72 bits
WeightGPT Perplexity	12.63
GPT-2
Metric	Result
Blocks	485,616
Token Entropy	7.02 bits
Huffman Average	7.05 bits
Top 100 Coverage	89.56%
BPE Reduction	28.83%
Key Findings
Motif Reuse
A very small subset of motifs dominates model structure.

For OPT-125M:

Top 100 motifs
=
96.34% of all blocks
Information Compression
Expected entropy for a 4096-token vocabulary:

12 bits/token
Observed:

5.47 bits/token
indicating strong motif reuse.

Predictable Sequences
Neighboring motifs contain predictive information:

Mutual Information
=
1.72 bits
Knowing the current motif reduces uncertainty about the next motif by:

31.4%
Motif Language Structure
Applying only 100 BPE merges:

38.28%
sequence-length reduction.

This suggests recurring motif phrases rather than isolated motif reuse.

WeightGPT
A small transformer trained on motif streams achieved:

Perplexity = 12.63
demonstrating that motif sequences are substantially predictable.

Repository Structure
dttq/
│
├── motif_analysis.py
├── motif_visualize.py
├── huffman_analysis.py
├── mutual_info.py
├── bpe_weight_tokens.py
│
├── upgrades/
│   ├── motif_dictionary_model.py
│   ├── predict_motifs.py
│   └── weight_gpt.py
│
├── advanced/
│   ├── residual_motifs.py
│   └── motif_transformer.py
│
├── outputs/
│   ├── motif_stats.json
│   ├── motif_layer_stats.json
│   ├── motif_pca.png
│   └── motif_heatmaps.png
│
└── paper/
Design Upgrades
1. Motif Dictionary Model
Represent model blocks as:

Motif ID
+
Residual ID
rather than raw weights.

2. Predictive Motif Coding
Use motif transition probabilities:

P(next motif | current motif)
to encode prediction errors instead of full motif IDs.

3. WeightGPT
Train a language model directly on motif sequences:

3066 3881 3561 ...
and encode prediction residuals.

4. Motif Transformers
Train neural networks directly using motif dictionaries instead of dense weight tensors.

5. Residual Languages
Residuals become:

ResidualToken
+
Tag
making the entire model recursively tokenizable.

Research Hypothesis
DTTQ investigates the hypothesis that:

Transformer weight spaces contain reusable motif vocabularies with measurable entropy reduction, sequence predictability, and hierarchical structure.

The goal is not only improved compression but also a deeper understanding of how large neural networks organize their parameter space.

Current Status
Completed
Weight tokenization
Motif dictionary generation
Entropy analysis
Huffman coding analysis
Mutual information analysis
BPE motif compression
Motif visualization
Motif family clustering
WeightGPT prototype
In Progress
Cross-model validation

Llama
Gemma
Mistral
Qwen
Reconstruction perplexity evaluation

Residual vocabulary learning

Hierarchical motif family compression

Installation
git clone https://github.com/thomaspricetj-hash/DTTQ-Delta-Tagged-Tokenized-Quantization-for-Neural-Network-Compression.git

cd DTTQ-Delta-Tagged-Tokenized-Quantization-for-Neural-Network-Compression

pip install -r requirements.txt
Example Usage
Run the main DTTQ experiment:

python main.py \
  --model facebook/opt-125m \
  --block_size 16 \
  --vocab_size 4096 \
  --sample_blocks 200000
Run motif analysis:

python motif_analysis.py
Run motif visualization:

python motif_visualize.py
Run Huffman analysis:

python huffman_analysis.py
Run WeightGPT:

python upgrades/weight_gpt.py
Run all upgrades:

python upgrades/demo_upgrades.py \
  --model facebook/opt-125m
Citation
If you use this repository in research, please cite:

@misc{price2026dttq,
  title={DTTQ: Delta-Tagged Tokenized Quantization and Reusable Motif Vocabularies in Transformer Weight Spaces},
  author={Thomas Price},
  year={2026},
  note={Research Prototype}
}
License
Released under the MIT License.

See LICENSE.md for details.

Disclaimer
This repository is an experimental research project exploring the structure of transformer weight spaces. Results should be considered preliminary until validated across additional architectures and downstream evaluation benchmarks.
