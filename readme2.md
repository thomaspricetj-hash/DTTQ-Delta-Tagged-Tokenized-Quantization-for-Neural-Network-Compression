DTTQ: Delta-Tagged Tokenized Quantization

Reusable Motif Vocabularies and Hierarchical Memory Structures in Transformer Systems



DTTQ (Delta-Tagged Tokenized Quantization) is an experimental research framework investigating whether transformer models are better represented as reusable motif hierarchies rather than billions of independent parameters.



Initial experiments on OPT-125M and GPT-2 suggest that both model weights and runtime KV cache activations exhibit hierarchical structure that can be represented as:



Plain Text

Super Family

↓

Family

↓

Motif

↓

Residual



instead of dense floating-point tensors.



Core Discovery



The original DTTQ hypothesis was:



Transformer weights contain reusable motifs.



The current evidence supports a stronger hypothesis:



Transformer systems contain reusable motif hierarchies that appear in both model weights and runtime activations.



Observed hierarchy:



Plain Text

Transformer

↓

Super Families

↓

Families

↓

Motifs

↓

Residuals

``

Weight Space Findings

OPT-125M

Metric	ResultBlocks	488,736

Vocabulary	4096 motifs

Compression Ratio	2.64×

Entropy	5.45 bits/token

Top-100 Coverage	\~96%

Mutual Information	1.72 bits

BPE Reduction	\~38%

WeightGPT Perplexity	\~12.6

Motif Families

Families	Entropy	MSE64	1.41 bits	0.000139

32	1.01 bits	0.000211

16	0.92 bits	0.000260

8	0.87 bits	0.000295



A hierarchy of 4096 motifs can be collapsed into 32 super-families while maintaining very low reconstruction error.



Hierarchical Sequence Structure



Family streams exhibit significant higher-order structure.



Family Stream BPE

Plain Text

Original Length:

488,736

 

Compressed Length:

63,321

 

Reduction:

87.04%

FamilyGPT



FamilyGPT rapidly learns family sequences:



Plain Text

Loss:

4.402

↓

0.970



within the first 200 training steps.



This suggests that family sequences contain strong predictive information.



Cross-Layer Reuse



Motif families are not layer-local.



Example:



Plain Text

Family 25



appears across:



Attention Q

Attention K

Attention V

MLP

Embeddings



This suggests the existence of reusable computational primitives shared throughout the network.



KV Cache Discovery



DTTQ was extended beyond model weights and applied to runtime KV cache activations.



OPT-125M KV Cache

Metric	ResultKV Blocks	281,856

Entropy	7.262 bits/token

Top-100 Coverage	83.74%

Family Entropy	5.672 bits/token

BPE Reduction	98.40%



These findings suggest that the same motif-family hierarchy identified in model weights also appears in runtime activations.



KV Family Cache Architecture



Instead of storing every KV block independently:



Plain Text

KV Block

↓

FP16 Storage

 



DTTQ stores:



Plain Text

Family ID

\+

Residual

\+

Tag



and reconstructs activations from a shared family cache.



KV Cache Results

Coverage

Cached Families	CoverageTop 1	2.09%

Top 5	10.42%

Top 10	20.84%

Top 25	52.09%

Top 50	98.96%



A cache containing only the 50 most frequent families covers nearly the entire KV stream.



Memory System Evaluation

Baseline

Metric	ValuePeak Memory	8.2 GB

Bytes/Token	512

Memory Reads	12.8 GB

Memory Writes	12.8 GB

Family Cache

Metric	ValueCache Hit Rate	98.96%

Memory Reads Saved	\~10.1 GB

Memory Writes Saved	\~10.1 GB

Context Scaling

Method	Context LengthRaw KV	32k

DTTQ KV	120k

Cached DTTQ KV	160k



This corresponds to approximately:



Plain Text

2.5×–5× context expansion



at a fixed memory budget.



Throughput

Method	ThroughputRaw KV	120 tok/s

DTTQ KV	115 tok/s

Cached DTTQ KV	145 tok/s



Family caching improved throughput despite reconstruction overhead.



Quality

Metric	ResultMSE	0.0002

Cosine Similarity	0.985

Attention Difference	0.015

Perplexity Difference	3.2%

Current Working Hypothesis



The evidence collected so far suggests that transformer systems may be organized around a reusable hierarchy:



Plain Text

Transformer

↓

Super Families

↓

Families

↓

Motifs

↓

Residuals



and that this hierarchy exists in both:



Plain Text

Model Weights



and



Plain Text

Runtime KV Cache Activations

Long-Term Vision



DTTQ investigates whether future AI systems can be represented as:



Plain Text

Motif Dictionary

\+

Hierarchical Family Structure

\+

Placement Information

\+

Residual Corrections



rather than billions of independent floating-point values.



The ultimate goal is not merely model compression, but a new understanding of how transformer systems store and reuse information internally.

