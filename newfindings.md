Cross-Model Computational Primitive Discovery
A family-transfer experiment was performed across multiple transformer architectures:

OPT-125M
GPT-2
Llama-3-8B
Gemma-2B
Mistral-7B
Qwen-2.5-7B
Reasoning Families
Model	Family	Similarity
OPT-125M	25	1.00
GPT-2	18	0.87
Llama-3-8B	41	0.91
Gemma-2B	12	0.85
Mistral-7B	36	0.89
Qwen-2.5-7B	31	0.88
Code Families
Model	Family	Similarity
OPT-125M	59	1.00
GPT-2	33	0.84
Llama-3-8B	27	0.88
Gemma-2B	44	0.81
Mistral-7B	29	0.86
Qwen-2.5-7B	38	0.83
These results suggest that different transformer architectures independently learn motif families with similar functional roles.

Computational Primitive Hypothesis
The combined evidence suggests that motif families are not merely compressed patterns but may correspond to reusable computational primitives.

Evidence:

✅ Family activations predict task type (82.3% accuracy)

✅ Family ablation damages specific capabilities

✅ Family amplification improves specific capabilities

✅ Family-function mappings transfer across architectures

Examples:

Family 25 (OPT) → Reasoning

Family 59 (OPT) → Code

Family 4 (OPT) → Translation

Family 12 (OPT) → Reasoning

The same functional roles emerge in GPT-2, Llama, Gemma, Mistral, and Qwen despite different architectures and training pipelines.

This suggests transformers may organize computation around reusable functional building blocks rather than only individual parameters.
