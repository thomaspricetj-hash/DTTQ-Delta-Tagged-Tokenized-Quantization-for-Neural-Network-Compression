DTTQ: Delta-Tagged Tokenized Quantization for Neural Network Compression



Version 0.1 Concept Proposal



Author: Community Research Proposal

&#x20;Status: Exploratory Architecture



Abstract



This paper proposes Delta-Tagged Tokenized Quantization (DTTQ), a neural network compression architecture that combines:



Learned weight tokenization

Delta encoding

Multi-level tagging

Traditional quantization

Optional entropy coding



The central hypothesis is that neural network weights contain higher-order patterns that are not fully exploited by existing FP8, FP4, INT4, or NVFP4 quantization schemes.



Rather than viewing a model as billions of independent floating-point values, DTTQ treats model parameters as a structured language composed of recurring patterns.



The architecture represents weight blocks as reusable tokens and stores deviations through compact delta tags.



This approach potentially achieves:



Higher compression ratios

Lower storage requirements

Reduced memory bandwidth

Better preservation of important weight structures



while remaining compatible with future accelerator designs.



1\. Motivation



Current quantization methods operate on individual values:



Plain Text

weight → fp16

weight → fp8

weight → fp4

weight → int4

Show more lines



Even sophisticated methods such as NVFP4 ultimately represent each weight independently.



However, neural network weights exhibit significant redundancy:



Plain Text

0.52 0.51 0.50 0.53

0.51 0.52 0.49 0.52

Show more lines



Similar structures occur repeatedly throughout transformers.



A core question emerges:



What if neural network weights were treated as sequences of reusable tokens rather than isolated numbers?



2\. Core Concept



DTTQ introduces three layers:



Plain Text

Weight Matrix

 

↓

 

Block Extraction

 

↓

 

Token Assignment

 

↓

 

Delta Correction

 

↓

 

Tag Encoding

 

↓

 

Storage

Show more lines



The final representation becomes:



Plain Text

Token + Delta + Tag

Show more lines



instead of



Plain Text

Raw Weight

Show more lines

3\. Learned Weight Vocabulary

Block Formation



Partition a weight matrix into fixed-size blocks:



Plain Text

4x4

8x8

16x16

Show more lines



Example:



Plain Text

Block A

 

0.52 0.50

0.49 0.51

Show more lines

Vocabulary Construction



Using k-means or vector quantization:



Plain Text

Token 0

Token 1

Token 2

...

Token N

Show more lines



Each token represents a common pattern.



Example:



Plain Text

Token 42

 

0.50 0.50

0.50 0.50

 

Show more lines

4\. Delta Encoding



Not all blocks match tokens perfectly.



Instead store:



Plain Text

Block ≈ Token + Residual

Show more lines



Example:



Plain Text

Actual

 

0.51 0.49

0.52 0.50

Show more lines



Closest Token:



Plain Text

0.50 0.50

0.50 0.50

 

Show more lines



Residual:



Plain Text

+0.01 -0.01

+0.02 0.00

Show more lines

5\. Delta Tags



Different deltas require different storage precision.



A tag system controls interpretation.



Example:



Tag	Meaning00	No residual

01	Small delta

10	Medium delta

11	Raw block

Example

Plain Text

Token: 42

 

Tag: 01

 

Delta:

+1

\-1

+2

0

``

Show more lines

6\. Hierarchical Tokens



Tokens themselves may reference larger tokens.



Example:



Plain Text

Super Token

 

contains

 

Token 15

Token 22

Token 31

Token 15

Show more lines



This creates:



Plain Text

Vocabulary

↓

Token

↓

Super Token

Show more lines



similar to language tokenization.



7\. Entropy Coding Layer



After tokenization:



Plain Text

42

42

42

17

42

42

Show more lines



Some tokens become highly frequent.



Apply:



Huffman coding

Arithmetic coding

ANS coding



to reduce storage further.



This layer is optional.



8\. Mathematical Formulation



Given weight block:



WWW



Find nearest token:



TiT\_iTi​



such that:



Ti=arg⁡min⁡∣∣W−T∣∣T\_i = \\arg \\min ||W-T||Ti​=argmin∣∣W−T∣∣



Residual:



R=W−TiR=W-T\_iR=W−Ti​



Stored representation:



W=(Ti+Q(R))W=(T\_i + Q(R))W=(Ti​+Q(R))



where:



Q(R)Q(R)Q(R)



is quantized residual data.



Final compressed form:



(TokenID,Tag,Delta)(TokenID, Tag, Delta)(TokenID,Tag,Delta)

9\. Compression Estimate



Consider a 16-value block.



FP16

Plain Text

16 × 16

 

= 256 bits

Show more lines

INT4

Plain Text

16 × 4

 

= 64 bits

Show more lines

Hypothetical DTTQ

Plain Text

Token ID = 12 bits

 

Tag = 2 bits

 

Small Delta = 16 bits

 

Total:

 

30 bits

Show more lines



This is less than half of INT4 storage.



Actual values depend on vocabulary quality.



10\. Hardware Considerations



Current tensor cores expect:



Plain Text

weight × activation

Show more lines



directly.



DTTQ requires:



Plain Text

Token Lookup

↓

Delta Reconstruction

↓

Multiply

Show more lines



Thus today's hardware may perform worse despite higher compression.



Future accelerators could incorporate:



Plain Text

Token Cache

 

Delta Engine

 

Dictionary Tensor Core

Show more lines



to eliminate reconstruction costs.



11\. DTTQ Accelerator Proposal



A future accelerator might contain:



Token Memory



Stores codebooks.



Delta Decoder



Applies small corrections.



Tile Generator



Reconstructs weight tiles.



Dictionary GEMM Engine



Performs multiplication directly from tokenized blocks.



Pipeline:



Plain Text

Token

↓

Dictionary

↓

Residual

↓

Tensor Multiply

Show more lines

12\. Advantages

Higher Compression



Repeated structures are stored once.



Reduced Bandwidth



Token streams are smaller than raw weights.



Preservation of Weight Structure



Patterns survive compression.



Multi-Level Adaptation



Different regions use different token granularities.



13\. Limitations

Decode Overhead



Requires additional reconstruction.



Vocabulary Training



Codebooks must be learned.



Hardware Dependency



Full benefits require dedicated silicon.



Error Propagation



Poor token selection may introduce artifacts.



14\. Potential Research Directions

Dynamic Vocabularies



Learn codebooks during training.



Layer-Specific Dictionaries



Each transformer layer gets unique tokens.



Mixture-of-Experts Compression



Experts share token vocabularies.



Tokenized Training



Train models directly in token space.



Recursive Token Models



Tokens of tokens of tokens.



15\. Long-Term Vision



Modern LLMs tokenize language:



Plain Text

Words

↓

Tokens

Show more lines



DTTQ proposes treating model weights similarly:



Plain Text

Weights

↓

Blocks

↓

Tokens

↓

Delta Tags

``

Show more lines



The ultimate hypothesis is:



Neural network parameters may be more naturally represented as a compressed symbolic language of recurring patterns than as billions of independent floating-point numbers.



If true, future models could be stored, transmitted, and potentially computed using tokenized representations that achieve significantly higher compression than conventional quantization while preserving accuracy.



An especially novel variant of your idea would be "Neural Weight Language Modeling", where the model's weights themselves are tokenized and compressed using BPE-like vocabulary learning, with delta-tags acting as grammar rules. That is unusual enough that it could genuinely be worth putting in front of compression and accelerator researchers as a speculative research direction.

