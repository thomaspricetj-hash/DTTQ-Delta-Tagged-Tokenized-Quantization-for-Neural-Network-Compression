import argparse, torch, random
from transformers import AutoModelForCausalLM
from dttq.extract import extract_matrices
from dttq.blocks import create_blocks
from dttq.vocab import learn_codebook
from dttq.tokenize import tokenize_blocks
import numpy as np
from collections import Counter
import heapq

class Node:
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol=symbol; self.freq=freq; self.left=left; self.right=right
    def __lt__(self, other): return self.freq < other.freq

def build_huffman(freq_dict):
    heap=[Node(sym,f) for sym,f in freq_dict.items()]
    heapq.heapify(heap)
    while len(heap)>1:
        n1=heapq.heappop(heap); n2=heapq.heappop(heap)
        heapq.heappush(heap, Node(freq=n1.freq+n2.freq, left=n1, right=n2))
    return heap[0]

def get_codes(node, prefix='', codes=None):
    if codes is None: codes={}
    if node.symbol is not None: codes[node.symbol]=prefix or '0'
    else:
        get_codes(node.left, prefix+'0', codes)
        get_codes(node.right, prefix+'1', codes)
    return codes

def entropy(counter,total):
    vals=np.array(list(counter.values()), dtype=np.float64)
    probs=vals/total
    return -np.sum(probs*np.log2(probs+1e-12))

parser=argparse.ArgumentParser()
parser.add_argument('--model', required=True)
args=parser.parse_args()

print(f"Loading {args.model}")
model=AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
model.eval()
matrices=extract_matrices(model)
blocks=create_blocks(matrices,16)
print(f"Blocks: {len(blocks)}")
sampled=random.sample(blocks, min(200000,len(blocks)))
codebook=learn_codebook(sampled,4096)
token_ids,_=tokenize_blocks(blocks,codebook)
seq=np.array(token_ids)
counter=Counter(seq)
total=len(seq)
Hx=entropy(counter,total)
print(f"Token entropy: {Hx:.3f} bits/token")
freq_dict=dict(counter)
root=build_huffman(freq_dict)
codes=get_codes(root)
avg_bits=sum(counter[s]*len(codes[s]) for s in counter)/total
print(f"Huffman avg bits/token: {avg_bits:.3f}")
top100=sum(cnt for _,cnt in counter.most_common(100))
print(f"Top 100 coverage: {top100/total*100:.2f}%")
# Mutual info
single_x=Counter(seq[:-1])
single_y=Counter(seq[1:])
pairs=Counter(zip(seq[:-1],seq[1:]))
Hx=entropy(single_x,len(seq)-1)
Hy=entropy(single_y,len(seq)-1)
Hxy=entropy(pairs,len(pairs))
I=Hx+Hy-Hxy
print(f"Mutual information I(token;next): {I:.4f} bits")
def bpe_len(seq, merges=100):
    seq_list=seq.tolist()
    for _ in range(merges):
        pairs=Counter(zip(seq_list,seq_list[1:]))
        if not pairs: break
        pair,_=pairs.most_common(1)[0]
        new_token=max(seq_list)+1
        new_seq=[]; i=0
        while i<len(seq_list):
            if i<len(seq_list)-1 and seq_list[i]==pair[0] and seq_list[i+1]==pair[1]:
                new_seq.append(new_token); i+=2
            else:
                new_seq.append(seq_list[i]); i+=1
        seq_list=new_seq
    return len(seq_list)
merged_len=bpe_len(seq,100)
print(f"BPE 100 merges length reduction: {(1-merged_len/total)*100:.2f}%")
