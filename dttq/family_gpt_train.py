"""
FamilyGPT: train small transformer on family stream
"""
import torch
from transformers import AutoModelForCausalLM
from dttq.extract import extract_matrices
from dttq.blocks import create_blocks
from dttq.vocab import learn_codebook
from dttq.tokenize import tokenize_blocks
import random, numpy as np
from sklearn.cluster import MiniBatchKMeans
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn as nn

model_name = 'facebook/opt-125m'
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model.eval()
matrices = extract_matrices(model)
blocks = create_blocks(matrices, 16)
sampled = random.sample(blocks, min(200000, len(blocks)))
codebook = learn_codebook(sampled, 4096)
token_ids, residuals = tokenize_blocks(blocks, codebook)

kmeans_64 = MiniBatchKMeans(n_clusters=64, random_state=0, batch_size=256)
family_ids = kmeans_64.fit_predict(codebook)
token_to_family = {i:int(family_ids[i]) for i in range(len(family_ids))}
family_stream = [token_to_family[tid] for tid in token_ids]

vocab_size = 64
max_len = 256
data = torch.tensor(family_stream, dtype=torch.long)
dataset = []
for i in range(0, len(data)-max_len, max_len):
    chunk = data[i:i+max_len]
    if len(chunk) < max_len: break
    dataset.append(chunk.unsqueeze(0))

class FamilyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4, max_len=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(max_len, d_model))
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512, batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        emb = self.embed(x) + self.pos[:x.size(1)]
        out = self.encoder(emb)
        return self.head(out)

device = 'cpu'
model_gpt = FamilyGPT(vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_gpt.parameters(), lr=1e-3)

print('Training FamilyGPT for 200 steps...')
model_gpt.train()
for step in range(200):
    x = dataset[step % len(dataset)][0].unsqueeze(0).to(device)
    y = torch.roll(x, -1, dims=1)
    logits = model_gpt(x[:, :-1])
    loss = criterion(logits.view(-1, vocab_size), y[:, 1:].reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 50 == 0:
        print(f'Step {step}, loss {loss.item():.3f}')

print('Training complete')
