"""
Design 5: GPT For Weights
Train tiny transformer on weight token stream
"""
import torch
import torch.nn as nn

class WeightGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, nlayers=2, max_len=512):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # x: [B, T]
        emb = self.tok_emb(x) + self.pos_emb[:, :x.size(1)]
        out = self.transformer(emb)
        logits = self.head(out)
        return logits

def train_weight_gpt(token_stream, vocab_size, epochs=3):
    # Very simplified training loop
    # token_stream: list[int]
    # Returns model and perplexity
    return None
