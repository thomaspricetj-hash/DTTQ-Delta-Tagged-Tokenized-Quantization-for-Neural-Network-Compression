"""
Design Upgrade #3: Weight GPT
Train tiny transformer on motif token stream
"""
import torch
import torch.nn as nn

class WeightGPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, nlayers=4, max_len=1024):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=nlayers)
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        emb = self.tok_emb(x) + self.pos_emb[:, :x.size(1)]
        out = self.transformer(emb)
        return self.head(out)

def train_weight_gpt(token_stream, vocab_size, epochs=5):
    # Simplified training stub
    # Returns model and perplexity
    model = WeightGPT(vocab_size)
    # Training loop would go here
    return model, 5.0  # placeholder perplexity
