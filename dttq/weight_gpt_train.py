"""
WeightGPT training and perplexity evaluation
"""
import torch, torch.nn as nn, math, random
from transformers import AutoModelForCausalLM
from dttq.extract import extract_matrices
from dttq.blocks import create_blocks
from dttq.vocab import learn_codebook
from dttq.tokenize import tokenize_blocks

class WeightGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, nlayers=2, max_len=512):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        emb = self.tok_emb(x) + self.pos_emb[:, :x.size(1)]
        out = self.transformer(emb)
        return self.head(out)

def get_batch(tokens, seq_len=256, batch_size=32):
    # Simple random sampling
    n = len(tokens) - seq_len
    idx = random.randint(0, n)
    x = torch.tensor(tokens[idx:idx+seq_len], dtype=torch.long)
    y = torch.tensor(tokens[idx+1:idx+seq_len+1], dtype=torch.long)
    # batch
    xs = []
    ys = []
    for _ in range(batch_size):
        i = random.randint(0, n)
        xs.append(torch.tensor(tokens[i:i+seq_len], dtype=torch.long))
        ys.append(torch.tensor(tokens[i+1:i+seq_len+1], dtype=torch.long))
    return torch.stack(xs), torch.stack(ys)

def main():
    model_name = 'facebook/opt-125m'
    print(f"Loading {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    matrices = extract_matrices(model)
    blocks = create_blocks(matrices, 16)
    import random
    sampled = random.sample(blocks, min(200000, len(blocks)))
    codebook = learn_codebook(sampled, 4096)
    token_ids, _ = tokenize_blocks(blocks, codebook)
    tokens = token_ids
    print(f"Token stream length: {len(tokens)}")
    
    vocab_size = 4096
    net = WeightGPT(vocab_size)
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    net.train()
    seq_len = 256
    batch_size = 32
    steps = 500
    for step in range(steps):
        x, y = get_batch(tokens, seq_len, batch_size)
        logits = net(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"Step {step} loss {loss.item():.4f}")
    
    # Evaluate perplexity on held-out
    net.eval()
    with torch.no_grad():
        total_loss = 0.0
        n = 0
        for _ in range(200):
            x, y = get_batch(tokens, seq_len, batch_size)
            logits = net(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            total_loss += loss.item() * x.numel()
            n += x.numel()
        ppl = math.exp(total_loss / n)
        print(f"WeightGPT perplexity: {ppl:.2f}")

if __name__ == "__main__":
    main()
