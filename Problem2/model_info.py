"""
Report number of parameters and model size (MB) for Vanilla RNN.
Run: python model_info.py
"""

import os
import torch
import torch.nn as nn

# ── Rebuild model architecture (must match task1_train_models.py) ─────────────
class VanillaRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_xh = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x, h):
        return torch.tanh(self.W_xh(x) + self.W_hh(h))

class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.hidden_size  = hidden_size
        self.embedding    = nn.Embedding(vocab_size, embed_dim)
        self.rnn_cell     = VanillaRNNCell(embed_dim, hidden_size)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(1, self.hidden_size)
        embeds  = self.embedding(x)
        outputs = []
        for t in range(embeds.size(0)):
            h = self.rnn_cell(embeds[t].unsqueeze(0), h)
            outputs.append(h)
        out    = torch.cat(outputs, dim=0)
        logits = self.output_layer(out)
        return logits, h

# ── Hyperparameters (must match task1_train_models.py) ────────────────────────
VOCAB_SIZE  = 28    # from your training output
EMBED_DIM   = 32
HIDDEN_SIZE = 128

model = VanillaRNN(VOCAB_SIZE, EMBED_DIM, HIDDEN_SIZE)

# Load saved weights
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "saved_models", "vanilla_rnn.pt")
model.load_state_dict(torch.load(model_path, map_location="cpu"))

# ── Count parameters ──────────────────────────────────────────────────────────
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# ── Model size in MB ─────────────────────────────────────────────────────────
# Each parameter is stored as float32 (4 bytes)
size_mb = (total_params * 4) / (1024 ** 2)

# Also check actual file size on disk
file_size_mb = os.path.getsize(model_path) / (1024 ** 2)

print(f"Total parameters     : {total_params:,}")
print(f"Trainable parameters : {trainable_params:,}")
print(f"Model size (calc)    : {size_mb:.4f} MB")
print(f"Model size (on disk) : {file_size_mb:.4f} MB")