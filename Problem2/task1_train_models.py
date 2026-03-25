"""
Task 1: Model Implementation and Training
Implements and trains three character-level name generation models from scratch:
  1. Vanilla RNN
  2. Bidirectional LSTM (BLSTM)
  3. RNN with Basic Attention Mechanism
Each model is trained on Indian names from TrainingNames.txt.

Run  : python task1_train_models.py
Needs: pip install torch
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import random

# ─────────────────────────────────────────────────────────────────────────────
# PATHS AND DEVICE
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_FILE   = os.path.join(BASE_DIR, "TrainingNames.txt")
MODELS_DIR  = os.path.join(BASE_DIR, "saved_models")
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD AND PREPARE DATASET
# Each name is wrapped with start token '<' and end token '>'.
# The model learns to predict the next character given the previous ones.
# ─────────────────────────────────────────────────────────────────────────────
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    names = [line.strip() for line in f if line.strip()]

START_TOKEN = '<'
END_TOKEN   = '>'
names_seq   = [START_TOKEN + name.lower() + END_TOKEN for name in names]

# Build character vocabulary from all unique characters in the dataset
all_chars  = sorted(set(c for name in names_seq for c in name))
vocab_size = len(all_chars)
char2idx   = {c: i for i, c in enumerate(all_chars)}
idx2char   = {i: c for c, i in char2idx.items()}

print(f"Loaded {len(names)} names")
print(f"Vocabulary size: {vocab_size} characters: {''.join(all_chars)}\n")

def name_to_tensor(name):
    """Convert a name string to a LongTensor of character indices."""
    return torch.tensor([char2idx[c] for c in name], dtype=torch.long)

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
EMBED_DIM     = 32
HIDDEN_SIZE   = 128
NUM_LAYERS    = 1
TEMPERATURE   = 0.8   # controls randomness during sampling (lower = more conservative)

# Model-specific hyperparameters — tuned per architecture:
# BLSTM uses fewer epochs to prevent memorization (novelty rate was low at 50 epochs)
# Attention uses a lower learning rate to prevent divergence (loss was increasing)
RNN_LR       = 0.003
RNN_EPOCHS   = 50
BLSTM_LR     = 0.003
BLSTM_EPOCHS = 30    # reduced from 50 — prevents overfitting/memorization
ATTN_LR      = 0.001 # reduced from 0.003 — prevents loss divergence
ATTN_EPOCHS  = 50

# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL 1: VANILLA RNN
#
# Architecture:
#   Embedding → RNN Cell (manual) → Linear → Softmax
#
# The RNN cell is implemented manually:
#   h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b)
# where x_t is the embedded character at time t, h_t is the hidden state.
# The hidden state captures context of all previous characters.
# ─────────────────────────────────────────────────────────────────────────────
class VanillaRNNCell(nn.Module):
    """
    Manual implementation of a single RNN cell.
    h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b)
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_xh = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x, h):
        # x: (batch, input_size), h: (batch, hidden_size)
        return torch.tanh(self.W_xh(x) + self.W_hh(h))


class VanillaRNN(nn.Module):
    """
    Character-level language model using a manually implemented RNN cell.
    At each time step, predicts the next character given all previous characters.
    """
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.hidden_size  = hidden_size
        self.embedding    = nn.Embedding(vocab_size, embed_dim)
        self.rnn_cell     = VanillaRNNCell(embed_dim, hidden_size)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        """
        x: (seq_len,) character indices
        Returns logits (seq_len, vocab_size) and final hidden state.
        """
        if h is None:
            h = torch.zeros(1, self.hidden_size, device=x.device)

        embeds  = self.embedding(x)
        outputs = []
        for t in range(embeds.size(0)):
            h = self.rnn_cell(embeds[t].unsqueeze(0), h)
            outputs.append(h)

        out    = torch.cat(outputs, dim=0)
        logits = self.output_layer(out)
        return logits, h

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL 2: BIDIRECTIONAL LSTM (BLSTM)
#
# Architecture:
#   Embedding → Bidirectional LSTM → Linear → Softmax
#
# LSTM gates:
#   f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)   — forget gate
#   i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)   — input gate
#   c_t = f_t * c_{t-1} + i_t * tanh(W_c * [h_{t-1}, x_t] + b_c)
#   o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)   — output gate
#   h_t = o_t * tanh(c_t)
#
# IMPORTANT: To avoid loss=0 overfitting (bidirectional sees future tokens),
# we train using ONLY the forward LSTM direction output. The backward LSTM
# weights are still trained (they share the same model) but only forward
# hidden states are used for prediction — consistent with generation time.
# ─────────────────────────────────────────────────────────────────────────────
class BLSTMModel(nn.Module):
    """
    Character-level language model using a Bidirectional LSTM.
    Trained and evaluated using only the forward direction to avoid
    the loss=0 overfitting that occurs when backward direction leaks future info.
    """
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size    = embed_dim,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            bidirectional = True,
            batch_first   = False,
        )
        # Output layer projects forward hidden states (hidden_size) to vocab.
        # We use only the forward half to keep training consistent with generation.
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        Training and generation: uses ONLY forward direction hidden states.
        This ensures training and generation behavior are consistent,
        avoiding the loss=0 issue caused by backward direction leaking future tokens.
        """
        embeds = self.embedding(x).unsqueeze(1)          # (seq_len, 1, embed_dim)
        out, hidden = self.lstm(embeds, hidden)           # (seq_len, 1, 2*hidden_size)
        # Extract only forward direction: first hidden_size dimensions
        forward_out = out[:, 0, :self.hidden_size]        # (seq_len, hidden_size)
        logits = self.output_layer(forward_out)           # (seq_len, vocab_size)
        return logits, hidden

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL 3: RNN WITH BASIC ATTENTION MECHANISM
#
# Architecture:
#   Embedding → RNN Cell (manual) → Attention → Linear → Softmax
#
# Attention (additive / Bahdanau style):
#   At each time step t, compute a weighted sum over ALL past hidden states.
#   Score:   e_i = v_a · tanh(W_a · h_i)
#   Weights: alpha = softmax(e)
#   Context: c_t = sum(alpha_i * h_i)
#   Output:  linear(concat(h_t, c_t)) → vocab scores
#
# This lets the model attend to relevant earlier characters when predicting
# the next one (e.g., knowing a name started with 'Ra' helps decide 'j'/'m').
# ─────────────────────────────────────────────────────────────────────────────
class RNNWithAttention(nn.Module):
    """
    Character-level language model using a manually implemented RNN
    with additive (Bahdanau-style) attention over past hidden states.
    """
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.hidden_size  = hidden_size
        self.embedding    = nn.Embedding(vocab_size, embed_dim)
        self.rnn_cell     = VanillaRNNCell(embed_dim, hidden_size)
        # Attention parameters
        self.W_a          = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_a          = nn.Linear(hidden_size, 1, bias=False)
        # Project h_t + context to vocab
        self.output_layer = nn.Linear(hidden_size * 2, vocab_size)

    def attention(self, hidden_states, current_h):
        """
        Compute attention-weighted context vector over all past hidden states.
        hidden_states : (t, hidden_size)
        Returns context vector (1, hidden_size).
        """
        scores  = self.v_a(torch.tanh(self.W_a(hidden_states)))   # (t, 1)
        weights = torch.softmax(scores, dim=0)                      # (t, 1)
        context = (weights * hidden_states).sum(dim=0, keepdim=True)  # (1, hidden_size)
        return context

    def forward(self, x, h=None):
        """
        x: (seq_len,) character indices
        Returns logits (seq_len, vocab_size) and final hidden state.
        """
        if h is None:
            h = torch.zeros(1, self.hidden_size, device=x.device)

        embeds        = self.embedding(x)
        outputs       = []
        hidden_states = []

        for t in range(embeds.size(0)):
            h = self.rnn_cell(embeds[t].unsqueeze(0), h)
            hidden_states.append(h)

            H       = torch.cat(hidden_states, dim=0)
            context = self.attention(H, h)
            combined = torch.cat([h, context], dim=1)   # (1, 2*hidden_size)
            outputs.append(combined)

        out    = torch.cat(outputs, dim=0)
        logits = self.output_layer(out)
        return logits, h

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAINING FUNCTION
# Uses teacher forcing: feed true previous character as input at each step.
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model, names_seq, epochs, lr, model_name):
    """Train using cross-entropy loss. Input: chars[0..T-1], Target: chars[1..T]."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        random.shuffle(names_seq)

        for name in names_seq:
            if len(name) < 2:
                continue
            tensor     = name_to_tensor(name).to(DEVICE)
            input_seq  = tensor[:-1]
            target_seq = tensor[1:]

            optimizer.zero_grad()
            logits, _ = model(input_seq)
            loss = criterion(logits, target_seq)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  [{model_name}] Epoch {epoch+1}/{epochs} — Loss: {total_loss/len(names_seq):.4f}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 6. NAME GENERATION FUNCTION
# Generates a name character by character starting from the START token.
# Temperature scaling controls randomness:
#   lower = more conservative, higher = more diverse.
# Special tokens (< >) are stripped from output.
# ─────────────────────────────────────────────────────────────────────────────
def generate_name(model, max_len=20, temperature=TEMPERATURE):
    """Generate a single name by sampling character by character."""
    model.eval()
    with torch.no_grad():
        current_char = torch.tensor([char2idx[START_TOKEN]], dtype=torch.long).to(DEVICE)
        h    = None
        name = ""

        for _ in range(max_len):
            logits, h = model(current_char, h)

            # Temperature-scaled softmax for controlled randomness
            probs    = torch.softmax(logits[-1] / temperature, dim=0)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            next_char = idx2char[next_idx]

            if next_char == END_TOKEN:
                break   # stop at end token

            # Skip start token if accidentally generated mid-sequence
            if next_char != START_TOKEN:
                name += next_char

            current_char = torch.tensor([next_idx], dtype=torch.long).to(DEVICE)

    # Final cleanup: remove any stray special tokens
    name = name.replace(START_TOKEN, '').replace(END_TOKEN, '').strip()
    return name


# ─────────────────────────────────────────────────────────────────────────────
# 7. TRAIN ALL THREE MODELS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Training Vanilla RNN")
print("=" * 60)
rnn_model = VanillaRNN(vocab_size, EMBED_DIM, HIDDEN_SIZE).to(DEVICE)
print(f"  Architecture    : Embedding({vocab_size},{EMBED_DIM}) → RNN Cell → Linear({vocab_size})")
print(f"  Hidden size     : {HIDDEN_SIZE}")
print(f"  Layers          : {NUM_LAYERS}")
print(f"  Learning rate   : {RNN_LR}")
print(f"  Epochs          : {RNN_EPOCHS}")
print(f"  Trainable params: {rnn_model.count_parameters():,}")
train_model(rnn_model, names_seq, RNN_EPOCHS, RNN_LR, "VanillaRNN")
torch.save(rnn_model.state_dict(), os.path.join(MODELS_DIR, "vanilla_rnn.pt"))

print("\n" + "=" * 60)
print("Training BLSTM")
print("=" * 60)
blstm_model = BLSTMModel(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
print(f"  Architecture    : Embedding({vocab_size},{EMBED_DIM}) → BiLSTM({HIDDEN_SIZE}x2, forward-only output) → Linear({vocab_size})")
print(f"  Hidden size     : {HIDDEN_SIZE} per direction")
print(f"  Layers          : {NUM_LAYERS}")
print(f"  Learning rate   : {BLSTM_LR}")
print(f"  Epochs          : {BLSTM_EPOCHS}")
print(f"  Trainable params: {blstm_model.count_parameters():,}")
train_model(blstm_model, names_seq, BLSTM_EPOCHS, BLSTM_LR, "BLSTM")
torch.save(blstm_model.state_dict(), os.path.join(MODELS_DIR, "blstm.pt"))

print("\n" + "=" * 60)
print("Training RNN with Attention")
print("=" * 60)
attn_model = RNNWithAttention(vocab_size, EMBED_DIM, HIDDEN_SIZE).to(DEVICE)
print(f"  Architecture    : Embedding({vocab_size},{EMBED_DIM}) → RNN Cell → Attention → Linear({vocab_size})")
print(f"  Hidden size     : {HIDDEN_SIZE}")
print(f"  Layers          : {NUM_LAYERS}")
print(f"  Learning rate   : {ATTN_LR}")
print(f"  Epochs          : {ATTN_EPOCHS}")
print(f"  Trainable params: {attn_model.count_parameters():,}")
train_model(attn_model, names_seq, ATTN_EPOCHS, ATTN_LR, "RNN+Attention")
torch.save(attn_model.state_dict(), os.path.join(MODELS_DIR, "rnn_attention.pt"))

# ─────────────────────────────────────────────────────────────────────────────
# 8. GENERATE AND SAVE SAMPLE NAMES FROM EACH MODEL
# Save 200 generated names per model for use in Tasks 2 and 3.
# ─────────────────────────────────────────────────────────────────────────────
NUM_GENERATE = 200

for model, model_name in [
    (rnn_model,   "vanilla_rnn"),
    (blstm_model, "blstm"),
    (attn_model,  "rnn_attention"),
]:
    generated = []
    while len(generated) < NUM_GENERATE:
        name = generate_name(model)
        if len(name) >= 2:   # skip empty or single-char outputs
            generated.append(name)

    out_path = os.path.join(BASE_DIR, f"generated_{model_name}.txt")
    with open(out_path, 'w', encoding='utf-8') as f:
        for name in generated:
            f.write(name + '\n')
    print(f"\nSaved {len(generated)} names → generated_{model_name}.txt")

print("\nTraining complete. Models saved to saved_models/")