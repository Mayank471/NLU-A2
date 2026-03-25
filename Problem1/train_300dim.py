"""
Train a single 300-dimensional Word2Vec from scratch (NumPy only) on the IITJ corpus
and print the full embedding vector for a chosen word.

Run  : python train_300dim.py
Needs: pip install numpy
"""

import os
import numpy as np
from collections import Counter

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CORPUS_FILE = os.path.join(BASE_DIR, "corpus.txt")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Load corpus ───────────────────────────────────────────────────────────────
sentences = []
with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        tokens = line.strip().split()
        if tokens:
            sentences.append(tokens)

print(f"Loaded {len(sentences)} sentences.\n")

# ── Build vocabulary ──────────────────────────────────────────────────────────
MIN_COUNT = 2
word_freq  = Counter(w for sent in sentences for w in sent)
word_freq  = {w: c for w, c in word_freq.items() if c >= MIN_COUNT}
vocab      = sorted(word_freq.keys())
word2idx   = {w: i for i, w in enumerate(vocab)}
idx2word   = {i: w for w, i in word2idx.items()}
vocab_size = len(vocab)

print(f"Vocabulary size: {vocab_size}")

filtered_sentences = [
    [w for w in sent if w in word2idx]
    for sent in sentences
]
filtered_sentences = [s for s in filtered_sentences if s]

# ── Negative sampling table ───────────────────────────────────────────────────
# Each word occupies slots proportional to freq^(3/4) — Word2Vec standard trick
def build_neg_table(table_size=1_000_000):
    freqs  = np.array([word_freq[idx2word[i]] for i in range(vocab_size)], dtype=np.float64)
    freqs  = freqs ** 0.75
    freqs /= freqs.sum()
    return np.random.choice(vocab_size, size=table_size, p=freqs)

neg_table = build_neg_table()

def get_negative_samples(target_idx, num_neg):
    """Sample num_neg indices, excluding the target word."""
    negs = []
    while len(negs) < num_neg:
        s = neg_table[np.random.randint(0, len(neg_table))]
        if s != target_idx:
            negs.append(s)
    return negs

# ── From-scratch CBOW with Negative Sampling ─────────────────────────────────
# Architecture:
#   W_in  : (vocab_size x embed_dim) — context word embeddings
#   W_out : (vocab_size x embed_dim) — center word embeddings
# Forward:  h = mean(W_in[context]),  score = sigmoid(h . W_out[target])
# Loss:     -log sigmoid(h.W_out[pos]) - sum log sigmoid(-h.W_out[neg])
# Backward: manual SGD gradients w.r.t W_in and W_out

EMBED_DIM   = 300   # 300-dimensional embeddings
NEG_SAMPLES = 5
LR          = 0.01
EPOCHS      = 3     # from-scratch is slow; 3 epochs is sufficient for demonstration
WINDOW      = 5

print(f"\nTraining from-scratch CBOW | dim=300 | window={WINDOW} | neg={NEG_SAMPLES}")
print(f"Learnable parameters: 2 x {vocab_size} x {EMBED_DIM} = {2*vocab_size*EMBED_DIM:,}\n")

# Initialize weight matrices with small random values
W_in  = np.random.uniform(-0.1, 0.1, (vocab_size, EMBED_DIM))
W_out = np.random.uniform(-0.1, 0.1, (vocab_size, EMBED_DIM))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

# Generate CBOW training pairs
pairs = []
for sent in filtered_sentences:
    indices = [word2idx[w] for w in sent]
    for i, center in enumerate(indices):
        context = [
            indices[j]
            for j in range(max(0, i - WINDOW), min(len(indices), i + WINDOW + 1))
            if j != i
        ]
        if context:
            pairs.append((context, center))

print(f"Total training pairs: {len(pairs):,}")

# Training loop
for epoch in range(EPOCHS):
    np.random.shuffle(pairs)
    total_loss = 0.0

    for context_indices, center_idx in pairs:
        # Hidden layer: mean of context embeddings
        h = W_in[context_indices].mean(axis=0)

        neg_indices = get_negative_samples(center_idx, NEG_SAMPLES)
        grad_h = np.zeros_like(h)

        # Positive sample gradient
        err_pos = sigmoid(h @ W_out[center_idx]) - 1.0
        grad_h += err_pos * W_out[center_idx]
        W_out[center_idx] -= LR * err_pos * h
        total_loss -= np.log(sigmoid(-err_pos - 1.0) + 1e-10)

        # Negative sample gradients
        for neg_idx in neg_indices:
            err_neg = sigmoid(h @ W_out[neg_idx])
            grad_h += err_neg * W_out[neg_idx]
            W_out[neg_idx] -= LR * err_neg * h

        # Update context word embeddings equally
        grad_ctx = grad_h / len(context_indices)
        for ctx_idx in context_indices:
            W_in[ctx_idx] -= LR * grad_ctx

    print(f"  Epoch {epoch+1}/{EPOCHS} done")

# Save W_in matrix
save_path = os.path.join(MODELS_DIR, "scratch_CBOW_dim300_w5_neg5_Win.npy")
np.save(save_path, W_in)
print(f"\nW_in matrix saved to: {save_path}")

# ── Print embedding vector for chosen word ────────────────────────────────────
word = "engineering"   # change to any word in your vocabulary

if word in word2idx:
    vector  = W_in[word2idx[word]]
    vec_str = ", ".join(f"{v:.4f}" for v in vector)
    print(f"\nEmbedding dimension: {len(vector)}")
    print(f"\n{word} - {vec_str}")
else:
    print(f"'{word}' not found in vocabulary.")