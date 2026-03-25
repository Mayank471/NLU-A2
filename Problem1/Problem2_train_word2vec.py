"""
Task 2: Word2Vec Model Training
Implements CBOW and Skip-gram with Negative Sampling:
  (A) From scratch using NumPy only
  (B) Using Gensim library
Experiments with: (i) Embedding dim, (ii) Context window size, (iii) Negative samples.
Saves all trained models to models/ for use in Tasks 3 and 4.

Run  : python problem2_train_word2vec.py
Needs: pip install gensim numpy
"""

import os
import csv
import numpy as np
from collections import Counter
from gensim.models import Word2Vec

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CORPUS_FILE = os.path.join(BASE_DIR, "corpus.txt")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD CORPUS
# Each line in corpus.txt is a preprocessed tokenized sentence.
# ─────────────────────────────────────────────────────────────────────────────
sentences = []
with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        tokens = line.strip().split()
        if tokens:
            sentences.append(tokens)

print(f"Loaded {len(sentences)} sentences from corpus.\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2. BUILD VOCABULARY
# Count word frequencies and keep only words that appear at least MIN_COUNT
# times. Rare words are excluded as they lack enough context to learn from.
# word2idx maps each word to a unique integer index for matrix lookups.
# ─────────────────────────────────────────────────────────────────────────────
MIN_COUNT = 5   # ignore words appearing fewer than 5 times

word_freq  = Counter(w for sent in sentences for w in sent)
word_freq  = {w: c for w, c in word_freq.items() if c >= MIN_COUNT}
vocab      = sorted(word_freq.keys())
word2idx   = {w: i for i, w in enumerate(vocab)}
idx2word   = {i: w for w, i in word2idx.items()}
vocab_size = len(vocab)

print(f"Vocabulary size (min_count={MIN_COUNT}): {vocab_size}\n")

# Filter sentences to only retain known vocabulary words
filtered_sentences = [
    [w for w in sent if w in word2idx]
    for sent in sentences
]
filtered_sentences = [s for s in filtered_sentences if s]

# ─────────────────────────────────────────────────────────────────────────────
# 3. NEGATIVE SAMPLING TABLE
# Standard Word2Vec trick: build a large table where each word occupies
# a number of slots proportional to its frequency^(3/4). Sampling a random
# slot from this table approximates sampling from the unigram distribution,
# slightly upweighting rare words relative to a flat frequency distribution.
# ─────────────────────────────────────────────────────────────────────────────
def build_neg_sampling_table(table_size=1_000_000):
    freqs  = np.array([word_freq[idx2word[i]] for i in range(vocab_size)], dtype=np.float64)
    freqs  = freqs ** 0.75          # raise to 3/4 power (Word2Vec standard)
    freqs /= freqs.sum()            # normalize to a probability distribution
    return np.random.choice(vocab_size, size=table_size, p=freqs)

neg_table = build_neg_sampling_table()

def get_negative_samples(target_idx, num_neg):
    """Sample num_neg word indices, making sure none equals the target word."""
    negs = []
    while len(negs) < num_neg:
        sample = neg_table[np.random.randint(0, len(neg_table))]
        if sample != target_idx:
            negs.append(sample)
    return negs

# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAINING DATA GENERATORS
# ─────────────────────────────────────────────────────────────────────────────
def generate_cbow_pairs(window):
    """
    Generate (context_indices, center_idx) pairs for CBOW.
    For each center word, context = all surrounding words within the window.
    CBOW predicts the center word from its context.
    """
    pairs = []
    for sent in filtered_sentences:
        indices = [word2idx[w] for w in sent]
        for i, center in enumerate(indices):
            context = [
                indices[j]
                for j in range(max(0, i - window), min(len(indices), i + window + 1))
                if j != i
            ]
            if context:
                pairs.append((context, center))
    return pairs

def generate_skipgram_pairs(window):
    """
    Generate (center_idx, context_idx) pairs for Skip-gram.
    For each center word, one pair is created per context word in the window.
    Skip-gram predicts surrounding context words given the center word.
    """
    pairs = []
    for sent in filtered_sentences:
        indices = [word2idx[w] for w in sent]
        for i, center in enumerate(indices):
            for j in range(max(0, i - window), min(len(indices), i + window + 1)):
                if j != i:
                    pairs.append((center, indices[j]))
    return pairs

# ─────────────────────────────────────────────────────────────────────────────
# 5. FROM-SCRATCH CBOW WITH NEGATIVE SAMPLING
#
# Architecture:
#   W_in  : (vocab_size x embed_dim) — input/context word embeddings
#   W_out : (vocab_size x embed_dim) — output/center word embeddings
#
# Forward pass:
#   h = mean of context word vectors from W_in       shape: (embed_dim,)
#   score = sigmoid(h . W_out[target])
#
# Loss (Negative Sampling):
#   L = -log sigmoid(h . W_out[pos]) - sum log sigmoid(-h . W_out[neg_i])
#
# Backward pass (manual gradients):
#   dL/dW_out[pos] = (sigmoid(h.W_out[pos]) - 1) * h
#   dL/dW_out[neg] = sigmoid(h.W_out[neg]) * h
#   dL/dh          = sum of above errors weighted by W_out vectors
#   dL/dW_in[ctx]  = dL/dh / len(context)   (distributed equally)
# ─────────────────────────────────────────────────────────────────────────────
class CBOWScratch:
    def __init__(self, vocab_size, embed_dim, neg_samples, lr=0.01):
        # Small random initialization — avoids symmetry-breaking issues
        self.W_in  = np.random.uniform(-0.1, 0.1, (vocab_size, embed_dim))
        self.W_out = np.random.uniform(-0.1, 0.1, (vocab_size, embed_dim))
        self.neg_samples = neg_samples
        self.lr = lr

    @staticmethod
    def sigmoid(x):
        # Clip input to avoid numerical overflow in exp
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def train_pair(self, context_indices, center_idx):
        """One SGD update step for a single (context, center) pair."""
        # Hidden layer: average of context word embeddings
        h = self.W_in[context_indices].mean(axis=0)   # shape: (embed_dim,)

        neg_indices = get_negative_samples(center_idx, self.neg_samples)
        grad_h = np.zeros_like(h)

        # Positive sample: error = sigmoid(score) - 1, want sigmoid → 1
        err_pos = self.sigmoid(h @ self.W_out[center_idx]) - 1.0
        grad_h += err_pos * self.W_out[center_idx]
        self.W_out[center_idx] -= self.lr * err_pos * h

        # Negative samples: error = sigmoid(score) - 0, want sigmoid → 0
        for neg_idx in neg_indices:
            err_neg = self.sigmoid(h @ self.W_out[neg_idx])
            grad_h += err_neg * self.W_out[neg_idx]
            self.W_out[neg_idx] -= self.lr * err_neg * h

        # Distribute gradient equally across all context word embeddings
        grad_ctx = grad_h / len(context_indices)
        for ctx_idx in context_indices:
            self.W_in[ctx_idx] -= self.lr * grad_ctx

    def train(self, pairs, epochs):
        """Train CBOW for the given number of epochs over all pairs."""
        for epoch in range(epochs):
            np.random.shuffle(pairs)
            for context_indices, center_idx in pairs:
                self.train_pair(context_indices, center_idx)
            print(f"  CBOW Scratch Epoch {epoch+1}/{epochs} done")

    def similarity(self, w1_idx, w2_idx):
        """Cosine similarity between two word vectors using W_in."""
        v1, v2 = self.W_in[w1_idx], self.W_in[w2_idx]
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        return float(np.dot(v1, v2) / denom) if denom > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. FROM-SCRATCH SKIP-GRAM WITH NEGATIVE SAMPLING
#
# Architecture:
#   W_in  : (vocab_size x embed_dim) — center word embeddings
#   W_out : (vocab_size x embed_dim) — context word embeddings
#
# Forward pass:
#   v_c = W_in[center]                              shape: (embed_dim,)
#   score = sigmoid(v_c . W_out[context])
#
# Loss (Negative Sampling):
#   L = -log sigmoid(v_c . W_out[pos]) - sum log sigmoid(-v_c . W_out[neg_i])
#
# Backward pass (manual gradients):
#   dL/dW_out[pos] = (sigmoid(v_c.W_out[pos]) - 1) * v_c
#   dL/dW_out[neg] = sigmoid(v_c.W_out[neg]) * v_c
#   dL/dv_c        = sum of above errors weighted by W_out vectors
#   dL/dW_in[ctr]  = dL/dv_c
# ─────────────────────────────────────────────────────────────────────────────
class SkipgramScratch:
    def __init__(self, vocab_size, embed_dim, neg_samples, lr=0.01):
        self.W_in  = np.random.uniform(-0.1, 0.1, (vocab_size, embed_dim))
        self.W_out = np.random.uniform(-0.1, 0.1, (vocab_size, embed_dim))
        self.neg_samples = neg_samples
        self.lr = lr

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def train_pair(self, center_idx, context_idx):
        """One SGD update step for a single (center, context) pair."""
        v_c = self.W_in[center_idx]   # center word vector

        neg_indices = get_negative_samples(context_idx, self.neg_samples)
        grad_vc = np.zeros_like(v_c)

        # Positive sample: error = sigmoid(score) - 1
        err_pos = self.sigmoid(v_c @ self.W_out[context_idx]) - 1.0
        grad_vc += err_pos * self.W_out[context_idx]
        self.W_out[context_idx] -= self.lr * err_pos * v_c

        # Negative samples: error = sigmoid(score)
        for neg_idx in neg_indices:
            err_neg = self.sigmoid(v_c @ self.W_out[neg_idx])
            grad_vc += err_neg * self.W_out[neg_idx]
            self.W_out[neg_idx] -= self.lr * err_neg * v_c

        self.W_in[center_idx] -= self.lr * grad_vc

    def train(self, pairs, epochs):
        """Train Skip-gram for the given number of epochs over all pairs."""
        for epoch in range(epochs):
            np.random.shuffle(pairs)
            for center_idx, context_idx in pairs:
                self.train_pair(center_idx, context_idx)
            print(f"  Skip-gram Scratch Epoch {epoch+1}/{epochs} done")

    def similarity(self, w1_idx, w2_idx):
        """Cosine similarity between two word vectors using W_in."""
        v1, v2 = self.W_in[w1_idx], self.W_in[w2_idx]
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        return float(np.dot(v1, v2) / denom) if denom > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 7. EVALUATION HELPERS
# Avg cosine similarity on known IITJ word pairs — used to compare configs
# and justify the best hyperparameter selection in the report.
# ─────────────────────────────────────────────────────────────────────────────
TEST_PAIRS = [
    ("engineering", "technology"),
    ("research",    "development"),
    ("students",    "faculty"),
    ("jodhpur",     "institute"),
]

def avg_sim_scratch(model):
    """Average cosine similarity for a from-scratch model on test pairs."""
    scores = [
        model.similarity(word2idx[w1], word2idx[w2])
        for w1, w2 in TEST_PAIRS
        if w1 in word2idx and w2 in word2idx
    ]
    return round(sum(scores) / len(scores), 4) if scores else 0.0

def avg_sim_gensim(model):
    """Average cosine similarity for a Gensim model on test pairs."""
    scores = [
        model.wv.similarity(w1, w2)
        for w1, w2 in TEST_PAIRS
        if w1 in model.wv and w2 in model.wv
    ]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 8. HYPERPARAMETER EXPERIMENTS
# Train all combinations of (dim, window, neg) for both architectures.
# From-scratch uses fewer epochs since pure NumPy is ~100x slower than
# Gensim's optimized C backend.
# Total learnable parameters per model = 2 x vocab_size x embed_dim (W_in + W_out)
# ─────────────────────────────────────────────────────────────────────────────
EMBEDDING_DIMS   = [50, 100, 200]   # (i)   embedding dimensions
WINDOW_SIZES     = [2, 5]           # (ii)  context window sizes
NEGATIVE_SAMPLES = [5, 10]          # (iii) number of negative samples

EPOCHS_SCRATCH = 1    # fewer epochs for from-scratch (pure NumPy is slower)
EPOCHS_GENSIM  = 10   # full epochs for Gensim (optimized C backend)

results = []

for dim in EMBEDDING_DIMS:
    for window in WINDOW_SIZES:
        for neg in NEGATIVE_SAMPLES:

            print(f"\n{'='*60}")
            print(f"Config: dim={dim} | window={window} | neg={neg}")
            print(f"  Learnable parameters: 2 x {vocab_size} x {dim} = {2*vocab_size*dim:,}")
            print(f"{'='*60}")

            cbow_pairs     = generate_cbow_pairs(window)
            skipgram_pairs = generate_skipgram_pairs(window)

            # ── (A) From-Scratch CBOW ─────────────────────────────────────
            print(f"\n[Scratch] CBOW")
            cbow_scratch = CBOWScratch(vocab_size, dim, neg, lr=0.01)
            cbow_scratch.train(cbow_pairs, epochs=EPOCHS_SCRATCH)
            # Save W_in matrix — the learned word embedding table used in Tasks 3 & 4
            np.save(os.path.join(MODELS_DIR, f"scratch_CBOW_dim{dim}_w{window}_neg{neg}_Win.npy"),
                    cbow_scratch.W_in)

            # ── (B) From-Scratch Skip-gram ────────────────────────────────
            print(f"\n[Scratch] Skip-gram")
            sg_scratch = SkipgramScratch(vocab_size, dim, neg, lr=0.01)
            sg_scratch.train(skipgram_pairs, epochs=EPOCHS_SCRATCH)
            np.save(os.path.join(MODELS_DIR, f"scratch_Skipgram_dim{dim}_w{window}_neg{neg}_Win.npy"),
                    sg_scratch.W_in)

            # ── (C) Gensim CBOW ───────────────────────────────────────────
            print(f"\n[Gensim] CBOW")
            cbow_gensim = Word2Vec(
                sentences   = filtered_sentences,
                vector_size = dim,
                window      = window,
                sg          = 0,        # 0 = CBOW
                negative    = neg,
                hs          = 0,        # use negative sampling, not hierarchical softmax
                min_count   = MIN_COUNT,
                workers     = os.cpu_count(),   # use all available CPU cores
                epochs      = EPOCHS_GENSIM,
                seed        = 42,
            )
            cbow_gensim.save(os.path.join(MODELS_DIR, f"gensim_CBOW_dim{dim}_w{window}_neg{neg}.model"))

            # ── (D) Gensim Skip-gram ──────────────────────────────────────
            print(f"\n[Gensim] Skip-gram")
            sg_gensim = Word2Vec(
                sentences   = filtered_sentences,
                vector_size = dim,
                window      = window,
                sg          = 1,        # 1 = Skip-gram
                negative    = neg,
                hs          = 0,
                min_count   = MIN_COUNT,
                workers     = os.cpu_count(),
                epochs      = EPOCHS_GENSIM,
                seed        = 42,
            )
            sg_gensim.save(os.path.join(MODELS_DIR, f"gensim_Skipgram_dim{dim}_w{window}_neg{neg}.model"))

            results.append({
                "Embed Dim"    : dim,
                "Window"       : window,
                "Neg Samples"  : neg,
                "Params"       : 2 * vocab_size * dim,
                "CBOW Scratch" : avg_sim_scratch(cbow_scratch),
                "SG Scratch"   : avg_sim_scratch(sg_scratch),
                "CBOW Gensim"  : avg_sim_gensim(cbow_gensim),
                "SG Gensim"    : avg_sim_gensim(sg_gensim),
            })

# ─────────────────────────────────────────────────────────────────────────────
# 9. PRINT AND SAVE RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────
header = ["Embed Dim", "Window", "Neg Samples", "Params",
          "CBOW Scratch", "SG Scratch", "CBOW Gensim", "SG Gensim"]

print(f"\n\n{'─'*95}")
print("TASK 2 RESULTS — Avg Cosine Similarity on IITJ Word Pairs")
print(f"{'─'*95}")
print('  '.join(f"{h:<16}" for h in header))
print(f"{'─'*95}")
for row in results:
    print('  '.join(f"{str(row[h]):<16}" for h in header))
print(f"{'─'*95}")

with open(os.path.join(BASE_DIR, "training_results.csv"), "w", newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults saved to : training_results.csv")
print(f"Models saved to  : models/")