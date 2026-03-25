"""
Task 3: Semantic Analysis
Loads saved models from Task 2 and performs:
  1. Top-5 nearest neighbors for query words using cosine similarity
  2. Word analogy experiments using vector arithmetic

Run  : python problem3_semantic_analysis.py
Needs: pip install gensim numpy
Note : Run problem2_train_word2vec.py first to generate saved models.
"""

import os
import numpy as np
from collections import Counter
from gensim.models import Word2Vec

# ─────────────────────────────────────────────────────────────────────────────
# 1. REBUILD VOCABULARY FROM CORPUS
# Must match exactly what was built during Task 2 (same MIN_COUNT).
# word2idx is needed to index into the from-scratch W_in matrices.
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CORPUS_FILE = os.path.join(BASE_DIR, "corpus.txt")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
MIN_COUNT   = 2   # must match Task 2 — change this if Task 2 used a different value

sentences = []
with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        tokens = line.strip().split()
        if tokens:
            sentences.append(tokens)

word_freq  = Counter(w for sent in sentences for w in sent)
word_freq  = {w: c for w, c in word_freq.items() if c >= MIN_COUNT}
vocab      = sorted(word_freq.keys())
word2idx   = {w: i for i, w in enumerate(vocab)}
idx2word   = {i: w for w, i in word2idx.items()}
vocab_size = len(vocab)

print(f"Vocabulary size: {vocab_size}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD SAVED MODELS
# Best configuration: dim=100, window=5, neg=5
# This is the standard Word2Vec baseline from the original Mikolov et al. paper.
# ─────────────────────────────────────────────────────────────────────────────
BEST_DIM    = 100
BEST_WINDOW = 5
BEST_NEG    = 5

# Load from-scratch W_in matrices saved as .npy files during Task 2
cbow_scratch_W = np.load(
    os.path.join(MODELS_DIR, f"scratch_CBOW_dim{BEST_DIM}_w{BEST_WINDOW}_neg{BEST_NEG}_Win.npy"))
sg_scratch_W   = np.load(
    os.path.join(MODELS_DIR, f"scratch_Skipgram_dim{BEST_DIM}_w{BEST_WINDOW}_neg{BEST_NEG}_Win.npy"))

# Load Gensim models
cbow_gensim = Word2Vec.load(
    os.path.join(MODELS_DIR, f"gensim_CBOW_dim{BEST_DIM}_w{BEST_WINDOW}_neg{BEST_NEG}.model"))
sg_gensim   = Word2Vec.load(
    os.path.join(MODELS_DIR, f"gensim_Skipgram_dim{BEST_DIM}_w{BEST_WINDOW}_neg{BEST_NEG}.model"))

print("All models loaded.\n")

# ─────────────────────────────────────────────────────────────────────────────
# 3. COSINE SIMILARITY HELPERS FOR FROM-SCRATCH MODELS
# Cosine similarity measures the angle between two vectors:
#   sim(a, b) = (a · b) / (||a|| * ||b||)
# Range: -1 (opposite directions) to +1 (identical direction).
# Words with similar meanings should have vectors pointing in similar directions,
# resulting in high cosine similarity scores.
# ─────────────────────────────────────────────────────────────────────────────

def top_k_neighbors_scratch(W, query_word, k=5):
    """
    Find top-k nearest neighbors for a query word using cosine similarity.
    Vectorized approach: normalizes the entire W matrix once, then uses a
    single matrix-vector dot product to compute all similarities at once.
    """
    if query_word not in word2idx:
        return None

    query_idx = word2idx[query_word]
    query_vec = W[query_idx]

    # Normalize all embedding vectors for cosine comparison
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    W_norm = W / norms

    query_norm   = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    similarities = W_norm @ query_norm   # shape: (vocab_size,)

    similarities[query_idx] = -1.0       # exclude the query word itself
    top_k_idx = np.argsort(similarities)[::-1][:k]
    return [(idx2word[i], float(similarities[i])) for i in top_k_idx]


def analogy_scratch(W, pos1, pos2, neg, topn=3):
    """
    Word analogy via vector arithmetic: pos1 - neg + pos2 ≈ answer.
    Example: 'chemistry' - 'mechanical' + 'engineering' ≈ 'physics'

    Steps:
      1. Compute result vector = W[pos1] - W[neg] + W[pos2]
      2. Find vocabulary words with highest cosine similarity to result vector
      3. Exclude the three input words from results
    """
    missing = [w for w in [pos1, pos2, neg] if w not in word2idx]
    if missing:
        return None

    result_vec = W[word2idx[pos1]] - W[word2idx[neg]] + W[word2idx[pos2]]

    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    W_norm = W / norms

    result_norm  = result_vec / (np.linalg.norm(result_vec) + 1e-10)
    similarities = W_norm @ result_norm

    for w in [pos1, pos2, neg]:
        similarities[word2idx[w]] = -1.0

    top_idx = np.argsort(similarities)[::-1][:topn]
    return [(idx2word[i], float(similarities[i])) for i in top_idx]


def analogy_gensim(model, pos1, pos2, neg, topn=3):
    """Gensim word analogy: pos1 - neg + pos2 ≈ answer."""
    missing = [w for w in [pos1, pos2, neg] if w not in model.wv]
    if missing:
        return None
    return model.wv.most_similar(positive=[pos1, pos2], negative=[neg], topn=topn)


# ─────────────────────────────────────────────────────────────────────────────
# 4. TOP-5 NEAREST NEIGHBORS
# Query words as specified in the assignment.
# ─────────────────────────────────────────────────────────────────────────────
QUERY_WORDS = ["research", "student", "phd", "exam"]

def print_neighbors(label, get_fn):
    """Print top-5 nearest neighbors for all query words."""
    print(f"\n{'='*60}")
    print(f"  Top-5 Nearest Neighbors — {label}")
    print(f"{'='*60}")
    for word in QUERY_WORDS:
        result = get_fn(word)
        if result is None:
            print(f"\n  '{word}': NOT IN VOCABULARY")
            continue
        print(f"\n  '{word}':")
        for rank, (neighbor, score) in enumerate(result, 1):
            print(f"    {rank}. {neighbor:<20} cosine similarity: {score:.4f}")

print_neighbors("CBOW (From Scratch)",
                lambda w: top_k_neighbors_scratch(cbow_scratch_W, w))
print_neighbors("Skip-gram (From Scratch)",
                lambda w: top_k_neighbors_scratch(sg_scratch_W, w))
print_neighbors("CBOW (Gensim)",
                lambda w: cbow_gensim.wv.most_similar(w, topn=5) if w in cbow_gensim.wv else None)
print_neighbors("Skip-gram (Gensim)",
                lambda w: sg_gensim.wv.most_similar(w, topn=5) if w in sg_gensim.wv else None)

# ─────────────────────────────────────────────────────────────────────────────
# 5. ANALOGY EXPERIMENTS
# Three IITJ-relevant analogies using vector arithmetic: a - b + c ≈ d
#
# (1) UG : BTech :: PG : ?
#     Skipped if 'ug'/'pg'/'btech' not in vocabulary (appeared < MIN_COUNT times)
#
# (2) professor : research :: student : ?
#     student - professor + research ≈ thesis/project
#     (student's equivalent of a professor's primary activity)
#
# (3) mechanical : engineering :: chemistry : ?
#     chemistry - mechanical + engineering ≈ physics/science
#     (domain-level concept for chemistry analogous to engineering for mechanical)
#
# (4) research : development :: exam : ?
#     exam - research + development ≈ interview/assessment
#     Added since analogy (1) is skipped due to OOV words.
# ─────────────────────────────────────────────────────────────────────────────
ANALOGIES = [
    # (pos1,       pos2,          neg,          description)
    ("pg",        "btech",       "ug",          "UG : BTech :: PG : ?"),
    ("student",   "research",    "professor",   "professor : research :: student : ?"),
    ("chemistry", "engineering", "mechanical",  "mechanical : engineering :: chemistry : ?"),
    ("exam",      "development", "research",    "research : development :: exam : ?"),
]

def print_analogies(label, get_fn):
    """Print analogy results for all experiments."""
    print(f"\n{'='*60}")
    print(f"  Analogy Experiments — {label}")
    print(f"{'='*60}")
    for pos1, pos2, neg, description in ANALOGIES:
        print(f"\n  {description}")
        result = get_fn(pos1, pos2, neg)
        if result is None:
            print(f"    Skipped — one or more words not in vocabulary")
            continue
        for rank, (word, score) in enumerate(result, 1):
            print(f"    {rank}. {word:<20} cosine similarity: {score:.4f}")

print_analogies("CBOW (From Scratch)",
                lambda p1, p2, n: analogy_scratch(cbow_scratch_W, p1, p2, n))
print_analogies("Skip-gram (From Scratch)",
                lambda p1, p2, n: analogy_scratch(sg_scratch_W, p1, p2, n))
print_analogies("CBOW (Gensim)",
                lambda p1, p2, n: analogy_gensim(cbow_gensim, p1, p2, n))
print_analogies("Skip-gram (Gensim)",
                lambda p1, p2, n: analogy_gensim(sg_gensim, p1, p2, n))

print("\nDone.")