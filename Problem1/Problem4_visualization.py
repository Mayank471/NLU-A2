"""
Task 4: Visualization of Word Embeddings
Loads saved models from Task 2 and projects word embeddings into 2D using
PCA and t-SNE. Visualizes semantic clusters and compares CBOW vs Skip-gram
for both from-scratch and Gensim implementations.

Run  : python problem4_visualization.py
Needs: pip install gensim numpy scikit-learn matplotlib
Note : Run problem2_train_word2vec.py first to generate saved models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
vocab_size = len(vocab)

print(f"Vocabulary size: {vocab_size}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD SAVED MODELS
# Best configuration: dim=100, window=5, neg=5
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
# 3. WORD CLUSTERS TO VISUALIZE
# Words grouped by semantic theme to test whether the learned embeddings
# place semantically related words near each other in the 2D projected space.
# A good embedding model should show words from the same cluster grouped
# together, forming visually distinct regions in the plot.
# ─────────────────────────────────────────────────────────────────────────────
WORD_CLUSTERS = {
    "Academic Roles" : ["professor", "assistant", "faculty", "director", "registrar", "dean"],
    "Programs"       : ["btech", "mtech", "phd", "mba", "degree", "program"],
    "Departments"    : ["mechanical", "electrical", "chemistry", "physics", "mathematics", "computer"],
    "Research"       : ["research", "publication", "thesis", "laboratory", "project", "development"],
    "Student Life"   : ["student", "campus", "hostel", "exam", "course", "semester"],
}

# One distinct color per cluster for visual clarity
COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]


def get_vectors_scratch(W):
    """Fetch embedding vectors from the from-scratch W_in matrix for cluster words."""
    words, vectors, colors = [], [], []
    for color, (_, word_list) in zip(COLORS, WORD_CLUSTERS.items()):
        for word in word_list:
            if word in word2idx:
                words.append(word)
                vectors.append(W[word2idx[word]])   # fetch row from W_in matrix
                colors.append(color)
    return words, np.array(vectors), colors


def get_vectors_gensim(model):
    """Fetch embedding vectors from a Gensim model's word vector lookup."""
    words, vectors, colors = [], [], []
    for color, (_, word_list) in zip(COLORS, WORD_CLUSTERS.items()):
        for word in word_list:
            if word in model.wv:
                words.append(word)
                vectors.append(model.wv[word])      # fetch from Gensim keyed vectors
                colors.append(color)
    return words, np.array(vectors), colors


def plot_embeddings(coords_2d, words, colors, title, output_path):
    """
    Plot 2D projected word embeddings with cluster coloring.
    Each word is shown as a labeled dot colored by its semantic cluster.
    Good embeddings should show words from the same cluster grouped together.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    for i, (word, color) in enumerate(zip(words, colors)):
        x, y = coords_2d[i]
        ax.scatter(x, y, color=color, s=80, zorder=2)
        # Slight offset so label does not overlap the dot
        ax.annotate(word, (x, y), fontsize=9, xytext=(5, 3), textcoords='offset points')

    legend_patches = [
        mpatches.Patch(color=color, label=name)
        for color, name in zip(COLORS, WORD_CLUSTERS.keys())
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. PCA PROJECTIONS
# PCA (Principal Component Analysis) is a linear dimensionality reduction.
# It finds the 2 directions of maximum variance in the embedding space and
# projects all vectors onto those axes. Fast and deterministic.
# The explained variance ratio tells how much information is retained in 2D.
# ─────────────────────────────────────────────────────────────────────────────
print("Generating PCA visualizations...")

models_to_plot = [
    ("CBOW (Scratch)",      *get_vectors_scratch(cbow_scratch_W), "pca_cbow_scratch.png"),
    ("Skip-gram (Scratch)", *get_vectors_scratch(sg_scratch_W),   "pca_sg_scratch.png"),
    ("CBOW (Gensim)",       *get_vectors_gensim(cbow_gensim),     "pca_cbow_gensim.png"),
    ("Skip-gram (Gensim)",  *get_vectors_gensim(sg_gensim),       "pca_sg_gensim.png"),
]

for label, words, vectors, colors, filename in models_to_plot:
    if len(vectors) < 2:
        print(f"  Skipping {label} — not enough words in vocabulary.")
        continue
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(vectors)
    var    = pca.explained_variance_ratio_
    print(f"  {label} — Variance explained: PC1={var[0]:.2%}, PC2={var[1]:.2%}")
    plot_embeddings(coords, words, colors,
                    title=f"PCA — {label}",
                    output_path=os.path.join(BASE_DIR, filename))

# ─────────────────────────────────────────────────────────────────────────────
# 5. t-SNE PROJECTIONS
# t-SNE (t-Distributed Stochastic Neighbor Embedding) is a non-linear method.
# It preserves local neighborhood structure — words that are close in the
# high-dimensional embedding space stay close in 2D. Better than PCA for
# revealing tight semantic clusters but is non-deterministic.
# perplexity ≈ effective number of neighbors; must be < number of data points.
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating t-SNE visualizations...")

tsne_models = [
    ("CBOW (Scratch)",      *get_vectors_scratch(cbow_scratch_W), "tsne_cbow_scratch.png"),
    ("Skip-gram (Scratch)", *get_vectors_scratch(sg_scratch_W),   "tsne_sg_scratch.png"),
    ("CBOW (Gensim)",       *get_vectors_gensim(cbow_gensim),     "tsne_cbow_gensim.png"),
    ("Skip-gram (Gensim)",  *get_vectors_gensim(sg_gensim),       "tsne_sg_gensim.png"),
]

for label, words, vectors, colors, filename in tsne_models:
    if len(vectors) < 2:
        print(f"  Skipping {label} — not enough words in vocabulary.")
        continue
    perplexity = min(15, len(vectors) - 1)  # perplexity must be < number of samples
    tsne   = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(vectors)
    plot_embeddings(coords, words, colors,
                    title=f"t-SNE — {label}",
                    output_path=os.path.join(BASE_DIR, filename))

print("\nAll visualizations saved.")
print("Files: pca_*.png, tsne_*.png")