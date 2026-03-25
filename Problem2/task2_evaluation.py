"""
Task 2: Quantitative Evaluation
Computes the following metrics for each model's generated names:
  1. Novelty Rate  — percentage of generated names NOT in the training set
  2. Diversity     — number of unique generated names / total generated names

Run  : python task2_evaluation.py
Note : Run task1_train_models.py first to generate the name files.
"""

import os

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "TrainingNames.txt")

# Generated name files from Task 1
GENERATED_FILES = {
    "Vanilla RNN"   : os.path.join(BASE_DIR, "generated_vanilla_rnn.txt"),
    "BLSTM"         : os.path.join(BASE_DIR, "generated_blstm.txt"),
    "RNN+Attention" : os.path.join(BASE_DIR, "generated_rnn_attention.txt"),
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD TRAINING NAMES
# Stored as a lowercase set for fast O(1) membership lookup.
# ─────────────────────────────────────────────────────────────────────────────
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    training_names = set(line.strip().lower() for line in f if line.strip())

print(f"Training set size: {len(training_names)} names\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2. EVALUATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def novelty_rate(generated_names, training_names):
    """
    Novelty Rate = number of generated names NOT in training set
                   ─────────────────────────────────────────────
                        total number of generated names

    A high novelty rate means the model is generating new names
    rather than memorizing and reproducing training examples.
    Range: 0.0 (all names seen before) to 1.0 (all names are new).
    """
    novel = sum(1 for name in generated_names if name.lower() not in training_names)
    return novel / len(generated_names) if generated_names else 0.0


def diversity(generated_names):
    """
    Diversity = number of unique generated names
                ─────────────────────────────────
                total number of generated names

    A high diversity means the model produces varied names rather
    than repeating the same few names over and over.
    Range: 0.0 (all names identical) to 1.0 (all names unique).
    """
    unique = set(name.lower() for name in generated_names)
    return len(unique) / len(generated_names) if generated_names else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. COMPUTE AND PRINT METRICS FOR EACH MODEL
# ─────────────────────────────────────────────────────────────────────────────
print(f"{'Model':<20} {'Total':>8} {'Unique':>8} {'Novelty Rate':>15} {'Diversity':>12}")
print("─" * 70)

results = {}

for model_name, filepath in GENERATED_FILES.items():
    if not os.path.exists(filepath):
        print(f"{model_name:<20} File not found: {filepath}")
        continue

    with open(filepath, 'r', encoding='utf-8') as f:
        generated = [line.strip() for line in f if line.strip()]

    n_rate  = novelty_rate(generated, training_names)
    d_score = diversity(generated)
    unique  = len(set(name.lower() for name in generated))

    results[model_name] = {
        "total"        : len(generated),
        "unique"       : unique,
        "novelty_rate" : n_rate,
        "diversity"    : d_score,
    }

    print(f"{model_name:<20} {len(generated):>8} {unique:>8} {n_rate:>14.2%} {d_score:>11.2%}")

print("─" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 4. SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
if results:
    best_novelty   = max(results, key=lambda m: results[m]["novelty_rate"])
    best_diversity = max(results, key=lambda m: results[m]["diversity"])
    print(f"\nBest Novelty Rate : {best_novelty} ({results[best_novelty]['novelty_rate']:.2%})")
    print(f"Best Diversity    : {best_diversity} ({results[best_diversity]['diversity']:.2%})")

print("\nDone.")
