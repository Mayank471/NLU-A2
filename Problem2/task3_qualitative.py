"""
Task 3: Qualitative Analysis
Analyzes generated names for:
  1. Realism — do the names sound like plausible Indian names?
  2. Common failure modes — what goes wrong?
  3. Representative samples from each model

Run  : python task3_qualitative.py
Note : Run task1_train_models.py first to generate the name files.
"""

import os
import re
from collections import Counter

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "TrainingNames.txt")

GENERATED_FILES = {
    "Vanilla RNN"   : os.path.join(BASE_DIR, "generated_vanilla_rnn.txt"),
    "BLSTM"         : os.path.join(BASE_DIR, "generated_blstm.txt"),
    "RNN+Attention" : os.path.join(BASE_DIR, "generated_rnn_attention.txt"),
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    training_names = [line.strip().lower() for line in f if line.strip()]

training_set = set(training_names)

# ─────────────────────────────────────────────────────────────────────────────
# 2. REALISM HEURISTICS
# Indian names can be single names or first+last (with a space).
# We approximate realism using these rules:
#   - Reasonable length (2–20 characters including space)
#   - Only alphabetic characters and spaces (no digits or symbols)
#   - No special tokens (< or >) leaked into output
#   - Not a single repeated character (e.g., "aaaa")
# True realism requires human evaluation — these are proxies only.
# ─────────────────────────────────────────────────────────────────────────────
def is_realistic(name):
    """
    Heuristic check for whether a name looks like a plausible Indian name.
    Allows spaces to accommodate first+last name format (e.g., "Rahul Singh").
    """
    if not name or len(name) < 2 or len(name) > 20:
        return False   # too short or too long
    if not re.match(r'^[a-zA-Z ]+$', name):
        return False   # contains digits, symbols, or special tokens
    if name.startswith(' ') or name.endswith(' '):
        return False   # leading or trailing space
    if '  ' in name:
        return False   # double space
    if len(set(name.lower().replace(' ', ''))) == 1:
        return False   # all same character e.g. "aaaa"
    return True


def realism_rate(names):
    """Percentage of generated names passing the realism heuristic."""
    realistic = sum(1 for n in names if is_realistic(n))
    return realistic / len(names) if names else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. FAILURE MODE DETECTION
# Common failure modes in character-level RNN name generation:
#   - Too short  : less than 2 characters
#   - Too long   : more than 20 characters
#   - Repetition : same character repeated (e.g., "aaaaaa")
#   - Bad chars  : contains special tokens or non-alphabetic characters
#   - Duplicates : same name generated multiple times (low diversity)
# ─────────────────────────────────────────────────────────────────────────────
def detect_failure_modes(names):
    """Categorize generated names by their failure mode."""
    failures = {
        "too_short"  : 0,
        "too_long"   : 0,
        "repetition" : 0,
        "bad_chars"  : 0,   # special tokens or non-alpha/space characters
        "duplicates" : 0,
    }
    name_counts = Counter(n.lower() for n in names)

    for name in names:
        if not name or len(name) < 2:
            failures["too_short"] += 1
        elif len(name) > 20:
            failures["too_long"] += 1
        elif len(set(name.lower().replace(' ', ''))) == 1:
            failures["repetition"] += 1
        elif not re.match(r'^[a-zA-Z ]+$', name):
            failures["bad_chars"] += 1

    failures["duplicates"] = sum(1 for count in name_counts.values() if count > 1)
    return failures


# ─────────────────────────────────────────────────────────────────────────────
# 4. PRINT ANALYSIS FOR EACH MODEL
# ─────────────────────────────────────────────────────────────────────────────
for model_name, filepath in GENERATED_FILES.items():
    if not os.path.exists(filepath):
        print(f"{model_name}: File not found.\n")
        continue

    with open(filepath, 'r', encoding='utf-8') as f:
        generated = [line.strip() for line in f if line.strip()]

    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")

    # ── Representative samples ──────────────────────────────────────────────
    print(f"\n  Representative Samples (20 of {len(generated)}):")
    samples = generated[:10] + generated[-10:]
    for i, name in enumerate(samples, 1):
        print(f"    {i:>2}. {name.title()}")

    # ── Realism ─────────────────────────────────────────────────────────────
    r_rate = realism_rate(generated)
    print(f"\n  Realism Rate (heuristic): {r_rate:.2%}")
    print(f"  (Names passing length, alphabetic+space, non-repetition checks)")

    # ── Failure modes ────────────────────────────────────────────────────────
    failures = detect_failure_modes(generated)
    print(f"\n  Failure Modes:")
    for mode, count in failures.items():
        pct = count / len(generated) * 100
        print(f"    {mode:<15} : {count:>4} ({pct:.1f}%)")

    # ── Length distribution ──────────────────────────────────────────────────
    lengths = [len(n) for n in generated]
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    print(f"\n  Name Length: min={min(lengths)}, max={max(lengths)}, avg={avg_len:.1f}")

    # ── Most common generated names ──────────────────────────────────────────
    name_freq = Counter(n.lower() for n in generated)
    print(f"\n  Most Frequently Generated:")
    for name, count in name_freq.most_common(5):
        print(f"    '{name.title()}' — {count} times")

print("\nDone.")