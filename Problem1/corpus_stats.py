"""
Print corpus statistics from corpus.txt:
- Total sentences, total tokens, vocabulary size
- Top 20 most frequent words

Run: python corpus_stats.py
"""

import os
from collections import Counter

CORPUS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.txt")

all_tokens = []
total_sentences = 0

with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        tokens = line.strip().split()
        if tokens:
            all_tokens.extend(tokens)
            total_sentences += 1

freq = Counter(all_tokens)

print(f"Total Documents (pages+PDFs scraped) : see iitj_data.json")
print(f"Total Sentences                      : {total_sentences}")
print(f"Total Tokens                         : {len(all_tokens)}")
print(f"Vocabulary Size                      : {len(freq)}")

print(f"\nTop 20 most frequent words:")
for word, count in freq.most_common(20):
    print(f"  {word:<25} {count}")

# Print in submission format
print("\nSubmission format (top 10):")
top10 = freq.most_common(10)
print(", ".join(f"{w}, {c}" for w, c in top10))