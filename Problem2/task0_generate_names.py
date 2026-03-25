"""
Task 0: Dataset Generation
Uses the Anthropic API to generate 1000 Indian names and saves to TrainingNames.txt.

Run  : python task0_generate_names.py
Needs: pip install anthropic
"""

import anthropic
import os
import re

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TrainingNames.txt")

# ─────────────────────────────────────────────────────────────────────────────
# Generate names in batches using the Anthropic API.
# We request batches of 100 names at a time to stay within token limits.
# Names cover both male and female, across major Indian regions and religions.
# ─────────────────────────────────────────────────────────────────────────────
client = anthropic.Anthropic()

all_names = []
BATCH_SIZE = 100
NUM_BATCHES = 10   # 10 x 100 = 1000 names total

for batch in range(NUM_BATCHES):
    print(f"Generating batch {batch+1}/{NUM_BATCHES}...")

    message = client.messages.create(
        model      = "claude-sonnet-4-20250514",
        max_tokens = 1024,
        messages   = [
            {
                "role"   : "user",
                "content": (
                    f"Generate exactly {BATCH_SIZE} unique Indian names (mix of male and female, "
                    f"covering North, South, East and West India, different religions). "
                    f"Return ONLY the names, one per line, no numbering, no extra text. "
                    f"Do not repeat names from this list if any: {all_names[-50:] if all_names else []}"
                )
            }
        ]
    )

    # Extract names from the response — one name per line
    raw = message.content[0].text.strip()
    batch_names = [
        # Strip any numbering or punctuation that might appear despite instructions
        re.sub(r'^[\d\.\)\-\s]+', '', line).strip()
        for line in raw.splitlines()
        if line.strip()
    ]
    all_names.extend(batch_names)
    print(f"  Got {len(batch_names)} names. Total so far: {len(all_names)}")

# ─────────────────────────────────────────────────────────────────────────────
# Deduplicate while preserving order, then trim to exactly 1000.
# ─────────────────────────────────────────────────────────────────────────────
seen = set()
unique_names = []
for name in all_names:
    name_clean = name.strip()
    if name_clean and name_clean.lower() not in seen:
        seen.add(name_clean.lower())
        unique_names.append(name_clean)

unique_names = unique_names[:1000]

# Save to file — one name per line
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for name in unique_names:
        f.write(name + '\n')

print(f"\nSaved {len(unique_names)} unique Indian names to: {OUTPUT_FILE}")
