#!/usr/bin/env python3
"""
Sample top 1M sentences from en.txt
"""

INPUT_FILE = './corpora/en.txt'
OUTPUT_FILE = './examples/language-modeling/corpora/en.txt'
NUM_SENTENCES = 2_000_000

print(f"📖 Reading first {NUM_SENTENCES:,} sentences from {INPUT_FILE}...")

with open(INPUT_FILE, 'r', encoding='utf-8') as f_in:
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for i, line in enumerate(f_in):
            if i >= NUM_SENTENCES:
                break
            f_out.write(line)

print(f"✅ Created {OUTPUT_FILE} with {NUM_SENTENCES:,} sentences")
