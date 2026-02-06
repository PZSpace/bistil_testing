# Quick language detection script
import re

def is_igbo(line):
    # Check for Igbo-specific characters
    igbo_chars = ['ọ', 'ụ', 'ị', 'Ọ', 'Ụ', 'Ị', 'ń', 'ḿ']
    return any(char in line for char in igbo_chars)

with open('./examples/language-modeling/corpora/ibo.txt', 'r') as f:
    lines = f.readlines()

en_lines = []
ig_lines = []

for line in lines:
    line = line.strip()
    if not line:
        continue
    if is_igbo(line):
        ig_lines.append(line)
    else:
        en_lines.append(line)
    
with open('./corpora/ibo.txt', 'w') as f:
    f.write('\n'.join(ig_lines))