from datasets import load_dataset

def create_corpus(language_code, output_file):
    """
    Create a corpus file from Wikipedia

    Args:
        language_code: ISO language code (e.g., 'en', 'es', 'ar')
        output_file: Output file path
        num_sentences: Number of sentences to extract
    """
    print(f"📥 Downloading {language_code} Wikipedia...")

    dataset = load_dataset("wikimedia/wikipedia", f"{language_code}", split="train")

    print(f"✍️ Writing to {output_file}...")
    count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            # Simple sentence splitting
            sentences = item['text'].split('.')
            for sent in sentences:
                sent = sent.strip()
                if len(sent) > 10:  # Filter very short sentences
                    f.write(sent + '\n')
                    count += 1

    print(f"✅ Created {output_file} with {count} sentences")

# Create English corpus
create_corpus('20231101.en', './corpora/en.txt')

# Create Igbo corpus
# create_corpus('20231101.ig', './corpora/ibo.txt')