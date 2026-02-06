# Running trim_and_distil on Google Colab

This guide shows you how to run the bilingual model training on Google Colab.

## 🚀 Quick Start

### 1. Open Google Colab
Go to [colab.research.google.com](https://colab.research.google.com)

### 2. Enable GPU
- Click **Runtime** → **Change runtime type**
- Select **T4 GPU** or **A100 GPU** (if available)
- Click **Save**

### 3. Mount Google Drive (Optional but Recommended)
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 4. Clone the Repository
```bash
!git clone https://github.com/AlanAnsell/bistil.git
%cd bistil
```

### 5. Install Dependencies
```bash
!pip install -q torch
!pip install -q -e .
```

### 6. Prepare Your Corpus Files

**Option A: Upload directly to Colab (temporary)**
```python
import os
os.makedirs('/content/corpora', exist_ok=True)

from google.colab import files
print("Upload en.txt:")
uploaded = files.upload()
!mv en.txt /content/corpora/

print("Upload your target language corpus (e.g., mt.txt):")
uploaded = files.upload()
!mv *.txt /content/corpora/
```

**Option B: Use Google Drive (permanent)**
```bash
# First, upload your corpus files to Google Drive: MyDrive/corpora/
# Then run:
!mkdir -p /content/drive/MyDrive/corpora
# Upload en.txt and target_lang.txt to that folder via Drive UI
```

### 7. Edit Configuration (if needed)
```bash
# Open the script
%cd examples/language-modeling

# To change the target language, edit line 6 in the script:
# N=0  -> for mt (Maltese)
# N=1  -> for fo (Faroese)
# N=2  -> for ibo (Igbo)
# N=3  -> for kin (Kinyarwanda)

# Or change line 7 to your own language:
# TARGET_LANGS=("es" "ar" "de")  # Spanish, Arabic, German
```

### 8. Run the Training Script
```bash
!bash trim_and_distil_colab.sh
```

---

## 📁 Directory Structure

### Using `/content` (temporary - deleted after session)
```
/content/
├── bistil/                          # Cloned repository
├── corpora/                         # Your corpus files
│   ├── en.txt
│   └── mt.txt
└── bistil_outputs/                  # Training outputs
    ├── mbert-en-mt-trimmed-vocab/   # Step 1 output
    └── mbert-en-mt-6-layers-trimmed-vocab/  # Final model
```

### Using Google Drive (permanent - saved to your Drive)
```
/content/drive/MyDrive/
├── corpora/                         # Your corpus files
│   ├── en.txt
│   └── mt.txt
└── bistil_outputs/                  # Training outputs
    ├── mbert-en-mt-trimmed-vocab/
    └── mbert-en-mt-6-layers-trimmed-vocab/
```

To use Google Drive storage, edit line 14 in `trim_and_distil_colab.sh`:
```bash
BASE_DIR="/content/drive/MyDrive"  # Instead of "/content"
```

---

## ⚙️ Configuration Options

Edit these variables in `trim_and_distil_colab.sh`:

| Variable | Line | Description | Default |
|----------|------|-------------|---------|
| `N` | 6 | Language selection (0-3) | `0` |
| `SOURCE_LANG` | 11 | Source language | `en` |
| `TARGET_LANGS` | 7 | Available target languages | `("mt" "fo" "ibo" "kin")` |
| `BASE_DIR` | 14 | Output directory | `/content` |

---

## 📊 Expected Training Time

On Colab GPU (T4):
- **Step 1 (Vocab trimming)**: ~5-10 minutes
- **Step 2 (Distillation)**: ~24-48 hours (200k steps)

For testing, reduce `max_steps` in line 99:
```bash
--max_steps 1000 \  # Quick test (instead of 200000)
```

---

## ⚠️ Common Issues

### 1. Out of Memory
Reduce batch size in lines 94-95:
```bash
--per_device_train_batch_size 4 \  # Reduced from 8
--per_device_eval_batch_size 4 \
```

### 2. Colab Timeout (12-hour limit on free tier)
- Use **Colab Pro** for longer sessions
- Or save checkpoints more frequently by changing line 98:
```bash
--save_steps 5000 \  # Save every 5k steps instead of 10M
```

Then resume from checkpoint in a new session.

### 3. Teacher SFT Model Not Found
Check if the language has a pre-trained SFT model at:
https://huggingface.co/cambridgeltl

If not available, you'll need to train without SFT (see `bilingual_mlm.sh` instead).

---

## 💾 Downloading Your Trained Model

```python
# Zip the final model
!cd /content/bistil_outputs && zip -r model.zip mbert-en-mt-6-layers-trimmed-vocab/

# Download to your computer
from google.colab import files
files.download('/content/bistil_outputs/model.zip')
```

---

## 🎯 Next Steps

After training completes, use your model for downstream tasks:
- **NER**: See `examples/token-classification/train_ner.sh`
- **Text Classification**: See `examples/text-classification/run_nli.py`
- **Question Answering**: See `examples/question-answering/`

Your model path will be:
```
/content/bistil_outputs/mbert-en-mt-6-layers-trimmed-vocab/
```
