#!/bin/bash
# Colab-compatible version of trim_and_distil.sh
# This script is optimized for Google Colab environment

# Set language (change N to select different language from array)
N=2
TARGET_LANGS=("mt" "fo" "ibo" "kin")
TARGET_LANG=${TARGET_LANGS[N]}
VALIDATION_SPLIT_PERCENTAGES=(5 5 5 5 5 5 5 5)
VALIDATION_SPLIT_PERCENTAGE=${VALIDATION_SPLIT_PERCENTAGES[N]}
SOURCE_LANG=en

# Colab directory structure
# Use /content/drive/MyDrive if you want to save to Google Drive (recommended)
# Or use /content for temporary storage (will be deleted after session)
BASE_DIR="/content/drive/MyDrive/Bistil_Modified_Resources"  # Change to "/content/drive/MyDrive" to save to Google Drive

# Create necessary directories
CORPUS_DIR="${BASE_DIR}/data"
OUTPUT_BASE="${BASE_DIR}/bistil_outputs"
VOCAB_DIR="${OUTPUT_BASE}/mbert-${SOURCE_LANG}-${TARGET_LANG}-trimmed-vocab"
FINAL_DIR="${OUTPUT_BASE}/mbert-${SOURCE_LANG}-${TARGET_LANG}-6-layers-trimmed-vocab"

mkdir -p ${CORPUS_DIR}
mkdir -p ${VOCAB_DIR}
mkdir -p ${FINAL_DIR}

# Check if corpus files exist
if [ ! -f "${CORPUS_DIR}/${SOURCE_LANG}.txt" ]; then
    echo "❌ Error: Missing corpus file: ${CORPUS_DIR}/${SOURCE_LANG}.txt"
    echo "Please upload your corpus files to ${CORPUS_DIR}/"
    exit 1
fi

if [ ! -f "${CORPUS_DIR}/${TARGET_LANG}.txt" ]; then
    echo "❌ Error: Missing corpus file: ${CORPUS_DIR}/${TARGET_LANG}.txt"
    echo "Please upload your corpus files to ${CORPUS_DIR}/"
    exit 1
fi

echo "✅ Corpus files found"
echo "📁 Source corpus: ${CORPUS_DIR}/${SOURCE_LANG}.txt"
echo "📁 Target corpus: ${CORPUS_DIR}/${TARGET_LANG}.txt"
echo ""

# ========================================
# Step 1: Vocabulary Trimming
# ========================================
echo "🔧 Step 1/2: Trimming vocabulary..."
echo "Output: ${VOCAB_DIR}"

python trim_vocab.py \
  --model_name_or_path bert-base-multilingual-cased \
  --source_file "${CORPUS_DIR}/${SOURCE_LANG}.txt" \
  --target_file "${CORPUS_DIR}/${TARGET_LANG}.txt" \
  --output_dir ${VOCAB_DIR} \
  --overwrite_output_dir \
  --preprocessing_num_workers 2

if [[ $? != 0 ]]; then
    echo "❌ Vocabulary trimming failed!"
    exit 1
fi

# Remove tokenizer.json for mBERT
rm -f "${VOCAB_DIR}/tokenizer.json"
echo "✅ Step 1 complete: Vocabulary trimmed"
echo ""

# ========================================
# Sampling English data for distillation
# ========================================
echo "📊 Sampling 1M English sentences for distillation..."
SOURCE_SAMPLED="${CORPUS_DIR}/${SOURCE_LANG}_1m.txt"
head -n 1000000 "${CORPUS_DIR}/${SOURCE_LANG}.txt" > "${SOURCE_SAMPLED}"
echo "✅ Created ${SOURCE_SAMPLED} with 1M sentences"
echo ""

# ========================================
# Step 2: Knowledge Distillation
# ========================================
echo "🔧 Step 2/2: Knowledge distillation..."

SOURCE_SFT=cambridgeltl/mbert-lang-sft-${SOURCE_LANG}-small
TARGET_SFT=cambridgeltl/mbert-lang-sft-${TARGET_LANG}-small

echo "📚 Source SFT: ${SOURCE_SFT}"
echo "📚 Target SFT: ${TARGET_SFT}"
echo "📁 Output: ${FINAL_DIR}"
echo ""

python distil_mlm.py \
  --model_name_or_path ${VOCAB_DIR} \
  --sft_tokenizer bert-base-multilingual-cased \
  --source_file "${SOURCE_SAMPLED}" \
  --source_sft ${SOURCE_SFT} \
  --target_file "${CORPUS_DIR}/${TARGET_LANG}.txt" \
  --target_sft ${TARGET_SFT} \
  --layer_reduction_factor 2 \
  --hidden_size_reduction_factor 1 \
  --output_dir ${FINAL_DIR} \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --max_seq_length 256 \
  --save_steps 1000 \
  --max_steps 50000 \
  --overwrite_output_dir \
  --learning_rate 1e-4 \
  --eval_strategy steps \
  --eval_steps 1000 \
  --logging_steps 500 \
  --validation_split_percentage ${VALIDATION_SPLIT_PERCENTAGE} \
  --preprocessing_num_workers 2 \
  --remove_unused_columns no \
  --save_total_limit 2

if [[ $? != 0 ]]; then
    echo "❌ Distillation failed!"
    exit 1
fi

echo ""
echo "✅ ✅ ✅ Training complete! ✅ ✅ ✅"
echo ""
echo "📁 Final model saved to: ${FINAL_DIR}"
echo "🎉 You can now use this bilingual ${SOURCE_LANG}-${TARGET_LANG} model for downstream tasks!"
