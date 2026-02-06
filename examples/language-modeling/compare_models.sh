#!/bin/bash
# Script to compare base model vs distilled model
# This evaluates multiple models and saves results for comparison

# Configuration
SOURCE_LANG=en
TARGET_LANG=ibo
CORPUS_DIR="./corpora"  # Must match training script
RESULTS_DIR="./evaluation_results"

# Create results directory
mkdir -p ${RESULTS_DIR}

echo "=================================================="
echo "Model Comparison Script"
echo "=================================================="
echo ""

# Models to compare
# Uncomment/modify based on your setup

# 1. Base mBERT model (original)
echo "1/3 Evaluating base mBERT model..."
python evaluate_mlm.py \
  --model_name_or_path bert-base-multilingual-cased \
  --source_file "${CORPUS_DIR}/${SOURCE_LANG}.txt" \
  --target_file "${CORPUS_DIR}/${TARGET_LANG}.txt" \
  --validation_split_percentage 5 \
  --max_seq_length 256 \
  --batch_size 8 \
  --output_file "${RESULTS_DIR}/base_mbert_results.json"

echo ""
echo "2/3 Evaluating vocab-trimmed mBERT..."
# 2. Vocabulary-trimmed model (after trim_vocab.py)
# Note: Tokenizer is included in the model directory, no need to specify separately
python evaluate_mlm.py \
  --model_name_or_path /content/bistil_outputs/mbert-${SOURCE_LANG}-${TARGET_LANG}-trimmed-vocab \
  --source_file "${CORPUS_DIR}/${SOURCE_LANG}.txt" \
  --target_file "${CORPUS_DIR}/${TARGET_LANG}.txt" \
  --validation_split_percentage 5 \
  --max_seq_length 256 \
  --batch_size 8 \
  --output_file "${RESULTS_DIR}/vocab_trimmed_results.json"

echo ""
echo "3/3 Evaluating distilled student model..."
# 3. Distilled student model (final output)
# NOTE: Explicitly using vocab-trimmed tokenizer (27,132 tokens) to ensure correct vocabulary
#       If your model was trained with the updated distil_mlm.py, the tokenizer is already saved
#       with the model, but we specify it here for safety and backwards compatibility
python evaluate_mlm.py \
  --model_name_or_path /content/bistil_outputs/mbert-${SOURCE_LANG}-${TARGET_LANG}-6-layers-trimmed-vocab \
  --tokenizer_name_or_path /content/bistil_outputs/mbert-${SOURCE_LANG}-${TARGET_LANG}-trimmed-vocab \
  --source_file "${CORPUS_DIR}/${SOURCE_LANG}.txt" \
  --target_file "${CORPUS_DIR}/${TARGET_LANG}.txt" \
  --validation_split_percentage 5 \
  --max_seq_length 256 \
  --batch_size 8 \
  --output_file "${RESULTS_DIR}/distilled_student_results.json"

echo ""
echo "=================================================="
echo "Evaluation Complete!"
echo "=================================================="
echo "Results saved to: ${RESULTS_DIR}/"
echo ""
echo "Compare results with:"
echo "  python compare_results.py ${RESULTS_DIR}/*.json"
echo ""
