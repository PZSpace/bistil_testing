#!/bin/bash
# Evaluate original mBERT base model on Igbo NER
# This is the baseline - full 12-layer model with full vocabulary

LANG=ibo
BASE_MODEL=bert-base-multilingual-cased
TOKENIZER=bert-base-multilingual-cased
OUTPUT_DIR=/content/drive/MyDrive/Bistil_Modified_Resources/eval_results/base_model

python run_token_classification.py \
  --model_name_or_path $BASE_MODEL \
  --tokenizer_name $TOKENIZER \
  --dataset_name ner_dataset.py \
  --dataset_config_name $LANG \
  --output_dir $OUTPUT_DIR \
  --do_eval \
  --label_column_name ner_tags \
  --per_device_eval_batch_size 4 \
  --task_name ner \
  --overwrite_output_dir \
  --eval_split test \
  --max_seq_length 128

echo "Base model evaluation complete! Results saved to: $OUTPUT_DIR/eval_results.json"
