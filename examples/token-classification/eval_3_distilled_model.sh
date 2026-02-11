#!/bin/bash
# Evaluate final distilled NER model on Igbo NER
# This is after Step 2: Task-SFT distilled model for NER

LANG=ibo
DISTILLED_MODEL=/content/drive/MyDrive/Bistil_Modified_Resources/bistil_outputs2/mbert-en-${LANG}-6-layers-trimmed-vocab-distilled
TOKENIZER=/content/drive/MyDrive/Bistil_Modified_Resources/bistil_outputs/mbert-en-${LANG}-trimmed-vocab
OUTPUT_DIR=/content/drive/MyDrive/Bistil_Modified_Resources/eval_results/distilled_model

python run_token_classification.py \
  --model_name_or_path $DISTILLED_MODEL \
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

echo "Distilled model evaluation complete! Results saved to: $OUTPUT_DIR/eval_results.json"
