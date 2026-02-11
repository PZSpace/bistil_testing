#!/bin/bash
# Quick test script to verify compute_metrics fix

N=0
LANGS=("ibo" "kin" "kin" "lug")
TARGET_LANG=${LANGS[N]}
SOURCE_LANG=en

STUDENT_MODEL="/content/drive/MyDrive/Bistil_Modified_Resources/bistil_outputs/mbert-${SOURCE_LANG}-${TARGET_LANG}-6-layers-trimmed-vocab"
STUDENT_MODEL_TOKENIZER="/content/drive/MyDrive/Bistil_Modified_Resources/bistil_outputs/mbert-${SOURCE_LANG}-${TARGET_LANG}-trimmed-vocab"
TEACHER_MODEL=bert-base-multilingual-cased
LANG_SFT=cambridgeltl/mbert-lang-sft-${SOURCE_LANG}-small
TASK_SFT=cambridgeltl/mbert-task-sft-masakhaner
OUTPUT_DIR="/content/drive/MyDrive/Bistil_Modified_Resources/bistil_outputs2/test-ner-metrics"
mkdir -p $OUTPUT_DIR

rm -f $STUDENT_MODEL/trainer_state.json

python distil_token_classification.py \
  --student_model_name_or_path $STUDENT_MODEL \
  --student_tokenizer_name $STUDENT_MODEL_TOKENIZER \
  --teacher_model_name_or_path $TEACHER_MODEL \
  --lang_ft $LANG_SFT \
  --task_ft $TASK_SFT \
  --dataset_name ner_dataset.py \
  --dataset_config_name $SOURCE_LANG \
  --label_names labels \
  --output_dir $OUTPUT_DIR \
  --do_train \
  --do_eval \
  --label_column_name ner_tags \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --task_name ner \
  --max_seq_length 128 \
  --overwrite_output_dir \
  --overwrite_cache \
  --max_steps 50 \
  --eval_steps 25 \
  --save_steps 1000 \
  --eval_strategy steps \
  --save_strategy steps \
  --freeze_layer_norm \
  --learning_rate 5e-5 \
  --metric_for_best_model eval_f1 \
  --eval_split validation \
  --remove_unused_columns no \
  --save_total_limit 1

# Key test parameters:
# --max_steps 50           # Only train for 50 steps (~1-2 minutes)
# --eval_steps 25          # Evaluate after 25 steps to test compute_metrics
# --eval_strategy steps    # Use step-based eval instead of epoch-based
# --save_total_limit 1     # Keep only 1 checkpoint to save space
