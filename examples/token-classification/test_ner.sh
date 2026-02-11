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
  --full_ft_max_epochs_per_iteration 1 \
  --sparse_ft_max_epochs_per_iteration 1 \
  --save_steps 5000 \
  --ft_params_proportion 0.08 \
  --eval_strategy epoch \
  --save_strategy epoch \
  --freeze_layer_norm \
  --learning_rate 5e-5 \
  --metric_for_best_model eval_f1 \
  --load_best_model_at_end \
  --eval_split validation \
  --remove_unused_columns no \
  --save_total_limit 1

# Key test parameters:
# --full_ft_max_epochs_per_iteration 1    # Only 1 epoch full fine-tuning
# --sparse_ft_max_epochs_per_iteration 1  # Only 1 epoch sparse fine-tuning
# Total: 2 epochs = ~3-4 minutes to test compute_metrics fix
# --eval_strategy epoch                   # Will evaluate after each epoch
