#!/bin/bash
LANG=ibo # Hausa
#PROP=$2
#LANG_FT=cambridgeltl/mbert-lang-sft-${LANG}-small
#TASK_FT=cambridgeltl/mbert-task-sft-masakhaner
#BASE_MODEL=$HDD/experiments/distil/ner/distil/mbert-en-${LANG}-6-layers-trimmed-vocab-density-0.5-distilled
BASE_MODEL=/content/drive/MyDrive/Bistil_Modified_Resources/bistil_outputs2/mbert-en-${LANG}-6-layers-trimmed-vocab-distilled
#TOKENIZER=$HDD/experiments/distil/language-modeling/distil/mbert-en-${LANG}-trimmed-vocab
TOKENIZER=/content/drive/MyDrive/Bistil_Modified_Resources/bistil_outputs/mbert-en-${LANG}-trimmed-vocab
#TASK_FT=$HDD/experiments/distil/ner/mbert-${LANG}-en-${PROP}-8p
#TASK_FT=models/ner/en

python run_token_classification.py \
  --model_name_or_path $BASE_MODEL \
  --tokenizer_name $TOKENIZER \
  --dataset_name ner_dataset.py \
  --dataset_config_name $LANG \
  --output_dir results/ner/${LANG} \
  --do_eval \
  --label_column_name ner_tags \
  --per_device_eval_batch_size 4 \
  --task_name ner \
  --overwrite_output_dir \
  --eval_split test
  #--lang_ft $LANG_FT \
  #--task_ft $TASK_FT \
