#!/bin/bash
# Run all three evaluations sequentially and compare results
# Usage: bash eval_all_compare.sh

echo "=========================================="
echo "Evaluating all 3 models on Igbo NER"
echo "=========================================="

echo ""
echo "1/3: Evaluating BASE MODEL (mBERT 12-layer)..."
bash eval_1_base_model.sh

echo ""
echo "2/3: Evaluating TRIMMED VOCAB MODEL (6-layer student)..."
bash eval_2_trimmed_vocab.sh

echo ""
echo "3/3: Evaluating DISTILLED MODEL (NER-trained)..."
bash eval_3_distilled_model.sh

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="
echo ""
echo "Results locations:"
echo "  Base model:       /content/drive/MyDrive/Bistil_Modified_Resources/eval_results/base_model/eval_results.json"
echo "  Trimmed vocab:    /content/drive/MyDrive/Bistil_Modified_Resources/eval_results/trimmed_vocab_model/eval_results.json"
echo "  Distilled model:  /content/drive/MyDrive/Bistil_Modified_Resources/eval_results/distilled_model/eval_results.json"
echo ""
echo "To compare results:"
echo "  cat /content/drive/MyDrive/Bistil_Modified_Resources/eval_results/*/eval_results.json"
