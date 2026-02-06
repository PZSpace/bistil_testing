#!/usr/bin/env python3
"""
Evaluate MLM models on bilingual data
This script evaluates any BERT-like model on masked language modeling tasks
and can be used to compare base models with distilled models.
"""

import argparse
import logging
import math
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class EvalArguments:
    """Arguments for evaluation"""
    
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    source_file: Optional[str] = field(
        default=None,
        metadata={"help": "Source language corpus file (e.g., en.txt)"}
    )
    target_file: Optional[str] = field(
        default=None,
        metadata={"help": "Target language corpus file (e.g., ibo.txt)"}
    )
    validation_split_percentage: int = field(
        default=5,
        metadata={"help": "Percentage of data to use for evaluation"}
    )
    max_seq_length: int = field(
        default=256,
        metadata={"help": "Maximum sequence length"}
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask"}
    )
    batch_size: int = field(
        default=8,
        metadata={"help": "Batch size for evaluation"}
    )
    preprocessing_num_workers: int = field(
        default=2,
        metadata={"help": "Number of workers for preprocessing"}
    )
    output_file: Optional[str] = field(
        default=None,
        metadata={"help": "Output file to save results (default: print to stdout)"}
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate MLM models")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                       help="Path to model to evaluate")
    parser.add_argument("--source_file", type=str, default=None,
                       help="Source language corpus")
    parser.add_argument("--target_file", type=str, default=None,
                       help="Target language corpus")
    parser.add_argument("--validation_split_percentage", type=int, default=5,
                       help="Percentage for validation split")
    parser.add_argument("--max_seq_length", type=int, default=256,
                       help="Maximum sequence length")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                       help="Masking probability")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--preprocessing_num_workers", type=int, default=2,
                       help="Number of preprocessing workers")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file for results")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None,
                       help="Path to tokenizer (if different from model). Use this when model directory doesn't have tokenizer files.")
    
    args = parser.parse_args()
    
    logger.info(f"Evaluating model: {args.model_name_or_path}")
    
    # Load tokenizer and model
    logger.info("Loading tokenizer and model...")
    tokenizer_path = args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
    
    # Model info
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {num_params:,} parameters")
    logger.info(f"Vocab size: {len(tokenizer)}")
    logger.info(f"Hidden layers: {model.config.num_hidden_layers}")
    logger.info(f"Hidden size: {model.config.hidden_size}")
    
    # Load datasets
    logger.info("Loading evaluation data...")
    datasets = {}
    
    if args.source_file:
        logger.info(f"Loading source file: {args.source_file}")
        source_data = load_dataset(
            'text',
            data_files={'train': args.source_file},
            split=f'train[:{args.validation_split_percentage}%]',
        )
        datasets['source'] = source_data
        logger.info(f"Source: {len(source_data)} examples")
    
    if args.target_file:
        logger.info(f"Loading target file: {args.target_file}")
        target_data = load_dataset(
            'text',
            data_files={'train': args.target_file},
            split=f'train[:{args.validation_split_percentage}%]',
        )
        datasets['target'] = target_data
        logger.info(f"Target: {len(target_data)} examples")
    
    if not datasets:
        logger.error("No data files provided! Use --source_file and/or --target_file")
        sys.exit(1)
    
    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(examples['text'], return_special_tokens_mask=True)
    
    logger.info("Tokenizing datasets...")
    tokenized_datasets = {}
    for name, dataset in datasets.items():
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=dataset.column_names,
            desc=f"Tokenizing {name}",
        )
        tokenized_datasets[name] = tokenized
    
    # Group texts into chunks
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        total_length = (total_length // args.max_seq_length) * args.max_seq_length
        result = {
            k: [t[i:i + args.max_seq_length] for i in range(0, total_length, args.max_seq_length)]
            for k, t in concatenated.items()
        }
        return result
    
    logger.info("Grouping texts...")
    eval_datasets = {}
    for name, dataset in tokenized_datasets.items():
        grouped = dataset.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            desc=f"Grouping {name}",
        )
        eval_datasets[name] = grouped
        logger.info(f"{name}: {len(grouped)} batches")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=args.mlm_probability,
    )
    
    # Training args (for Trainer, but we only eval)
    training_args = TrainingArguments(
        output_dir="/tmp/eval_output",
        per_device_eval_batch_size=args.batch_size,
        do_train=False,
        do_eval=True,
    )
    
    # Results
    results = {
        'model': args.model_name_or_path,
        'num_parameters': num_params,
        'vocab_size': len(tokenizer),
        'num_layers': model.config.num_hidden_layers,
        'hidden_size': model.config.hidden_size,
    }
    
    # Evaluate on each dataset
    logger.info("\n" + "="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    
    for name, eval_dataset in eval_datasets.items():
        logger.info(f"\nEvaluating on {name}...")
        
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        
        metrics = trainer.evaluate()
        
        # Calculate perplexity
        if "eval_loss" in metrics:
            try:
                perplexity = math.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")
            metrics["perplexity"] = perplexity
        
        results[f'{name}_loss'] = metrics.get('eval_loss', None)
        results[f'{name}_perplexity'] = metrics.get('perplexity', None)
        results[f'{name}_samples'] = len(eval_dataset)
        
        logger.info(f"{name.upper()} Results:")
        logger.info(f"  Loss: {metrics.get('eval_loss', 'N/A'):.4f}")
        logger.info(f"  Perplexity: {metrics.get('perplexity', 'N/A'):.4f}")
        logger.info(f"  Samples: {len(eval_dataset)}")
    
    # Calculate average if both languages exist
    if 'source_loss' in results and 'target_loss' in results:
        avg_loss = (results['source_loss'] + results['target_loss']) / 2
        avg_perplexity = (results['source_perplexity'] + results['target_perplexity']) / 2
        results['average_loss'] = avg_loss
        results['average_perplexity'] = avg_perplexity
        
        logger.info("\nAVERAGE (Source + Target):")
        logger.info(f"  Loss: {avg_loss:.4f}")
        logger.info(f"  Perplexity: {avg_perplexity:.4f}")
    
    logger.info("="*60)
    
    # Save results
    if args.output_file:
        import json
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to: {args.output_file}")
    
    return results


if __name__ == "__main__":
    main()
