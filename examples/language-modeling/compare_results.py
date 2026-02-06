#!/usr/bin/env python3
"""
Compare evaluation results from multiple models
Reads JSON files produced by evaluate_mlm.py and displays a comparison table
"""

import argparse
import json
import sys
from pathlib import Path


def format_number(num, decimals=4):
    """Format number with comma separators and fixed decimals"""
    if num is None:
        return "N/A"
    if isinstance(num, float):
        return f"{num:,.{decimals}f}"
    return f"{num:,}"


def main():
    parser = argparse.ArgumentParser(description="Compare model evaluation results")
    parser.add_argument("result_files", nargs="+", help="JSON result files to compare")
    args = parser.parse_args()
    
    # Load all results
    results = []
    for file_path in args.result_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                data['_file'] = Path(file_path).stem
                results.append(data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}", file=sys.stderr)
    
    if not results:
        print("No valid result files found!", file=sys.stderr)
        sys.exit(1)
    
    # Sort by average perplexity (lower is better)
    results.sort(key=lambda x: x.get('average_perplexity', float('inf')))
    
    print("\n" + "="*100)
    print("MODEL COMPARISON RESULTS")
    print("="*100)
    print()
    
    # Model info table
    print("MODEL INFORMATION")
    print("-"*100)
    header = f"{'Model':<40} {'Parameters':<15} {'Vocab':<10} {'Layers':<8} {'Hidden':<8}"
    print(header)
    print("-"*100)
    
    for r in results:
        model_name = Path(r['model']).name if '/' in r['model'] else r['model']
        if len(model_name) > 37:
            model_name = model_name[:34] + "..."
        
        row = (
            f"{model_name:<40} "
            f"{format_number(r.get('num_parameters', 0), 0):<15} "
            f"{format_number(r.get('vocab_size', 0), 0):<10} "
            f"{r.get('num_layers', 'N/A'):<8} "
            f"{r.get('hidden_size', 'N/A'):<8}"
        )
        print(row)
    
    print()
    
    # Performance table - Source Language
    print("SOURCE LANGUAGE PERFORMANCE")
    print("-"*100)
    header = f"{'Model':<40} {'Loss':<12} {'Perplexity':<15} {'Samples':<10}"
    print(header)
    print("-"*100)
    
    for r in results:
        model_name = Path(r['model']).name if '/' in r['model'] else r['model']
        if len(model_name) > 37:
            model_name = model_name[:34] + "..."
        
        loss = r.get('source_loss', None)
        ppl = r.get('source_perplexity', None)
        samples = r.get('source_samples', None)
        
        row = (
            f"{model_name:<40} "
            f"{format_number(loss) if loss else 'N/A':<12} "
            f"{format_number(ppl) if ppl else 'N/A':<15} "
            f"{format_number(samples, 0) if samples else 'N/A':<10}"
        )
        print(row)
    
    print()
    
    # Performance table - Target Language
    print("TARGET LANGUAGE PERFORMANCE")
    print("-"*100)
    header = f"{'Model':<40} {'Loss':<12} {'Perplexity':<15} {'Samples':<10}"
    print(header)
    print("-"*100)
    
    for r in results:
        model_name = Path(r['model']).name if '/' in r['model'] else r['model']
        if len(model_name) > 37:
            model_name = model_name[:34] + "..."
        
        loss = r.get('target_loss', None)
        ppl = r.get('target_perplexity', None)
        samples = r.get('target_samples', None)
        
        row = (
            f"{model_name:<40} "
            f"{format_number(loss) if loss else 'N/A':<12} "
            f"{format_number(ppl) if ppl else 'N/A':<15} "
            f"{format_number(samples, 0) if samples else 'N/A':<10}"
        )
        print(row)
    
    print()
    
    # Average performance
    print("AVERAGE PERFORMANCE (Source + Target)")
    print("-"*100)
    header = f"{'Model':<40} {'Avg Loss':<12} {'Avg Perplexity':<15} {'Rank':<10}"
    print(header)
    print("-"*100)
    
    for rank, r in enumerate(results, 1):
        model_name = Path(r['model']).name if '/' in r['model'] else r['model']
        if len(model_name) > 37:
            model_name = model_name[:34] + "..."
        
        avg_loss = r.get('average_loss', None)
        avg_ppl = r.get('average_perplexity', None)
        
        # Add indicator for best model
        indicator = " ⭐ BEST" if rank == 1 else ""
        
        row = (
            f"{model_name:<40} "
            f"{format_number(avg_loss) if avg_loss else 'N/A':<12} "
            f"{format_number(avg_ppl) if avg_ppl else 'N/A':<15} "
            f"#{rank}{indicator}"
        )
        print(row)
    
    print("="*100)
    print()
    
    # Summary statistics
    if len(results) > 1:
        best = results[0]
        worst = results[-1]
        
        print("SUMMARY")
        print("-"*100)
        print(f"Best Model: {Path(best['model']).name}")
        print(f"  Average Perplexity: {format_number(best.get('average_perplexity'))}")
        print(f"  Parameters: {format_number(best.get('num_parameters'), 0)}")
        print()
        
        # Improvement over worst
        if worst.get('average_perplexity') and best.get('average_perplexity'):
            improvement = ((worst['average_perplexity'] - best['average_perplexity']) / 
                          worst['average_perplexity'] * 100)
            print(f"Improvement over worst model: {improvement:.2f}% lower perplexity")
        
        # Size reduction
        if worst.get('num_parameters') and best.get('num_parameters'):
            size_reduction = ((worst['num_parameters'] - best['num_parameters']) / 
                            worst['num_parameters'] * 100)
            print(f"Parameter reduction: {size_reduction:.2f}% fewer parameters")
        
        print("="*100)
        print()


if __name__ == "__main__":
    main()
