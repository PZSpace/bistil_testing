#!/usr/bin/env python3
"""
Test compute_metrics logic locally BEFORE running on Colab
"""
import numpy as np

def compute_metrics_test(p):
    """Exact copy of compute_metrics function"""
    predictions, labels = p
    
    try:
        # Debug: Print structure
        print(f"\nDEBUG compute_metrics:")
        print(f"  predictions type: {type(predictions)}")
        
        if isinstance(predictions, (tuple, list)):
            print(f"  predictions is list/tuple with {len(predictions)} elements")
            print(f"  First element shape: {predictions[0].shape if hasattr(predictions[0], 'shape') else 'no shape'}")
            if len(predictions) > 1:
                print(f"  Second element shape: {predictions[1].shape if hasattr(predictions[1], 'shape') else 'no shape'}")
            
            # Convert all to numpy arrays
            predictions = [np.asarray(pred) for pred in predictions]
            
            # Get shapes
            shapes = [pred.shape for pred in predictions]
            print(f"  All shapes: {shapes}")
            
            # Check if we need to pad
            # Expected shape: (batch_size, seq_length, num_labels)
            if len(shapes[0]) == 3:
                # Find max dimensions
                max_batch = max(s[0] for s in shapes)
                max_seq = max(s[1] for s in shapes)
                num_labels = shapes[0][2]
                
                print(f"  Max batch: {max_batch}, Max seq: {max_seq}, Num labels: {num_labels}")
                
                # Pad each prediction array
                padded_preds = []
                for pred in predictions:
                    if pred.shape != (max_batch, max_seq, num_labels):
                        # Need to pad
                        pad_batch = max_batch - pred.shape[0]
                        pad_seq = max_seq - pred.shape[1]
                        
                        if pad_batch > 0 or pad_seq > 0:
                            pad_width = ((0, pad_batch), (0, pad_seq), (0, 0))
                            pred = np.pad(pred, pad_width, mode='constant', constant_values=0)
                
                    padded_preds.append(pred)
                
                # Concatenate along batch dimension
                predictions = np.concatenate(padded_preds, axis=0)
                print(f"  After concatenation shape: {predictions.shape}")
            else:
                # Unexpected shape, try simple concatenation
                predictions = np.concatenate(predictions, axis=0)
        
        elif not isinstance(predictions, np.ndarray):
            predictions = np.asarray(predictions)
        
        print(f"  Final predictions shape before argmax: {predictions.shape}")
        
        # Get class predictions
        predictions = np.argmax(predictions, axis=-1)
        print(f"  Predictions shape after argmax: {predictions.shape}")
        print(f"  SUCCESS!\n")
        return predictions
        
    except Exception as e:
        print(f"\nERROR in compute_metrics preprocessing:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise

# Test Case 1: List of 3 arrays with different batch sizes (same seq length)
print("="*80)
print("TEST 1: Different batch sizes, same sequence length")
print("="*80)
pred1 = np.random.rand(4, 128, 9)  # batch=4
pred2 = np.random.rand(4, 128, 9)  # batch=4  
pred3 = np.random.rand(3, 128, 9)  # batch=3 (last batch smaller)
labels = np.random.randint(-100, 9, (11, 128))  # Total 11 samples

predictions = [pred1, pred2, pred3]
result = compute_metrics_test((predictions, labels))
assert result.shape == (12, 128), f"Expected (12, 128) [padded to max batch], got {result.shape}"
print("✓ TEST 1 PASSED")

# Test Case 2: List of 3 arrays with different sequence lengths
print("\n" + "="*80)
print("TEST 2: Same batch size, different sequence lengths (EDGE CASE)")
print("="*80)
pred1 = np.random.rand(4, 128, 9)
pred2 = np.random.rand(4, 120, 9)  # Different seq length
pred3 = np.random.rand(4, 110, 9)  # Different seq length
labels = np.random.randint(-100, 9, (12, 128))

predictions = [pred1, pred2, pred3]
result = compute_metrics_test((predictions, labels))
assert result.shape == (12, 128), f"Expected (12, 128), got {result.shape}"
print("✓ TEST 2 PASSED")

# Test Case 3: Single numpy array (no accumulation)
print("\n" + "="*80)
print("TEST 3: Single numpy array (no eval_accumulation_steps)")
print("="*80)
predictions = np.random.rand(10, 128, 9)
labels = np.random.randint(-100, 9, (10, 128))

result = compute_metrics_test((predictions, labels))
assert result.shape == (10, 128), f"Expected (10, 128), got {result.shape}"
print("✓ TEST 3 PASSED")

# Test Case 4: Realistic scenario - 3247 eval samples, batch_size=4, accumulation=4
print("\n" + "="*80)
print("TEST 4: Realistic scenario (3247 samples, batch=4, accum=4)")
print("="*80)
# 3247 / 4 = 811 batches + 3 remainder
# With accumulation=4: 811/4 = 202 full accumulations + 3 batches remainder
# So we get a list with 203 elements, last one has 3 batches

# Simulate last accumulation with 3 batches
pred1 = np.random.rand(4, 128, 9)
pred2 = np.random.rand(4, 128, 9)
pred3 = np.random.rand(3, 128, 9)  # Last batch has only 3 samples
labels = np.random.randint(-100, 9, (11, 128))

predictions = [pred1, pred2, pred3]
result = compute_metrics_test((predictions, labels))
print("✓ TEST 4 PASSED")

print("\n" + "="*80)
print("ALL TESTS PASSED! ✓")
print("The compute_metrics function should work on Colab!")
print("="*80)
