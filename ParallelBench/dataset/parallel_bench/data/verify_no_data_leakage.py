#!/usr/bin/env python3
"""
Verify that test-aug data has no overlap with train data.

Usage:
    python verify_no_data_leakage.py
"""

import json
from pathlib import Path
from collections import defaultdict


def load_data_samples(file_path, max_samples=1000):
    """Load samples from a jsonl file."""
    samples = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            if line.strip():
                data = json.loads(line)
                # Extract key information for comparison
                if 'input' in data and 'context' in data['input']:
                    context = data['input']['context']
                    # Also check answer if available
                    answer = data.get('answer', {})
                    if isinstance(answer, dict):
                        example = answer.get('example', '')
                    else:
                        example = str(answer)
                    samples.append({
                        'context': context,
                        'example': example,
                        'full_data': data
                    })
    return samples


def compare_datasets(train_file, test_file, task_name):
    """Compare train and test datasets for overlaps."""
    print(f"\n{'='*60}")
    print(f"Comparing {task_name}")
    print(f"{'='*60}")
    
    train_samples = load_data_samples(train_file, max_samples=5000)
    test_samples = load_data_samples(test_file, max_samples=5000)
    
    print(f"Train samples loaded: {len(train_samples)}")
    print(f"Test-aug samples loaded: {len(test_samples)}")
    
    # Create sets for fast lookup
    train_contexts = {s['context'] for s in train_samples}
    train_examples = {s['example'] for s in train_samples}
    
    test_contexts = {s['context'] for s in test_samples}
    test_examples = {s['example'] for s in test_samples}
    
    # Check for overlaps
    context_overlap = train_contexts & test_contexts
    example_overlap = train_examples & test_examples
    
    print(f"\nContext overlaps: {len(context_overlap)}")
    if context_overlap:
        print(f"⚠ WARNING: Found {len(context_overlap)} overlapping contexts!")
        print("Sample overlapping contexts:")
        for ctx in list(context_overlap)[:5]:
            print(f"  - {ctx}")
    else:
        print("✓ No context overlaps found")
    
    print(f"\nExample/Answer overlaps: {len(example_overlap)}")
    if example_overlap:
        print(f"⚠ WARNING: Found {len(example_overlap)} overlapping examples!")
        print("Sample overlapping examples:")
        for ex in list(example_overlap)[:5]:
            print(f"  - {ex[:100]}...")
    else:
        print("✓ No example overlaps found")
    
    # Check full data matches (more strict)
    train_full = {json.dumps(s['full_data'], sort_keys=True) for s in train_samples}
    test_full = {json.dumps(s['full_data'], sort_keys=True) for s in test_samples}
    full_overlap = train_full & test_full
    
    print(f"\nFull data overlaps: {len(full_overlap)}")
    if full_overlap:
        print(f"⚠ WARNING: Found {len(full_overlap)} identical samples!")
    else:
        print("✓ No identical samples found")
    
    return len(context_overlap) == 0 and len(example_overlap) == 0 and len(full_overlap) == 0


def main():
    data_dir = Path(__file__).parent / "output"
    train_dir = data_dir / "train" / "waiting_line"
    test_aug_dir = data_dir / "test-aug" / "waiting_line"
    
    # Get all task files
    train_files = list(train_dir.glob("*.jsonl"))
    test_aug_files = list(test_aug_dir.glob("*.jsonl"))
    
    print(f"Found {len(train_files)} train files")
    print(f"Found {len(test_aug_files)} test-aug files")
    
    all_clean = True
    
    # Compare each task
    for train_file in train_files:
        task_name = train_file.stem
        test_file = test_aug_dir / train_file.name
        
        if not test_file.exists():
            print(f"\n⚠ Warning: No corresponding test-aug file for {task_name}")
            continue
        
        is_clean = compare_datasets(train_file, test_file, task_name)
        if not is_clean:
            all_clean = False
    
    print(f"\n{'='*60}")
    if all_clean:
        print("✓ ALL CHECKS PASSED: No data leakage detected!")
    else:
        print("✗ WARNING: Data leakage detected! Please regenerate test-aug data.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

