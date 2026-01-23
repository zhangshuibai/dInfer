#!/usr/bin/env python3
"""
Evaluation script: Compute metric scores from jsonl output directory

Usage:
    python eval_from_jsonl.py <output_dir> [--verbose]

Examples:
    python eval_from_jsonl.py outputs/waiting_line_shuffle_threshold_0_90
    python eval_from_jsonl.py outputs/waiting_line_copy --verbose

Logic:
    1. Extract task name from directory name (e.g., "waiting_line_shuffle" from "waiting_line_shuffle_threshold_0_90")
    2. Select appropriate metric function based on task type (TASK_METRIC_MAP)
    3. Load predictions from rank_0.jsonl in the output directory
    4. Load ground truth references from the corresponding data file (TASK_DATA_MAP)
    5. Compute scores for each sample using the selected metric
    6. Calculate and display statistics (average, min, max)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "tasks" / "parallel_bench"))
from adapter import (
    list_match_score_metric,
    list_shuffle_score_metric,
    list_random_insert_score_metric,
    list_random_remove_score_metric,
    list_random_replace_score_metric,
)

# Task name -> metric function mapping
# Different tasks use different metrics based on their evaluation requirements
TASK_METRIC_MAP = {
    "waiting_line_shuffle": list_shuffle_score_metric,      # Shuffle: checks if order is different
    "waiting_line_copy": list_match_score_metric,            # Copy: exact match required
    "waiting_line_reverse": list_match_score_metric,         # Reverse: exact match required
    "waiting_line_sort": list_match_score_metric,            # Sort: exact match required
    "waiting_line_insert": list_random_insert_score_metric, # Insert: checks if element inserted correctly
    "waiting_line_remove": list_random_remove_score_metric, # Remove: checks if element removed correctly
    "waiting_line_replace": list_random_replace_score_metric, # Replace: checks if element replaced correctly
}

# Task name -> ground truth data file path mapping
TASK_DATA_MAP = {
    "waiting_line_shuffle": "/data/dInfer/ParallelBench/dataset/parallel_bench/data/output/test/waiting_line/shuffle.jsonl",
    "waiting_line_copy": "/data/dInfer/ParallelBench/dataset/parallel_bench/data/output/test/waiting_line/copy.jsonl",
    "waiting_line_reverse": "/data/dInfer/ParallelBench/dataset/parallel_bench/data/output/test/waiting_line/reverse.jsonl",
    "waiting_line_sort": "/data/dInfer/ParallelBench/dataset/parallel_bench/data/output/test/waiting_line/sort.jsonl",
}


def extract_task_name(dir_path: str) -> str:
    """Extract task name from directory path"""
    dir_name = Path(dir_path).name
    # Try exact match first
    for task_name in TASK_METRIC_MAP.keys():
        if task_name in dir_name:
            return task_name
    # Fallback: parse from directory name pattern (e.g., "waiting_line_shuffle_threshold_0_90")
    parts = dir_name.split('_')
    if 'waiting_line' in dir_name:
        idx = parts.index('waiting_line')
        return f"waiting_line_{parts[idx+1]}"
    raise ValueError(f"Cannot determine task from: {dir_path}")


def load_predictions(jsonl_path: str):
    """
    Load predictions from jsonl file
    
    Note: Predictions are loaded in line order (line 0, line 1, ...)
    Each line corresponds to one sample, indexed by line number.
    """
    with open(jsonl_path) as f:
        return [json.loads(line)['answer'] for line in f if line.strip()]


def load_references(data_file: str):
    """
    Load ground truth references from data file
    
    Note: References are loaded in line order (line 0, line 1, ...)
    Each line corresponds to one sample, indexed by line number.
    """
    with open(data_file) as f:
        return [json.loads(line)['answer'] for line in f if line.strip()]


def evaluate(output_dir: str, verbose: bool = False):
    """
    Evaluate results in output directory
    
    Workflow:
    1. Extract task name from directory name
    2. Select metric function based on task (TASK_METRIC_MAP)
    3. Load predictions and references
    4. Compute scores for each sample
    5. Calculate and display statistics
    
    Ground Truth Matching:
    - Predictions and references are matched by line index (positional matching)
    - predictions[i] corresponds to references[i] (i-th line in each file)
    - This assumes both files have the same order and same number of samples
    - If lengths differ, only the first min(len(predictions), len(references)) samples are evaluated
    """
    output_dir = Path(output_dir).resolve()
    jsonl_file = output_dir / "rank_0.jsonl"
    
    # Step 1: Identify task and select metric
    task_name = extract_task_name(str(output_dir))
    metric_func = TASK_METRIC_MAP[task_name]  # Different metric for different tasks
    data_file = TASK_DATA_MAP[task_name]
    
    # Step 2: Load data
    # Both loaded in line order: predictions[0] <-> references[0], predictions[1] <-> references[1], ...
    predictions = load_predictions(str(jsonl_file))
    references = load_references(data_file)
    
    # Step 3: Align lengths (use minimum to ensure 1-to-1 correspondence)
    min_len = min(len(predictions), len(references))
    predictions = predictions[:min_len]
    references = references[:min_len]
    
    # Step 4: Compute scores using the selected metric
    # Matching: predictions[i] is evaluated against references[i] (by line index)
    scores = []
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        score = metric_func([pred], [ref])  # Call task-specific metric function
        score = score[0] if isinstance(score, (list, tuple)) else score
        scores.append(float(score))
        if verbose:
            print(f"  Sample {i}: {score:.4f}")  # i is the line index (0-indexed)
    
    # Step 5: Calculate statistics
    avg_score = sum(scores) / len(scores)
    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"Metric: {metric_func.__name__}")  # Shows which metric was used
    print(f"Samples: {len(scores)}")
    print(f"Average: {avg_score:.4f} ({avg_score*100:.2f}%)")
    print(f"Min: {min(scores):.4f}, Max: {max(scores):.4f}")
    print(f"{'='*60}")
    
    return avg_score


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    output_dir = sys.argv[1]
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    evaluate(output_dir, verbose)


if __name__ == "__main__":
    main()
