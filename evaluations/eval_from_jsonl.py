#!/usr/bin/env python3
"""
Evaluation script: Compute metric scores from jsonl output directory

Usage:
    python eval_from_jsonl.py <output_dir> [--verbose] [--filter-format]

Options:
    --verbose, -v       : Show detailed scores for each sample
    --filter-format, -f: Filter out responses that don't follow format instructions
                         (e.g., filter explanations and keep only list-formatted outputs)

Examples:
    python eval_from_jsonl.py outputs/waiting_line_shuffle_threshold_0_90
    python eval_from_jsonl.py outputs/waiting_line_copy --verbose
    python eval_from_jsonl.py outputs/waiting_line_shuffle_threshold_0_90 --filter-format

Logic:
    1. Extract task name from directory name (e.g., "waiting_line_shuffle" from "waiting_line_shuffle_threshold_0_90")
    2. Select appropriate metric function based on task type (TASK_METRIC_MAP)
    3. Load predictions from rank_0.jsonl in the output directory
    4. Load ground truth references from the corresponding data file (TASK_DATA_MAP)
    5. (Optional) Filter out malformed responses if --filter-format is enabled
    6. Compute scores for each sample using the selected metric
    7. Calculate and display statistics (average, min, max)
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
    "waiting_line_insert_index": list_match_score_metric,    # Insert at index: exact match required
    "waiting_line_insert_random": list_random_insert_score_metric, # Insert random: checks if element inserted correctly
    "waiting_line_remove_index": list_match_score_metric,    # Remove at index: exact match required
    "waiting_line_remove_random": list_random_remove_score_metric, # Remove random: checks if element removed correctly
    "waiting_line_replace_index": list_match_score_metric,  # Replace at index: exact match required
    "waiting_line_replace_random": list_random_replace_score_metric, # Replace random: checks if element replaced correctly
}

# Task name -> ground truth data file path mapping
TASK_DATA_MAP = {
    "waiting_line_shuffle": "/data/dInfer/ParallelBench/dataset/parallel_bench/data/output/test-aug/waiting_line/shuffle.jsonl",
    "waiting_line_copy": "/data/dInfer/ParallelBench/dataset/parallel_bench/data/output/test-aug/waiting_line/copy.jsonl",
    "waiting_line_reverse": "/data/dInfer/ParallelBench/dataset/parallel_bench/data/output/test-aug/waiting_line/reverse.jsonl",
    "waiting_line_sort": "/data/dInfer/ParallelBench/dataset/parallel_bench/data/output/test-aug/waiting_line/sort.jsonl",
    "waiting_line_insert_index": "/data/dInfer/ParallelBench/dataset/parallel_bench/data/output/test-aug/waiting_line/insert_index.jsonl",
    "waiting_line_insert_random": "/data/dInfer/ParallelBench/dataset/parallel_bench/data/output/test-aug/waiting_line/insert_random.jsonl",
    "waiting_line_remove_index": "/data/dInfer/ParallelBench/dataset/parallel_bench/data/output/test-aug/waiting_line/remove_index.jsonl",
    "waiting_line_remove_random": "/data/dInfer/ParallelBench/dataset/parallel_bench/data/output/test-aug/waiting_line/remove_random.jsonl",
    "waiting_line_replace_index": "/data/dInfer/ParallelBench/dataset/parallel_bench/data/output/test-aug/waiting_line/replace_index.jsonl",
    "waiting_line_replace_random": "/data/dInfer/ParallelBench/dataset/parallel_bench/data/output/test-aug/waiting_line/replace_random.jsonl",
}


def extract_task_name_from_results(output_dir: Path) -> str:
    """Extract task name from results JSON file"""
    # Look for results JSON file in any __data__* subdirectory
    for subdir in output_dir.iterdir():
        if subdir.is_dir() and subdir.name.startswith("__data__"):
            # Find the most recent results JSON file
            results_files = list(subdir.glob("results_*.json"))
            if results_files:
                # Sort by modification time, get the most recent
                results_file = max(results_files, key=lambda p: p.stat().st_mtime)
                with open(results_file) as f:
                    results = json.load(f)
                    # Extract task name from configs
                    if "configs" in results and results["configs"]:
                        task_name = list(results["configs"].keys())[0]
                        config = results["configs"][task_name]
                        if "task" in config:
                            return config["task"]
                        return task_name
    return None


def extract_task_name(dir_path: str) -> str:
    """Extract task name from directory path (fallback method)"""
    dir_name = Path(dir_path).name
    
    # Remove common suffixes to get clean task name
    dir_name_clean = dir_name
    for suffix in ['_threshold_0_90', '_threshold_0_80', '_remask_True', '_remask_False']:
        if suffix in dir_name_clean:
            dir_name_clean = dir_name_clean.replace(suffix, '')
    # Handle remaining threshold patterns
    if '_threshold_' in dir_name_clean:
        dir_name_clean = dir_name_clean.split('_threshold_')[0]
    if '_remask_' in dir_name_clean:
        dir_name_clean = dir_name_clean.split('_remask_')[0]
    
    # Try to match known task names - sort by length (longest first) to avoid partial matches
    task_names = sorted(TASK_METRIC_MAP.keys(), key=len, reverse=True)
    for task_name in task_names:
        if dir_name_clean == task_name or dir_name_clean.startswith(task_name + '_'):
            return task_name
    
    # Fallback: try substring match in original dir_name
    for task_name in task_names:
        if task_name in dir_name:
            return task_name
    
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


def is_valid_list_format(prediction: str) -> bool:
    """
    Check if prediction follows format instructions (should be a JSON array)
    
    Returns True if prediction is a valid JSON array, False otherwise.
    This filters out explanations and other non-list responses.
    """
    prediction = prediction.strip()
    if not prediction.startswith('[') or not prediction.endswith(']'):
        return False
    try:
        parsed = json.loads(prediction)
        return isinstance(parsed, list)
    except (json.JSONDecodeError, ValueError):
        return False


def filter_by_format(predictions, references, verbose: bool = False):
    """
    Filter predictions and references to keep only valid list-formatted responses
    
    Returns:
        filtered_predictions: List of valid predictions
        filtered_references: List of corresponding references
        filtered_indices: Original indices of kept samples
        filtered_count: Number of filtered samples
    """
    filtered_predictions = []
    filtered_references = []
    filtered_indices = []
    
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        if is_valid_list_format(pred):
            filtered_predictions.append(pred)
            filtered_references.append(ref)
            filtered_indices.append(i)
        elif verbose:
            print(f"  Sample {i}: FILTERED (not in list format): {pred[:60]}...")
    
    filtered_count = len(predictions) - len(filtered_predictions)
    return filtered_predictions, filtered_references, filtered_indices, filtered_count


def evaluate(output_dir: str, verbose: bool = False, filter_format: bool = False):
    """
    Evaluate results in output directory
    
    Workflow:
    1. Extract task name from directory name
    2. Select metric function based on task (TASK_METRIC_MAP)
    3. Load predictions and references
    4. (Optional) Filter out malformed responses if filter_format=True
    5. Compute scores for each sample
    6. Calculate and display statistics
    
    Ground Truth Matching:
    - Predictions and references are matched by line index (positional matching)
    - predictions[i] corresponds to references[i] (i-th line in each file)
    - This assumes both files have the same order and same number of samples
    - If lengths differ, only the first min(len(predictions), len(references)) samples are evaluated
    """
    output_dir = Path(output_dir).resolve()
    jsonl_file = output_dir / "rank_0.jsonl"
    
    # Step 1: Identify task and select metric
    # Try to get task name from results JSON first, fallback to directory name
    task_name = extract_task_name_from_results(output_dir)
    if task_name is None:
        # Fallback to extracting from directory name
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
    
    # Step 4: Filter format if enabled
    original_count = len(predictions)
    filtered_indices = list(range(len(predictions)))  # Default: all indices
    if filter_format:
        predictions, references, filtered_indices, filtered_count = filter_by_format(
            predictions, references, verbose
        )
        if filtered_count > 0:
            print(f"\nFiltered {filtered_count} malformed responses")
            print(f"Evaluating {len(predictions)} valid samples (out of {original_count} total)")
    else:
        filtered_count = 0
        valid_count = original_count
    
    # Step 5: Compute scores using the selected metric
    # Matching: predictions[i] is evaluated against references[i] (by line index)
    scores = []
    for idx, (pred, ref) in enumerate(zip(predictions, references)):
        sample_idx = filtered_indices[idx] if filter_format else idx
        score = metric_func([pred], [ref])  # Call task-specific metric function
        score = score[0] if isinstance(score, (list, tuple)) else score
        scores.append(float(score))
        if verbose:
            print(f"  Sample {sample_idx}: {score:.4f}")  # Show original index if filtered
    
    # Step 6: Calculate statistics
    if len(scores) == 0:
        print("\nNo valid samples to evaluate after filtering!")
        return None
    
    avg_score = sum(scores) / len(scores)
    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"Metric: {metric_func.__name__}")  # Shows which metric was used
    if filter_format and filtered_count > 0:
        print(f"Total samples: {original_count}")
        print(f"Valid samples: {len(scores)} (filtered {filtered_count})")
    else:
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
    filter_format = '--filter-format' in sys.argv or '-f' in sys.argv
    evaluate(output_dir, verbose, filter_format)


if __name__ == "__main__":
    main()
