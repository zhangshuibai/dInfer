"""
Adapter utilities for integrating ParallelBench tasks into lm-eval-harness.
"""
import sys
from pathlib import Path
from datasets import Dataset
import pandas as pd

# Add ParallelBench to path
PARALLEL_BENCH_PATH = Path(__file__).parent.parent.parent.parent / "ParallelBench"
PARALLEL_BENCH_PATH_STR = str(PARALLEL_BENCH_PATH.resolve())

# Lazy import to avoid circular import issues
def _import_parallel_bench():
    """Lazy import ParallelBench to avoid circular imports."""
    import sys as sys_module
    
    # Save original state
    original_path = sys.path[:]
    original_modules = {}
    
    # Temporarily remove 'utils' from sys.modules if it exists and is our module
    # This allows ParallelBench to import its own utils package
    utils_module_key = None
    for key in list(sys.modules.keys()):
        if key == 'utils' and hasattr(sys.modules[key], '__file__'):
            mod_file = Path(sys.modules[key].__file__)
            if 'parallel_bench' in str(mod_file) and mod_file.name == 'utils.py':
                # This is our utils.py, save and remove it
                original_modules[key] = sys.modules.pop(key)
                utils_module_key = key
                break
    
    # Remove current directory and parent directories that might conflict
    current_file_dir = str(Path(__file__).parent.resolve())
    current_file_parent = str(Path(__file__).parent.parent.resolve())
    current_file_grandparent = str(Path(__file__).parent.parent.parent.resolve())
    
    # Remove conflicting paths (our utils.py locations)
    paths_to_remove = [current_file_dir, current_file_parent, current_file_grandparent]
    for path in paths_to_remove:
        while path in sys.path:
            sys.path.remove(path)
    
    # Ensure ParallelBench root is at the front of sys.path
    if PARALLEL_BENCH_PATH_STR in sys.path:
        sys.path.remove(PARALLEL_BENCH_PATH_STR)
    sys.path.insert(0, PARALLEL_BENCH_PATH_STR)
    
    try:
        # Now import - Python will find ParallelBench's utils package (directory) first
        from dataset.parallel_bench import ParallelBench
        from dataset.parallel_bench.metrics import parallel_bench_metric_func_map
        return ParallelBench, parallel_bench_metric_func_map
    finally:
        # Restore original sys.path
        sys.path[:] = original_path
        # Restore our utils module if it was removed
        if utils_module_key and utils_module_key in original_modules:
            sys.modules[utils_module_key] = original_modules[utils_module_key]

# Global cache for ParallelBench instances
_task_cache = {}


def load_parallel_bench_task(task_name, split="test"):
    """Load ParallelBench task instance with caching."""
    ParallelBench, _ = _import_parallel_bench()
    cache_key = f"{task_name}_{split}"
    if cache_key not in _task_cache:
        pb = ParallelBench(task_name, split=split)
        _task_cache[cache_key] = pb
    return _task_cache[cache_key]


def _extract_task_name_from_path(file_path):
    """Extract ParallelBench task name from file path."""
    path = Path(file_path)
    # Extract path parts: .../test/waiting_line/copy.jsonl -> waiting_line/copy
    parts = path.parts
    if 'test' in parts:
        idx = parts.index('test')
        if idx + 2 < len(parts):
            return f"{parts[idx+1]}/{parts[idx+2].replace('.jsonl', '')}"
    # Fallback: try to extract from any path structure
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1].replace('.jsonl', '')}"
    return None


def process_docs(dataset):
    """
    Convert ParallelBench data format to lm-eval-harness format.
    
    Args:
        dataset: HuggingFace Dataset loaded from JSONL
    
    Returns:
        Processed dataset with 'text' and 'answer' fields
    """
    # Try to infer task_name from various sources
    task_name = None
    
    # Method 1: Check config_name in dataset attributes (set via dataset_kwargs in YAML)
    # In lm-eval, dataset_kwargs are passed to load_dataset, but config_name might
    # be stored in dataset.info or as an attribute
    if hasattr(dataset, 'config_name') and dataset.config_name:
        task_name = dataset.config_name
    
    # Method 2: Try to get from dataset info
    if task_name is None and hasattr(dataset, 'info'):
        if hasattr(dataset.info, 'config_name') and dataset.info.config_name:
            task_name = dataset.info.config_name
        # Also check if info has description or other fields
        if task_name is None and hasattr(dataset.info, 'description'):
            task_name = _extract_task_name_from_path(dataset.info.description)
    
    # Method 3: Try to extract from split name
    if task_name is None and hasattr(dataset, 'split') and dataset.split:
        if '/' in dataset.split:
            task_name = dataset.split
    
    # Method 4: Try to extract from file path in dataset's internal structure
    if task_name is None:
        # Check if we can access the original file path
        # HuggingFace datasets sometimes store this in _data_files
        if hasattr(dataset, '_data_files') and dataset._data_files:
            for file_path in dataset._data_files:
                extracted = _extract_task_name_from_path(file_path)
                if extracted:
                    task_name = extracted
                    break
    
    if task_name is None:
        raise ValueError(
            "Could not infer task_name from dataset. "
            "Please ensure 'config_name' is set in dataset_kwargs in the YAML file, "
            "or the data file path follows the pattern: .../test/task_category/task_name.jsonl"
        )
    
    return _process_docs_impl(dataset, task_name)

def _process_docs_impl(dataset, task_name):
    """Internal implementation of process_docs with explicit task_name."""
    
    pb = load_parallel_bench_task(task_name)
    
    def _process(doc, idx):
        # Get sample from ParallelBench using index
        if idx >= len(pb):
            idx = idx % len(pb)
        
        sample = pb[idx]
        
        # Extract messages (already formatted by ParallelBench)
        messages = sample["input"]["messages"]
        
        # Build prompt text with few-shot examples if present
        prompt_parts = []
        for i in range(0, len(messages) - 1, 2):
            if i + 1 < len(messages):
                prompt_parts.append(f"User: {messages[i]['content']}")
                prompt_parts.append(f"Assistant: {messages[i+1]['content']}")
                prompt_parts.append("")
        
        # Add the final user message
        prompt_parts.append(f"User: {messages[-1]['content']}")
        prompt_text = "\n".join(prompt_parts)
        
        # Extract answer
        answer = sample["label"]
        if isinstance(answer, dict):
            answer = answer.get("example", answer.get("result", str(answer)))
        
        return {
            "text": prompt_text,
            "answer": answer,
            "_task": task_name,
            "_metadata": sample.get("metadata", {}),
            "_original_idx": idx
        }
    
    return dataset.map(_process, with_indices=True)


def doc_to_text(doc):
    """Extract text field for lm-eval."""
    return doc["text"]


def doc_to_target(doc):
    """Extract target answer for lm-eval."""
    return doc["answer"]


# Metric wrappers for lm-eval-harness
def _wrap_metric(metric_func, metric_name):
    """Wrap ParallelBench metric function for lm-eval format."""
    def metric_fn(predictions, references):
        scores = []
        for pred, ref in zip(predictions, references):
            # Handle dict references (for tasks like shuffle)
            if isinstance(ref, dict):
                score = metric_func(pred, ref, strict=False)
            else:
                score = metric_func(pred, ref, strict=False)
            
            # Handle dict return values
            if isinstance(score, dict):
                score = score.get("score", 0.0)
            
            scores.append(float(score))
        
        return scores
    
    metric_fn.__name__ = metric_name
    return metric_fn



# Metric functions - initialized lazily on first access
_metric_funcs_cache = {}

def _get_metric_func(name):
    """Get a metric function by name, initializing lazily."""
    if name not in _metric_funcs_cache:
        _, parallel_bench_metric_func_map = _import_parallel_bench()
        if name == "summary_score" or name == "paraphrase_score":
            # These need to be instantiated
            metric_func = parallel_bench_metric_func_map[name]()
        else:
            metric_func = parallel_bench_metric_func_map[name]
        _metric_funcs_cache[name] = _wrap_metric(metric_func, name)
    return _metric_funcs_cache[name]

# Create metric function accessors (lazy)
def list_match_score_metric(predictions, references):
    return _get_metric_func("list_match_score")(predictions, references)

def list_shuffle_score_metric(predictions, references):
    return _get_metric_func("list_shuffle_score")(predictions, references)

def list_random_insert_score_metric(predictions, references):
    return _get_metric_func("list_random_insert_score")(predictions, references)

def list_random_remove_score_metric(predictions, references):
    return _get_metric_func("list_random_remove_score")(predictions, references)

def list_random_replace_score_metric(predictions, references):
    return _get_metric_func("list_random_replace_score")(predictions, references)

def latin_square_score_metric(predictions, references):
    return _get_metric_func("latin_square_score")(predictions, references)

def sudoku_score_metric(predictions, references):
    return _get_metric_func("sudoku_score")(predictions, references)

def summary_score_metric(predictions, references):
    return _get_metric_func("summary_score")(predictions, references)

def paraphrase_score_metric(predictions, references):
    return _get_metric_func("paraphrase_score")(predictions, references)

def sentence_to_words_score_metric(predictions, references):
    return _get_metric_func("sentence_to_words_score")(predictions, references)

