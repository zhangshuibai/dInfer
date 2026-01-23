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
# Simplified ParallelBench import
def _import_parallel_bench():
    """Import ParallelBench with manual utils setup."""
    # Set up ParallelBench path
    if PARALLEL_BENCH_PATH_STR not in sys.path:
        sys.path.insert(0, PARALLEL_BENCH_PATH_STR)
    
    # Create utils module manually to avoid conflicts
    import types
    utils_module = types.ModuleType("utils")
    
    # Load grammar_check into utils module
    grammar_path = PARALLEL_BENCH_PATH / "utils" / "grammar_check.py"
    with open(grammar_path, "r") as f:
        grammar_code = f.read()
    exec(grammar_code, utils_module.__dict__)
    
    # Add to sys.modules
    sys.modules["utils"] = utils_module
    sys.modules["utils.grammar_check"] = utils_module
    
    # Now import ParallelBench
    from dataset.parallel_bench import ParallelBench
    from dataset.parallel_bench.metrics import parallel_bench_metric_func_map
    
    return ParallelBench, parallel_bench_metric_func_map

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


# Mapping from YAML task names to ParallelBench task names
_TASK_NAME_MAP = {
    "waiting_line_copy": "waiting_line/copy",
    "waiting_line_reverse": "waiting_line/reverse",
    "waiting_line_sort": "waiting_line/sort",
    "waiting_line_shuffle": "waiting_line/shuffle",
    "sudoku_n4_12": "puzzle/sudoku_n4_12",
    "samsum": "paraphrase_summarize/samsum",
}

def process_docs(dataset):
    """
    Convert ParallelBench data format to lm-eval-harness format.
    
    Args:
        dataset: HuggingFace Dataset loaded from JSONL
    
    Returns:
        Processed dataset with 'text' and 'answer' fields
    """
    # Extract task_name from YAML file path in call stack (most reliable method)
    task_name = None
    import inspect
    for frame in inspect.stack():
        filename = frame.filename
        if 'parallel_bench' in filename and filename.endswith('.yaml'):
            yaml_task = Path(filename).stem
            task_name = _TASK_NAME_MAP.get(yaml_task)
            if task_name:
                break
        
        # Also try to get task name from config object in frame locals
        frame_locals = frame.frame.f_locals
        if 'self' in frame_locals:
            config = getattr(frame_locals['self'], 'config', None)
            if config is not None:
                task_config = getattr(config, 'task', None)
                if task_config:
                    task_name = _TASK_NAME_MAP.get(task_config)
                    if task_name:
                        break
    
    # Fallback: try to extract from dataset's data_files
    if task_name is None:
        builder = getattr(dataset, '_builder', None)
        if builder is not None and hasattr(builder, 'data_files'):
            data_files = builder.data_files
            if isinstance(data_files, dict):
                for split_files in data_files.values():
                    files = split_files if isinstance(split_files, list) else [split_files]
                    for file_path in files:
                        task_name = _extract_task_name_from_path(str(file_path))
                        if task_name:
                            break
                    if task_name:
                        break
            elif isinstance(data_files, list):
                for file_path in data_files:
                    task_name = _extract_task_name_from_path(str(file_path))
                    if task_name:
                        break
            else:
                task_name = _extract_task_name_from_path(str(data_files))
    
    # Additional fallback: try to get from dataset's _data_files attribute
    if task_name is None:
        if hasattr(dataset, '_data_files') and dataset._data_files:
            files = dataset._data_files if isinstance(dataset._data_files, list) else [dataset._data_files]
            for file_path in files:
                task_name = _extract_task_name_from_path(str(file_path))
                if task_name:
                    break
    
    # Additional fallback: try to get from dataset info
    if task_name is None:
        info = getattr(dataset, 'info', None)
        if info is not None:
            # Check if dataset has a description or other metadata with file path
            description = getattr(info, 'description', '')
            if description:
                task_name = _extract_task_name_from_path(description)
    
    # Final fallback: try to extract from any string attributes in dataset
    if task_name is None:
        # Check dataset's cache_files attribute if available
        cache_files = getattr(dataset, 'cache_files', None)
        if cache_files:
            for cache_file in cache_files if isinstance(cache_files, list) else [cache_files]:
                if cache_file:
                    task_name = _extract_task_name_from_path(str(cache_file))
                    if task_name:
                        break
    
    if task_name is None:
        raise ValueError(
            "Could not determine task_name. "
            "Please ensure the YAML file name is in _TASK_NAME_MAP or the data file path "
            "follows pattern: .../test/task_category/task_name.jsonl"
        )
    
    return _process_docs_impl(dataset, task_name)

def _process_docs_impl(dataset, task_name):
    """Internal implementation of process_docs with explicit task_name."""
    
    # Load ParallelBench task to get the prompt template (with instruction)
    pb_task = load_parallel_bench_task(task_name, split="test")
    prompt_template = pb_task.prompt
    
    # Process dataset directly - convert from ParallelBench JSONL format to lm_eval format
    def _process(doc):
        # Extract input and answer from the dataset document
        input_data = doc.get("input", {})
        answer = doc.get("answer", "")
        
        # Handle different input formats
        if "messages" in input_data:
            # Format with messages (chat format) - use messages as-is
            messages = input_data["messages"]
            prompt_parts = []
            for i in range(0, len(messages) - 1, 2):
                if i + 1 < len(messages):
                    prompt_parts.append(f"User: {messages[i]['content']}")
                    prompt_parts.append(f"Assistant: {messages[i+1]['content']}")
                    prompt_parts.append("")
            
            # Add the final user message
            if len(messages) > 0:
                prompt_parts.append(f"User: {messages[-1]['content']}")
            prompt_text = "\n".join(prompt_parts)
        elif "context" in input_data:
            # Format with context and ICL examples using ParallelBench's prompt template
            context = input_data.get("context", "")
            icl_examples = input_data.get("icl_examples", [])
            
            # Format few-shot examples using the prompt template
            few_shot_text = ""
            for ex in icl_examples:
                # Format each example using the prompt template
                ex_input_dict = ex.get("input", {})
                ex_prompt = prompt_template.format(**ex_input_dict).replace("\\n", "\n")
                ex_answer = ex.get("answer", "")
                if isinstance(ex_answer, dict):
                    ex_answer = ex_answer.get("example", str(ex_answer))
                few_shot_text += f"{ex_prompt}\n{ex_answer}\n\n"
            
            # Format current task using the prompt template
            current_prompt = prompt_template.format(**input_data).replace("\\n", "\n")
            
            # Combine few-shot examples and current task
            prompt_text = few_shot_text + current_prompt
        else:
            # Fallback: use the input as-is
            prompt_text = str(input_data)
        
        return {
            "text": prompt_text,
            "answer": answer,
            "_task": task_name,
            "_metadata": doc.get("metadata", {}),
            "output_format": doc.get("output_format", "")
        }
    
    return dataset.map(_process)


def doc_to_text(doc):
    """Extract text field for lm-eval."""
    return doc["text"]


def doc_to_target(doc):
    """Extract target answer for lm-eval."""
    answer = doc["answer"]
    # For tasks like shuffle, answer is a dict with "input" and "example"
    # Return the dict as-is so metric functions can access the "input" field
    return answer


# Metric wrappers for lm-eval-harness
def _wrap_metric(metric_func, metric_name):
    """Wrap ParallelBench metric function for lm-eval format."""
    def metric_fn(predictions, references):
        # Ensure inputs are lists
        if not isinstance(predictions, (list, tuple)):
            predictions = [predictions]
        if not isinstance(references, (list, tuple)):
            references = [references]
        
        # Ensure we have matching lengths
        min_len = min(len(predictions), len(references))
        predictions = predictions[:min_len]
        references = references[:min_len]
        
        scores = []
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            # Debug: print input types
            print(f"DEBUG {metric_name} sample {i}: pred type={type(pred).__name__}, ref type={type(ref).__name__}")
            print(f"DEBUG {metric_name} sample {i}: pred[:100]={str(pred)[:100]}")
            print(f"DEBUG {metric_name} sample {i}: ref[:200]={str(ref)[:200]}")
            
            # Clean prediction - remove "Output: " prefix if present
            if isinstance(pred, str) and pred.startswith("Output:"):
                pred = pred.replace("Output:", "").strip()
            
            # Handle reference format - try to convert to dict
            original_ref_type = type(ref).__name__
            if isinstance(ref, str):
                # Try JSON first
                try:
                    import json
                    ref = json.loads(ref)
                    print(f"DEBUG {metric_name} sample {i}: Parsed ref from string as JSON, new type={type(ref).__name__}")
                except Exception:
                    # Try Python literal_eval for Python repr format (e.g., {'input': [...], 'example': '...'})
                    try:
                        import ast
                        ref = ast.literal_eval(ref)
                        print(f"DEBUG {metric_name} sample {i}: Parsed ref from string as Python literal, new type={type(ref).__name__}")
                    except Exception as e:
                        print(f"DEBUG {metric_name} sample {i}: Failed to parse ref: {e}")
                        pass
            
            # Unwrap single-element lists
            if isinstance(ref, list) and len(ref) == 1:
                ref = ref[0]
                print(f"DEBUG {metric_name} sample {i}: Unwrapped single-element list, new type={type(ref).__name__}")
            
            # Handle different reference formats based on task type
            # For shuffle tasks: ref should be dict with 'input' and 'example'
            # For copy tasks: ref might be a list (the expected output)
            if not isinstance(ref, dict):
                # For copy/match tasks, if ref is a list, convert to dict format
                if ('copy' in metric_name.lower() or 'match' in metric_name.lower()) and isinstance(ref, list):
                    # For copy/match tasks, the reference is the expected output list
                    # Convert to dict format: {'input': ref, 'example': ref} (same for copy)
                    # The metric function expects: input (list) and example (list or JSON string)
                    import json
                    ref = {
                        'input': ref,  # Original list
                        'example': ref  # Expected output (same as input for copy)
                    }
                    print(f"DEBUG {metric_name} sample {i}: Converted list ref to dict format for copy/match task")
                else:
                    print(f"WARNING in {metric_name} sample {i}: Reference is not dict (type: {type(ref).__name__}, original: {original_ref_type}), value: {str(ref)[:200]}")
                    scores.append(0.0)
                    continue
            
            if isinstance(ref, dict):
                print(f"DEBUG {metric_name} sample {i}: ref is dict with keys: {list(ref.keys())}")
            
            # Call metric function
            try:
                score = metric_func(pred, ref, strict=False)
            except Exception as e:
                score = 0.0
            
            # Handle dict return values
            if isinstance(score, dict):
                score = score.get("score", 0.0)
            
            # Handle list/tuple return values - recursively flatten and take average if numeric
            if isinstance(score, (list, tuple)):
                if len(score) == 0:
                    score = 0.0
                else:
                    # Recursively flatten nested lists
                    def flatten_nested_list(lst):
                        result = []
                        for item in lst:
                            if isinstance(item, (list, tuple)):
                                result.extend(flatten_nested_list(item))
                            elif isinstance(item, (int, float)):
                                result.append(float(item))
                        return result
                    
                    numeric_scores = flatten_nested_list(score)
                    if numeric_scores:
                        score = sum(numeric_scores) / len(numeric_scores)
                    else:
                        score = 0.0
            
            # Convert to float
            try:
                if score is None:
                    score = 0.0
                elif isinstance(score, bool):
                    score = 1.0 if score else 0.0
                else:
                    score = float(score)
            except (ValueError, TypeError):
                score = 0.0
            
            scores.append(score)
        
        # Return list of scores - ensure all are floats and no nested structures
        result = []
        for s in scores:
            if isinstance(s, (list, tuple)):
                # This shouldn't happen, but handle it defensively
                print(f"WARNING in {metric_name}: Found list in scores: {s}, converting to 0.0")
                result.append(0.0)
            else:
                try:
                    result.append(float(s))
                except (ValueError, TypeError):
                    print(f"WARNING in {metric_name}: Could not convert score to float: {s} (type: {type(s)}), using 0.0")
                    result.append(0.0)
        
        # Final validation - ensure result is a flat list of floats
        final_result = []
        for i, val in enumerate(result):
            if isinstance(val, (list, tuple)):
                print(f"ERROR in {metric_name}: Found nested structure at index {i}: {val}")
                final_result.append(0.0)
            elif not isinstance(val, (int, float)):
                print(f"ERROR in {metric_name}: Found non-numeric value at index {i}: {val} (type: {type(val)})")
                final_result.append(0.0)
            else:
                final_result.append(float(val))
        
        # CRITICAL FIX: If lm-eval calls this function per-sample and expects a single value,
        # return the first (and only) score as a single float, not a list
        # But if we have multiple samples, return the list
        if len(final_result) == 1:
            # Return single value to avoid nesting when lm-eval collects results
            ret_value = final_result[0]
            print(f"DEBUG {metric_name}: Returning single value: {ret_value} (type: {type(ret_value).__name__})")
            print(f"DEBUG {metric_name}: Input lengths - predictions: {len(predictions)}, references: {len(references)}")
            return ret_value
        else:
            types_str = ', '.join([type(x).__name__ for x in final_result[:5]])
            print(f"DEBUG {metric_name}: Returning {len(final_result)} scores: {final_result[:5]}... (types: {types_str})")
            print(f"DEBUG {metric_name}: Input lengths - predictions: {len(predictions)}, references: {len(references)}")
            return final_result
    
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

