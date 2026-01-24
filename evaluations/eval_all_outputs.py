#!/usr/bin/env python3
"""
Evaluate all task outputs in outputs directory and print results table

Usage:
    python eval_all_outputs.py [--filter-format] [--verbose]
"""

import sys
from pathlib import Path
from eval_from_jsonl import evaluate
from collections import defaultdict
import argparse


def find_all_task_dirs(outputs_dir: Path):
    """Find all task output directories"""
    task_dirs = []
    for model_dir in sorted(outputs_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for task_dir in sorted(model_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            # 检查是否有rank_0.jsonl文件
            if (task_dir / "rank_0.jsonl").exists():
                task_dirs.append((model_name, task_dir))
    return task_dirs


def extract_task_info(task_dir: Path):
    """Extract task information from task directory name"""
    name = task_dir.name
    # 格式: waiting_line_copy_threshold_0_80_remask_False
    parts = name.split('_')
    
    # 提取任务名 (waiting_line_copy)
    task_name = None
    for i in range(len(parts)):
        if parts[i] == 'threshold':
            task_name = '_'.join(parts[:i])
            break
    
    # 提取threshold
    threshold = None
    remask = None
    if 'threshold' in name:
        threshold_idx = name.find('threshold_')
        if threshold_idx != -1:
            threshold_part = name[threshold_idx + len('threshold_'):]
            threshold = threshold_part.split('_remask_')[0].replace('_', '.')
    
    # 提取remask
    if '_remask_' in name:
        remask_part = name.split('_remask_')[-1]
        remask = remask_part == 'True'
    
    return task_name, threshold, remask


def main():
    parser = argparse.ArgumentParser(description='Evaluate all task outputs in outputs directory')
    parser.add_argument('--filter-format', '-f', action='store_true',
                        help='Filter out malformed samples')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show verbose output')
    parser.add_argument('--outputs-dir', type=str, default='outputs',
                        help='Outputs directory path (default: outputs)')
    args = parser.parse_args()
    
    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.exists():
        print(f"Error: outputs directory does not exist: {outputs_dir}")
        sys.exit(1)
    
    # Find all task directories
    task_dirs = find_all_task_dirs(outputs_dir)
    
    if not task_dirs:
        print(f"No task output directories found")
        sys.exit(1)
    
    print(f"Found {len(task_dirs)} task output directories\n")
    
    # 评估所有任务
    results = []
    for idx, (model_name, task_dir) in enumerate(task_dirs, 1):
        task_name, threshold, remask = extract_task_info(task_dir)
        
        print(f"[{idx}/{len(task_dirs)}] Evaluating: {model_name}/{task_dir.name}")
        
        try:
            # Evaluate (suppress detailed output unless verbose is specified)
            import io
            from contextlib import redirect_stdout
            from eval_from_jsonl import load_predictions, load_references, filter_by_format, TASK_METRIC_MAP, TASK_DATA_MAP, extract_task_name_from_results, extract_task_name
            from pathlib import Path as PathLib
            
            # Get task name and data file
            task_dir_path = PathLib(task_dir).resolve()
            task_name_eval = extract_task_name_from_results(task_dir_path)
            if task_name_eval is None:
                task_name_eval = extract_task_name(str(task_dir_path))
            data_file = TASK_DATA_MAP[task_name_eval]
            
            # Load data
            jsonl_file = task_dir_path / "rank_0.jsonl"
            predictions = load_predictions(str(jsonl_file))
            references = load_references(data_file)
            
            # Calculate filtered count
            filtered_count = 0
            if args.filter_format:
                original_count = len(predictions)
                filtered_predictions, filtered_references, filtered_indices, filtered_count = filter_by_format(
                    predictions, references, False
                )
            else:
                filtered_count = 0
            
            # Evaluate
            if args.verbose:
                score = evaluate(str(task_dir), verbose=True, filter_format=args.filter_format)
            else:
                # Capture output
                f = io.StringIO()
                with redirect_stdout(f):
                    score = evaluate(str(task_dir), verbose=False, filter_format=args.filter_format)
            
            results.append({
                'model': model_name,
                'task': task_name or task_dir.name,
                'threshold': threshold or 'N/A',
                'remask': remask,
                'score': score,
                'filtered_count': filtered_count,
                'status': 'success'
            })
            
            if score is not None:
                print(f"  ✓ Accuracy: {score*100:.2f}%\n")
            else:
                print(f"  ✗ Evaluation failed\n")
                results[-1]['status'] = 'failed'
                
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            results.append({
                'model': model_name,
                'task': task_name or task_dir.name,
                'threshold': threshold or 'N/A',
                'remask': remask,
                'score': None,
                'filtered_count': 0,
                'status': 'error',
                'error': str(e)
            })
    
    # Print results table
    print("\n" + "="*100)
    print("Evaluation Results Summary")
    print("="*100)
    
    if not results:
        print("No results to display")
        return
    
    # Group by model and task
    grouped = defaultdict(list)
    for r in results:
        key = (r['model'], r['task'])
        grouped[key].append(r)
    
    # Print table
    print(f"\n{'Model':<20} {'Task':<30} {'Threshold':<12} {'Remask':<8} {'Accuracy':<12} {'Filtered':<10} {'Status':<10}")
    print("-" * 110)
    
    for (model, task), items in sorted(grouped.items()):
        for item in sorted(items, key=lambda x: (x['threshold'], x['remask'])):
            threshold_str = str(item['threshold'])
            remask_str = 'True' if item['remask'] else 'False' if item['remask'] is False else 'N/A'
            score_str = f"{item['score']*100:.2f}%" if item['score'] is not None else "N/A"
            filtered_str = str(item.get('filtered_count', 0))
            status_str = item['status']
            
            print(f"{model:<20} {task:<30} {threshold_str:<12} {remask_str:<8} {score_str:<12} {filtered_str:<10} {status_str:<10}")
    
    # Statistics by task
    print("\n" + "-" * 100)
    task_stats = defaultdict(list)
    for r in results:
        if r['score'] is not None:
            task_stats[r['task']].append(r['score'])
    
    if task_stats:
        print("Statistics by Task:")
        for task, scores in sorted(task_stats.items()):
            avg = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            print(f"  {task:<30} Avg: {avg*100:.2f}%, Min: {min_score*100:.2f}%, Max: {max_score*100:.2f}%")
    
    # Overall statistics
    print("\n" + "-" * 100)
    success_count = sum(1 for r in results if r['status'] == 'success' and r['score'] is not None)
    failed_count = len(results) - success_count
    
    if success_count > 0:
        avg_score = sum(r['score'] for r in results if r['score'] is not None) / success_count
        print(f"Total: {len(results)} tasks, Success: {success_count}, Failed: {failed_count}")
        print(f"Average Accuracy: {avg_score*100:.2f}%")
    else:
        print(f"Total: {len(results)} tasks, Success: {success_count}, Failed: {failed_count}")
    
    print("="*100)


if __name__ == "__main__":
    main()

