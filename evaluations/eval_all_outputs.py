#!/usr/bin/env python3
"""
Evaluate all task outputs in outputs directory and print results table

Usage:
    python eval_all_outputs.py [--filter-format] [--verbose] [--csv CSV_FILE]
"""

import sys
from pathlib import Path
from eval_from_jsonl import evaluate
from collections import defaultdict
import argparse
import csv


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
    parser.add_argument('--csv', type=str, default=None,
                        help='Export results to CSV file (default: None, no export)')
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
    results_all = []  # Results without filtering
    results_filtered = []  # Results with filtering
    
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
            
            # Evaluate without filtering
            if args.verbose:
                score_all = evaluate(str(task_dir), verbose=True, filter_format=False)
            else:
                f = io.StringIO()
                with redirect_stdout(f):
                    score_all = evaluate(str(task_dir), verbose=False, filter_format=False)
            
            # Evaluate with filtering
            filtered_count = 0
            score_filtered = None
            if args.filter_format:
                original_count = len(predictions)
                filtered_predictions, filtered_references, filtered_indices, filtered_count = filter_by_format(
                    predictions, references, False
                )
                if args.verbose:
                    score_filtered = evaluate(str(task_dir), verbose=True, filter_format=True)
                else:
                    f = io.StringIO()
                    with redirect_stdout(f):
                        score_filtered = evaluate(str(task_dir), verbose=False, filter_format=True)
            else:
                # If filter_format is not enabled, use the same score for both
                score_filtered = score_all
            
            # Add results for both filtered and unfiltered
            result_base = {
                'model': model_name,
                'task': task_name or task_dir.name,
                'threshold': threshold or 'N/A',
                'remask': remask,
                'filtered_count': filtered_count,
            }
            
            results_all.append({
                **result_base,
                'score': score_all,
                'status': 'success' if score_all is not None else 'failed'
            })
            
            results_filtered.append({
                **result_base,
                'score': score_filtered,
                'status': 'success' if score_filtered is not None else 'failed'
            })
            
            if score_all is not None:
                print(f"  ✓ Accuracy (All): {score_all*100:.2f}%")
            if score_filtered is not None:
                print(f"  ✓ Accuracy (Filtered): {score_filtered*100:.2f}% (Filtered: {filtered_count})")
            if score_all is None and score_filtered is None:
                print(f"  ✗ Evaluation failed\n")
                
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            result_base = {
                'model': model_name,
                'task': task_name or task_dir.name,
                'threshold': threshold or 'N/A',
                'remask': remask,
                'score': None,
                'filtered_count': 0,
                'status': 'error',
                'error': str(e)
            }
            results_all.append(result_base)
            results_filtered.append(result_base)
    
    # Print results table for both filtered and unfiltered
    print("\n" + "="*100)
    print("Evaluation Results Summary")
    print("="*100)
    
    if not results_all:
        print("No results to display")
        return
    
    # Function to generate comparison table
    def generate_comparison_table(results, title_suffix=""):
        # Group by task, threshold, and remask for comparison
        # Structure: {(task, threshold, remask): {model: result}}
        comparison = defaultdict(dict)
        for r in results:
            if r['score'] is not None:
                key = (r['task'], r['threshold'], r['remask'])
                comparison[key][r['model']] = r
        
        # Prepare data for CSV export
        csv_rows = []
        
        # Print comparison table
        print(f"\n{'Task':<35} {'Threshold':<12} {'Remask':<8} {'Base':<12} {'Vanilla':<12} {'Mixture':<12} {'Filtered':<10}")
        print("-" * 120)
        
        # Sort by task, threshold, remask
        for (task, threshold, remask) in sorted(comparison.keys()):
            models = comparison[(task, threshold, remask)]
            base_result = models.get('llada2-base')
            vanilla_result = models.get('llada2-vanilla')
            mixture_result = models.get('llada2-mixture')
            
            threshold_str = str(threshold)
            remask_str = 'True' if remask else 'False' if remask is False else 'N/A'
            
            base_score = f"{base_result['score']*100:.2f}%" if base_result and base_result['score'] is not None else "N/A"
            vanilla_score = f"{vanilla_result['score']*100:.2f}%" if vanilla_result and vanilla_result['score'] is not None else "N/A"
            mixture_score = f"{mixture_result['score']*100:.2f}%" if mixture_result and mixture_result['score'] is not None else "N/A"
            
            # Filtered count (show from mixture if available, else vanilla, else base)
            filtered_result = mixture_result if mixture_result else (vanilla_result if vanilla_result else base_result)
            filtered_str = str(filtered_result.get('filtered_count', 0)) if filtered_result else "0"
            filtered_count = filtered_result.get('filtered_count', 0) if filtered_result else 0
            
            print(f"{task:<35} {threshold_str:<12} {remask_str:<8} {base_score:<12} {vanilla_score:<12} {mixture_score:<12} {filtered_str:<10}")
            
            # Prepare CSV row
            csv_rows.append({
                'Task': task,
                'Threshold': threshold_str,
                'Remask': remask_str,
                'Base_Accuracy': base_result['score'] * 100 if base_result and base_result['score'] is not None else None,
                'Vanilla_Accuracy': vanilla_result['score'] * 100 if vanilla_result and vanilla_result['score'] is not None else None,
                'Mixture_Accuracy': mixture_result['score'] * 100 if mixture_result and mixture_result['score'] is not None else None,
                'Filtered_Count': filtered_count
            })
        
        return csv_rows, comparison
    
    # Generate table for all results (no filtering)
    print("\n" + "="*100)
    print("Results WITHOUT Format Filtering")
    print("="*100)
    csv_rows_all, comparison_all = generate_comparison_table(results_all, " (All Samples)")
    
    # Generate table for filtered results
    print("\n" + "="*100)
    print("Results WITH Format Filtering")
    print("="*100)
    csv_rows_filtered, comparison_filtered = generate_comparison_table(results_filtered, " (Filtered)")
    
    # Export to CSV if requested
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export unfiltered results
        csv_path_all = csv_path.parent / f"{csv_path.stem}_all{csv_path.suffix}"
        with open(csv_path_all, 'w', newline='') as f:
            if csv_rows_all:
                fieldnames = ['Task', 'Threshold', 'Remask', 'Base_Accuracy', 'Vanilla_Accuracy', 'Mixture_Accuracy', 'Filtered_Count']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows_all)
        print(f"\n✓ Results (All) exported to CSV: {csv_path_all}")
        
        # Export filtered results
        csv_path_filtered = csv_path.parent / f"{csv_path.stem}_filtered{csv_path.suffix}"
        with open(csv_path_filtered, 'w', newline='') as f:
            if csv_rows_filtered:
                fieldnames = ['Task', 'Threshold', 'Remask', 'Base_Accuracy', 'Vanilla_Accuracy', 'Mixture_Accuracy', 'Filtered_Count']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows_filtered)
        print(f"✓ Results (Filtered) exported to CSV: {csv_path_filtered}")
    
    # Statistics by model (for both filtered and unfiltered)
    def print_statistics(results, title):
        print(f"\n{title}")
        print("-" * 100)
        model_stats = defaultdict(list)
        for r in results:
            if r['score'] is not None:
                model_stats[r['model']].append(r['score'])
        
        if model_stats:
            print("Statistics by Model:")
            for model in sorted(model_stats.keys()):
                scores = model_stats[model]
                avg = sum(scores) / len(scores)
                min_score = min(scores)
                max_score = max(scores)
                print(f"  {model:<20} Avg: {avg*100:.2f}%, Min: {min_score*100:.2f}%, Max: {max_score*100:.2f}% ({len(scores)} tasks)")
        
        # Overall comparison summary
        print("\n" + "-" * 100)
        base_scores = [r['score'] for r in results if r['model'] == 'llada2-base' and r['score'] is not None]
        vanilla_scores = [r['score'] for r in results if r['model'] == 'llada2-vanilla' and r['score'] is not None]
        mixture_scores = [r['score'] for r in results if r['model'] == 'llada2-mixture' and r['score'] is not None]
        
        print(f"Overall Comparison:")
        if base_scores:
            base_avg = sum(base_scores) / len(base_scores)
            print(f"  Base: {base_avg*100:.2f}% ({len(base_scores)} tasks)")
        if vanilla_scores:
            vanilla_avg = sum(vanilla_scores) / len(vanilla_scores)
            print(f"  Vanilla: {vanilla_avg*100:.2f}% ({len(vanilla_scores)} tasks)")
        if mixture_scores:
            mixture_avg = sum(mixture_scores) / len(mixture_scores)
            print(f"  Mixture: {mixture_avg*100:.2f}% ({len(mixture_scores)} tasks)")
        
        # Calculate differences if multiple models available
        if base_scores and vanilla_scores:
            base_avg = sum(base_scores) / len(base_scores)
            vanilla_avg = sum(vanilla_scores) / len(vanilla_scores)
            print(f"  Vanilla - Base: {(vanilla_avg - base_avg)*100:+.2f}%")
        if vanilla_scores and mixture_scores:
            vanilla_avg = sum(vanilla_scores) / len(vanilla_scores)
            mixture_avg = sum(mixture_scores) / len(mixture_scores)
            print(f"  Mixture - Vanilla: {(mixture_avg - vanilla_avg)*100:+.2f}%")
        if base_scores and mixture_scores:
            base_avg = sum(base_scores) / len(base_scores)
            mixture_avg = sum(mixture_scores) / len(mixture_scores)
            print(f"  Mixture - Base: {(mixture_avg - base_avg)*100:+.2f}%")
        
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
    
    print_statistics(results_all, "Statistics WITHOUT Format Filtering")
    print("\n" + "="*100)
    print_statistics(results_filtered, "Statistics WITH Format Filtering")
    print("="*100)


if __name__ == "__main__":
    main()

