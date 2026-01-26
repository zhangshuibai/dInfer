#!/usr/bin/env python3
"""
Analyze remask and threshold effects on ParallelBench tasks

Usage:
    python analyze.py [--all-csv ALL_CSV] [--filtered-csv FILTERED_CSV] [--output-dir OUTPUT_DIR]
"""

import argparse
import csv
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Task categories
CERTAIN_TASKS = [
    'waiting_line_copy',
    'waiting_line_reverse',
    'waiting_line_sort',
    'waiting_line_insert_index',
    'waiting_line_remove_index',
    'waiting_line_replace_index'
]

UNCERTAIN_TASKS = [
    'waiting_line_shuffle',
    'waiting_line_insert_random',
    'waiting_line_remove_random',
    'waiting_line_replace_random'
]

ALL_TASKS = CERTAIN_TASKS + UNCERTAIN_TASKS


def load_csv(csv_path):
    """Load CSV file and return DataFrame"""
    df = pd.read_csv(csv_path)
    return df


def calculate_averages(df, tasks, models=['Base_Accuracy', 'Vanilla_Accuracy', 'Mixture_Accuracy']):
    """Calculate average accuracy for given tasks and models"""
    results = {}
    for model in models:
        model_col = model
        values = []
        for task in tasks:
            task_rows = df[df['Task'] == task]
            if not task_rows.empty:
                value = task_rows[model_col].iloc[0]
                if pd.notna(value):
                    values.append(value)
        if values:
            results[model] = round(sum(values) / len(values), 2)
        else:
            results[model] = None
    return results


def generate_table(df_data, remask, threshold, filter_mode, output_dir):
    """Generate analysis table for specific remask, threshold, and filter mode"""
    
    # Filter data - use direct comparison for better accuracy
    # Handle remask (could be bool or string)
    if isinstance(remask, bool):
        remask_filter = df_data['Remask'] == remask
    else:
        # Convert to bool if string
        remask_bool = str(remask).lower() == 'true'
        remask_filter = df_data['Remask'] == remask_bool
    
    # Handle threshold (use float comparison, handle both 0.8 and 0.80)
    threshold_filter = (df_data['Threshold'] == threshold) | (df_data['Threshold'] == float(threshold))
    
    # Combine filters
    filter_condition = remask_filter & threshold_filter
    
    df_filtered = df_data[filter_condition].copy()
    
    if df_filtered.empty:
        print(f"  No data for remask={remask}, threshold={threshold}, filter={filter_mode}")
        return
    
    # Prepare table data
    table_data = []
    
    # Process each task
    for task in ALL_TASKS:
        row = df_filtered[df_filtered['Task'] == task]
        
        if row.empty:
            continue
        
        # Get values (use first row if exists) and round to 2 decimal places
        base = round(row['Base_Accuracy'].iloc[0], 2) if pd.notna(row['Base_Accuracy'].iloc[0]) else None
        vanilla = round(row['Vanilla_Accuracy'].iloc[0], 2) if pd.notna(row['Vanilla_Accuracy'].iloc[0]) else None
        mixture = round(row['Mixture_Accuracy'].iloc[0], 2) if pd.notna(row['Mixture_Accuracy'].iloc[0]) else None
        
        table_data.append({
            'Task': task,
            'Category': 'Certain' if task in CERTAIN_TASKS else 'Uncertain',
            'Base_Accuracy': base,
            'Vanilla_Accuracy': vanilla,
            'Mixture_Accuracy': mixture,
        })
    
    # Calculate averages
    # All tasks average
    all_avg = calculate_averages(df_filtered, ALL_TASKS)
    
    # Certain tasks average
    certain_avg = calculate_averages(df_filtered, CERTAIN_TASKS)
    
    # Uncertain tasks average
    uncertain_avg = calculate_averages(df_filtered, UNCERTAIN_TASKS)
    
    # Add average rows
    table_data.append({
        'Task': '=== Average (All Tasks) ===',
        'Category': '',
        'Base_Accuracy': all_avg.get('Base_Accuracy'),
        'Vanilla_Accuracy': all_avg.get('Vanilla_Accuracy'),
        'Mixture_Accuracy': all_avg.get('Mixture_Accuracy'),
    })
    
    table_data.append({
        'Task': '=== Average (Certain Tasks) ===',
        'Category': 'Certain',
        'Base_Accuracy': certain_avg.get('Base_Accuracy'),
        'Vanilla_Accuracy': certain_avg.get('Vanilla_Accuracy'),
        'Mixture_Accuracy': certain_avg.get('Mixture_Accuracy'),
    })
    
    table_data.append({
        'Task': '=== Average (Uncertain Tasks) ===',
        'Category': 'Uncertain',
        'Base_Accuracy': uncertain_avg.get('Base_Accuracy'),
        'Vanilla_Accuracy': uncertain_avg.get('Vanilla_Accuracy'),
        'Mixture_Accuracy': uncertain_avg.get('Mixture_Accuracy'),
    })
    
    # Create DataFrame
    df_table = pd.DataFrame(table_data)
    
    # Format output filename
    remask_str = 'True' if remask else 'False'
    threshold_str = str(threshold).replace('.', '_')
    filename = f"remask_{remask_str}_threshold_{threshold_str}_{filter_mode}.csv"
    output_path = output_dir / filename
    
    # Save to CSV
    df_table.to_csv(output_path, index=False)
    print(f"  Saved: {filename}")
    
    # Also print summary
    print(f"    All Tasks - Base: {all_avg.get('Base_Accuracy', 'N/A'):.2f}%, Vanilla: {all_avg.get('Vanilla_Accuracy', 'N/A'):.2f}%, Mixture: {all_avg.get('Mixture_Accuracy', 'N/A'):.2f}%")
    print(f"    Certain Tasks - Base: {certain_avg.get('Base_Accuracy', 'N/A'):.2f}%, Vanilla: {certain_avg.get('Vanilla_Accuracy', 'N/A'):.2f}%, Mixture: {certain_avg.get('Mixture_Accuracy', 'N/A'):.2f}%")
    print(f"    Uncertain Tasks - Base: {uncertain_avg.get('Base_Accuracy', 'N/A'):.2f}%, Vanilla: {uncertain_avg.get('Vanilla_Accuracy', 'N/A'):.2f}%, Mixture: {uncertain_avg.get('Mixture_Accuracy', 'N/A'):.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Analyze remask and threshold effects on ParallelBench tasks')
    parser.add_argument('--all-csv', type=str, default='/data/dInfer/evaluations/results_all.csv',
                        help='Path to results_all.csv (default: /data/dInfer/evaluations/results_all.csv)')
    parser.add_argument('--filtered-csv', type=str, default='/data/dInfer/evaluations/results_filtered.csv',
                        help='Path to results_filtered.csv (default: /data/dInfer/evaluations/results_filtered.csv)')
    parser.add_argument('--output-dir', type=str, default='/data/dInfer/evaluations/analyze_remask_parallelbench',
                        help='Output directory for analysis tables (default: /data/dInfer/evaluations/analyze_remask_parallelbench)')
    args = parser.parse_args()
    
    # Load CSV files
    print("Loading CSV files...")
    df_all = load_csv(args.all_csv)
    df_filtered = load_csv(args.filtered_csv)
    print(f"  Loaded {len(df_all)} rows from {args.all_csv}")
    print(f"  Loaded {len(df_filtered)} rows from {args.filtered_csv}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Get unique remask and threshold values
    # Handle remask values (could be string or boolean)
    remask_unique = df_all['Remask'].unique()
    remask_values = []
    for r in remask_unique:
        if pd.notna(r):
            if isinstance(r, bool):
                remask_values.append(r)
            elif str(r).lower() == 'true':
                remask_values.append(True)
            elif str(r).lower() == 'false':
                remask_values.append(False)
    remask_values = sorted(set(remask_values))
    
    # Handle threshold values
    threshold_values = []
    for t in df_all['Threshold'].unique():
        if pd.notna(t):
            try:
                threshold_values.append(float(t))
            except (ValueError, TypeError):
                pass
    threshold_values = sorted(set(threshold_values))
    
    print(f"\nRemask values: {remask_values}")
    print(f"Threshold values: {threshold_values}")
    
    # Generate tables for each combination
    # Create subdirectory for remask-threshold tables
    remask_threshold_dir = output_dir / "remask_threshold"
    remask_threshold_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating remask-threshold analysis tables...")
    for remask in remask_values:
        for threshold in threshold_values:
            # Generate table for all samples (no filtering)
            print(f"\nProcessing: remask={remask}, threshold={threshold}, filter=all")
            generate_table(df_all, remask, threshold, 'all', remask_threshold_dir)
            
            # Generate table for filtered samples
            print(f"\nProcessing: remask={remask}, threshold={threshold}, filter=filtered")
            generate_table(df_filtered, remask, threshold, 'filtered', remask_threshold_dir)
    
    print(f"\n✓ Remask-threshold analysis complete! Tables saved to {remask_threshold_dir}")
    
    # Generate cross-remask comparison tables
    print("\n" + "="*100)
    print("Generating cross-remask comparison tables...")
    print("="*100)
    
    cross_output_dir = output_dir / "cross_remask"
    cross_output_dir.mkdir(parents=True, exist_ok=True)
    
    for threshold in threshold_values:
        for filter_mode in ['all', 'filtered']:
            print(f"\nProcessing: threshold={threshold}, filter={filter_mode}")
            generate_cross_remask_table(df_all if filter_mode == 'all' else df_filtered, threshold, filter_mode, cross_output_dir)
    
    print(f"\n✓ Cross-remask comparison tables saved to {cross_output_dir}")


def generate_cross_remask_table(df_data, threshold, filter_mode, output_dir):
    """Generate cross-remask comparison table for specific threshold and filter mode"""
    
    # Filter by threshold
    threshold_filter = (df_data['Threshold'] == threshold) | (df_data['Threshold'] == float(threshold))
    df_filtered = df_data[threshold_filter].copy()
    
    if df_filtered.empty:
        print(f"  No data for threshold={threshold}, filter={filter_mode}")
        return
    
    # Prepare table data
    table_data = []
    
    # Process each task
    for task in ALL_TASKS:
        # Get data for remask=False and remask=True
        row_false = df_filtered[(df_filtered['Task'] == task) & (df_filtered['Remask'] == False)]
        row_true = df_filtered[(df_filtered['Task'] == task) & (df_filtered['Remask'] == True)]
        
        if row_false.empty and row_true.empty:
            continue
        
        # Extract values and round to 2 decimal places
        base_false = round(row_false['Base_Accuracy'].iloc[0], 2) if not row_false.empty and pd.notna(row_false['Base_Accuracy'].iloc[0]) else None
        base_true = round(row_true['Base_Accuracy'].iloc[0], 2) if not row_true.empty and pd.notna(row_true['Base_Accuracy'].iloc[0]) else None
        
        vanilla_false = round(row_false['Vanilla_Accuracy'].iloc[0], 2) if not row_false.empty and pd.notna(row_false['Vanilla_Accuracy'].iloc[0]) else None
        vanilla_true = round(row_true['Vanilla_Accuracy'].iloc[0], 2) if not row_true.empty and pd.notna(row_true['Vanilla_Accuracy'].iloc[0]) else None
        
        mixture_false = round(row_false['Mixture_Accuracy'].iloc[0], 2) if not row_false.empty and pd.notna(row_false['Mixture_Accuracy'].iloc[0]) else None
        mixture_true = round(row_true['Mixture_Accuracy'].iloc[0], 2) if not row_true.empty and pd.notna(row_true['Mixture_Accuracy'].iloc[0]) else None
        
        table_data.append({
            'Task': task,
            'Category': 'Certain' if task in CERTAIN_TASKS else 'Uncertain',
            'Base_Remask_False': base_false,
            'Base_Remask_True': base_true,
            'Vanilla_Remask_False': vanilla_false,
            'Vanilla_Remask_True': vanilla_true,
            'Mixture_Remask_False': mixture_false,
            'Mixture_Remask_True': mixture_true,
        })
    
    # Calculate averages for each combination
    def calculate_cross_avg(tasks, remask_value, model_col):
        values = []
        for task in tasks:
            task_rows = df_filtered[(df_filtered['Task'] == task) & (df_filtered['Remask'] == remask_value)]
            if not task_rows.empty:
                value = task_rows[model_col].iloc[0]
                if pd.notna(value):
                    values.append(value)
        return round(sum(values) / len(values), 2) if values else None
    
    # All tasks averages
    all_base_false = calculate_cross_avg(ALL_TASKS, False, 'Base_Accuracy')
    all_base_true = calculate_cross_avg(ALL_TASKS, True, 'Base_Accuracy')
    all_vanilla_false = calculate_cross_avg(ALL_TASKS, False, 'Vanilla_Accuracy')
    all_vanilla_true = calculate_cross_avg(ALL_TASKS, True, 'Vanilla_Accuracy')
    all_mixture_false = calculate_cross_avg(ALL_TASKS, False, 'Mixture_Accuracy')
    all_mixture_true = calculate_cross_avg(ALL_TASKS, True, 'Mixture_Accuracy')
    
    # Certain tasks averages
    certain_base_false = calculate_cross_avg(CERTAIN_TASKS, False, 'Base_Accuracy')
    certain_base_true = calculate_cross_avg(CERTAIN_TASKS, True, 'Base_Accuracy')
    certain_vanilla_false = calculate_cross_avg(CERTAIN_TASKS, False, 'Vanilla_Accuracy')
    certain_vanilla_true = calculate_cross_avg(CERTAIN_TASKS, True, 'Vanilla_Accuracy')
    certain_mixture_false = calculate_cross_avg(CERTAIN_TASKS, False, 'Mixture_Accuracy')
    certain_mixture_true = calculate_cross_avg(CERTAIN_TASKS, True, 'Mixture_Accuracy')
    
    # Uncertain tasks averages
    uncertain_base_false = calculate_cross_avg(UNCERTAIN_TASKS, False, 'Base_Accuracy')
    uncertain_base_true = calculate_cross_avg(UNCERTAIN_TASKS, True, 'Base_Accuracy')
    uncertain_vanilla_false = calculate_cross_avg(UNCERTAIN_TASKS, False, 'Vanilla_Accuracy')
    uncertain_vanilla_true = calculate_cross_avg(UNCERTAIN_TASKS, True, 'Vanilla_Accuracy')
    uncertain_mixture_false = calculate_cross_avg(UNCERTAIN_TASKS, False, 'Mixture_Accuracy')
    uncertain_mixture_true = calculate_cross_avg(UNCERTAIN_TASKS, True, 'Mixture_Accuracy')
    
    # Add average rows
    table_data.append({
        'Task': '=== Average (All Tasks) ===',
        'Category': '',
        'Base_Remask_False': all_base_false,
        'Base_Remask_True': all_base_true,
        'Vanilla_Remask_False': all_vanilla_false,
        'Vanilla_Remask_True': all_vanilla_true,
        'Mixture_Remask_False': all_mixture_false,
        'Mixture_Remask_True': all_mixture_true,
    })
    
    table_data.append({
        'Task': '=== Average (Certain Tasks) ===',
        'Category': 'Certain',
        'Base_Remask_False': certain_base_false,
        'Base_Remask_True': certain_base_true,
        'Vanilla_Remask_False': certain_vanilla_false,
        'Vanilla_Remask_True': certain_vanilla_true,
        'Mixture_Remask_False': certain_mixture_false,
        'Mixture_Remask_True': certain_mixture_true,
    })
    
    table_data.append({
        'Task': '=== Average (Uncertain Tasks) ===',
        'Category': 'Uncertain',
        'Base_Remask_False': uncertain_base_false,
        'Base_Remask_True': uncertain_base_true,
        'Vanilla_Remask_False': uncertain_vanilla_false,
        'Vanilla_Remask_True': uncertain_vanilla_true,
        'Mixture_Remask_False': uncertain_mixture_false,
        'Mixture_Remask_True': uncertain_mixture_true,
    })
    
    # Create DataFrame
    df_table = pd.DataFrame(table_data)
    
    # Format output filename
    threshold_str = str(threshold).replace('.', '_')
    filename = f"threshold_{threshold_str}_{filter_mode}.csv"
    output_path = output_dir / filename
    
    # Save to CSV
    df_table.to_csv(output_path, index=False)
    print(f"  Saved: {filename}")
    
    # Print summary
    print(f"    All Tasks - Base(False): {all_base_false:.2f}%, Base(True): {all_base_true:.2f}%")
    print(f"    All Tasks - Vanilla(False): {all_vanilla_false:.2f}%, Vanilla(True): {all_vanilla_true:.2f}%")
    print(f"    All Tasks - Mixture(False): {all_mixture_false:.2f}%, Mixture(True): {all_mixture_true:.2f}%")


if __name__ == "__main__":
    main()

