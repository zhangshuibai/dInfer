#!/usr/bin/env python3
"""
Regenerate training data for waiting_line/insert_random task with 5000 samples.

Usage:
    python regenerate_insert_random_train.py
"""

import sys
from pathlib import Path
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from dataset.parallel_bench.data.task import create_parallel_bench_task


def main():
    # Load training task config
    train_config_path = Path(__file__).parent / "output" / "train" / "waiting_line" / "task_config.yaml"
    
    print(f"Loading config from: {train_config_path}")
    with open(train_config_path, "r") as f:
        task_configs = yaml.safe_load(f)
    
    # Get insert_random task config
    task_name = "waiting_line/insert_random"
    if task_name not in task_configs:
        raise ValueError(f"Task {task_name} not found in config file")
    
    task_config = task_configs[task_name].copy()
    
    # Ensure num_samples is 5000
    task_config["num_samples"] = 5000
    print(f"\nTask: {task_name}")
    print(f"Number of samples: {task_config['num_samples']}")
    print(f"Seed: {task_config['seed']}")
    
    # Set output file
    output_file = Path(__file__).parent / "output" / "train" / "waiting_line" / "insert_random.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating {task_config['num_samples']} samples...")
    print(f"Output file: {output_file}")
    
    # Generate the data
    create_parallel_bench_task(
        split="train",
        task=task_config,
        output_file=str(output_file)
    )
    
    # Verify the generated file
    if output_file.exists():
        with open(output_file, 'r') as f:
            line_count = sum(1 for _ in f)
        print(f"\n✓ Successfully generated {line_count} samples")
        if line_count == task_config['num_samples']:
            print("✓ Sample count matches expected value")
        else:
            print(f"⚠ Warning: Expected {task_config['num_samples']} samples, got {line_count}")
    else:
        print("✗ Error: Output file was not created")


if __name__ == "__main__":
    main()

