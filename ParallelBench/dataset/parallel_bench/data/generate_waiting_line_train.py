#!/usr/bin/env python3
"""
Generate training data for waiting_line tasks.

Usage:
    python generate_waiting_line_train.py --total_samples 50000 --seed_offset 100000
"""

import argparse
import sys
import yaml
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from dataset.parallel_bench.data.task import create_parallel_bench_task


def load_test_config():
    """Load test configuration as template."""
    test_config_path = Path(__file__).parent / "output" / "test" / "waiting_line" / "task_config.yaml"
    with open(test_config_path, "r") as f:
        return yaml.safe_load(f)


def generate_train_data(total_samples=50000, seed_offset=100000, task_ratios=None):
    """
    Generate training data for all waiting_line tasks.
    
    Args:
        total_samples: Total number of samples across all tasks
        seed_offset: Offset to add to test seed to ensure no overlap
        task_ratios: Dict of task_name -> ratio (default: equal for all tasks)
    """
    test_config = load_test_config()
    
    # Filter waiting_line tasks
    waiting_line_tasks = {
        name: config for name, config in test_config.items()
        if name.startswith("waiting_line/")
    }
    
    num_tasks = len(waiting_line_tasks)
    
    # Calculate samples per task
    if task_ratios is None:
        samples_per_task = total_samples // num_tasks
        samples_dict = {name: samples_per_task for name in waiting_line_tasks.keys()}
    else:
        total_ratio = sum(task_ratios.values())
        samples_dict = {
            name: int(total_samples * ratio / total_ratio)
            for name, ratio in task_ratios.items()
        }
    
    # Generate data for each task
    output_dir = Path(__file__).parent / "output" / "train" / "waiting_line"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for task_name, task_config in waiting_line_tasks.items():
        # Create train task config
        train_task = task_config.copy()
        
        # Modify seed
        train_task["seed"] = task_config["seed"] + seed_offset
        
        # Modify num_samples
        train_task["num_samples"] = samples_dict[task_name]
        
        # Remove samples_per_length for faster generation (use random sampling instead)
        train_task.pop("samples_per_length", None)
        
        # Generate data
        task_file_name = task_name.split("/")[-1] + ".jsonl"
        output_file = output_dir / task_file_name
        
        print(f"Generating {task_name}: {train_task['num_samples']} samples (seed: {train_task['seed']})")
        create_parallel_bench_task(
            split="train",
            task=train_task,
            output_file=str(output_file)
        )
    
    print(f"\nTraining data generated in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate waiting_line training data")
    parser.add_argument(
        "--total_samples",
        type=int,
        default=50000,
        help="Total number of samples across all tasks (default: 50000)"
    )
    parser.add_argument(
        "--seed_offset",
        type=int,
        default=100000,
        help="Offset to add to test seed (default: 100000)"
    )
    parser.add_argument(
        "--task_ratios",
        type=str,
        default=None,
        help="Task ratios as YAML file path or YAML string"
    )
    
    args = parser.parse_args()
    
    # Parse task ratios if provided
    task_ratios = None
    if args.task_ratios:
        ratios_path = Path(args.task_ratios)
        if ratios_path.exists():
            with open(ratios_path, "r") as f:
                task_ratios = yaml.safe_load(f)
        else:
            task_ratios = yaml.safe_load(args.task_ratios)
    
    generate_train_data(
        total_samples=args.total_samples,
        seed_offset=args.seed_offset,
        task_ratios=task_ratios
    )


if __name__ == "__main__":
    main()

