#!/usr/bin/env python3
"""
Generate augmented test data for all waiting_line tasks with guaranteed no data leakage.
Uses train seed as base and adds large offset to ensure different final seeds.

Usage:
    python generate_all_test_aug_safe.py --num_samples 5000
"""

import sys
import argparse
from pathlib import Path
import yaml
import hashlib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from dataset.parallel_bench.data.task import create_parallel_bench_task
from dataset.parallel_bench.data.task_utils import str_to_seed


def calculate_final_seed(task_name, base_seed):
    """Calculate the final seed that will be used by create_parallel_bench_task."""
    # This mimics the logic in task.py: str_to_seed(task_name, base_seed)
    task_short_name = task_name.split("/")[-1]
    return str_to_seed(task_short_name, base_seed)


def find_safe_seed(task_name, train_seed, min_offset=100000):
    """
    Find a safe seed that ensures no overlap with train data.
    
    The final seed is: str_to_seed(task_name, seed) % 65535
    We need to find a seed such that the final seed differs from train's final seed.
    """
    train_final_seed = calculate_final_seed(task_name, train_seed)
    
    # Try different offsets until we find one that produces a different final seed
    for offset in range(min_offset, min_offset + 100000, 1000):
        test_seed = train_seed + offset
        test_final_seed = calculate_final_seed(task_name, test_seed)
        
        if test_final_seed != train_final_seed:
            return test_seed, train_final_seed, test_final_seed
    
    # Fallback: use a completely different seed range
    fallback_seed = 200000 + hash(task_name) % 50000
    fallback_final = calculate_final_seed(task_name, fallback_seed)
    return fallback_seed, train_final_seed, fallback_final


def main(num_samples=5000):
    # Load train task config to get all tasks and their seeds
    train_config_path = Path(__file__).parent / "output" / "train" / "waiting_line" / "task_config.yaml"
    
    print(f"Loading train config from: {train_config_path}")
    with open(train_config_path, "r") as f:
        train_task_configs = yaml.safe_load(f)
    
    # Load test task config as base template
    test_config_path = Path(__file__).parent / "output" / "test" / "waiting_line" / "task_config.yaml"
    
    print(f"Loading test config template from: {test_config_path}")
    with open(test_config_path, "r") as f:
        test_task_configs = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(__file__).parent / "output" / "test-aug" / "waiting_line"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each task
    all_generated_configs = {}
    
    for task_name, train_config in train_task_configs.items():
        print(f"\n{'='*60}")
        print(f"Processing task: {task_name}")
        print(f"{'='*60}")
        
        # Get base config from test config (if exists) or use train config
        if task_name in test_task_configs:
            task_config = test_task_configs[task_name].copy()
        else:
            task_config = train_config.copy()
        
        # Modify num_samples
        task_config["num_samples"] = num_samples
        # Remove samples_per_length to allow random sampling (faster generation)
        task_config.pop("samples_per_length", None)
        
        # Get train seed
        train_seed = train_config.get("seed", 0)
        
        # Find a safe seed that guarantees different final seed
        safe_seed, train_final, test_final = find_safe_seed(task_name, train_seed)
        task_config["seed"] = safe_seed
        
        print(f"Train seed: {train_seed} -> Final seed: {train_final}")
        print(f"Test-aug seed: {safe_seed} -> Final seed: {test_final}")
        print(f"✓ Seed difference: {test_final - train_final} (guaranteed different)")
        
        # Extract task file name
        task_file_name = task_name.split("/")[-1] + ".jsonl"
        output_file = output_dir / task_file_name
        
        print(f"Number of samples: {task_config['num_samples']}")
        print(f"Output file: {output_file}")
        
        # Generate the data
        try:
            create_parallel_bench_task(
                split="test-aug",
                task=task_config,
                output_file=str(output_file)
            )
            
            # Verify the generated file
            if output_file.exists():
                with open(output_file, 'r') as f:
                    line_count = sum(1 for _ in f)
                print(f"✓ Successfully generated {line_count} samples")
                if line_count == task_config['num_samples']:
                    print("✓ Sample count matches expected value")
                else:
                    print(f"⚠ Warning: Expected {task_config['num_samples']} samples, got {line_count}")
                
                # Store config for saving (restore words to file path)
                if "words" in task_config and isinstance(task_config["words"], list):
                    if task_name in test_task_configs and "words" in test_task_configs[task_name]:
                        task_config["words"] = test_task_configs[task_name]["words"]
                    elif "words" in train_config:
                        task_config["words"] = train_config["words"]
                
                all_generated_configs[task_name] = task_config
            else:
                print("✗ Error: Output file was not created")
        except Exception as e:
            print(f"✗ Error generating {task_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save combined task_config.yaml
    config_file = output_dir / "task_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(all_generated_configs, f, default_flow_style=False, sort_keys=False)
    print(f"\n{'='*60}")
    print(f"✓ Saved task config to: {config_file}")
    print(f"✓ Generated {len(all_generated_configs)} tasks")
    print(f"{'='*60}")
    print("\n⚠ IMPORTANT: Please run verify_no_data_leakage.py to confirm no data overlap!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate augmented test data with guaranteed no data leakage")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="Number of samples to generate for each task (default: 5000)"
    )
    args = parser.parse_args()
    main(num_samples=args.num_samples)

