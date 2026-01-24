#!/usr/bin/env python3
"""
Generate augmented test data for all waiting_line tasks.
For each task in train/waiting_line, generate corresponding test data in test-aug/waiting_line
with different seeds.

Usage:
    python generate_all_test_aug.py --num_samples 5000
"""

import sys
import argparse
from pathlib import Path
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from dataset.parallel_bench.data.task import create_parallel_bench_task


def main(num_samples=5000):
    # Load train task config to get all tasks
    train_config_path = Path(__file__).parent / "output" / "train" / "waiting_line" / "task_config.yaml"
    
    print(f"Loading config from: {train_config_path}")
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
            print(f"Using test config as base (seed: {task_config.get('seed', 'N/A')})")
        else:
            task_config = train_config.copy()
            print(f"Using train config as base (seed: {task_config.get('seed', 'N/A')})")
        
        # Modify num_samples
        task_config["num_samples"] = num_samples
        # Remove samples_per_length to allow random sampling (faster generation)
        task_config.pop("samples_per_length", None)
        
        # Use a completely different seed from train to ensure no overlap
        # Strategy: Use a large offset (100000+) to ensure seed range separation
        # The str_to_seed function uses: hash(task_name) + seed, then mod (2^16-1)
        # So we use a large offset to ensure different final seeds even after modulo
        if "seed" in task_config:
            original_seed = task_config["seed"]
            # Get train seed for comparison
            train_seed = train_config.get("seed", 0)
            # Use a large offset (100000) to ensure complete separation
            # This ensures even after str_to_seed processing, seeds will be different
            task_config["seed"] = original_seed + 100000
            print(f"Train seed: {train_seed}, Test seed: {original_seed} -> New seed: {task_config['seed']}")
            print(f"  (Offset: +100000 to ensure no overlap with train data)")
        else:
            # If no seed in config, use a completely different range
            task_config["seed"] = 100000 + hash(task_name) % 10000
            print(f"Generated seed: {task_config['seed']} (using separate range)")
        
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
                
                # Store config for saving
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate augmented test data for all waiting_line tasks")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="Number of samples to generate for each task (default: 5000)"
    )
    args = parser.parse_args()
    main(num_samples=args.num_samples)

