#!/usr/bin/env python3
"""
Test script to observe token distribution across experts in LLaDA2 MoE model.

This script:
1. Loads a LLaDA2 MoE model
2. Enables token distribution statistics
3. Runs forward passes on sample inputs
4. Displays token distribution for each expert in each MoE layer

Usage:
    # Simple test with dummy data
    python test_expert_token_distribution.py

    # Test with a specific model
    python test_expert_token_distribution.py --model_path /data/models/LLaDA2.0-mini-preview

    # Test with custom batch size and sequence length
    python test_expert_token_distribution.py --batch_size 4 --seq_length 128
"""

import argparse
import torch
import sys
import os
from datetime import datetime

# Add the dInfer python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

from dinfer.model.modeling_llada2_moe_sglang import LLaDA2SGLangLM
from dinfer.model.configuration_llada2_moe import LLaDA2MoeConfig


# Global log file handle
LOG_FILE = None
ORIGINAL_STDOUT = None


class TeeOutput:
    """Class to write to both console and log file."""
    def __init__(self, console, logfile):
        self.console = console
        self.logfile = logfile

    def write(self, message):
        self.console.write(message)
        if self.logfile:
            self.logfile.write(message)
            self.logfile.flush()

    def flush(self):
        self.console.flush()
        if self.logfile:
            self.logfile.flush()


def log_print(*args, **kwargs):
    """Print to both console and log file (uses regular print)."""
    # Just use builtin print, which will go through TeeOutput if enabled
    __builtins__.print(*args, **kwargs)


def test_with_dummy_data(config, device='cuda'):
    """Test with randomly generated dummy data."""
    log_print("\n" + "="*70)
    log_print("Testing with dummy data")
    log_print("="*70)

    # Create a simple model for testing
    log_print("Creating model...")
    model = LLaDA2SGLangLM(config)
    model = model.to(device)
    model.eval()

    # Enable token statistics and incast statistics for all MoE layers
    log_print("\nEnabling token distribution and incast statistics...")
    moe_layers = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'enable_token_statistics'):
            layer.mlp.enable_token_statistics()
            layer.mlp.reset_token_statistics()
            # Enable incast statistics with 8 experts per node
            layer.mlp.enable_incast_statistics(experts_per_node=8)
            layer.mlp.reset_incast_statistics()
            moe_layers.append((i, layer.mlp))
            log_print(f"  Layer {i}: MoE layer enabled (with incast tracking)")

    if not moe_layers:
        log_print("Warning: No MoE layers found in the model!")
        return

    log_print(f"\nFound {len(moe_layers)} MoE layers")

    # Create dummy input
    batch_size = 2
    seq_length = 64
    log_print(f"\nCreating dummy input: batch_size={batch_size}, seq_length={seq_length}")
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length)).to(device)

    # Run forward pass
    log_print("Running forward pass...")
    with torch.no_grad():
        outputs = model(input_ids)

    log_print(f"Output shape: {outputs.logits.shape}")

    # Display statistics
    display_statistics(moe_layers, config)


def test_with_model_path(model_path, batch_size=2, seq_length=64, device='cuda', experts_per_node=8):
    """Test with a loaded model from path."""
    log_print("\n" + "="*70)
    log_print("Testing with model from path")
    log_print("="*70)
    log_print(f"Model path: {model_path}")
    log_print(f"Experts per node: {experts_per_node}")

    try:
        from transformers import AutoTokenizer

        # Load tokenizer
        log_print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load config
        log_print("Loading model configuration...")
        config = LLaDA2MoeConfig.from_pretrained(model_path)

        log_print(f"\nModel Configuration:")
        log_print(f"  Number of layers: {config.num_hidden_layers}")
        log_print(f"  Number of experts: {config.num_experts}")
        log_print(f"  Experts per token (top-k): {config.num_experts_per_tok}")
        log_print(f"  Hidden size: {config.hidden_size}")
        log_print(f"  MoE intermediate size: {config.moe_intermediate_size}")
        if hasattr(config, 'n_group') and config.n_group > 0:
            log_print(f"  Expert groups: {config.n_group}")
            log_print(f"  Top-k per group: {config.topk_group}")

        # Load model
        log_print("\nLoading model weights...")
        model = LLaDA2SGLangLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        model = model.to(device)
        model.eval()

        # Enable token statistics and incast statistics
        log_print("\nEnabling token distribution and incast statistics...")
        moe_layers = []
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'enable_token_statistics'):
                layer.mlp.enable_token_statistics()
                layer.mlp.reset_token_statistics()
                # Enable incast statistics
                layer.mlp.enable_incast_statistics(experts_per_node=experts_per_node)
                layer.mlp.reset_incast_statistics()
                moe_layers.append((i, layer.mlp))
                log_print(f"  Layer {i}: MoE layer enabled (with incast tracking)")

        if not moe_layers:
            log_print("Warning: No MoE layers found in the model!")
            return

        log_print(f"\nFound {len(moe_layers)} MoE layers")

        # Create sample input
        sample_text = "The quick brown fox jumps over the lazy dog."
        log_print(f"\nTokenizing sample text: '{sample_text}'")
        inputs = tokenizer(sample_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        log_print(f"Input shape: {input_ids.shape}")

        # Repeat to create multiple samples if needed
        if batch_size > 1:
            input_ids = input_ids.repeat(batch_size, 1)
            log_print(f"Repeated to batch_size={batch_size}: {input_ids.shape}")

        # Run forward pass
        log_print("\nRunning forward pass...")
        with torch.no_grad():
            outputs = model(input_ids)

        log_print(f"Output shape: {outputs.logits.shape}")

        # Display statistics
        display_statistics(moe_layers, config)

    except Exception as e:
        log_print(f"\nError loading model from path: {e}")
        log_print("Falling back to dummy data test...")
        log_print("\nCreating default config for testing...")
        config = create_test_config()
        test_with_dummy_data(config, device)


def display_statistics(moe_layers, config):
    """Display token distribution statistics."""
    # Display statistics for each MoE layer
    log_print("\n" + "="*70)
    log_print("TOKEN DISTRIBUTION ACROSS EXPERTS")
    log_print("="*70)

    for layer_id, moe_layer in moe_layers:
        moe_layer.print_token_statistics()

    # Display incast statistics
    log_print("\n" + "="*70)
    log_print("CROSS-NODE INCAST STATISTICS (PER LAYER)")
    log_print("="*70)

    for layer_id, moe_layer in moe_layers:
        moe_layer.print_incast_statistics(top_n=10, show_all_forwards=False)

    # Aggregate incast statistics across all layers
    log_print("\n" + "="*70)
    log_print("AGGREGATED INCAST ANALYSIS (ACROSS ALL LAYERS)")
    log_print("="*70)

    from collections import defaultdict
    all_incast_degrees = []
    expert_max_incast = defaultdict(int)
    expert_total_incast = defaultdict(int)
    expert_appearances = defaultdict(int)

    # Collect incast data from all layers
    for layer_id, moe_layer in moe_layers:
        incast_stats = moe_layer.get_incast_statistics()

        # Process each forward pass
        for forward_data in incast_stats['per_forward_incast']:
            incast_data = forward_data['incast_data']
            for expert_id, data in incast_data.items():
                incast = data['num_remote_senders']
                all_incast_degrees.append(incast)
                expert_max_incast[expert_id] = max(expert_max_incast[expert_id], incast)
                expert_total_incast[expert_id] += incast
                expert_appearances[expert_id] += 1

    if all_incast_degrees:
        log_print(f"\nOverall Incast Statistics:")
        log_print(f"  Total forward passes analyzed: {len(moe_layers)} layers")
        log_print(f"  Max incast degree: {max(all_incast_degrees)}")
        log_print(f"  Min incast degree: {min(all_incast_degrees)}")
        log_print(f"  Average incast degree: {sum(all_incast_degrees) / len(all_incast_degrees):.2f}")
        log_print(f"  Median incast degree: {sorted(all_incast_degrees)[len(all_incast_degrees)//2]}")

        # Top experts with highest max incast
        log_print(f"\nTop 20 Experts with Highest Max Incast:\n")
        log_print(f"{'Expert':<8} {'Node':<6} {'Max':<8} {'Avg':<8} {'Freq':<8}")
        log_print(f"{'ID':<8} {'ID':<6} {'Incast':<8} {'Incast':<8} {'Count':<8}")
        log_print(f"{'-'*50}")

        expert_stats = []
        for expert_id in expert_max_incast:
            avg_incast = expert_total_incast[expert_id] / expert_appearances[expert_id]
            expert_stats.append({
                'id': expert_id,
                'node': expert_id // 8,  # Assuming 8 experts per node
                'max': expert_max_incast[expert_id],
                'avg': avg_incast,
                'freq': expert_appearances[expert_id]
            })

        expert_stats.sort(key=lambda x: x['max'], reverse=True)

        for stat in expert_stats[:20]:
            log_print(f"{stat['id']:<8} {stat['node']:<6} {stat['max']:<8} "
                      f"{stat['avg']:<8.1f} {stat['freq']:<8}")
    else:
        log_print("\nNo incast data collected. Make sure incast statistics are enabled.")

    log_print(f"\n{'='*70}\n")

    # Summary across all layers
    log_print("\n" + "="*70)
    log_print("SUMMARY ACROSS ALL MOE LAYERS")
    log_print("="*70)

    total_tokens_all_layers = 0
    expert_totals = torch.zeros(config.num_experts, dtype=torch.long)

    for layer_id, moe_layer in moe_layers:
        stats = moe_layer.get_token_statistics()
        total_tokens_all_layers += stats['total_tokens']
        expert_totals += torch.tensor(stats['expert_token_count'], dtype=torch.long)

    log_print(f"Total tokens processed across all MoE layers: {total_tokens_all_layers}")
    log_print(f"Total MoE layers: {len(moe_layers)}")
    log_print(f"\nAggregated tokens per expert across all layers:")
    log_print(f"{'Expert ID':<12} {'Total Tokens':<15} {'Percentage':<12}")
    log_print(f"{'-'*40}")

    total_selections = expert_totals.sum().item()
    for expert_id in range(config.num_experts):
        count = expert_totals[expert_id].item()
        percentage = (count / total_selections * 100) if total_selections > 0 else 0
        log_print(f"{expert_id:<12} {count:<15} {percentage:>6.2f}%")

    log_print(f"\nTotal expert selections: {total_selections}")
    log_print(f"Average selections per expert: {total_selections / config.num_experts:.2f}")

    # Calculate load balance metrics
    mean_count = expert_totals.float().mean().item()
    std_count = expert_totals.float().std().item()
    cv = (std_count / mean_count) if mean_count > 0 else 0

    log_print(f"\nLoad Balance Metrics:")
    log_print(f"  Mean tokens per expert: {mean_count:.2f}")
    log_print(f"  Std deviation: {std_count:.2f}")
    log_print(f"  Coefficient of variation: {cv:.4f}")
    log_print(f"  Min tokens: {expert_totals.min().item()}")
    log_print(f"  Max tokens: {expert_totals.max().item()}")
    log_print(f"  Range: {expert_totals.max().item() - expert_totals.min().item()}")
    log_print(f"{'='*70}\n")


def create_test_config():
    """Create a minimal test configuration."""
    config = LLaDA2MoeConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1024,
        moe_intermediate_size=1024,
        num_hidden_layers=4,  # Only 4 layers for quick testing
        num_attention_heads=8,
        num_key_value_heads=8,
        num_experts=8,
        num_experts_per_tok=2,
        num_shared_experts=0,
        n_group=4,
        topk_group=2,
    )
    return config


def main():
    parser = argparse.ArgumentParser(description="Test expert token distribution")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the LLaDA2 MoE model (optional)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for testing"
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=64,
        help="Sequence length for testing"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda or cpu)"
    )
    parser.add_argument(
        "--experts_per_node",
        type=int,
        default=8,
        help="Number of experts per node for incast analysis (default: 8)"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Log file path to save all outputs (default: auto-generate with timestamp)"
    )
    args = parser.parse_args()

    # Setup log file
    global LOG_FILE, ORIGINAL_STDOUT
    if args.log_file is None:
        # Auto-generate log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = f"expert_distribution_{timestamp}.log"

    LOG_FILE = open(args.log_file, 'w', encoding='utf-8')
    ORIGINAL_STDOUT = sys.stdout
    sys.stdout = TeeOutput(ORIGINAL_STDOUT, LOG_FILE)

    log_print(f"Logging to: {args.log_file}\n")

    log_print(f"\n{'='*70}")
    log_print("Expert Token Distribution Test")
    log_print(f"{'='*70}")
    log_print(f"Device: {args.device}")
    log_print(f"Batch size: {args.batch_size}")
    log_print(f"Sequence length: {args.seq_length}")

    if not torch.cuda.is_available() and args.device == "cuda":
        log_print("Warning: CUDA not available, falling back to CPU")
        args.device = "cpu"

    try:
        if args.model_path and os.path.exists(args.model_path):
            test_with_model_path(args.model_path, args.batch_size, args.seq_length, args.device, args.experts_per_node)
        else:
            if args.model_path:
                log_print(f"Warning: Model path '{args.model_path}' not found")
            log_print("Creating test configuration for dummy data test...")
            config = create_test_config()
            test_with_dummy_data(config, args.device)
    finally:
        # Restore stdout and close log file
        if ORIGINAL_STDOUT is not None:
            sys.stdout = ORIGINAL_STDOUT
        if LOG_FILE is not None:
            log_print(f"\nLog saved to: {args.log_file}")
            LOG_FILE.close()


if __name__ == "__main__":
    main()
