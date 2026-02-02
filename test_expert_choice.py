#!/usr/bin/env python3
"""
Test script to compare token-choice vs expert-choice routing.

This script:
1. Loads LLaDA2 MoE model with configurable routing strategy
2. Runs forward passes with both strategies
3. Compares incast patterns between token-choice and expert-choice

Usage:
    # Test with token-choice (default)
    python test_expert_choice.py --routing_strategy token_choice

    # Test with expert-choice
    python test_expert_choice.py --routing_strategy expert_choice --expert_capacity 10

    # Compare both strategies
    python test_expert_choice.py --compare_strategies
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


def test_routing_strategy(
    routing_strategy="token_choice",
    expert_capacity=None,
    batch_size=2,
    seq_length=64,
    device='cuda',
    experts_per_node=8
):
    """Test a specific routing strategy."""
    print(f"\n{'='*80}")
    print(f"Testing {routing_strategy.upper()} Routing")
    print(f"{'='*80}")

    # Create test configuration
    config = LLaDA2MoeConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1024,
        moe_intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
        num_experts=64,  # 64 experts = 8 nodes * 8 experts/node
        num_experts_per_tok=8,
        num_shared_experts=0,
        n_group=8,
        topk_group=4,
        # Expert-choice configuration
        routing_strategy=routing_strategy,
        expert_capacity=expert_capacity,
    )

    print(f"\nConfiguration:")
    print(f"  Routing strategy: {config.routing_strategy}")
    print(f"  Number of experts: {config.num_experts}")
    print(f"  Top-k per token: {config.num_experts_per_tok}")
    print(f"  Expert capacity: {expert_capacity if expert_capacity else 'N/A (token-choice)'}")
    print(f"  Experts per node: {experts_per_node}")

    # Create model
    print("\nCreating model...")
    model = LLaDA2SGLangLM(config)
    model = model.to(device)
    model.eval()

    # Enable statistics
    print("\nEnabling statistics...")
    moe_layers = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'enable_incast_statistics'):
            layer.mlp.enable_incast_statistics(experts_per_node=experts_per_node)
            layer.mlp.reset_incast_statistics()
            moe_layers.append((i, layer.mlp))

    print(f"Found {len(moe_layers)} MoE layers")

    # Create input
    print(f"\nCreating input: batch_size={batch_size}, seq_length={seq_length}")
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length)).to(device)

    # Run forward pass
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(input_ids)

    print(f"Output shape: {outputs.logits.shape}")

    # Collect and display incast statistics
    print(f"\n{'='*80}")
    print("INCAST STATISTICS")
    print(f"{'='*80}")

    from collections import defaultdict
    all_incast_degrees = []
    expert_max_incast = defaultdict(int)
    expert_total_incast = defaultdict(int)
    expert_appearances = defaultdict(int)

    for layer_id, moe_layer in moe_layers:
        incast_stats = moe_layer.get_incast_statistics()
        for forward_data in incast_stats['per_forward_incast']:
            incast_data = forward_data['incast_data']
            for expert_id, data in incast_data.items():
                incast = data['num_remote_senders']
                all_incast_degrees.append(incast)
                expert_max_incast[expert_id] = max(expert_max_incast[expert_id], incast)
                expert_total_incast[expert_id] += incast
                expert_appearances[expert_id] += 1

    if all_incast_degrees:
        print(f"\nOverall Incast Statistics:")
        print(f"  Max incast degree: {max(all_incast_degrees)}")
        print(f"  Min incast degree: {min(all_incast_degrees)}")
        print(f"  Average incast degree: {sum(all_incast_degrees) / len(all_incast_degrees):.2f}")
        print(f"  Median incast degree: {sorted(all_incast_degrees)[len(all_incast_degrees)//2]}")
        print(f"  Total measurements: {len(all_incast_degrees)}")

        # Distribution of incast degrees
        from collections import Counter
        incast_counter = Counter(all_incast_degrees)
        print(f"\nIncast degree distribution:")
        for degree in sorted(incast_counter.keys())[:10]:
            count = incast_counter[degree]
            pct = count / len(all_incast_degrees) * 100
            print(f"    Incast={degree}: {count} times ({pct:.1f}%)")
    else:
        print("\nNo incast data collected")

    print(f"\n{'='*80}\n")

    return {
        'routing_strategy': routing_strategy,
        'max_incast': max(all_incast_degrees) if all_incast_degrees else 0,
        'avg_incast': sum(all_incast_degrees) / len(all_incast_degrees) if all_incast_degrees else 0,
        'median_incast': sorted(all_incast_degrees)[len(all_incast_degrees)//2] if all_incast_degrees else 0,
        'total_measurements': len(all_incast_degrees),
    }


def main():
    parser = argparse.ArgumentParser(description="Test expert-choice routing")
    parser.add_argument(
        "--routing_strategy",
        type=str,
        default="token_choice",
        choices=["token_choice", "expert_choice"],
        help="Routing strategy to use"
    )
    parser.add_argument(
        "--expert_capacity",
        type=int,
        default=None,
        help="Expert capacity for expert-choice routing (default: same as top_k)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size"
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=64,
        help="Sequence length"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    parser.add_argument(
        "--experts_per_node",
        type=int,
        default=8,
        help="Number of experts per node"
    )
    parser.add_argument(
        "--compare_strategies",
        action="store_true",
        help="Compare token-choice vs expert-choice"
    )

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("EXPERT-CHOICE ROUTING TEST")
    print(f"{'='*80}")
    print(f"Device: {args.device}")

    if args.compare_strategies:
        # Test both strategies
        print("\n\nCOMPARING TOKEN-CHOICE VS EXPERT-CHOICE")
        print(f"{'='*80}\n")

        results = []

        # Test token-choice
        result1 = test_routing_strategy(
            routing_strategy="token_choice",
            expert_capacity=None,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            device=args.device,
            experts_per_node=args.experts_per_node,
        )
        results.append(result1)

        # Test expert-choice
        result2 = test_routing_strategy(
            routing_strategy="expert_choice",
            expert_capacity=args.expert_capacity or 8,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            device=args.device,
            experts_per_node=args.experts_per_node,
        )
        results.append(result2)

        # Compare results
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}\n")
        print(f"{'Strategy':<20} {'Max Incast':<15} {'Avg Incast':<15} {'Median':<15}")
        print(f"{'-'*65}")
        for r in results:
            print(f"{r['routing_strategy']:<20} {r['max_incast']:<15} {r['avg_incast']:<15.2f} {r['median_incast']:<15}")

        improvement = (results[0]['avg_incast'] - results[1]['avg_incast']) / results[0]['avg_incast'] * 100
        print(f"\nAverage incast improvement: {improvement:.1f}%")

    else:
        # Test single strategy
        test_routing_strategy(
            routing_strategy=args.routing_strategy,
            expert_capacity=args.expert_capacity,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            device=args.device,
            experts_per_node=args.experts_per_node,
        )


if __name__ == "__main__":
    main()
