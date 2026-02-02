#!/usr/bin/env python3
"""
Continuous monitoring of cross-node incast patterns over multiple forward passes.

This script runs multiple simulations and tracks how incast patterns change,
helping identify:
- Consistency of incast hotspots
- Variability of network load
- Worst-case vs average-case scenarios

Usage:
    python test_incast_continuous.py --num_iterations 100 --log_file continuous_incast.log
"""

import argparse
import torch
import sys
import os
from datetime import datetime
from collections import defaultdict
import time

# Add the dInfer python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))


def simulate_one_forward(num_tokens=128, num_experts=256, top_k=8, experts_per_node=8, seed=None):
    """
    Simulate one forward pass and return incast statistics.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Simulate router logits: [num_tokens, num_experts]
    router_logits = torch.randn(num_tokens, num_experts)

    # Get top-k experts for each token: [num_tokens, top_k]
    topk_values, topk_indices = torch.topk(router_logits, top_k, dim=1)

    # Calculate incast for each expert
    expert_incast = {}

    for receiver_expert in range(num_experts):
        receiver_node = receiver_expert // experts_per_node

        # Find all tokens that route to this expert
        token_mask = (topk_indices == receiver_expert).any(dim=1)
        num_tokens_received = token_mask.sum().item()

        if num_tokens_received == 0:
            expert_incast[receiver_expert] = {
                'tokens': 0,
                'remote_senders': 0,
                'local_senders': 0,
            }
            continue

        # For these tokens, find all experts they also route to
        sender_experts = topk_indices[token_mask].unique()

        # Count remote vs local senders
        remote_senders = []
        local_senders = []

        for sender in sender_experts.tolist():
            sender_node = sender // experts_per_node
            if sender_node == receiver_node:
                local_senders.append(sender)
            else:
                remote_senders.append(sender)

        expert_incast[receiver_expert] = {
            'tokens': num_tokens_received,
            'remote_senders': len(remote_senders),
            'local_senders': len(local_senders),
        }

    return expert_incast


def run_continuous_monitoring(args):
    """Run continuous monitoring and track statistics over time."""

    print(f"\n{'='*80}")
    print("CONTINUOUS INCAST MONITORING")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Iterations: {args.num_iterations}")
    print(f"  Tokens per iteration: {args.num_tokens}")
    print(f"  Experts: {args.num_experts}")
    print(f"  Top-k: {args.top_k}")
    print(f"  Experts per node: {args.experts_per_node}")
    print(f"  Total nodes: {args.num_experts // args.experts_per_node}")
    print(f"{'='*80}\n")

    # Track statistics over time
    all_iterations = []
    expert_incast_history = defaultdict(list)  # expert_id -> list of incast degrees

    print("Running simulations...")
    start_time = time.time()

    for iteration in range(args.num_iterations):
        # Run one forward pass
        seed = args.seed + iteration if args.seed is not None else None
        incast_stats = simulate_one_forward(
            num_tokens=args.num_tokens,
            num_experts=args.num_experts,
            top_k=args.top_k,
            experts_per_node=args.experts_per_node,
            seed=seed
        )

        # Collect statistics
        iteration_data = {
            'iteration': iteration,
            'max_incast': 0,
            'avg_incast': 0,
            'experts_with_traffic': 0,
        }

        remote_incasts = []
        for expert_id, stats in incast_stats.items():
            remote_senders = stats['remote_senders']
            if remote_senders > 0:
                remote_incasts.append(remote_senders)
                expert_incast_history[expert_id].append(remote_senders)
                iteration_data['experts_with_traffic'] += 1

        if remote_incasts:
            iteration_data['max_incast'] = max(remote_incasts)
            iteration_data['avg_incast'] = sum(remote_incasts) / len(remote_incasts)

        all_iterations.append(iteration_data)

        # Progress update every 10 iterations
        if (iteration + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (iteration + 1) / elapsed
            eta = (args.num_iterations - iteration - 1) / rate if rate > 0 else 0
            print(f"  [{iteration+1:4d}/{args.num_iterations}] "
                  f"Max incast: {iteration_data['max_incast']:3d}, "
                  f"Avg: {iteration_data['avg_incast']:5.1f}, "
                  f"Rate: {rate:5.1f} iter/s, ETA: {eta:5.1f}s")

    elapsed = time.time() - start_time
    print(f"\nCompleted {args.num_iterations} iterations in {elapsed:.2f}s "
          f"({args.num_iterations/elapsed:.1f} iter/s)\n")

    # Analyze results
    print(f"\n{'='*80}")
    print("ANALYSIS ACROSS ALL ITERATIONS")
    print(f"{'='*80}\n")

    # Overall statistics
    max_incasts = [it['max_incast'] for it in all_iterations]
    avg_incasts = [it['avg_incast'] for it in all_iterations]

    print(f"Max Incast Degree Across Iterations:")
    print(f"  Overall maximum: {max(max_incasts)}")
    print(f"  Overall minimum: {min(max_incasts)}")
    print(f"  Average: {sum(max_incasts) / len(max_incasts):.2f}")
    print(f"  Std deviation: {torch.tensor(max_incasts).float().std().item():.2f}")

    print(f"\nAverage Incast Degree Across Iterations:")
    print(f"  Maximum: {max(avg_incasts):.2f}")
    print(f"  Minimum: {min(avg_incasts):.2f}")
    print(f"  Average: {sum(avg_incasts) / len(avg_incasts):.2f}")
    print(f"  Std deviation: {torch.tensor(avg_incasts).float().std().item():.2f}")

    # Per-expert analysis
    print(f"\n{'='*80}")
    print("PER-EXPERT INCAST ANALYSIS")
    print(f"{'='*80}\n")

    # Calculate per-expert statistics
    expert_stats = []
    for expert_id in range(args.num_experts):
        if expert_id in expert_incast_history:
            history = expert_incast_history[expert_id]
            expert_stats.append({
                'expert_id': expert_id,
                'node_id': expert_id // args.experts_per_node,
                'max_incast': max(history),
                'avg_incast': sum(history) / len(history),
                'min_incast': min(history),
                'std_incast': torch.tensor(history).float().std().item(),
                'appearances': len(history),
            })

    # Sort by max incast
    expert_stats.sort(key=lambda x: x['max_incast'], reverse=True)

    print(f"Top 20 Experts with Highest Max Incast (Hotspots)\n")
    print(f"{'Expert':<8} {'Node':<6} {'Max':<8} {'Avg':<8} {'Min':<8} {'Std':<8} {'Freq':<8}")
    print(f"{'ID':<8} {'ID':<6} {'Incast':<8} {'Incast':<8} {'Incast':<8} {'Dev':<8} {'%':<8}")
    print(f"{'-'*70}")

    for stat in expert_stats[:20]:
        freq_pct = (stat['appearances'] / args.num_iterations) * 100
        print(f"{stat['expert_id']:<8} {stat['node_id']:<6} "
              f"{stat['max_incast']:<8} {stat['avg_incast']:<8.1f} "
              f"{stat['min_incast']:<8} {stat['std_incast']:<8.1f} "
              f"{freq_pct:<7.1f}%")

    # Incast stability analysis
    print(f"\n{'='*80}")
    print("INCAST STABILITY ANALYSIS")
    print(f"{'='*80}\n")

    # Find most stable vs most variable experts
    stable_experts = sorted(expert_stats, key=lambda x: x['std_incast'])[:10]
    variable_experts = sorted(expert_stats, key=lambda x: x['std_incast'], reverse=True)[:10]

    print("Most Stable Experts (Low Variability):\n")
    print(f"{'Expert':<8} {'Node':<6} {'Avg':<8} {'Std':<8} {'Range':<10}")
    print(f"{'-'*45}")
    for stat in stable_experts:
        range_val = stat['max_incast'] - stat['min_incast']
        print(f"{stat['expert_id']:<8} {stat['node_id']:<6} "
              f"{stat['avg_incast']:<8.1f} {stat['std_incast']:<8.1f} "
              f"{range_val:<10}")

    print(f"\nMost Variable Experts (High Variability):\n")
    print(f"{'Expert':<8} {'Node':<6} {'Avg':<8} {'Std':<8} {'Range':<10}")
    print(f"{'-'*45}")
    for stat in variable_experts:
        range_val = stat['max_incast'] - stat['min_incast']
        print(f"{stat['expert_id']:<8} {stat['node_id']:<6} "
              f"{stat['avg_incast']:<8.1f} {stat['std_incast']:<8.1f} "
              f"{range_val:<10}")

    # Node-level aggregation
    print(f"\n{'='*80}")
    print("NODE-LEVEL AGGREGATED STATISTICS")
    print(f"{'='*80}\n")

    node_stats = defaultdict(lambda: {'max': [], 'avg': []})
    for stat in expert_stats:
        node_id = stat['node_id']
        node_stats[node_id]['max'].append(stat['max_incast'])
        node_stats[node_id]['avg'].append(stat['avg_incast'])

    print(f"{'Node':<6} {'Experts':<10} {'Avg Max':<10} {'Avg Avg':<10} {'Total':<10}")
    print(f"{'ID':<6} {'Count':<10} {'Incast':<10} {'Incast':<10} {'Load':<10}")
    print(f"{'-'*55}")

    for node_id in sorted(node_stats.keys()):
        data = node_stats[node_id]
        avg_max = sum(data['max']) / len(data['max'])
        avg_avg = sum(data['avg']) / len(data['avg'])
        total_load = sum(data['avg'])
        print(f"{node_id:<6} {len(data['max']):<10} {avg_max:<10.1f} "
              f"{avg_avg:<10.1f} {total_load:<10.1f}")

    print(f"\n{'='*80}\n")

    return all_iterations, expert_stats


def main():
    parser = argparse.ArgumentParser(description="Continuous monitoring of cross-node incast patterns")
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=100,
        help="Number of forward passes to simulate (default: 100)"
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=128,
        help="Number of tokens per forward pass (default: 128)"
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        default=256,
        help="Total number of experts (default: 256)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=8,
        help="Number of experts each token selects (default: 8)"
    )
    parser.add_argument(
        "--experts_per_node",
        type=int,
        default=8,
        help="Number of experts per node (default: 8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Log file to save results"
    )

    args = parser.parse_args()

    # Setup logging if requested
    if args.log_file:
        class TeeOutput:
            def __init__(self, *files):
                self.files = files
            def write(self, message):
                for f in self.files:
                    f.write(message)
                    f.flush()
            def flush(self):
                for f in self.files:
                    f.flush()

        log_file = open(args.log_file, 'w')
        sys.stdout = TeeOutput(sys.__stdout__, log_file)
        print(f"Logging to: {args.log_file}\n")

    # Run continuous monitoring
    all_iterations, expert_stats = run_continuous_monitoring(args)

    if args.log_file:
        print(f"\nResults saved to: {args.log_file}")
        log_file.close()


if __name__ == "__main__":
    main()
