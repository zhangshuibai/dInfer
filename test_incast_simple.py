#!/usr/bin/env python3
"""
Simplified test script to observe cross-node incast patterns in token-choice routing.

This simulates a multi-node scenario where:
- 8 experts per node (e.g., Node 0: experts 0-7, Node 1: experts 8-15, etc.)
- For each expert, we count how many REMOTE experts (from other nodes) send tokens to it
- This helps analyze potential network incast congestion

Usage:
    python test_incast_simple.py --model_path /data/models/LLaDA2.0-mini-preview
"""

import argparse
import torch
import sys
import os
from datetime import datetime
from collections import defaultdict

# Add the dInfer python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))


def simulate_token_routing(num_tokens=128, num_experts=256, top_k=8, experts_per_node=8):
    """
    Simulate token-choice routing and analyze cross-node communication.

    In token-choice: each token selects top_k experts.
    We want to know: for each expert (receiver), how many remote experts (senders)
    send tokens to it.

    Args:
        num_tokens: Number of tokens in the batch
        num_experts: Total number of experts
        top_k: Each token selects top_k experts
        experts_per_node: Number of experts per node (default: 8)
    """
    # Simulate router logits: [num_tokens, num_experts]
    torch.manual_seed(42)
    router_logits = torch.randn(num_tokens, num_experts)

    # Get top-k experts for each token: [num_tokens, top_k]
    topk_values, topk_indices = torch.topk(router_logits, top_k, dim=1)

    print(f"\nSimulation Setup:")
    print(f"  Tokens: {num_tokens}")
    print(f"  Experts: {num_experts}")
    print(f"  Top-k: {top_k}")
    print(f"  Experts per node: {experts_per_node}")
    print(f"  Total nodes: {num_experts // experts_per_node}")

    # Analyze cross-node communication for each expert
    print(f"\n{'='*80}")
    print("CROSS-NODE INCAST ANALYSIS (Token-Choice Routing)")
    print(f"{'='*80}\n")

    incast_stats = []

    for receiver_expert in range(num_experts):
        receiver_node = receiver_expert // experts_per_node

        # Find all tokens that route to this expert
        token_mask = (topk_indices == receiver_expert).any(dim=1)
        num_tokens_received = token_mask.sum().item()

        if num_tokens_received == 0:
            continue

        # For these tokens, find all experts they also route to (potential senders)
        sender_experts = topk_indices[token_mask].unique()

        # Count remote senders (from different nodes)
        remote_senders = []
        for sender in sender_experts.tolist():
            sender_node = sender // experts_per_node
            if sender_node != receiver_node:  # Cross-node
                remote_senders.append(sender)

        num_remote_senders = len(remote_senders)

        if num_remote_senders > 0:
            incast_stats.append({
                'expert_id': receiver_expert,
                'node_id': receiver_node,
                'num_tokens': num_tokens_received,
                'num_remote_senders': num_remote_senders,
                'remote_senders': remote_senders[:10],  # Show first 10
            })

    # Sort by incast degree (num_remote_senders)
    incast_stats.sort(key=lambda x: x['num_remote_senders'], reverse=True)

    # Display top experts with highest incast
    print(f"Top 20 Experts with Highest Cross-Node Incast Degree\n")
    print(f"{'Expert':<10} {'Node':<8} {'Tokens':<10} {'Remote':<10} {'Incast':<10} {'Remote Sender IDs'}")
    print(f"{'ID':<10} {'ID':<8} {'Recv':<10} {'Senders':<10} {'Degree':<10}")
    print(f"{'-'*80}")

    for stat in incast_stats[:20]:
        expert_id = stat['expert_id']
        node_id = stat['node_id']
        num_tokens = stat['num_tokens']
        num_remote = stat['num_remote_senders']
        senders = stat['remote_senders']
        senders_str = f"{senders[:5]}..." if len(senders) > 5 else str(senders)

        print(f"{expert_id:<10} {node_id:<8} {num_tokens:<10} {num_remote:<10} {num_remote:<10} {senders_str}")

    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")

    incast_degrees = [s['num_remote_senders'] for s in incast_stats]
    if incast_degrees:
        print(f"\nIncast Degree Statistics:")
        print(f"  Experts with cross-node traffic: {len(incast_stats)} / {num_experts}")
        print(f"  Max incast degree: {max(incast_degrees)}")
        print(f"  Min incast degree: {min(incast_degrees)}")
        print(f"  Avg incast degree: {sum(incast_degrees) / len(incast_degrees):.2f}")
        print(f"  Median incast degree: {sorted(incast_degrees)[len(incast_degrees)//2]}")

    # Distribution analysis
    print(f"\nIncast Degree Distribution:")
    degree_counts = defaultdict(int)
    for degree in incast_degrees:
        degree_counts[degree] += 1

    for degree in sorted(degree_counts.keys()):
        count = degree_counts[degree]
        percentage = count / len(incast_stats) * 100
        bar = '█' * int(percentage / 2)
        print(f"  Degree {degree:2d}: {count:3d} experts ({percentage:5.1f}%) {bar}")

    # Node-level analysis
    print(f"\nNode-Level Traffic Analysis:")
    node_traffic = defaultdict(lambda: {'experts': [], 'total_incast': 0})

    for stat in incast_stats:
        node_id = stat['node_id']
        expert_id = stat['expert_id']
        incast = stat['num_remote_senders']

        node_traffic[node_id]['experts'].append(expert_id)
        node_traffic[node_id]['total_incast'] += incast

    print(f"\n{'Node':<8} {'Experts':<15} {'Total':<15} {'Avg Incast':<15}")
    print(f"{'ID':<8} {'Affected':<15} {'Incast':<15} {'Per Expert':<15}")
    print(f"{'-'*60}")

    for node_id in sorted(node_traffic.keys()):
        data = node_traffic[node_id]
        num_experts = len(data['experts'])
        total = data['total_incast']
        avg = total / num_experts if num_experts > 0 else 0
        print(f"{node_id:<8} {num_experts:<15} {total:<15} {avg:<15.2f}")

    print(f"\n{'='*80}\n")

    return incast_stats


def main():
    parser = argparse.ArgumentParser(description="Simulate and analyze cross-node incast patterns")
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=128,
        help="Number of tokens to simulate (default: 128)"
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
        "--log_file",
        type=str,
        default=None,
        help="Log file to save results"
    )

    args = parser.parse_args()

    # Setup logging if requested
    if args.log_file:
        import sys
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

    print(f"\n{'='*80}")
    print("Cross-Node Incast Pattern Analyzer")
    print("(Simulating Token-Choice Routing)")
    print(f"{'='*80}")

    # Run simulation
    incast_stats = simulate_token_routing(
        num_tokens=args.num_tokens,
        num_experts=args.num_experts,
        top_k=args.top_k,
        experts_per_node=args.experts_per_node
    )

    if args.log_file:
        print(f"\nResults saved to: {args.log_file}")
        log_file.close()


if __name__ == "__main__":
    main()
