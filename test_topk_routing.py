#!/usr/bin/env python3
"""
Simple test to verify expert-choice TopK routing logic.

This directly tests the TopK implementation without loading the full model.
"""

import torch
import sys
import os

# Add the dInfer python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

from dinfer.model.topk_expert_choice import expert_choice_topk_gpu, grouped_topk_gpu
from collections import defaultdict


def calculate_incast(topk_ids, num_experts=64, experts_per_node=8):
    """
    Calculate cross-node incast for routing decisions.

    Args:
        topk_ids: [N, K] tensor of expert IDs selected for each token
        num_experts: Total number of experts
        experts_per_node: Number of experts per node

    Returns:
        dict with incast statistics
    """
    num_tokens, top_k = topk_ids.shape

    expert_incast = {}

    for receiver_expert in range(num_experts):
        receiver_node = receiver_expert // experts_per_node

        # Find tokens routing to this expert
        token_mask = (topk_ids == receiver_expert).any(dim=1)
        num_tokens_received = token_mask.sum().item()

        if num_tokens_received == 0:
            continue

        # Find all experts these tokens also route to
        sender_experts = topk_ids[token_mask].unique()

        # Count remote senders
        remote_senders = []
        for sender in sender_experts.tolist():
            sender_node = sender // experts_per_node
            if sender_node != receiver_node:
                remote_senders.append(sender)

        if len(remote_senders) > 0:
            expert_incast[receiver_expert] = {
                'tokens': num_tokens_received,
                'num_remote_senders': len(remote_senders),
            }

    return expert_incast


def test_token_choice_routing(num_tokens=128, num_experts=64, top_k=8, experts_per_node=8):
    """Test standard token-choice routing."""
    print(f"\n{'='*80}")
    print("TOKEN-CHOICE ROUTING")
    print(f"{'='*80}")
    print(f"Tokens: {num_tokens}, Experts: {num_experts}, Top-k: {top_k}")

    # Create dummy inputs
    hidden_states = torch.randn(num_tokens, 512).cuda()
    router_logits = torch.randn(num_tokens, num_experts).cuda()

    # Use grouped topk (similar to LLaDA2)
    topk_weights, topk_ids = grouped_topk_gpu(
        hidden_states=hidden_states,
        gating_output=router_logits,
        topk=top_k,
        renormalize=True,
        num_expert_group=8,
        topk_group=4,
    )

    print(f"\nOutput shapes:")
    print(f"  topk_weights: {topk_weights.shape}")
    print(f"  topk_ids: {topk_ids.shape}")

    # Calculate incast
    expert_incast = calculate_incast(topk_ids, num_experts, experts_per_node)

    incast_degrees = [data['num_remote_senders'] for data in expert_incast.values()]

    if incast_degrees:
        print(f"\nIncast Statistics:")
        print(f"  Max incast: {max(incast_degrees)}")
        print(f"  Min incast: {min(incast_degrees)}")
        print(f"  Avg incast: {sum(incast_degrees) / len(incast_degrees):.2f}")
        print(f"  Experts with remote senders: {len(expert_incast)}")

    return {
        'strategy': 'token_choice',
        'max_incast': max(incast_degrees) if incast_degrees else 0,
        'avg_incast': sum(incast_degrees) / len(incast_degrees) if incast_degrees else 0,
    }


def test_expert_choice_routing(num_tokens=128, num_experts=64, top_k=8, capacity=16, experts_per_node=8):
    """Test expert-choice routing."""
    print(f"\n{'='*80}")
    print("EXPERT-CHOICE ROUTING")
    print(f"{'='*80}")
    print(f"Tokens: {num_tokens}, Experts: {num_experts}, Top-k: {top_k}, Capacity: {capacity}")

    # Create dummy inputs
    hidden_states = torch.randn(num_tokens, 512).cuda()
    router_logits = torch.randn(num_tokens, num_experts).cuda()

    # Use expert-choice routing
    topk_weights, topk_ids = expert_choice_topk_gpu(
        hidden_states=hidden_states,
        gating_output=router_logits,
        capacity=capacity,
        num_experts=num_experts,
        top_k=top_k,
        renormalize=True,
    )

    print(f"\nOutput shapes:")
    print(f"  topk_weights: {topk_weights.shape}")
    print(f"  topk_ids: {topk_ids.shape}")

    # Check how many tokens have valid experts
    valid_mask = topk_ids >= 0
    tokens_with_experts = valid_mask.any(dim=1).sum().item()
    print(f"  Tokens with at least one expert: {tokens_with_experts}/{num_tokens}")

    # Calculate incast
    expert_incast = calculate_incast(topk_ids, num_experts, experts_per_node)

    incast_degrees = [data['num_remote_senders'] for data in expert_incast.values()]

    if incast_degrees:
        print(f"\nIncast Statistics:")
        print(f"  Max incast: {max(incast_degrees)}")
        print(f"  Min incast: {min(incast_degrees)}")
        print(f"  Avg incast: {sum(incast_degrees) / len(incast_degrees):.2f}")
        print(f"  Experts with remote senders: {len(expert_incast)}")

    return {
        'strategy': 'expert_choice',
        'max_incast': max(incast_degrees) if incast_degrees else 0,
        'avg_incast': sum(incast_degrees) / len(incast_degrees) if incast_degrees else 0,
    }


def main():
    print(f"\n{'='*80}")
    print("TOPK ROUTING COMPARISON TEST")
    print(f"{'='*80}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    # Test parameters
    num_tokens = 128
    num_experts = 64
    top_k = 8
    capacity = 16  # Each expert can handle 16 tokens
    experts_per_node = 8

    # Test token-choice
    result1 = test_token_choice_routing(num_tokens, num_experts, top_k, experts_per_node)

    # Test expert-choice
    result2 = test_expert_choice_routing(num_tokens, num_experts, top_k, capacity, experts_per_node)

    # Compare
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}\n")
    print(f"{'Strategy':<20} {'Max Incast':<15} {'Avg Incast':<15}")
    print(f"{'-'*50}")
    print(f"{result1['strategy']:<20} {result1['max_incast']:<15} {result1['avg_incast']:<15.2f}")
    print(f"{result2['strategy']:<20} {result2['max_incast']:<15} {result2['avg_incast']:<15.2f}")

    if result1['avg_incast'] > 0:
        improvement = (result1['avg_incast'] - result2['avg_incast']) / result1['avg_incast'] * 100
        print(f"\nIncast reduction: {improvement:.1f}%")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
