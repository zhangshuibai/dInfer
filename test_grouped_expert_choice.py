#!/usr/bin/env python3
"""
Test grouped expert choice routing to verify group boundaries are respected.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

from dinfer.model.topk_expert_choice import grouped_expert_choice_topk_gpu


def test_grouped_expert_choice():
    """Test that grouped expert choice respects group boundaries."""
    print("="*80)
    print("Testing Grouped Expert Choice Routing")
    print("="*80)

    # Configuration
    num_tokens = 128
    num_experts = 256
    num_expert_group = 8
    experts_per_group = num_experts // num_expert_group  # 32
    topk = 8
    topk_group = 4
    capacity = 8

    print(f"\nConfiguration:")
    print(f"  Tokens: {num_tokens}")
    print(f"  Experts: {num_experts}")
    print(f"  Expert groups: {num_expert_group}")
    print(f"  Experts per group: {experts_per_group}")
    print(f"  Top-k per token: {topk}")
    print(f"  Top-k groups per token: {topk_group}")
    print(f"  Expert capacity: {capacity}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    # Create fake router logits
    torch.manual_seed(42)
    hidden_states = torch.randn(num_tokens, 512, device=device)
    gating_output = torch.randn(num_tokens, num_experts, device=device)

    print("\nRunning grouped expert choice routing...")

    # Run grouped expert choice
    topk_weights, topk_ids = grouped_expert_choice_topk_gpu(
        hidden_states=hidden_states,
        gating_output=gating_output,
        capacity=capacity,
        num_experts=num_experts,
        topk=topk,
        renormalize=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
    )

    print(f"\nOutput shapes:")
    print(f"  topk_weights: {topk_weights.shape}")
    print(f"  topk_ids: {topk_ids.shape}")

    # Verify group boundaries are respected
    print(f"\nVerifying group boundaries...")

    violations = 0
    for token_idx in range(num_tokens):
        selected_experts = topk_ids[token_idx].cpu().numpy()
        # Filter out invalid experts (-1)
        valid_experts = [e for e in selected_experts if e >= 0]

        if len(valid_experts) == 0:
            continue

        # Determine which groups these experts belong to
        selected_groups = set()
        for expert_id in valid_experts:
            group_id = expert_id // experts_per_group
            selected_groups.add(group_id)

        # Should have at most topk_group different groups
        if len(selected_groups) > topk_group:
            violations += 1
            if violations <= 5:  # Print first 5 violations
                print(f"  Token {token_idx}: selected {len(selected_groups)} groups (expected ≤{topk_group})")
                print(f"    Selected groups: {sorted(selected_groups)}")
                print(f"    Selected experts: {valid_experts}")

    print(f"\n{'='*80}")
    if violations == 0:
        print("✓ SUCCESS! All tokens respect group boundaries.")
        print(f"  Every token selected experts from at most {topk_group} groups.")
    else:
        print(f"✗ FAILURE! {violations}/{num_tokens} tokens violated group boundaries.")
        print(f"  Some tokens selected experts from more than {topk_group} groups.")
    print(f"{'='*80}")

    # Check capacity constraint
    print(f"\nVerifying capacity constraint...")
    expert_loads = {}
    for token_idx in range(num_tokens):
        for expert_id in topk_ids[token_idx].cpu().numpy():
            if expert_id >= 0:
                expert_loads[expert_id] = expert_loads.get(expert_id, 0) + 1

    if expert_loads:
        max_load = max(expert_loads.values())
        num_experts_used = len(expert_loads)
        print(f"  Max expert load: {max_load}")
        print(f"  Capacity limit: {capacity}")
        print(f"  Experts used: {num_experts_used}/{num_experts}")

        if max_load <= capacity:
            print(f"  ✓ Capacity constraint satisfied!")
        else:
            print(f"  ✗ Capacity constraint violated!")

    print()
    return violations == 0


if __name__ == "__main__":
    success = test_grouped_expert_choice()
    sys.exit(0 if success else 1)
