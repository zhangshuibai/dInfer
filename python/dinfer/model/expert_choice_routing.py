# Copyright 2024 dInfer Team
# Expert Choice Routing for SGLang Triton Kernel Format
#
# This module provides native [E, C] expert choice routing that directly
# outputs RoutingData/GatherIndx/ScatterIndx format for SGLang's triton kernels.
#
# Key advantage: No information loss from converting [E,C] -> [N,topk]

from __future__ import annotations

from typing import Optional, Tuple
import torch

try:
    from triton_kernels.routing import (
        GatherIndx,
        RoutingData,
        ScatterIndx,
        ExptData,
    )
    from triton_kernels.target_info import is_hip
    TRITON_KERNELS_AVAILABLE = True
except ImportError:
    TRITON_KERNELS_AVAILABLE = False
    is_hip = lambda: False


def compute_expt_data_torch(hist: torch.Tensor, n_expts_tot: int, n_gates: int) -> "ExptData":
    """
    Compute ExptData using torch reference implementation.

    This is adapted from triton_kernels.routing.compute_expt_data_torch
    """
    device = hist.device

    # Offset for each expert
    token_offs_raw = torch.cumsum(hist, dim=0)
    token_offs_raw = torch.cat((torch.zeros(1, device=device, dtype=torch.int32), token_offs_raw))
    token_offs_raw = token_offs_raw.int()

    # Block sizes for tiled matmul
    block_ms = [16, 32, 64, 128]
    if is_hip():
        block_ms.append(256)

    # Maximum number of tiles
    if n_gates <= n_expts_tot:
        max_n_tiles = n_gates
    else:
        max_n_tiles = n_expts_tot - 1 - ((n_expts_tot - n_gates - 1) // min(block_ms))

    # Fill up tile offset/infos for each block size
    token_offs_pad = dict()
    block_pid_map = dict()

    for block_m in block_ms:
        n_tiles = (hist + block_m - 1) // block_m  # matmul blocks needed per expert
        token_offs_pad[block_m] = torch.cumsum(n_tiles, dim=0)
        token_offs_pad[block_m] = torch.cat((torch.zeros(1, device=device), token_offs_pad[block_m]))
        token_offs_pad[block_m] = token_offs_pad[block_m].int()

        # Compute block -> (expert, block_within_expert) mapping
        block_pid_map[block_m] = -torch.ones(max_n_tiles, device=device, dtype=torch.int32)
        for e in range(n_expts_tot):
            offset = token_offs_pad[block_m][e].item()
            for b in range(n_tiles[e].item()):
                block_pid_map[block_m][int(offset + b)] = (b << 16) + e

    return ExptData(hist, token_offs_raw, token_offs_pad, block_pid_map)


def expert_choice_routing(
    gating_output: torch.Tensor,
    capacity: int,
    renormalize: bool = True,
) -> Tuple["RoutingData", "GatherIndx", "ScatterIndx"]:
    """
    Expert Choice routing that directly returns RoutingData, GatherIndx, ScatterIndx.

    This is a native [E, C] implementation that avoids converting to [N, topk] format,
    which would cause information loss when tokens are selected by multiple experts.

    Key insight: In expert choice, each expert selects exactly `capacity` tokens,
    resulting in PERFECTLY BALANCED load across all experts.

    Args:
        gating_output: [N, E] router logits
        capacity: Tokens per expert (C) - each expert selects exactly C tokens
        renormalize: Whether to renormalize weights after selection

    Returns:
        routing_data: RoutingData containing gate scales and expert histogram
        gather_indx: GatherIndx for gathering tokens
        scatter_indx: ScatterIndx for scattering results

    Example:
        >>> gating_output = torch.randn(128, 256)  # 128 tokens, 256 experts
        >>> capacity = 8  # each expert selects 8 tokens
        >>> routing_data, gather_idx, scatter_idx = expert_choice_routing(gating_output, capacity)
        >>> # routing_data.expt_hist will be [8, 8, 8, ..., 8] - perfectly balanced!
    """
    if not TRITON_KERNELS_AVAILABLE:
        raise ImportError("triton_kernels package is required for expert_choice_routing")

    num_tokens, num_experts = gating_output.shape
    device = gating_output.device
    dtype = gating_output.dtype

    # Expert Choice: softmax over tokens (dim=0) - experts compete for tokens
    # This is the KEY difference from token choice which uses softmax(dim=-1)
    scores = torch.softmax(gating_output.float(), dim=0)  # [N, E]

    # Transpose to expert-centric view: [E, N]
    scores_t = scores.t()  # [E, N]

    # Each expert selects exactly `capacity` tokens
    actual_capacity = min(capacity, num_tokens)

    # Expert choice: each expert picks top-C tokens
    expert_scores, expert_tokens = torch.topk(scores_t, k=actual_capacity, dim=1, sorted=False)
    # expert_tokens: [E, C] - token indices selected by each expert
    # expert_scores: [E, C] - corresponding scores

    # Sort each expert's token selections by token index for consistent ordering
    expert_tokens_sorted, sort_indices = torch.sort(expert_tokens, dim=1)
    expert_scores_sorted = torch.gather(expert_scores, 1, sort_indices)

    # Total number of (token, expert) pairs = E * C
    n_gates = num_experts * actual_capacity

    # Flatten to create expert-sorted arrays
    # These are already in expert-major order: [expert0_token0, expert0_token1, ..., expert1_token0, ...]
    gate_scal = expert_scores_sorted.flatten()  # [E*C]
    topk_indx = expert_tokens_sorted.flatten().int()  # [E*C]

    # gate_indx: inverse permutation
    # We need to create mapping from flat index to sorted position
    gate_indx = torch.argsort(topk_indx, stable=True).int()

    # Renormalize if requested
    if renormalize:
        # For expert choice, renormalize per-token across all experts that selected it
        # Build token -> total_score mapping
        token_weight_sum = torch.zeros(num_tokens, device=device, dtype=scores.dtype)
        token_weight_sum.scatter_add_(0, topk_indx.long(), gate_scal)

        # Avoid division by zero
        token_weight_sum = torch.where(
            token_weight_sum > 0,
            token_weight_sum,
            torch.ones_like(token_weight_sum)
        )

        # Renormalize gate_scal
        gate_scal = gate_scal / token_weight_sum[topk_indx.long()]

    # Histogram: number of tokens per expert
    # For expert choice, this is PERFECTLY UNIFORM: every expert gets exactly C tokens
    expt_hist = torch.full((num_experts,), actual_capacity, dtype=torch.int32, device=device)

    # Compute ExptData for the matmul kernels
    expt_data = compute_expt_data_torch(expt_hist, num_experts, n_gates)

    # n_expts_act: This is used by triton kernel for buffer sizing
    # Triton kernel allocates buffer of size (M * n_expts_act)
    # For expert choice, total assignments = E * C, so we need:
    # M * n_expts_act >= E * C  =>  n_expts_act >= (E * C) / M
    # Use ceiling division to ensure enough buffer space
    n_expts_act = (n_gates + num_tokens - 1) // num_tokens

    # Pack the data structures
    routing_data = RoutingData(
        gate_scal=gate_scal.to(dtype),
        expt_hist=expt_hist,
        n_expts_tot=num_experts,
        n_expts_act=n_expts_act,  # ceil(E * C / M) for proper buffer sizing
        expt_data=expt_data,
    )

    gather_indx = GatherIndx(src_indx=topk_indx, dst_indx=gate_indx)
    scatter_indx = ScatterIndx(src_indx=gate_indx, dst_indx=topk_indx)

    return routing_data, gather_indx, scatter_indx


def grouped_expert_choice_routing(
    gating_output: torch.Tensor,
    capacity: int,
    num_expert_group: int = 8,
    topk_group: int = 4,
    renormalize: bool = True,
) -> Tuple["RoutingData", "GatherIndx", "ScatterIndx"]:
    """
    Grouped Expert Choice routing.

    For pure expert choice, groups don't affect the selection - each expert
    independently selects its tokens. This is a wrapper for compatibility.

    Args:
        gating_output: [N, E] router logits
        capacity: Tokens per expert (C)
        num_expert_group: Number of expert groups (unused for expert choice)
        topk_group: Number of groups to select (unused for expert choice)
        renormalize: Whether to renormalize weights

    Returns:
        routing_data, gather_indx, scatter_indx
    """
    # For expert choice, groups don't matter - each expert independently selects
    return expert_choice_routing(gating_output, capacity, renormalize)


class ExpertChoiceTopKOutput:
    """
    TopK output wrapper for expert choice routing.

    This provides the same interface as TritonKernelTopKOutput for compatibility
    with existing MoE layer code.
    """

    def __init__(
        self,
        routing_data: "RoutingData",
        gather_indx: "GatherIndx",
        scatter_indx: "ScatterIndx",
    ):
        self.routing_data = routing_data
        self.gather_indx = gather_indx
        self.scatter_indx = scatter_indx

    def __iter__(self):
        """Allow tuple unpacking: routing_data, gather_idx, scatter_idx = output"""
        return iter((self.routing_data, self.gather_indx, self.scatter_indx))

    @property
    def format(self):
        """Compatible with TopKOutputFormat.TRITON_KERNEL"""
        from sglang.srt.layers.moe.topk import TopKOutputFormat
        return TopKOutputFormat.TRITON_KERNEL


def create_expert_choice_routing_function(capacity: int, renormalize: bool = True):
    """
    Factory function to create an expert choice routing function with fixed capacity.

    NOTE: This returns ExpertChoiceTopKOutput which has format=TRITON_KERNEL.
    Make sure your MoE layer is using triton kernel backend.

    Args:
        capacity: Tokens per expert
        renormalize: Whether to renormalize weights

    Returns:
        A callable that returns ExpertChoiceTopKOutput (TRITON_KERNEL format)

    Example:
        >>> from dinfer.model.expert_choice_routing import create_expert_choice_routing_function
        >>> routing_fn = create_expert_choice_routing_function(capacity=8)
        >>> # Use as custom_routing_function in your MoE config
    """
    def routing_function(
        hidden_states: torch.Tensor = None,
        gating_output: torch.Tensor = None,
        topk: int = None,
        renormalize: bool = renormalize,
        **kwargs,
    ) -> ExpertChoiceTopKOutput:
        routing_data, gather_idx, scatter_idx = expert_choice_routing(
            gating_output=gating_output,
            capacity=capacity,
            renormalize=renormalize,
        )
        return ExpertChoiceTopKOutput(routing_data, gather_idx, scatter_idx)

    return routing_function


# ----------------------- Direct TopK Replacement -----------------------


class ExpertChoiceTopK:
    """
    Expert Choice TopK module - can be used as drop-in replacement for SGLang's TopK.

    This directly returns TritonKernelTopKOutput format, using native [E, C] routing
    without any conversion to [N, topk] format.

    Usage in modeling code:

        # Replace:
        # self.topk = TopK(top_k=8, ...)

        # With:
        from dinfer.model.expert_choice_routing import ExpertChoiceTopK
        self.topk = ExpertChoiceTopK(capacity=8, renormalize=True)

    Requirements:
        - Triton kernels must be enabled (get_moe_runner_backend().is_triton_kernel() == True)
        - Model must use bfloat16 (triton kernel limitation)
    """

    def __init__(
        self,
        capacity: int,
        renormalize: bool = True,
    ):
        """
        Args:
            capacity: Number of tokens each expert selects (C)
            renormalize: Whether to renormalize routing weights per-token
        """
        self.capacity = capacity
        self.renormalize = renormalize

    def __call__(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        **kwargs,
    ) -> ExpertChoiceTopKOutput:
        """
        Perform expert choice routing.

        Args:
            hidden_states: [N, hidden_dim] - token features (unused, for API compatibility)
            router_logits: [N, E] - router logits

        Returns:
            ExpertChoiceTopKOutput with format=TRITON_KERNEL
        """
        routing_data, gather_idx, scatter_idx = expert_choice_routing(
            gating_output=router_logits,
            capacity=self.capacity,
            renormalize=self.renormalize,
        )
        return ExpertChoiceTopKOutput(routing_data, gather_idx, scatter_idx)


# ----------------------- Statistics / Debug Utilities -----------------------


def get_expert_choice_statistics(
    gating_output: torch.Tensor,
    capacity: int,
) -> dict:
    """
    Get statistics about expert choice routing for debugging.

    Args:
        gating_output: [N, E] router logits
        capacity: Tokens per expert

    Returns:
        Dictionary with statistics:
        - tokens_per_expert: [E] - should all be equal to capacity
        - experts_per_token: [N] - how many experts selected each token
        - max_experts_per_token: maximum overlap
        - min_experts_per_token: minimum overlap
        - avg_experts_per_token: average overlap
    """
    num_tokens, num_experts = gating_output.shape
    actual_capacity = min(capacity, num_tokens)

    # Expert choice selection
    scores = torch.softmax(gating_output.float(), dim=0)
    scores_t = scores.t()
    _, expert_tokens = torch.topk(scores_t, k=actual_capacity, dim=1, sorted=False)

    # Count how many experts selected each token
    experts_per_token = torch.zeros(num_tokens, dtype=torch.int32, device=gating_output.device)
    for e in range(num_experts):
        experts_per_token[expert_tokens[e]] += 1

    # Tokens per expert (should all be capacity for expert choice)
    tokens_per_expert = torch.full((num_experts,), actual_capacity, dtype=torch.int32, device=gating_output.device)

    return {
        "tokens_per_expert": tokens_per_expert,
        "experts_per_token": experts_per_token,
        "max_experts_per_token": experts_per_token.max().item(),
        "min_experts_per_token": experts_per_token.min().item(),
        "avg_experts_per_token": experts_per_token.float().mean().item(),
        "total_assignments": num_experts * actual_capacity,
        "capacity": actual_capacity,
        "num_experts": num_experts,
        "num_tokens": num_tokens,
    }


def verify_expert_balance(routing_data: "RoutingData") -> bool:
    """
    Verify that expert choice achieved perfect load balance.

    Args:
        routing_data: RoutingData from expert_choice_routing

    Returns:
        True if all experts have exactly the same number of tokens
    """
    hist = routing_data.expt_hist
    return (hist == hist[0]).all().item()


def verify_triton_kernel_config(model: "torch.nn.Module") -> dict:
    """
    Verify that the model is correctly configured to use triton kernel backend.

    This should be called after model loading and before inference to ensure
    expert choice routing will work correctly with FusedMoE.

    Args:
        model: The loaded model (e.g., LLaDA2ForCausalLM)

    Returns:
        Dictionary with verification results:
        - moe_runner_backend: Current MOE_RUNNER_BACKEND setting
        - is_triton_kernel: Whether backend is TRITON_KERNEL
        - moe_modules: List of MoE modules found and their triton kernel status
        - all_triton_kernels: True if all MoE modules use triton kernels

    Example:
        >>> from dinfer.model.expert_choice_routing import verify_triton_kernel_config
        >>> result = verify_triton_kernel_config(model)
        >>> print(f"Backend: {result['moe_runner_backend']}")
        >>> print(f"All using triton kernels: {result['all_triton_kernels']}")
    """
    import sglang.srt.layers.moe.utils as moe_utils

    result = {
        "moe_runner_backend": str(moe_utils.MOE_RUNNER_BACKEND),
        "is_triton_kernel": (
            moe_utils.MOE_RUNNER_BACKEND is not None
            and moe_utils.MOE_RUNNER_BACKEND.is_triton_kernel()
        ),
        "moe_modules": [],
        "all_triton_kernels": True,
    }

    for name, module in model.named_modules():
        # Check for MoE layers with FusedMoE experts
        if hasattr(module, 'experts') and hasattr(module.experts, 'quant_method'):
            qm = module.experts.quant_method
            use_triton = getattr(qm, 'use_triton_kernels', None)

            module_info = {
                "name": name,
                "use_triton_kernels": use_triton,
            }
            result["moe_modules"].append(module_info)

            if use_triton is False:
                result["all_triton_kernels"] = False

    return result


def print_triton_kernel_config(model: "torch.nn.Module") -> None:
    """
    Print triton kernel configuration for debugging.

    Call this after model loading to verify the configuration.

    Args:
        model: The loaded model
    """
    import sglang.srt.layers.moe.utils as moe_utils

    print(f"MOE_RUNNER_BACKEND = {moe_utils.MOE_RUNNER_BACKEND}")

    found_moe = False
    for name, module in model.named_modules():
        if hasattr(module, 'experts') and hasattr(module.experts, 'quant_method'):
            qm = module.experts.quant_method
            use_triton = getattr(qm, 'use_triton_kernels', 'N/A')
            print(f"{name}: use_triton_kernels={use_triton}")
            found_moe = True
            break

    if not found_moe:
        print("No MoE modules with quant_method found in model")
