# Expert Choice MoE Implementation
# Native [E, C] format - each expert selects C tokens
#
# Key difference from Token Choice:
# - Token Choice: [N, K] - each token selects K experts
# - Expert Choice: [E, C] - each expert selects C tokens
#
# This implementation directly works with [E, C] format without conversion

from __future__ import annotations

import functools
import os
from typing import TYPE_CHECKING, List, Optional, Tuple, NamedTuple

import torch
import triton
import triton.language as tl

from sglang.srt.utils import (
    get_bool_env_var,
    is_cuda,
    is_hip,
)

_is_hip = is_hip()
_is_cuda = is_cuda()

if _is_cuda or _is_hip:
    from sgl_kernel import gelu_and_mul, silu_and_mul


# ================================ Data Structures ================================

class ExpertChoiceRoutingOutput(NamedTuple):
    """
    Expert Choice routing output in native [E, C] format.

    Attributes:
        expert_tokens: [E, C] - token indices selected by each expert
        expert_weights: [E, C] - routing weights for each selection
        num_experts: int - total number of experts (E)
        capacity: int - tokens per expert (C)
    """
    expert_tokens: torch.Tensor   # [E, C] int32 - token indices
    expert_weights: torch.Tensor  # [E, C] float32 - weights
    num_experts: int
    capacity: int


# ================================ Routing Function ================================

def expert_choice_routing(
    gating_output: torch.Tensor,
    capacity: int,
    renormalize: bool = True,
) -> ExpertChoiceRoutingOutput:
    """
    Expert Choice routing - each expert selects top-C tokens.

    Args:
        gating_output: [N, E] router logits
        capacity: C - number of tokens each expert selects
        renormalize: whether to renormalize weights per-token

    Returns:
        ExpertChoiceRoutingOutput with [E, C] format
    """
    num_tokens, num_experts = gating_output.shape
    device = gating_output.device

    # Expert Choice: softmax over tokens (dim=0) - experts compete for tokens
    scores = torch.softmax(gating_output.float(), dim=0)  # [N, E]

    # Transpose to expert-centric view: [E, N]
    scores_t = scores.t()  # [E, N]

    # Each expert selects top-C tokens
    actual_capacity = min(capacity, num_tokens)
    expert_weights, expert_tokens = torch.topk(
        scores_t, k=actual_capacity, dim=1, sorted=False
    )
    # expert_tokens: [E, C] - token indices selected by each expert
    # expert_weights: [E, C] - corresponding scores

    if renormalize:
        # Renormalize per-token: sum weights across all experts that selected this token
        # Build token -> total_weight mapping
        token_weight_sum = torch.zeros(num_tokens, device=device, dtype=torch.float32)
        token_weight_sum.scatter_add_(
            0,
            expert_tokens.flatten().long(),
            expert_weights.flatten()
        )

        # Avoid division by zero
        token_weight_sum = torch.clamp(token_weight_sum, min=1e-6)

        # Renormalize each weight by the token's total weight
        expert_weights = expert_weights / token_weight_sum[expert_tokens.long()]

    return ExpertChoiceRoutingOutput(
        expert_tokens=expert_tokens.to(torch.int32),
        expert_weights=expert_weights.to(torch.float32),
        num_experts=num_experts,
        capacity=actual_capacity,
    )


# ================================ Fused Routing Kernel ================================

@triton.jit
def _fused_softmax_topk_kernel(
    # Inputs
    gating_ptr,          # [N, E] router logits
    # Outputs
    expert_tokens_ptr,   # [E, C] token indices
    expert_weights_ptr,  # [E, C] weights
    # Dims
    N: tl.constexpr,
    C: tl.constexpr,
    # Strides
    stride_g_n, stride_g_e,
    stride_et_e, stride_et_c,
    stride_ew_e, stride_ew_c,
    # Block
    BLOCK_N: tl.constexpr,
):
    """
    Fused softmax(dim=0) + top-C selection for expert choice routing.
    One program per expert. Replaces: softmax → transpose → topk (3 kernel launches → 1).
    """
    pid_e = tl.program_id(0)

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    # Load gating column for this expert: gating_output[:, e]
    g_ptrs = gating_ptr + offs_n * stride_g_n + pid_e * stride_g_e
    vals = tl.load(g_ptrs, mask=mask_n, other=float('-inf')).to(tl.float32)

    # Softmax over tokens (dim=0)
    max_val = tl.max(vals, axis=0)
    exp_vals = tl.exp(vals - max_val)
    exp_vals = tl.where(mask_n, exp_vals, 0.0)
    sum_val = tl.sum(exp_vals, axis=0)
    scores = exp_vals / sum_val

    # Iterative top-C: find max, store, mask out
    for _i in range(C):
        cur_max = tl.max(scores, axis=0)
        # Use >= with small epsilon for robust float comparison across GPU architectures
        is_max = (scores >= cur_max - 1e-6) & mask_n
        cand = tl.where(is_max, offs_n, N)
        idx = tl.min(cand, axis=0)
        # Clamp to valid range to prevent out-of-bounds scatter
        idx = tl.minimum(idx, N - 1)

        tl.store(
            expert_tokens_ptr + pid_e * stride_et_e + _i * stride_et_c,
            idx.to(tl.int32),
        )
        tl.store(
            expert_weights_ptr + pid_e * stride_ew_e + _i * stride_ew_c,
            cur_max,
        )
        scores = tl.where(offs_n == idx, -1.0, scores)


def expert_choice_routing_fused(
    gating_output: torch.Tensor,
    capacity: int,
    renormalize: bool = True,
) -> ExpertChoiceRoutingOutput:
    """
    Fused expert choice routing: softmax + topk in a single Triton kernel.
    Drop-in replacement for expert_choice_routing().
    """
    num_tokens, num_experts = gating_output.shape
    device = gating_output.device
    actual_capacity = min(capacity, num_tokens)

    expert_tokens = torch.empty((num_experts, actual_capacity), device=device, dtype=torch.int32)
    expert_weights = torch.empty((num_experts, actual_capacity), device=device, dtype=torch.float32)

    BLOCK_N = triton.next_power_of_2(num_tokens)

    _fused_softmax_topk_kernel[(num_experts,)](
        gating_output,
        expert_tokens,
        expert_weights,
        num_tokens,
        actual_capacity,
        gating_output.stride(0), gating_output.stride(1),
        expert_tokens.stride(0), expert_tokens.stride(1),
        expert_weights.stride(0), expert_weights.stride(1),
        BLOCK_N=BLOCK_N,
    )

    if renormalize:
        token_weight_sum = torch.zeros(num_tokens, device=device, dtype=torch.float32)
        token_weight_sum.scatter_add_(
            0,
            expert_tokens.flatten().long(),
            expert_weights.flatten(),
        )
        token_weight_sum = torch.clamp(token_weight_sum, min=1e-6)
        expert_weights = expert_weights / token_weight_sum[expert_tokens.long()]

    return ExpertChoiceRoutingOutput(
        expert_tokens=expert_tokens,
        expert_weights=expert_weights,
        num_experts=num_experts,
        capacity=actual_capacity,
    )


# ================================ Fused Scatter Kernel ================================

@triton.jit
def _fused_scatter_add_kernel(
    # Inputs
    src_ptr,         # [total, D] source (flattened expert output)
    indices_ptr,     # [total] token indices (int32)
    # Output
    dst_ptr,         # [N, D] destination (pre-zeroed, float32)
    # Dims
    total,           # E * C
    D,               # hidden_dim
    # Strides
    stride_src_t, stride_src_d,
    stride_dst_n, stride_dst_d,
    # Block
    BLOCK_D: tl.constexpr,
):
    """
    Scatter-add with Triton atomics.
    Replaces: output.index_add_(0, flat_tokens, flat_output)
    Grid: (E*C, ceil(D/BLOCK_D))
    """
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)

    if pid_t >= total:
        return

    token_idx = tl.load(indices_ptr + pid_t).to(tl.int64)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    vals = tl.load(
        src_ptr + pid_t * stride_src_t + offs_d * stride_src_d,
        mask=mask_d, other=0.0,
    ).to(tl.float32)

    tl.atomic_add(
        dst_ptr + token_idx * stride_dst_n + offs_d * stride_dst_d,
        vals,
        mask=mask_d,
    )


def scatter_add_fused(
    expert_output: torch.Tensor,   # [E, C, D]
    expert_tokens: torch.Tensor,   # [E, C] int32
    num_tokens: int,
    hidden_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Fused scatter-add using Triton atomics."""
    E, C, D = expert_output.shape
    total = E * C

    # Accumulate in float32 for precision, then cast
    output = torch.zeros(num_tokens, hidden_dim, device=device, dtype=torch.float32)

    flat_output = expert_output.reshape(total, D).contiguous()
    flat_tokens = expert_tokens.reshape(total).contiguous()

    BLOCK_D = 128
    grid = (total, triton.cdiv(D, BLOCK_D))

    _fused_scatter_add_kernel[grid](
        flat_output, flat_tokens, output,
        total, D,
        flat_output.stride(0), flat_output.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_D=BLOCK_D,
    )

    return output.to(dtype)


# ================================ Triton GEMM Kernels ================================

@triton.jit
def expert_choice_gather_gemm_kernel(
    # Inputs
    hidden_ptr,          # [N, K] input hidden states
    weight_ptr,          # [E, N_out, K] expert weights (w1 or w13)
    expert_tokens_ptr,   # [E, C] token indices per expert
    # Output
    output_ptr,          # [E, C, N_out] intermediate output
    # Dimensions
    N: tl.constexpr,     # num tokens
    K: tl.constexpr,     # input hidden dim
    N_out: tl.constexpr, # output dim (intermediate_size * 2 for gate_up)
    E: tl.constexpr,     # num experts
    C: tl.constexpr,     # capacity (tokens per expert)
    # Strides
    stride_h_n, stride_h_k,
    stride_w_e, stride_w_n, stride_w_k,
    stride_et_e, stride_et_c,
    stride_o_e, stride_o_c, stride_o_n,
    # Block sizes
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Gather tokens and compute GEMM for Expert Choice.

    For each expert e:
        tokens = expert_tokens[e]  # [C]
        x = hidden_states[tokens]  # [C, K]
        output[e] = x @ weight[e].T  # [C, N_out]
    """
    # Program ID
    pid_e = tl.program_id(0)  # expert index
    pid_c = tl.program_id(1)  # token block within expert
    pid_n = tl.program_id(2)  # output dim block

    # Offsets
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Mask for valid tokens within capacity
    mask_c = offs_c < C
    mask_n = offs_n < N_out

    # Load token indices for this expert
    token_idx_ptr = expert_tokens_ptr + pid_e * stride_et_e + offs_c * stride_et_c
    token_indices = tl.load(token_idx_ptr, mask=mask_c, other=0)

    # Accumulator
    acc = tl.zeros((BLOCK_C, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        mask_k = k_offs < K

        # Load hidden states: gather from token_indices
        # hidden[token_indices, k_offs] -> [BLOCK_C, BLOCK_K]
        h_ptrs = hidden_ptr + token_indices[:, None] * stride_h_n + k_offs[None, :] * stride_h_k
        h = tl.load(h_ptrs, mask=mask_c[:, None] & mask_k[None, :], other=0.0)

        # Load weights: [BLOCK_K, BLOCK_N]
        w_ptrs = weight_ptr + pid_e * stride_w_e + k_offs[:, None] * stride_w_k + offs_n[None, :] * stride_w_n
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # GEMM
        acc += tl.dot(h, w)

    # Store output
    out_ptrs = output_ptr + pid_e * stride_o_e + offs_c[:, None] * stride_o_c + offs_n[None, :] * stride_o_n
    tl.store(out_ptrs, acc.to(output_ptr.dtype.element_ty), mask=mask_c[:, None] & mask_n[None, :])


@triton.jit
def expert_choice_gemm_kernel(
    # Inputs
    intermediate_ptr,    # [E, C, K] intermediate activations (after activation)
    weight_ptr,          # [E, N_out, K] expert weights (w2)
    expert_weights_ptr,  # [E, C] routing weights
    # Output
    output_ptr,          # [E, C, N_out] output per expert (before scatter)
    # Dimensions
    K: tl.constexpr,     # intermediate dim (after activation)
    N_out: tl.constexpr, # output hidden dim
    C: tl.constexpr,     # capacity
    # Strides
    stride_i_e, stride_i_c, stride_i_k,
    stride_w_e, stride_w_n, stride_w_k,
    stride_ew_e, stride_ew_c,
    stride_o_e, stride_o_c, stride_o_n,
    # Block sizes
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Compute GEMM for Expert Choice (w2 projection).

    For each expert e:
        weights = expert_weights[e]  # [C]
        y = intermediate[e] @ weight[e].T  # [C, N_out]
        output[e] = y * weights[:, None]
    """
    # Program ID
    pid_e = tl.program_id(0)  # expert index
    pid_c = tl.program_id(1)  # token block within expert
    pid_n = tl.program_id(2)  # output dim block

    # Offsets
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Mask
    mask_c = offs_c < C
    mask_n = offs_n < N_out

    # Load routing weights
    weight_ptr_r = expert_weights_ptr + pid_e * stride_ew_e + offs_c * stride_ew_c
    routing_weights = tl.load(weight_ptr_r, mask=mask_c, other=0.0)

    # Accumulator
    acc = tl.zeros((BLOCK_C, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        mask_k = k_offs < K

        # Load intermediate activations: [BLOCK_C, BLOCK_K]
        i_ptrs = intermediate_ptr + pid_e * stride_i_e + offs_c[:, None] * stride_i_c + k_offs[None, :] * stride_i_k
        i_val = tl.load(i_ptrs, mask=mask_c[:, None] & mask_k[None, :], other=0.0)

        # Load weights: [BLOCK_K, BLOCK_N]
        w_ptrs = weight_ptr + pid_e * stride_w_e + k_offs[:, None] * stride_w_k + offs_n[None, :] * stride_w_n
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # GEMM
        acc += tl.dot(i_val, w)

    # Apply routing weights
    acc = acc * routing_weights[:, None]

    # Store to per-expert output buffer
    out_ptrs = output_ptr + pid_e * stride_o_e + offs_c[:, None] * stride_o_c + offs_n[None, :] * stride_o_n
    tl.store(out_ptrs, acc.to(output_ptr.dtype.element_ty), mask=mask_c[:, None] & mask_n[None, :])


# ================================ Fused V2 Kernels ================================
# Reduce 4 kernel launches to 2:
#   Kernel 1: gather + GEMM1 + SiLU+mul  (replaces gather_gemm + silu_and_mul)
#   Kernel 2: GEMM2 + weighted scatter   (replaces gemm2 + scatter_add)

@triton.jit
def expert_choice_gather_gemm_silu_kernel(
    # Inputs
    hidden_ptr,          # [N, K] input hidden states
    weight_ptr,          # [E, K, N_gate_up] expert weights w1 (gate_up combined)
    expert_tokens_ptr,   # [E, C] token indices per expert
    # Output
    output_ptr,          # [E, C, HALF_N] intermediate output (after silu+mul)
    # Dimensions
    N: tl.constexpr,     # num tokens
    K: tl.constexpr,     # input hidden dim (e.g. 2048)
    HALF_N: tl.constexpr,  # intermediate_size (e.g. 512, half of gate_up)
    E: tl.constexpr,     # num experts
    C: tl.constexpr,     # capacity
    # Strides - hidden [N, K]
    stride_h_n, stride_h_k,
    # Strides - weight [E, K, gate_up] (SGLang triton format)
    stride_w_e, stride_w_n, stride_w_k,
    # Strides - expert_tokens [E, C]
    stride_et_e, stride_et_c,
    # Strides - output [E, C, HALF_N]
    stride_o_e, stride_o_c, stride_o_n,
    # Block sizes
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fused Gather + GEMM1 + SiLU+mul for Expert Choice.

    Each block computes paired column blocks from w1:
      - gate columns [pid_n*BLOCK_N, pid_n*BLOCK_N + BLOCK_N)
      - up columns   [pid_n*BLOCK_N + HALF_N, pid_n*BLOCK_N + HALF_N + BLOCK_N)
    Then applies: output = silu(gate) * up

    Output shape: [E, C, HALF_N] (half of gate_up)
    """
    pid_e = tl.program_id(0)  # expert index
    pid_c = tl.program_id(1)  # token block within expert
    pid_n = tl.program_id(2)  # output dim block (over HALF_N, not full gate_up)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_c = offs_c < C
    mask_n = offs_n < HALF_N

    # Load token indices for this expert
    token_idx_ptr = expert_tokens_ptr + pid_e * stride_et_e + offs_c * stride_et_c
    token_indices = tl.load(token_idx_ptr, mask=mask_c, other=0)

    # Two accumulators: gate and up halves
    acc_gate = tl.zeros((BLOCK_C, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_C, BLOCK_N), dtype=tl.float32)

    # Column offsets in w1 for gate and up halves
    offs_n_gate = offs_n         # columns [0, HALF_N)
    offs_n_up = offs_n + HALF_N  # columns [HALF_N, 2*HALF_N)

    # Loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        mask_k = k_offs < K

        # Load hidden states: gather from token_indices (shared for both halves)
        h_ptrs = hidden_ptr + token_indices[:, None] * stride_h_n + k_offs[None, :] * stride_h_k
        h = tl.load(h_ptrs, mask=mask_c[:, None] & mask_k[None, :], other=0.0)

        # Load gate weights: w1[e, k, n_gate]
        w_gate_ptrs = weight_ptr + pid_e * stride_w_e + k_offs[:, None] * stride_w_k + offs_n_gate[None, :] * stride_w_n
        w_gate = tl.load(w_gate_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Load up weights: w1[e, k, n_up]
        w_up_ptrs = weight_ptr + pid_e * stride_w_e + k_offs[:, None] * stride_w_k + offs_n_up[None, :] * stride_w_n
        w_up = tl.load(w_up_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # GEMM for both halves
        acc_gate += tl.dot(h, w_gate)
        acc_up += tl.dot(h, w_up)

    # Apply SiLU(gate) * up in fp32 registers
    # SiLU(x) = x * sigmoid(x)
    result = (acc_gate * tl.sigmoid(acc_gate)) * acc_up

    # Store output [E, C, HALF_N]
    out_ptrs = output_ptr + pid_e * stride_o_e + offs_c[:, None] * stride_o_c + offs_n[None, :] * stride_o_n
    tl.store(out_ptrs, result.to(output_ptr.dtype.element_ty), mask=mask_c[:, None] & mask_n[None, :])


@triton.jit
def expert_choice_gemm_scatter_kernel(
    # Inputs
    intermediate_ptr,    # [E, C, K] intermediate activations
    weight_ptr,          # [E, N_out, K] expert weights (w2)
    expert_weights_ptr,  # [E, C] routing weights
    expert_tokens_ptr,   # [E, C] token indices for scatter
    # Output
    output_ptr,          # [N, N_out] output (fp32, pre-zeroed)
    # Dimensions
    K: tl.constexpr,     # intermediate dim
    N_out: tl.constexpr, # output hidden dim
    C: tl.constexpr,     # capacity
    N_tokens: tl.constexpr,  # total number of tokens
    # Strides - intermediate [E, C, K]
    stride_i_e, stride_i_c, stride_i_k,
    # Strides - weight [E, N_out, K] (logical)
    stride_w_e, stride_w_n, stride_w_k,
    # Strides - expert_weights [E, C]
    stride_ew_e, stride_ew_c,
    # Strides - expert_tokens [E, C]
    stride_et_e, stride_et_c,
    # Strides - output [N, N_out]
    stride_o_n, stride_o_d,
    # Block sizes
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fused GEMM2 + weighted scatter for Expert Choice.

    Computes: output[token_idx] += (intermediate[e] @ w2[e].T) * routing_weight
    Uses tl.atomic_add to scatter directly to the output tensor.
    """
    pid_e = tl.program_id(0)  # expert index
    pid_c = tl.program_id(1)  # token block within expert
    pid_n = tl.program_id(2)  # output dim block

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_c = offs_c < C
    mask_n = offs_n < N_out

    # Load routing weights [BLOCK_C]
    ew_ptrs = expert_weights_ptr + pid_e * stride_ew_e + offs_c * stride_ew_c
    routing_weights = tl.load(ew_ptrs, mask=mask_c, other=0.0)

    # Load token indices for scatter [BLOCK_C]
    et_ptrs = expert_tokens_ptr + pid_e * stride_et_e + offs_c * stride_et_c
    token_indices = tl.load(et_ptrs, mask=mask_c, other=0).to(tl.int64)

    # Accumulator
    acc = tl.zeros((BLOCK_C, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension (intermediate_size)
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        mask_k = k_offs < K

        # Load intermediate activations [BLOCK_C, BLOCK_K]
        i_ptrs = intermediate_ptr + pid_e * stride_i_e + offs_c[:, None] * stride_i_c + k_offs[None, :] * stride_i_k
        i_val = tl.load(i_ptrs, mask=mask_c[:, None] & mask_k[None, :], other=0.0)

        # Load weights [BLOCK_K, BLOCK_N]
        w_ptrs = weight_ptr + pid_e * stride_w_e + k_offs[:, None] * stride_w_k + offs_n[None, :] * stride_w_n
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc += tl.dot(i_val, w)

    # Apply routing weights: acc[c, n] *= routing_weights[c]
    acc = acc * routing_weights[:, None]

    # Scatter-add to output[token_indices[c], offs_n] using vectorized atomic_add
    # token_indices: [BLOCK_C], offs_n: [BLOCK_N]
    # Compute 2D output pointer grid: [BLOCK_C, BLOCK_N]
    out_ptrs = output_ptr + token_indices[:, None] * stride_o_n + offs_n[None, :] * stride_o_d
    tl.atomic_add(out_ptrs, acc, mask=mask_c[:, None] & mask_n[None, :])


# ================================ Main Functions ================================
def expert_choice_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    routing_output: ExpertChoiceRoutingOutput,
    activation: str = "silu",
    use_triton_weight_format: bool = True,
) -> torch.Tensor:
    """
    Expert Choice MoE forward pass with native [E, C] format.

    Args:
        hidden_states: [N, hidden_dim] input
        w1: Expert weights for gate_up projection
            - If use_triton_weight_format=True: [E, hidden_dim, intermediate_size*2] (SGLang format)
            - If use_triton_weight_format=False: [E, intermediate_size*2, hidden_dim]
        w2: Expert weights for down projection
            - If use_triton_weight_format=True: [E, intermediate_size, hidden_dim] (SGLang format)
            - If use_triton_weight_format=False: [E, hidden_dim, intermediate_size]
        routing_output: ExpertChoiceRoutingOutput with [E, C] format
        activation: "silu" or "gelu"
        use_triton_weight_format: Whether weights are in SGLang triton kernel format

    Returns:
        output: [N, hidden_dim]
    """
    N, hidden_dim = hidden_states.shape
    E = routing_output.num_experts
    C = routing_output.capacity

    # No PyTorch fallback needed!
    # Even when C < 16, we use compact memory allocation with masked computation.
    # BLOCK_C is set to 16 (minimum for Tensor Core), but we only allocate C elements.
    # The mask (offs_c < C) protects all load/store operations from out-of-bounds access.
    # This is the standard approach for handling "tail effects" in high-performance computing.

    # Determine intermediate_size from w1 shape
    if use_triton_weight_format:
        # w1 is [E, hidden_dim, intermediate_size*2]
        intermediate_size = w1.shape[2] // 2
    else:
        # w1 is [E, intermediate_size*2, hidden_dim]
        intermediate_size = w1.shape[1] // 2

    expert_tokens = routing_output.expert_tokens  # [E, C]
    expert_weights = routing_output.expert_weights  # [E, C]

    device = hidden_states.device
    dtype = hidden_states.dtype

    # ========== Step 1: Gather + GEMM (w1) ==========
    # Output: [E, C, intermediate_size * 2]
    intermediate1 = torch.empty(
        (E, C, intermediate_size * 2),
        device=device,
        dtype=dtype
    )

    # Block sizes for gather_gemm kernel
    # Triton requires BLOCK sizes to be powers of 2 and >= 16 for Tensor Core
    # Even if C < 16, we use BLOCK_C = 16 with mask protection
    # The mask (offs_c < C) ensures only valid data is loaded/stored
    BLOCK_C = 16 if C < 32 else 32
    BLOCK_K = 64
    BLOCK_N = 64

    grid_gemm1 = (
        E,  # experts
        triton.cdiv(C, BLOCK_C),  # tokens per expert
        triton.cdiv(intermediate_size * 2, BLOCK_N),  # output dim
    )

    # w1 strides for kernel: (stride_w_e, stride_w_n, stride_w_k)
    # SGLang triton format: w1 = [E, K, N_out], so K stride = w1.stride(1), N_out stride = w1.stride(2)
    # Standard format: w1 = [E, N_out, K], so N_out stride = w1.stride(1), K stride = w1.stride(2)
    if use_triton_weight_format:
        w1_strides = (w1.stride(0), w1.stride(2), w1.stride(1))  # (E, N_out, K)
    else:
        w1_strides = (w1.stride(0), w1.stride(1), w1.stride(2))  # (E, N_out, K)

    expert_choice_gather_gemm_kernel[grid_gemm1](
        hidden_states, w1, expert_tokens,
        intermediate1,
        N, hidden_dim, intermediate_size * 2, E, C,
        hidden_states.stride(0), hidden_states.stride(1),
        w1_strides[0], w1_strides[1], w1_strides[2],
        expert_tokens.stride(0), expert_tokens.stride(1),
        intermediate1.stride(0), intermediate1.stride(1), intermediate1.stride(2),
        BLOCK_C=BLOCK_C, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
    )

    # ========== Step 2: Activation (SiLU or GELU) ==========
    # intermediate1: [E, C, intermediate_size * 2] -> intermediate2: [E*C, intermediate_size]
    intermediate1_flat = intermediate1.view(E * C, intermediate_size * 2)
    intermediate2 = torch.empty(
        (E * C, intermediate_size),
        device=device,
        dtype=dtype,
    )

    if activation == "silu":
        silu_and_mul(intermediate1_flat, intermediate2)
    elif activation == "gelu":
        gelu_and_mul(intermediate1_flat, intermediate2)
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    intermediate2 = intermediate2.view(E, C, intermediate_size)

    # ========== Step 3: GEMM (w2) ==========
    # Output: [E, C, hidden_dim] - per expert output before scatter
    expert_output = torch.empty(
        (E, C, hidden_dim),
        device=device,
        dtype=dtype
    )

    # Triton requires BLOCK sizes to be powers of 2 and >= 16 for Tensor Core
    # Even if C < 16, we use BLOCK_C = 16 with mask protection
    BLOCK_C = 16 if C < 32 else 32
    BLOCK_K = 64
    BLOCK_N = 64

    grid_gemm2 = (
        E,
        triton.cdiv(C, BLOCK_C),
        triton.cdiv(hidden_dim, BLOCK_N),
    )

    # w2 strides for kernel: (stride_w_e, stride_w_n, stride_w_k)
    # SGLang triton format: w2 = [E, K, N_out], where K=intermediate_size, N_out=hidden_dim
    # Standard format: w2 = [E, N_out, K], where N_out=hidden_dim, K=intermediate_size
    if use_triton_weight_format:
        w2_strides = (w2.stride(0), w2.stride(2), w2.stride(1))  # (E, N_out, K)
    else:
        w2_strides = (w2.stride(0), w2.stride(1), w2.stride(2))  # (E, N_out, K)

    expert_choice_gemm_kernel[grid_gemm2](
        intermediate2, w2, expert_weights,
        expert_output,
        intermediate_size, hidden_dim, C,
        intermediate2.stride(0), intermediate2.stride(1), intermediate2.stride(2),
        w2_strides[0], w2_strides[1], w2_strides[2],
        expert_weights.stride(0), expert_weights.stride(1),
        expert_output.stride(0), expert_output.stride(1), expert_output.stride(2),
        BLOCK_C=BLOCK_C, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
    )

    # ========== Step 4: Scatter-add to output ==========
    output = scatter_add_fused(expert_output, expert_tokens, N, hidden_dim, dtype, device)

    return output


def expert_choice_fused_experts_v2(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    routing_output: ExpertChoiceRoutingOutput,
    activation: str = "silu",
    use_triton_weight_format: bool = True,
) -> torch.Tensor:
    """
    Expert Choice MoE forward pass - V2 with 2 fused kernels.

    Kernel 1: gather + GEMM1 + SiLU+mul → [E, C, intermediate_size]
    Kernel 2: GEMM2 + weighted atomic scatter → [N, hidden_dim]

    Only 2 kernel launches instead of 4 in v1.
    """
    N, hidden_dim = hidden_states.shape
    E = routing_output.num_experts
    C = routing_output.capacity

    if use_triton_weight_format:
        intermediate_size = w1.shape[2] // 2
    else:
        intermediate_size = w1.shape[1] // 2

    expert_tokens = routing_output.expert_tokens
    expert_weights = routing_output.expert_weights

    device = hidden_states.device
    dtype = hidden_states.dtype

    # ========== Step 1: Gather + GEMM1 + SiLU+mul (fused) ==========
    intermediate = torch.empty(
        (E, C, intermediate_size),
        device=device,
        dtype=dtype,
    )

    BLOCK_C = 16 if C < 32 else 32
    BLOCK_K = 64
    BLOCK_N = 64

    grid_gemm1 = (
        E,
        triton.cdiv(C, BLOCK_C),
        triton.cdiv(intermediate_size, BLOCK_N),  # over HALF_N, not full gate_up
    )

    if use_triton_weight_format:
        w1_strides = (w1.stride(0), w1.stride(2), w1.stride(1))
    else:
        w1_strides = (w1.stride(0), w1.stride(1), w1.stride(2))

    expert_choice_gather_gemm_silu_kernel[grid_gemm1](
        hidden_states, w1, expert_tokens,
        intermediate,
        N, hidden_dim, intermediate_size, E, C,
        hidden_states.stride(0), hidden_states.stride(1),
        w1_strides[0], w1_strides[1], w1_strides[2],
        expert_tokens.stride(0), expert_tokens.stride(1),
        intermediate.stride(0), intermediate.stride(1), intermediate.stride(2),
        BLOCK_C=BLOCK_C, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
    )

    # ========== Step 2: GEMM2 + weighted scatter (fused) ==========
    output = torch.zeros(N, hidden_dim, device=device, dtype=torch.float32)

    grid_gemm2 = (
        E,
        triton.cdiv(C, BLOCK_C),
        triton.cdiv(hidden_dim, BLOCK_N),
    )

    if use_triton_weight_format:
        w2_strides = (w2.stride(0), w2.stride(2), w2.stride(1))
    else:
        w2_strides = (w2.stride(0), w2.stride(1), w2.stride(2))

    expert_choice_gemm_scatter_kernel[grid_gemm2](
        intermediate, w2, expert_weights, expert_tokens,
        output,
        intermediate_size, hidden_dim, C, N,
        intermediate.stride(0), intermediate.stride(1), intermediate.stride(2),
        w2_strides[0], w2_strides[1], w2_strides[2],
        expert_weights.stride(0), expert_weights.stride(1),
        expert_tokens.stride(0), expert_tokens.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_C=BLOCK_C, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
    )

    return output.to(dtype)


# ================================ PyTorch Reference Implementation ================================

def expert_choice_fused_experts_torch(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    routing_output: ExpertChoiceRoutingOutput,
    activation: str = "silu",
    use_triton_weight_format: bool = True,
) -> torch.Tensor:
    """
    PyTorch reference implementation for Expert Choice MoE.
    Slower but easier to debug.

    Args:
        hidden_states: [N, hidden_dim]
        w1: Expert weights for gate_up projection
            - If use_triton_weight_format=True: [E, hidden_dim, intermediate_size*2]
            - If use_triton_weight_format=False: [E, intermediate_size*2, hidden_dim]
        w2: Expert weights for down projection
            - If use_triton_weight_format=True: [E, intermediate_size, hidden_dim]
            - If use_triton_weight_format=False: [E, hidden_dim, intermediate_size]
        routing_output: ExpertChoiceRoutingOutput
        activation: "silu" or "gelu"
        use_triton_weight_format: Whether weights are in SGLang triton kernel format

    Returns:
        output: [N, hidden_dim]
    """
    N, hidden_dim = hidden_states.shape
    E = routing_output.num_experts
    C = routing_output.capacity

    expert_tokens = routing_output.expert_tokens  # [E, C]
    expert_weights = routing_output.expert_weights  # [E, C]

    output = torch.zeros_like(hidden_states)

    for e in range(E):
        # Get tokens selected by this expert
        token_indices = expert_tokens[e]  # [C]
        weights = expert_weights[e]  # [C]

        # Gather tokens
        x = hidden_states[token_indices.long()]  # [C, hidden_dim]

        # First GEMM: [C, hidden_dim] @ [hidden_dim, intermediate*2] -> [C, intermediate*2]
        if use_triton_weight_format:
            # w1[e] is [hidden_dim, intermediate*2], need to compute x @ w1[e]
            gate_up = x @ w1[e]  # [C, intermediate*2]
        else:
            # w1[e] is [intermediate*2, hidden_dim], need to compute x @ w1[e].t()
            gate_up = x @ w1[e].t()  # [C, intermediate*2]

        # Activation
        intermediate = gate_up.shape[-1] // 2
        gate = gate_up[:, :intermediate]
        up = gate_up[:, intermediate:]

        if activation == "silu":
            activated = torch.nn.functional.silu(gate) * up
        elif activation == "gelu":
            activated = torch.nn.functional.gelu(gate) * up
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Second GEMM: [C, intermediate] @ [intermediate, hidden_dim] -> [C, hidden_dim]
        if use_triton_weight_format:
            # w2[e] is [intermediate, hidden_dim], need to compute activated @ w2[e]
            y = activated @ w2[e]  # [C, hidden_dim]
        else:
            # w2[e] is [hidden_dim, intermediate], need to compute activated @ w2[e].t()
            y = activated @ w2[e].t()  # [C, hidden_dim]

        # Apply routing weights and scatter-add
        y = y * weights.unsqueeze(-1)  # [C, hidden_dim]
        output.index_add_(0, token_indices.long(), y.to(output.dtype))

    return output


# ================================ High-Level API ================================

def expert_choice_moe_forward(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    capacity: int,
    activation: str = "silu",
    renormalize: bool = True,
    use_torch_impl: bool = False,
    use_triton_weight_format: bool = True,
) -> torch.Tensor:
    """
    Complete Expert Choice MoE forward pass.

    Args:
        hidden_states: [N, hidden_dim] input tokens
        gating_output: [N, E] router logits
        w1: Expert weights (format depends on use_triton_weight_format)
        w2: Expert weights (format depends on use_triton_weight_format)
        capacity: C - tokens per expert
        activation: "silu" or "gelu"
        renormalize: whether to renormalize routing weights
        use_torch_impl: use PyTorch reference implementation (slower, for debugging)
        use_triton_weight_format: whether weights are in SGLang triton format

    Returns:
        output: [N, hidden_dim]
    """
    # Step 1: Compute routing
    routing_output = expert_choice_routing(
        gating_output=gating_output,
        capacity=capacity,
        renormalize=renormalize,
    )

    # Step 2: Apply experts
    if use_torch_impl:
        return expert_choice_fused_experts_torch(
            hidden_states, w1, w2, routing_output, activation, use_triton_weight_format
        )
    else:
        return expert_choice_fused_experts(
            hidden_states, w1, w2, routing_output, activation, use_triton_weight_format
        )


# ================================ Testing ================================

def test_expert_choice_moe():
    """Test Expert Choice MoE implementation."""
    torch.manual_seed(42)

    # Dimensions
    N = 128  # tokens
    hidden_dim = 512
    intermediate_size = 1024
    E = 8  # experts
    C = 32  # capacity (tokens per expert)

    # Create inputs
    hidden_states = torch.randn(N, hidden_dim, device='cuda', dtype=torch.bfloat16)
    gating_output = torch.randn(N, E, device='cuda', dtype=torch.float32)
    w1 = torch.randn(E, intermediate_size * 2, hidden_dim, device='cuda', dtype=torch.bfloat16) * 0.01
    w2 = torch.randn(E, hidden_dim, intermediate_size, device='cuda', dtype=torch.bfloat16) * 0.01

    # Test routing
    routing_output = expert_choice_routing(gating_output, capacity=C)
    print(f"expert_tokens shape: {routing_output.expert_tokens.shape}")
    print(f"expert_weights shape: {routing_output.expert_weights.shape}")

    # Test PyTorch implementation
    output_torch = expert_choice_moe_forward(
        hidden_states, gating_output, w1, w2,
        capacity=C, use_torch_impl=True
    )
    print(f"PyTorch output shape: {output_torch.shape}")
    print(f"PyTorch output norm: {output_torch.norm().item():.4f}")

    # Test Triton implementation
    output_triton = expert_choice_moe_forward(
        hidden_states, gating_output, w1, w2,
        capacity=C, use_torch_impl=False
    )
    print(f"Triton output shape: {output_triton.shape}")
    print(f"Triton output norm: {output_triton.norm().item():.4f}")

    # Compare
    diff = (output_torch - output_triton).abs().max().item()
    print(f"Max difference: {diff:.6f}")

    return diff < 0.01


if __name__ == "__main__":
    test_expert_choice_moe()
