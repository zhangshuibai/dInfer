#!/usr/bin/env python3
"""
Benchmark: Token Choice vs Expert Choice (Original) vs Expert Choice (Fused Triton)

Benchmarks the MoE layer forward path with three variants:
  Token Choice:        gate → TopK → FusedMoE (SGLang matmul_ogs)
  Expert Choice:       gate → expert_choice_routing (PyTorch) → expert_choice_fused_experts
  Expert Choice Fused: gate → expert_choice_routing_fused (Triton) → expert_choice_fused_experts (Triton scatter)

Usage:
    # Use real model weights (recommended for realistic routing patterns)
    python bench_moe_kernel.py --model_path /path/to/LLaDA2.0-mini-preview

    # Use random weights (no model needed, faster to start)
    python bench_moe_kernel.py --random_weights

    # Custom token counts
    python bench_moe_kernel.py --random_weights --num_tokens 512 1024 2048 4096
"""

import os
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

import argparse
import sys
import time
from collections import defaultdict

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))


# ─────────────────────── helpers ───────────────────────

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/LLaDA2.0-mini-preview")


def init_distributed(tp_size=1, model_path=None):
    """Minimal distributed init for SGLang components."""
    import sglang.srt.distributed as distributed
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.dp_attention import initialize_dp_attention

    # Use real model path for ServerArgs (it validates the path)
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", tp_size))
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29599")

    if not torch.distributed.is_initialized():
        distributed.init_distributed_environment(world_size, rank, "env://", rank, "nccl")
        distributed.initialize_model_parallel(tp_size, tp_size, 1, backend="nccl")

    # Monkey-patch all_reduce / all_gather (same as your eval script)
    def torch_all_reduce(tensor):
        torch.distributed.all_reduce(tensor)
        return tensor
    distributed.tensor_model_parallel_all_reduce = torch_all_reduce

    server_args = ServerArgs(
        model_path=model_path, enable_dp_attention=True,
        trust_remote_code=True, tp_size=tp_size, dp_size=1, pp_size=1,
    )
    try:
        from sglang.srt.server_args import set_global_server_args_for_scheduler
        set_global_server_args_for_scheduler(server_args)
    except ImportError:
        pass
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    initialize_dp_attention(server_args=server_args, model_config=model_config)
    initialize_moe_config(server_args)
    return server_args


def get_moe_layer(model):
    """Find the first MoE layer from the model."""
    for layer in model.model.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
            return layer.mlp
    raise RuntimeError("No MoE layer found")


# ─────────────────────── benchmark core ───────────────────────

@torch.no_grad()
def bench_token_choice(moe_layer, hidden_states, warmup=10, repeat=100):
    """Benchmark: gate → TopK → FusedMoE (SGLang path)."""
    N = hidden_states.shape[0]
    hs = hidden_states.view(-1, hidden_states.shape[-1])

    # warmup
    for _ in range(warmup):
        router_logits = moe_layer.gate(hs)
        topk_output = moe_layer.topk(hs, router_logits)
        _ = moe_layer.experts(hs, topk_output)
    torch.cuda.synchronize()

    # timed runs
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

    for i in range(repeat):
        start_events[i].record()
        router_logits = moe_layer.gate(hs)
        topk_output = moe_layer.topk(hs, router_logits)
        _ = moe_layer.experts(hs, topk_output)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return times


@torch.no_grad()
def bench_expert_choice(moe_layer, hidden_states, warmup=10, repeat=100):
    """Benchmark: gate → expert_choice_routing → expert_choice_fused_experts (custom path)."""
    from dinfer.model.fused_moe_ec import expert_choice_routing, expert_choice_fused_experts

    N = hidden_states.shape[0]
    hs = hidden_states.view(-1, hidden_states.shape[-1])
    top_k = moe_layer.top_k
    num_experts = moe_layer.num_experts

    w1 = moe_layer.experts.w13_weight
    w2 = moe_layer.experts.w2_weight
    hidden_dim = hs.shape[-1]
    use_triton_format = (w1.shape[1] == hidden_dim)

    # warmup
    for _ in range(warmup):
        router_logits = moe_layer.gate(hs)
        num_tokens = hs.shape[0]
        capacity = max(1, (num_tokens * top_k) // num_experts)
        routing_output = expert_choice_routing(
            gating_output=router_logits, capacity=capacity,
            renormalize=moe_layer.norm_topk_prob,
        )
        _ = expert_choice_fused_experts(
            hidden_states=hs, w1=w1, w2=w2,
            routing_output=routing_output, activation="silu",
            use_triton_weight_format=use_triton_format,
        )
    torch.cuda.synchronize()

    # timed runs
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

    for i in range(repeat):
        start_events[i].record()
        router_logits = moe_layer.gate(hs)
        num_tokens = hs.shape[0]
        capacity = max(1, (num_tokens * top_k) // num_experts)
        routing_output = expert_choice_routing(
            gating_output=router_logits, capacity=capacity,
            renormalize=moe_layer.norm_topk_prob,
        )
        _ = expert_choice_fused_experts(
            hidden_states=hs, w1=w1, w2=w2,
            routing_output=routing_output, activation="silu",
            use_triton_weight_format=use_triton_format,
        )
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return times


@torch.no_grad()
def bench_expert_choice_fused(moe_layer, hidden_states, warmup=10, repeat=100):
    """Benchmark: gate → fused_routing → fused_experts (fused Triton kernels)."""
    from dinfer.model.fused_moe_ec import expert_choice_routing_fused, expert_choice_fused_experts

    N = hidden_states.shape[0]
    hs = hidden_states.view(-1, hidden_states.shape[-1])
    top_k = moe_layer.top_k
    num_experts = moe_layer.num_experts

    w1 = moe_layer.experts.w13_weight
    w2 = moe_layer.experts.w2_weight
    hidden_dim = hs.shape[-1]
    use_triton_format = (w1.shape[1] == hidden_dim)

    # warmup
    for _ in range(warmup):
        router_logits = moe_layer.gate(hs)
        num_tokens = hs.shape[0]
        capacity = max(1, (num_tokens * top_k) // num_experts)
        routing_output = expert_choice_routing_fused(
            gating_output=router_logits, capacity=capacity,
            renormalize=moe_layer.norm_topk_prob,
        )
        _ = expert_choice_fused_experts(
            hidden_states=hs, w1=w1, w2=w2,
            routing_output=routing_output, activation="silu",
            use_triton_weight_format=use_triton_format,
        )
    torch.cuda.synchronize()

    # timed runs
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

    for i in range(repeat):
        start_events[i].record()
        router_logits = moe_layer.gate(hs)
        num_tokens = hs.shape[0]
        capacity = max(1, (num_tokens * top_k) // num_experts)
        routing_output = expert_choice_routing_fused(
            gating_output=router_logits, capacity=capacity,
            renormalize=moe_layer.norm_topk_prob,
        )
        _ = expert_choice_fused_experts(
            hidden_states=hs, w1=w1, w2=w2,
            routing_output=routing_output, activation="silu",
            use_triton_weight_format=use_triton_format,
        )
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return times


@torch.no_grad()
def bench_expert_choice_fused_v2(moe_layer, hidden_states, warmup=10, repeat=100):
    """Benchmark: gate → fused_routing → fused_experts_v2 (2 kernels: GEMM1+SiLU, GEMM2+Scatter)."""
    from dinfer.model.fused_moe_ec import expert_choice_routing_fused, expert_choice_fused_experts_v2

    hs = hidden_states.view(-1, hidden_states.shape[-1])
    top_k = moe_layer.top_k
    num_experts = moe_layer.num_experts
    w1 = moe_layer.experts.w13_weight
    w2 = moe_layer.experts.w2_weight
    hidden_dim = hs.shape[-1]
    use_triton_format = (w1.shape[1] == hidden_dim)

    # warmup
    for _ in range(warmup):
        router_logits = moe_layer.gate(hs)
        num_tokens = hs.shape[0]
        capacity = max(1, (num_tokens * top_k) // num_experts)
        routing_output = expert_choice_routing_fused(
            gating_output=router_logits, capacity=capacity,
            renormalize=moe_layer.norm_topk_prob,
        )
        _ = expert_choice_fused_experts_v2(
            hidden_states=hs, w1=w1, w2=w2,
            routing_output=routing_output, activation="silu",
            use_triton_weight_format=use_triton_format,
        )
    torch.cuda.synchronize()

    # timed runs
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

    for i in range(repeat):
        start_events[i].record()
        router_logits = moe_layer.gate(hs)
        num_tokens = hs.shape[0]
        capacity = max(1, (num_tokens * top_k) // num_experts)
        routing_output = expert_choice_routing_fused(
            gating_output=router_logits, capacity=capacity,
            renormalize=moe_layer.norm_topk_prob,
        )
        _ = expert_choice_fused_experts_v2(
            hidden_states=hs, w1=w1, w2=w2,
            routing_output=routing_output, activation="silu",
            use_triton_weight_format=use_triton_format,
        )
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return times


@torch.no_grad()
def bench_token_choice_breakdown(moe_layer, hidden_states, warmup=10, repeat=100):
    """Breakdown for Token Choice: gate / topk / experts (FusedMoE)."""
    hs = hidden_states.view(-1, hidden_states.shape[-1])

    # warmup
    for _ in range(warmup):
        rl = moe_layer.gate(hs)
        tk = moe_layer.topk(hs, rl)
        _ = moe_layer.experts(hs, tk)
    torch.cuda.synchronize()

    stages = ["gate", "routing", "experts"]
    results = {s: [] for s in stages}

    for _ in range(repeat):
        events = {s: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for s in stages}

        events["gate"][0].record()
        rl = moe_layer.gate(hs)
        events["gate"][1].record()

        events["routing"][0].record()
        tk = moe_layer.topk(hs, rl)
        events["routing"][1].record()

        events["experts"][0].record()
        _ = moe_layer.experts(hs, tk)
        events["experts"][1].record()

        torch.cuda.synchronize()
        for s in stages:
            results[s].append(events[s][0].elapsed_time(events[s][1]))

    return results


def _prepare_ec_common(moe_layer, hidden_states):
    """Prepare common EC parameters for breakdown benchmarks."""
    import triton
    hs = hidden_states.view(-1, hidden_states.shape[-1])
    top_k = moe_layer.top_k
    num_experts = moe_layer.num_experts
    w1 = moe_layer.experts.w13_weight
    w2 = moe_layer.experts.w2_weight
    hidden_dim = hs.shape[-1]
    use_triton_format = (w1.shape[1] == hidden_dim)
    if use_triton_format:
        intermediate_size = w1.shape[2] // 2
    else:
        intermediate_size = w1.shape[1] // 2
    num_tokens = hs.shape[0]
    capacity = max(1, (num_tokens * top_k) // num_experts)
    E = num_experts
    C = capacity
    device = hs.device
    dtype = hs.dtype
    BLOCK_C = 16 if C < 32 else 32
    grid1 = (E, triton.cdiv(C, BLOCK_C), triton.cdiv(intermediate_size * 2, 64))
    grid2 = (E, triton.cdiv(C, BLOCK_C), triton.cdiv(hidden_dim, 64))
    if use_triton_format:
        w1_strides = (w1.stride(0), w1.stride(2), w1.stride(1))
        w2_strides = (w2.stride(0), w2.stride(2), w2.stride(1))
    else:
        w1_strides = (w1.stride(0), w1.stride(1), w1.stride(2))
        w2_strides = (w2.stride(0), w2.stride(1), w2.stride(2))
    return dict(
        hs=hs, w1=w1, w2=w2, E=E, C=C, hidden_dim=hidden_dim,
        intermediate_size=intermediate_size, num_tokens=num_tokens,
        capacity=capacity, device=device, dtype=dtype, BLOCK_C=BLOCK_C,
        grid1=grid1, grid2=grid2, w1_strides=w1_strides, w2_strides=w2_strides,
    )


@torch.no_grad()
def bench_expert_choice_breakdown(moe_layer, hidden_states, warmup=10, repeat=100):
    """Breakdown: gate / routing(PyTorch) / gemm1 / activation / gemm2 / scatter(index_add_)."""
    from dinfer.model.fused_moe_ec import (
        expert_choice_routing,
        expert_choice_gather_gemm_kernel,
        expert_choice_gemm_kernel,
    )
    from sgl_kernel import silu_and_mul

    p = _prepare_ec_common(moe_layer, hidden_states)
    hs, w1, w2 = p['hs'], p['w1'], p['w2']
    E, C, hidden_dim = p['E'], p['C'], p['hidden_dim']
    intermediate_size = p['intermediate_size']
    num_tokens, capacity = p['num_tokens'], p['capacity']
    device, dtype = p['device'], p['dtype']
    BLOCK_C = p['BLOCK_C']
    grid1, grid2 = p['grid1'], p['grid2']
    w1_strides, w2_strides = p['w1_strides'], p['w2_strides']

    # warmup
    for _ in range(warmup):
        rl = moe_layer.gate(hs)
        ro = expert_choice_routing(rl, capacity, moe_layer.norm_topk_prob)
        intermediate1 = torch.empty((E, C, intermediate_size * 2), device=device, dtype=dtype)
        expert_choice_gather_gemm_kernel[grid1](
            hs, w1, ro.expert_tokens, intermediate1,
            num_tokens, hidden_dim, intermediate_size * 2, E, C,
            hs.stride(0), hs.stride(1),
            w1_strides[0], w1_strides[1], w1_strides[2],
            ro.expert_tokens.stride(0), ro.expert_tokens.stride(1),
            intermediate1.stride(0), intermediate1.stride(1), intermediate1.stride(2),
            BLOCK_C=BLOCK_C, BLOCK_K=64, BLOCK_N=64,
        )
        i1_flat = intermediate1.view(E * C, intermediate_size * 2)
        intermediate2 = torch.empty((E * C, intermediate_size), device=device, dtype=dtype)
        silu_and_mul(i1_flat, intermediate2)
        intermediate2 = intermediate2.view(E, C, intermediate_size)
        expert_output = torch.empty((E, C, hidden_dim), device=device, dtype=dtype)
        expert_choice_gemm_kernel[grid2](
            intermediate2, w2, ro.expert_weights, expert_output,
            intermediate_size, hidden_dim, C,
            intermediate2.stride(0), intermediate2.stride(1), intermediate2.stride(2),
            w2_strides[0], w2_strides[1], w2_strides[2],
            ro.expert_weights.stride(0), ro.expert_weights.stride(1),
            expert_output.stride(0), expert_output.stride(1), expert_output.stride(2),
            BLOCK_C=BLOCK_C, BLOCK_K=64, BLOCK_N=64,
        )
        output = torch.zeros(num_tokens, hidden_dim, device=device, dtype=dtype)
        output.index_add_(0, ro.expert_tokens.flatten().long(), expert_output.view(E * C, hidden_dim))
    torch.cuda.synchronize()

    stages = ["gate", "routing", "gemm1", "activation", "gemm2", "scatter"]
    results = {s: [] for s in stages}

    for _ in range(repeat):
        events = {s: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for s in stages}

        events["gate"][0].record()
        rl = moe_layer.gate(hs)
        events["gate"][1].record()

        events["routing"][0].record()
        ro = expert_choice_routing(rl, capacity, moe_layer.norm_topk_prob)
        events["routing"][1].record()

        intermediate1 = torch.empty((E, C, intermediate_size * 2), device=device, dtype=dtype)
        events["gemm1"][0].record()
        expert_choice_gather_gemm_kernel[grid1](
            hs, w1, ro.expert_tokens, intermediate1,
            num_tokens, hidden_dim, intermediate_size * 2, E, C,
            hs.stride(0), hs.stride(1),
            w1_strides[0], w1_strides[1], w1_strides[2],
            ro.expert_tokens.stride(0), ro.expert_tokens.stride(1),
            intermediate1.stride(0), intermediate1.stride(1), intermediate1.stride(2),
            BLOCK_C=BLOCK_C, BLOCK_K=64, BLOCK_N=64,
        )
        events["gemm1"][1].record()

        i1_flat = intermediate1.view(E * C, intermediate_size * 2)
        intermediate2 = torch.empty((E * C, intermediate_size), device=device, dtype=dtype)
        events["activation"][0].record()
        silu_and_mul(i1_flat, intermediate2)
        events["activation"][1].record()
        intermediate2 = intermediate2.view(E, C, intermediate_size)

        expert_output = torch.empty((E, C, hidden_dim), device=device, dtype=dtype)
        events["gemm2"][0].record()
        expert_choice_gemm_kernel[grid2](
            intermediate2, w2, ro.expert_weights, expert_output,
            intermediate_size, hidden_dim, C,
            intermediate2.stride(0), intermediate2.stride(1), intermediate2.stride(2),
            w2_strides[0], w2_strides[1], w2_strides[2],
            ro.expert_weights.stride(0), ro.expert_weights.stride(1),
            expert_output.stride(0), expert_output.stride(1), expert_output.stride(2),
            BLOCK_C=BLOCK_C, BLOCK_K=64, BLOCK_N=64,
        )
        events["gemm2"][1].record()

        output = torch.zeros(num_tokens, hidden_dim, device=device, dtype=dtype)
        events["scatter"][0].record()
        output.index_add_(0, ro.expert_tokens.flatten().long(), expert_output.view(E * C, hidden_dim))
        events["scatter"][1].record()

        torch.cuda.synchronize()
        for s in stages:
            results[s].append(events[s][0].elapsed_time(events[s][1]))

    return results


@torch.no_grad()
def bench_expert_choice_fused_breakdown(moe_layer, hidden_states, warmup=10, repeat=100):
    """Breakdown: gate / routing_fused(Triton) / gemm1 / activation / gemm2 / scatter_fused(Triton)."""
    from dinfer.model.fused_moe_ec import (
        expert_choice_routing_fused, scatter_add_fused,
        expert_choice_gather_gemm_kernel,
        expert_choice_gemm_kernel,
    )
    from sgl_kernel import silu_and_mul

    p = _prepare_ec_common(moe_layer, hidden_states)
    hs, w1, w2 = p['hs'], p['w1'], p['w2']
    E, C, hidden_dim = p['E'], p['C'], p['hidden_dim']
    intermediate_size = p['intermediate_size']
    num_tokens, capacity = p['num_tokens'], p['capacity']
    device, dtype = p['device'], p['dtype']
    BLOCK_C = p['BLOCK_C']
    grid1, grid2 = p['grid1'], p['grid2']
    w1_strides, w2_strides = p['w1_strides'], p['w2_strides']

    # warmup
    for _ in range(warmup):
        rl = moe_layer.gate(hs)
        ro = expert_choice_routing_fused(rl, capacity, moe_layer.norm_topk_prob)
        intermediate1 = torch.empty((E, C, intermediate_size * 2), device=device, dtype=dtype)
        expert_choice_gather_gemm_kernel[grid1](
            hs, w1, ro.expert_tokens, intermediate1,
            num_tokens, hidden_dim, intermediate_size * 2, E, C,
            hs.stride(0), hs.stride(1),
            w1_strides[0], w1_strides[1], w1_strides[2],
            ro.expert_tokens.stride(0), ro.expert_tokens.stride(1),
            intermediate1.stride(0), intermediate1.stride(1), intermediate1.stride(2),
            BLOCK_C=BLOCK_C, BLOCK_K=64, BLOCK_N=64,
        )
        i1_flat = intermediate1.view(E * C, intermediate_size * 2)
        intermediate2 = torch.empty((E * C, intermediate_size), device=device, dtype=dtype)
        silu_and_mul(i1_flat, intermediate2)
        intermediate2 = intermediate2.view(E, C, intermediate_size)
        expert_output = torch.empty((E, C, hidden_dim), device=device, dtype=dtype)
        expert_choice_gemm_kernel[grid2](
            intermediate2, w2, ro.expert_weights, expert_output,
            intermediate_size, hidden_dim, C,
            intermediate2.stride(0), intermediate2.stride(1), intermediate2.stride(2),
            w2_strides[0], w2_strides[1], w2_strides[2],
            ro.expert_weights.stride(0), ro.expert_weights.stride(1),
            expert_output.stride(0), expert_output.stride(1), expert_output.stride(2),
            BLOCK_C=BLOCK_C, BLOCK_K=64, BLOCK_N=64,
        )
        _ = scatter_add_fused(expert_output, ro.expert_tokens, num_tokens, hidden_dim, dtype, device)
    torch.cuda.synchronize()

    stages = ["gate", "routing", "gemm1", "activation", "gemm2", "scatter"]
    results = {s: [] for s in stages}

    for _ in range(repeat):
        events = {s: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for s in stages}

        events["gate"][0].record()
        rl = moe_layer.gate(hs)
        events["gate"][1].record()

        events["routing"][0].record()
        ro = expert_choice_routing_fused(rl, capacity, moe_layer.norm_topk_prob)
        events["routing"][1].record()

        intermediate1 = torch.empty((E, C, intermediate_size * 2), device=device, dtype=dtype)
        events["gemm1"][0].record()
        expert_choice_gather_gemm_kernel[grid1](
            hs, w1, ro.expert_tokens, intermediate1,
            num_tokens, hidden_dim, intermediate_size * 2, E, C,
            hs.stride(0), hs.stride(1),
            w1_strides[0], w1_strides[1], w1_strides[2],
            ro.expert_tokens.stride(0), ro.expert_tokens.stride(1),
            intermediate1.stride(0), intermediate1.stride(1), intermediate1.stride(2),
            BLOCK_C=BLOCK_C, BLOCK_K=64, BLOCK_N=64,
        )
        events["gemm1"][1].record()

        i1_flat = intermediate1.view(E * C, intermediate_size * 2)
        intermediate2 = torch.empty((E * C, intermediate_size), device=device, dtype=dtype)
        events["activation"][0].record()
        silu_and_mul(i1_flat, intermediate2)
        events["activation"][1].record()
        intermediate2 = intermediate2.view(E, C, intermediate_size)

        expert_output = torch.empty((E, C, hidden_dim), device=device, dtype=dtype)
        events["gemm2"][0].record()
        expert_choice_gemm_kernel[grid2](
            intermediate2, w2, ro.expert_weights, expert_output,
            intermediate_size, hidden_dim, C,
            intermediate2.stride(0), intermediate2.stride(1), intermediate2.stride(2),
            w2_strides[0], w2_strides[1], w2_strides[2],
            ro.expert_weights.stride(0), ro.expert_weights.stride(1),
            expert_output.stride(0), expert_output.stride(1), expert_output.stride(2),
            BLOCK_C=BLOCK_C, BLOCK_K=64, BLOCK_N=64,
        )
        events["gemm2"][1].record()

        events["scatter"][0].record()
        _ = scatter_add_fused(expert_output, ro.expert_tokens, num_tokens, hidden_dim, dtype, device)
        events["scatter"][1].record()

        torch.cuda.synchronize()
        for s in stages:
            results[s].append(events[s][0].elapsed_time(events[s][1]))

    return results


@torch.no_grad()
def bench_expert_choice_fused_v2_breakdown(moe_layer, hidden_states, warmup=10, repeat=100):
    """Breakdown: gate / routing_fused / gemm1_silu(fused) / gemm2_scatter(fused) — only 2 compute kernels."""
    from dinfer.model.fused_moe_ec import (
        expert_choice_routing_fused,
        expert_choice_gather_gemm_silu_kernel,
        expert_choice_gemm_scatter_kernel,
    )

    import triton
    hs = hidden_states.view(-1, hidden_states.shape[-1])
    top_k = moe_layer.top_k
    num_experts = moe_layer.num_experts
    w1 = moe_layer.experts.w13_weight
    w2 = moe_layer.experts.w2_weight
    hidden_dim = hs.shape[-1]
    use_triton_format = (w1.shape[1] == hidden_dim)
    if use_triton_format:
        intermediate_size = w1.shape[2] // 2
    else:
        intermediate_size = w1.shape[1] // 2
    num_tokens = hs.shape[0]
    capacity = max(1, (num_tokens * top_k) // num_experts)
    E = num_experts
    C = capacity
    device = hs.device
    dtype = hs.dtype

    BLOCK_C = 16 if C < 32 else 32
    grid1 = (E, triton.cdiv(C, BLOCK_C), triton.cdiv(intermediate_size, 64))
    grid2 = (E, triton.cdiv(C, BLOCK_C), triton.cdiv(hidden_dim, 64))
    if use_triton_format:
        w1_strides = (w1.stride(0), w1.stride(2), w1.stride(1))
        w2_strides = (w2.stride(0), w2.stride(2), w2.stride(1))
    else:
        w1_strides = (w1.stride(0), w1.stride(1), w1.stride(2))
        w2_strides = (w2.stride(0), w2.stride(1), w2.stride(2))

    # warmup
    for _ in range(warmup):
        rl = moe_layer.gate(hs)
        ro = expert_choice_routing_fused(rl, capacity, moe_layer.norm_topk_prob)
        intermediate = torch.empty((E, C, intermediate_size), device=device, dtype=dtype)
        expert_choice_gather_gemm_silu_kernel[grid1](
            hs, w1, ro.expert_tokens, intermediate,
            num_tokens, hidden_dim, intermediate_size, E, C,
            hs.stride(0), hs.stride(1),
            w1_strides[0], w1_strides[1], w1_strides[2],
            ro.expert_tokens.stride(0), ro.expert_tokens.stride(1),
            intermediate.stride(0), intermediate.stride(1), intermediate.stride(2),
            BLOCK_C=BLOCK_C, BLOCK_K=64, BLOCK_N=64,
        )
        output = torch.zeros(num_tokens, hidden_dim, device=device, dtype=torch.float32)
        expert_choice_gemm_scatter_kernel[grid2](
            intermediate, w2, ro.expert_weights, ro.expert_tokens, output,
            intermediate_size, hidden_dim, C, num_tokens,
            intermediate.stride(0), intermediate.stride(1), intermediate.stride(2),
            w2_strides[0], w2_strides[1], w2_strides[2],
            ro.expert_weights.stride(0), ro.expert_weights.stride(1),
            ro.expert_tokens.stride(0), ro.expert_tokens.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_C=BLOCK_C, BLOCK_K=64, BLOCK_N=64,
        )
    torch.cuda.synchronize()

    stages = ["gate", "routing", "gemm1_silu", "gemm2_scatter"]
    results = {s: [] for s in stages}

    for _ in range(repeat):
        events = {s: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for s in stages}

        events["gate"][0].record()
        rl = moe_layer.gate(hs)
        events["gate"][1].record()

        events["routing"][0].record()
        ro = expert_choice_routing_fused(rl, capacity, moe_layer.norm_topk_prob)
        events["routing"][1].record()

        intermediate = torch.empty((E, C, intermediate_size), device=device, dtype=dtype)
        events["gemm1_silu"][0].record()
        expert_choice_gather_gemm_silu_kernel[grid1](
            hs, w1, ro.expert_tokens, intermediate,
            num_tokens, hidden_dim, intermediate_size, E, C,
            hs.stride(0), hs.stride(1),
            w1_strides[0], w1_strides[1], w1_strides[2],
            ro.expert_tokens.stride(0), ro.expert_tokens.stride(1),
            intermediate.stride(0), intermediate.stride(1), intermediate.stride(2),
            BLOCK_C=BLOCK_C, BLOCK_K=64, BLOCK_N=64,
        )
        events["gemm1_silu"][1].record()

        output = torch.zeros(num_tokens, hidden_dim, device=device, dtype=torch.float32)
        events["gemm2_scatter"][0].record()
        expert_choice_gemm_scatter_kernel[grid2](
            intermediate, w2, ro.expert_weights, ro.expert_tokens, output,
            intermediate_size, hidden_dim, C, num_tokens,
            intermediate.stride(0), intermediate.stride(1), intermediate.stride(2),
            w2_strides[0], w2_strides[1], w2_strides[2],
            ro.expert_weights.stride(0), ro.expert_weights.stride(1),
            ro.expert_tokens.stride(0), ro.expert_tokens.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_C=BLOCK_C, BLOCK_K=64, BLOCK_N=64,
        )
        events["gemm2_scatter"][1].record()

        torch.cuda.synchronize()
        for s in stages:
            results[s].append(events[s][0].elapsed_time(events[s][1]))

    return results


# ─────────────────────── main ───────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark MoE Kernel: Token Choice vs Expert Choice vs Expert Choice Fused")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to LLaDA2 model (for real weights)")
    parser.add_argument("--random_weights", action="store_true",
                        help="Use random weights (no model loading needed)")
    parser.add_argument("--num_tokens", type=int, nargs="+", default=[256, 512, 1024, 2048],
                        help="Token counts to benchmark")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=200, help="Timed iterations")
    parser.add_argument("--breakdown", action="store_true",
                        help="Show per-stage breakdown comparison: TC vs EC vs EC-Fused")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda"
    torch.set_default_dtype(torch.bfloat16)

    # ---- resolve model path ----
    if args.model_path:
        model_path = args.model_path
    elif os.path.exists(DEFAULT_MODEL_PATH):
        model_path = DEFAULT_MODEL_PATH
    else:
        model_path = None

    # ---- init distributed (required by SGLang) ----
    server_args = init_distributed(tp_size=args.tp_size, model_path=model_path)

    # ---- load / create model ----
    from dinfer.model.modeling_llada2_moe_sglang import LLaDA2SGLangLM
    from transformers import AutoConfig

    if model_path and not args.random_weights:
        print(f"[Loading model from {model_path}]")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.routing_strategy = "token_choice"  # default, we'll test both
        from sglang.srt.layers.moe import initialize_moe_config
        initialize_moe_config(server_args)
        model = LLaDA2SGLangLM(config=config, expert_map_path=".").eval()
        model.load_weights(model_path, device=device)
        initialize_moe_config(server_args)
        model = model.to(device)
    else:
        print("[Using random weights]")
        from dinfer.model.configuration_llada2_moe import LLaDA2MoeConfig
        config = LLaDA2MoeConfig(
            vocab_size=157184,
            hidden_size=2048,
            intermediate_size=5120,
            moe_intermediate_size=512,
            num_hidden_layers=2,  # only need 1-2 layers for benchmark
            num_attention_heads=16,
            num_key_value_heads=4,
            num_experts=256,
            num_experts_per_tok=8,
            num_shared_experts=1,
            n_group=8,
            topk_group=4,
            first_k_dense_replace=1,
            routed_scaling_factor=2.5,
            norm_topk_prob=True,
            score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            routing_strategy="token_choice",
            head_dim=128,
        )
        from sglang.srt.layers.moe import initialize_moe_config
        initialize_moe_config(server_args)
        model = LLaDA2SGLangLM(config=config, expert_map_path=".").eval().to(device)

    moe_layer = get_moe_layer(model)
    print(f"\n[MoE Layer Config]")
    print(f"  num_experts={moe_layer.num_experts}, top_k={moe_layer.top_k}")
    print(f"  hidden_size={config.hidden_size}, moe_intermediate_size={config.moe_intermediate_size}")
    print(f"  w1 shape={moe_layer.experts.w13_weight.shape}")
    print(f"  w2 shape={moe_layer.experts.w2_weight.shape}")

    # ---- benchmark ----
    print(f"\n{'='*100}")
    print(f"  Benchmark: Token Choice vs Expert Choice (Original) vs Expert Choice (Fused Triton)")
    print(f"  warmup={args.warmup}, repeat={args.repeat}")
    print(f"{'='*100}")

    all_results = []

    for N in args.num_tokens:
        hidden_states = torch.randn(1, N, config.hidden_size, device=device, dtype=torch.bfloat16)
        capacity = max(1, (N * moe_layer.top_k) // moe_layer.num_experts)

        print(f"\n--- num_tokens={N}, capacity={capacity} ---")

        # Token Choice
        tc_times = bench_token_choice(moe_layer, hidden_states, warmup=args.warmup, repeat=args.repeat)
        tc_median = sorted(tc_times)[len(tc_times) // 2]
        tc_mean = sum(tc_times) / len(tc_times)
        tc_p10 = sorted(tc_times)[len(tc_times) // 10]
        tc_p90 = sorted(tc_times)[len(tc_times) * 9 // 10]

        # Expert Choice (original PyTorch routing)
        ec_times = bench_expert_choice(moe_layer, hidden_states, warmup=args.warmup, repeat=args.repeat)
        ec_median = sorted(ec_times)[len(ec_times) // 2]
        ec_mean = sum(ec_times) / len(ec_times)
        ec_p10 = sorted(ec_times)[len(ec_times) // 10]
        ec_p90 = sorted(ec_times)[len(ec_times) * 9 // 10]

        # Expert Choice Fused v1 (Triton routing + Triton scatter, 4 compute kernels)
        ecf_times = bench_expert_choice_fused(moe_layer, hidden_states, warmup=args.warmup, repeat=args.repeat)
        ecf_median = sorted(ecf_times)[len(ecf_times) // 2]
        ecf_mean = sum(ecf_times) / len(ecf_times)

        # Expert Choice Fused v2 (2 compute kernels: GEMM1+SiLU, GEMM2+Scatter)
        ecf2_times = bench_expert_choice_fused_v2(moe_layer, hidden_states, warmup=args.warmup, repeat=args.repeat)
        ecf2_median = sorted(ecf2_times)[len(ecf2_times) // 2]
        ecf2_mean = sum(ecf2_times) / len(ecf2_times)

        ec_tc = ec_median / tc_median if tc_median > 0 else float("inf")
        ecf_tc = ecf_median / tc_median if tc_median > 0 else float("inf")
        ecf2_tc = ecf2_median / tc_median if tc_median > 0 else float("inf")
        v2_vs_v1 = ecf_median / ecf2_median if ecf2_median > 0 else float("inf")
        v2_vs_ec = ec_median / ecf2_median if ecf2_median > 0 else float("inf")

        print(f"  Token  Choice:    median={tc_median:.3f}ms   mean={tc_mean:.3f}ms")
        print(f"  EC (original):    median={ec_median:.3f}ms   mean={ec_mean:.3f}ms   EC/TC={ec_tc:.2f}x")
        print(f"  EC-Fused v1:      median={ecf_median:.3f}ms   mean={ecf_mean:.3f}ms   v1/TC={ecf_tc:.2f}x")
        print(f"  EC-Fused v2:      median={ecf2_median:.3f}ms   mean={ecf2_mean:.3f}ms   v2/TC={ecf2_tc:.2f}x")
        print(f"  v2 speedup:       {v2_vs_v1:.2f}x over v1,  {v2_vs_ec:.2f}x over original EC")

        all_results.append({
            "N": N, "C": capacity,
            "tc_median": tc_median,
            "ec_median": ec_median,
            "ecf_median": ecf_median,
            "ecf2_median": ecf2_median,
            "ec_tc": ec_tc, "ecf_tc": ecf_tc, "ecf2_tc": ecf2_tc,
            "v2_vs_v1": v2_vs_v1,
        })

        # Breakdown: side-by-side comparison
        if args.breakdown:
            print(f"\n  [Per-Stage Breakdown] (N={N}, E={moe_layer.num_experts}, C={capacity})")

            tc_bd = bench_token_choice_breakdown(moe_layer, hidden_states, warmup=args.warmup, repeat=args.repeat)
            ec_bd = bench_expert_choice_breakdown(moe_layer, hidden_states, warmup=args.warmup, repeat=args.repeat)
            ecf2_bd = bench_expert_choice_fused_v2_breakdown(moe_layer, hidden_states, warmup=args.warmup, repeat=args.repeat)

            def get_median(times_list):
                s = sorted(times_list)
                return s[len(s) // 2]

            # Stages: TC has (gate, routing, experts); EC has (gate, routing, gemm1, activation, gemm2, scatter);
            # v2 has (gate, routing, gemm1_silu, gemm2_scatter)
            all_stages = ["gate", "routing", "gemm1", "activation", "gemm2", "scatter",
                          "gemm1_silu", "gemm2_scatter", "experts"]

            tc_medians = {s: get_median(tc_bd[s]) for s in tc_bd}
            ec_medians = {s: get_median(ec_bd[s]) for s in ec_bd}
            ecf2_medians = {s: get_median(ecf2_bd[s]) for s in ecf2_bd}

            tc_total = sum(tc_medians.values())
            ec_total = sum(ec_medians.values())
            ecf2_total = sum(ecf2_medians.values())

            print(f"    {'stage':>14s}  |  {'TC (SGLang)':>14s}  {'%':>5s}  |  {'EC (4 kern)':>14s}  {'%':>5s}  |  {'EC-v2 (2 kern)':>14s}  {'%':>5s}")
            print(f"    {'-'*14}  +  {'-'*14}  {'-'*5}  +  {'-'*14}  {'-'*5}  +  {'-'*14}  {'-'*5}")

            for stage in all_stages:
                tc_val = tc_medians.get(stage, None)
                ec_val = ec_medians.get(stage, None)
                ecf2_val = ecf2_medians.get(stage, None)

                # Skip stages that don't exist in any method
                if tc_val is None and ec_val is None and ecf2_val is None:
                    continue

                tc_str = f"{tc_val:.3f}ms" if tc_val is not None else "---"
                tc_pct = f"{tc_val/tc_total*100:.1f}%" if tc_val is not None else "---"
                ec_str = f"{ec_val:.3f}ms" if ec_val is not None else "---"
                ec_pct = f"{ec_val/ec_total*100:.1f}%" if ec_val is not None else "---"
                ecf2_str = f"{ecf2_val:.3f}ms" if ecf2_val is not None else "---"
                ecf2_pct = f"{ecf2_val/ecf2_total*100:.1f}%" if ecf2_val is not None else "---"

                print(f"    {stage:>14s}  |  {tc_str:>14s}  {tc_pct:>5s}  |  {ec_str:>14s}  {ec_pct:>5s}  |  {ecf2_str:>14s}  {ecf2_pct:>5s}")

            print(f"    {'-'*14}  +  {'-'*14}  {'-'*5}  +  {'-'*14}  {'-'*5}  +  {'-'*14}  {'-'*5}")
            print(f"    {'TOTAL':>14s}  |  {tc_total:>13.3f}ms  {'100%':>5s}  |  {ec_total:>13.3f}ms  {'100%':>5s}  |  {ecf2_total:>13.3f}ms  {'100%':>5s}")
            print(f"    {'vs TC':>14s}  |  {'1.00x':>14s}  {'':>5s}  |  {ec_total/tc_total:>13.2f}x  {'':>5s}  |  {ecf2_total/tc_total:>13.2f}x  {'':>5s}")

    # ---- summary ----
    print(f"\n{'='*110}")
    print(f"  SUMMARY")
    print(f"{'='*110}")
    print(f"  {'N':>6s}  {'C':>4s}  {'TC':>9s}  {'EC':>9s}  {'EC-v1':>9s}  {'EC-v2':>9s}  {'EC/TC':>7s}  {'v1/TC':>7s}  {'v2/TC':>7s}  {'v2/v1':>7s}")
    print(f"  {'-'*6}  {'-'*4}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
    for r in all_results:
        print(f"  {r['N']:>6d}  {r['C']:>4d}  {r['tc_median']:>8.3f}ms {r['ec_median']:>8.3f}ms {r['ecf_median']:>8.3f}ms {r['ecf2_median']:>8.3f}ms {r['ec_tc']:>6.2f}x {r['ecf_tc']:>6.2f}x {r['ecf2_tc']:>6.2f}x {r['v2_vs_v1']:>6.2f}x")
    print(f"{'='*110}")


if __name__ == "__main__":
    main()
