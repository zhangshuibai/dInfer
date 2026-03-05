import argparse
import csv
import json
import os
import sys
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

# Keep FlashInfer cache/log path writable in containerized environments.
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp")

def _patch_torch_compile_noop():
    # Work around some torch+sglang environments where importing modules with
    # @torch.compile(...) triggers inductor template duplicate registration.
    def _noop_compile(fn=None, *args, **kwargs):
        if fn is None:
            def _decorator(f):
                return f
            return _decorator
        return fn

    torch.compile = _noop_compile

_patch_torch_compile_noop()

# Ensure local package import works without `pip install -e .`.
REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from sglang.srt.server_args import ServerArgs
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8Config

from dinfer import BlockDiffusionLLM, BlockIteratorFactory, KVCacheFactory, ThresholdParallelDecoder
from dinfer.decoding.diffusion_runner import ModelRunner
from dinfer.model.modeling_llada2_moe_sglang import (
    LLaDA2SGLangLM,
    LLaDA2SparseMoeBlock,
)


def load_quant_config(config, model_path: str) -> bool:
    quant_config_path = os.path.join(model_path, "hf_quant_config.json")
    if not os.path.exists(quant_config_path):
        return False
    with open(quant_config_path, "r", encoding="utf-8") as f:
        quant_config_json = json.load(f)
    config.quant_config = ModelOptFp8Config.from_config(quant_config_json)
    return True


def extract_topk_ids_weights(topk_output) -> Tuple[torch.Tensor, torch.Tensor]:
    if hasattr(topk_output, "topk_ids") and hasattr(topk_output, "topk_weights"):
        return topk_output.topk_ids, topk_output.topk_weights
    if isinstance(topk_output, (tuple, list)) and len(topk_output) >= 2:
        topk_weights, topk_ids = topk_output[0], topk_output[1]
        return topk_ids, topk_weights
    raise TypeError(f"Unsupported topk output type: {type(topk_output)}")


@dataclass
class LayerStats:
    layer_id: int
    num_experts: int
    calls: int = 0
    topk_ms_sum: float = 0.0
    moe_ms_sum: float = 0.0
    total_assignments_sum: float = 0.0

    def __post_init__(self):
        self.expert_count_sum = torch.zeros(self.num_experts, dtype=torch.float64)
        self.expert_count_max = torch.zeros(self.num_experts, dtype=torch.int64)
        self.expert_nonzero_calls = torch.zeros(self.num_experts, dtype=torch.int64)
        self.expert_est_ms_sum = torch.zeros(self.num_experts, dtype=torch.float64)
        self.imbalance_ratios: List[float] = []


class TCExpertProfiler:
    def __init__(
        self,
        detailed_expert_timing: bool = False,
        detailed_warmup: int = 2,
        detailed_iters: int = 5,
        max_detailed_experts_per_layer: int = 0,
    ):
        self.layer_stats: Dict[int, LayerStats] = {}
        self.detailed_expert_timing = detailed_expert_timing
        self.detailed_warmup = detailed_warmup
        self.detailed_iters = detailed_iters
        self.max_detailed_experts_per_layer = max_detailed_experts_per_layer
        self.detailed_ms: Dict[Tuple[int, int], List[float]] = defaultdict(list)
        self._orig_forward_router = {}

    def _ensure_layer(self, layer_id: int, num_experts: int):
        if layer_id not in self.layer_stats:
            self.layer_stats[layer_id] = LayerStats(layer_id=layer_id, num_experts=num_experts)

    @staticmethod
    def _count_expert_assignments(topk_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
        valid = topk_ids >= 0
        if valid.any():
            return torch.bincount(topk_ids[valid].reshape(-1), minlength=num_experts)
        return torch.zeros(num_experts, device=topk_ids.device, dtype=torch.int64)

    @staticmethod
    def _run_single_expert_mlp(
        x: torch.Tensor,
        weights: torch.Tensor,
        w1_e: torch.Tensor,
        w2_e: torch.Tensor,
        use_triton_weight_format: bool,
    ) -> torch.Tensor:
        if x.shape[0] == 0:
            return x.new_zeros((0, w2_e.shape[-1] if use_triton_weight_format else w2_e.shape[0]))
        if use_triton_weight_format:
            gate_up = x @ w1_e
        else:
            gate_up = x @ w1_e.t()
        half = gate_up.shape[-1] // 2
        gate = gate_up[:, :half]
        up = gate_up[:, half:]
        hidden = F.silu(gate) * up
        if use_triton_weight_format:
            out = hidden @ w2_e
        else:
            out = hidden @ w2_e.t()
        return out * weights.unsqueeze(-1).to(out.dtype)

    def _profile_detailed_experts(
        self,
        layer_id: int,
        module: LLaDA2SparseMoeBlock,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        counts: torch.Tensor,
    ):
        if not self.detailed_expert_timing:
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
        if counts.numel() == 0 or counts.max().item() == 0:
            return

        w1 = module.experts.w13_weight
        w2 = module.experts.w2_weight
        hidden_dim = hidden_states.shape[-1]
        use_triton_weight_format = (w1.shape[1] == hidden_dim)

        experts = torch.nonzero(counts > 0, as_tuple=False).flatten().tolist()
        if self.max_detailed_experts_per_layer > 0:
            sorted_experts = sorted(experts, key=lambda e: int(counts[e]), reverse=True)
            experts = sorted_experts[: self.max_detailed_experts_per_layer]

        for e in experts:
            pairs = (topk_ids == e).nonzero(as_tuple=False)
            if pairs.numel() == 0:
                continue
            token_idx = pairs[:, 0]
            k_idx = pairs[:, 1]
            x = hidden_states[token_idx]
            ew = topk_weights[token_idx, k_idx]

            for _ in range(self.detailed_warmup):
                _ = self._run_single_expert_mlp(x, ew, w1[e], w2[e], use_triton_weight_format)

            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(self.detailed_iters):
                _ = self._run_single_expert_mlp(x, ew, w1[e], w2[e], use_triton_weight_format)
            end.record()
            torch.cuda.synchronize()
            avg_ms = start.elapsed_time(end) / float(self.detailed_iters)
            self.detailed_ms[(layer_id, e)].append(avg_ms)

    def record(
        self,
        layer_id: int,
        num_experts: int,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        hidden_states: torch.Tensor,
        topk_ms: float,
        moe_ms: float,
        module: LLaDA2SparseMoeBlock,
    ):
        self._ensure_layer(layer_id, num_experts)
        layer = self.layer_stats[layer_id]
        layer.calls += 1
        layer.topk_ms_sum += topk_ms
        layer.moe_ms_sum += moe_ms

        counts = self._count_expert_assignments(topk_ids, num_experts)
        counts_cpu = counts.detach().cpu()
        layer.expert_count_sum += counts_cpu.to(torch.float64)
        layer.expert_count_max = torch.maximum(layer.expert_count_max, counts_cpu.to(torch.int64))
        layer.expert_nonzero_calls += (counts_cpu > 0).to(torch.int64)

        total = float(counts_cpu.sum().item())
        layer.total_assignments_sum += total
        if total > 0.0:
            layer.expert_est_ms_sum += (counts_cpu.to(torch.float64) / total) * moe_ms
            mean_load = total / float(num_experts)
            max_load = float(counts_cpu.max().item())
            layer.imbalance_ratios.append(max_load / max(mean_load, 1e-9))
        else:
            layer.imbalance_ratios.append(0.0)

        self._profile_detailed_experts(
            layer_id=layer_id,
            module=module,
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            counts=counts,
        )

    def install_hooks(self, model: torch.nn.Module):
        for module in model.modules():
            if not isinstance(module, LLaDA2SparseMoeBlock):
                continue
            self._orig_forward_router[module] = module._forward_router_experts

            def wrapped_forward_router_experts(this, hidden_states: torch.Tensor):
                if this.routing_strategy == "expert_choice":
                    return self._orig_forward_router[this](hidden_states)

                is_capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
                router_logits = this.gate(hidden_states)

                if is_capturing:
                    topk_output = this.topk(hidden_states, router_logits)
                    topk_ms = 0.0
                else:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    topk_output = this.topk(hidden_states, router_logits)
                    end.record()
                    torch.cuda.synchronize()
                    topk_ms = start.elapsed_time(end)

                topk_ids, topk_weights = extract_topk_ids_weights(topk_output)

                if is_capturing:
                    out = this.experts(hidden_states, topk_output)
                    moe_ms = 0.0
                else:
                    start2 = torch.cuda.Event(enable_timing=True)
                    end2 = torch.cuda.Event(enable_timing=True)
                    start2.record()
                    out = this.experts(hidden_states, topk_output)
                    end2.record()
                    torch.cuda.synchronize()
                    moe_ms = start2.elapsed_time(end2)

                self.record(
                    layer_id=this.layer_id,
                    num_experts=this.num_experts,
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                    hidden_states=hidden_states,
                    topk_ms=topk_ms,
                    moe_ms=moe_ms,
                    module=this,
                )
                return out

            module._forward_router_experts = wrapped_forward_router_experts.__get__(
                module, module.__class__
            )

    def uninstall_hooks(self):
        for module, orig in self._orig_forward_router.items():
            module._forward_router_experts = orig
        self._orig_forward_router.clear()

    def dump(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        layer_csv = os.path.join(out_dir, "layer_summary.csv")
        expert_csv = os.path.join(out_dir, "expert_summary.csv")
        detailed_csv = os.path.join(out_dir, "expert_detailed_timing.csv")
        layer_internal_csv = os.path.join(out_dir, "layer_internal_straggler.csv")

        with open(layer_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "layer_id",
                    "calls",
                    "avg_topk_ms",
                    "avg_moe_ms",
                    "avg_total_assignments",
                    "avg_imbalance_ratio_max_over_mean",
                    "max_loaded_expert_id",
                    "max_loaded_expert_avg_assignments",
                ]
            )
            for layer_id in sorted(self.layer_stats):
                s = self.layer_stats[layer_id]
                avg_counts = s.expert_count_sum / max(s.calls, 1)
                max_e = int(torch.argmax(avg_counts).item())
                writer.writerow(
                    [
                        layer_id,
                        s.calls,
                        s.topk_ms_sum / max(s.calls, 1),
                        s.moe_ms_sum / max(s.calls, 1),
                        s.total_assignments_sum / max(s.calls, 1),
                        statistics.mean(s.imbalance_ratios) if s.imbalance_ratios else 0.0,
                        max_e,
                        float(avg_counts[max_e].item()),
                    ]
                )

        with open(expert_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "layer_id",
                    "expert_id",
                    "avg_assignments",
                    "max_assignments",
                    "nonzero_call_ratio",
                    "est_avg_moe_ms_share",
                    "est_tokens_per_ms",
                ]
            )
            for layer_id in sorted(self.layer_stats):
                s = self.layer_stats[layer_id]
                calls = max(s.calls, 1)
                avg_counts = s.expert_count_sum / calls
                for e in range(s.num_experts):
                    avg_assign = float(avg_counts[e].item())
                    est_ms = float((s.expert_est_ms_sum[e] / calls).item())
                    tpm = avg_assign / max(est_ms, 1e-9) if est_ms > 0.0 else 0.0
                    writer.writerow(
                        [
                            layer_id,
                            e,
                            avg_assign,
                            int(s.expert_count_max[e].item()),
                            float((s.expert_nonzero_calls[e].item()) / calls),
                            est_ms,
                            tpm,
                        ]
                    )

        # Per-layer internal comparison (within same layer only).
        # This is the key table to detect if one expert drags a layer.
        with open(layer_internal_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "layer_id",
                    "active_experts",
                    "slowest_expert_id",
                    "slowest_est_ms",
                    "fastest_expert_id",
                    "fastest_est_ms",
                    "mean_est_ms",
                    "p50_est_ms",
                    "p95_est_ms",
                    "slow_over_mean",
                    "slow_over_p50",
                    "slow_over_fast",
                ]
            )
            for layer_id in sorted(self.layer_stats):
                s = self.layer_stats[layer_id]
                calls = max(s.calls, 1)
                avg_counts = s.expert_count_sum / calls
                ests = []
                for e in range(s.num_experts):
                    if avg_counts[e].item() <= 0:
                        continue
                    est_ms = float((s.expert_est_ms_sum[e] / calls).item())
                    ests.append((e, est_ms))
                if not ests:
                    continue
                ests_sorted = sorted(ests, key=lambda x: x[1])
                vals = [x[1] for x in ests_sorted]
                mean_v = sum(vals) / len(vals)
                p50 = vals[len(vals) // 2]
                p95 = vals[min(len(vals) - 1, int(0.95 * len(vals)))]
                fast_e, fast_v = ests_sorted[0]
                slow_e, slow_v = ests_sorted[-1]
                writer.writerow(
                    [
                        layer_id,
                        len(vals),
                        slow_e,
                        slow_v,
                        fast_e,
                        fast_v,
                        mean_v,
                        p50,
                        p95,
                        slow_v / max(mean_v, 1e-12),
                        slow_v / max(p50, 1e-12),
                        slow_v / max(fast_v, 1e-12),
                    ]
                )

        if self.detailed_expert_timing and self.detailed_ms:
            with open(detailed_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "layer_id",
                        "expert_id",
                        "samples",
                        "avg_expert_compute_ms",
                        "p50_expert_compute_ms",
                        "p95_expert_compute_ms",
                    ]
                )
                for key in sorted(self.detailed_ms):
                    layer_id, expert_id = key
                    samples = self.detailed_ms[key]
                    samples_sorted = sorted(samples)
                    p50 = samples_sorted[len(samples_sorted) // 2]
                    p95 = samples_sorted[min(len(samples_sorted) - 1, int(0.95 * len(samples_sorted)))]
                    writer.writerow(
                        [
                            layer_id,
                            expert_id,
                            len(samples),
                            statistics.mean(samples),
                            p50,
                            p95,
                        ]
                    )

        print(f"[Profile] Wrote: {layer_csv}")
        print(f"[Profile] Wrote: {expert_csv}")
        print(f"[Profile] Wrote: {layer_internal_csv}")
        if self.detailed_expert_timing:
            print(f"[Profile] Wrote: {detailed_csv}")

    def print_top_stragglers(self, topn: int = 10):
        candidates = []
        for layer_id, s in self.layer_stats.items():
            calls = max(s.calls, 1)
            avg_counts = s.expert_count_sum / calls
            for e in range(s.num_experts):
                avg_assign = float(avg_counts[e].item())
                est_ms = float((s.expert_est_ms_sum[e] / calls).item())
                candidates.append((est_ms, avg_assign, layer_id, e))
        candidates.sort(reverse=True, key=lambda x: x[0])
        print("\n[Top Stragglers by Estimated MoE Time Share]")
        print("rank  layer  expert  est_ms  avg_assignments")
        for i, (est_ms, avg_assign, layer_id, expert_id) in enumerate(candidates[:topn], start=1):
            print(f"{i:>4}  {layer_id:>5}  {expert_id:>6}  {est_ms:>6.3f}  {avg_assign:>15.3f}")

    def print_layer_internal_stragglers(self, topn_layers: int = 10, min_avg_assignments: float = 1.0):
        rows = []
        for layer_id, s in self.layer_stats.items():
            calls = max(s.calls, 1)
            avg_counts = s.expert_count_sum / calls
            ests = []
            for e in range(s.num_experts):
                avg_assign = float(avg_counts[e].item())
                if avg_assign < min_avg_assignments:
                    continue
                est_ms = float((s.expert_est_ms_sum[e] / calls).item())
                ests.append((e, est_ms, avg_assign))
            if len(ests) < 2:
                continue
            vals = sorted(x[1] for x in ests)
            mean_v = sum(vals) / len(vals)
            p50 = vals[len(vals) // 2]
            slow_e, slow_v, slow_assign = max(ests, key=lambda x: x[1])
            fast_e, fast_v, fast_assign = min(ests, key=lambda x: x[1])
            rows.append(
                (
                    slow_v / max(mean_v, 1e-12),
                    layer_id,
                    slow_e,
                    slow_v,
                    slow_assign,
                    fast_e,
                    fast_v,
                    fast_assign,
                    mean_v,
                    p50,
                )
            )

        rows.sort(reverse=True, key=lambda x: x[0])
        print("\n[Layer-Internal Stragglers]")
        print("rank  layer  slow_e  slow_ms  slow_avg_assign  fast_e  fast_ms  mean_ms  p50_ms  slow/mean")
        for i, r in enumerate(rows[:topn_layers], start=1):
            _, layer_id, slow_e, slow_v, slow_assign, fast_e, fast_v, _, mean_v, p50 = r
            print(
                f"{i:>4}  {layer_id:>5}  {slow_e:>6}  {slow_v:>7.4f}  {slow_assign:>15.2f}  "
                f"{fast_e:>6}  {fast_v:>7.4f}  {mean_v:>7.4f}  {p50:>7.4f}  {slow_v/max(mean_v,1e-12):>9.2f}"
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Profile TC MoE per-layer/per-expert load and latency.")
    parser.add_argument("--model_name", type=str, required=True, help="Local model directory.")
    parser.add_argument("--mode", type=str, choices=["real", "synthetic"], default="real")
    parser.add_argument("--dataset", type=str, default=None, help="JSON dataset for real inference mode.")
    parser.add_argument("--gen_len", type=int, default=512, help="Target generation length in real mode.")
    parser.add_argument("--block_length", type=int, default=32, help="Block length for block diffusion generation.")
    parser.add_argument("--max_samples", type=int, default=0, help="0 means use all samples in dataset.")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--cache", type=str, default="prefix", choices=["prefix", "dual", "none"])
    parser.add_argument("--prefilling_limit", type=int, default=256)
    parser.add_argument("--mini_batch_size", type=int, default=1)
    parser.add_argument("--use_naive_batching", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--profile_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out_dir", type=str, default="runs/tc_expert_profile")
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--ep_size", type=int, default=1)
    parser.add_argument("--port", type=str, default="29550")
    parser.add_argument("--detailed_expert_timing", action="store_true")
    parser.add_argument("--detailed_warmup", type=int, default=2)
    parser.add_argument("--detailed_iters", type=int, default=5)
    parser.add_argument(
        "--max_detailed_experts_per_layer",
        type=int,
        default=0,
        help="0 means profile all active experts for detailed timing.",
    )
    return parser.parse_args()


def setup_dist(tp_size: int, ep_size: int, port: str):
    from sglang.srt import distributed

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    world_size = tp_size * ep_size
    rank = 0
    distributed.init_distributed_environment(world_size, rank, "env://", rank, "nccl")
    distributed.initialize_model_parallel(tp_size, ep_size, 1, backend="nccl")


def validate_model_name_or_path(model_name: str):
    if model_name.strip() == "/path/to/your/llada2-model":
        raise ValueError(
            "You are using the placeholder path '/path/to/your/llada2-model'. "
            "Please replace --model_name with your real local model directory."
        )
    if os.path.isabs(model_name) and not os.path.isdir(model_name):
        raise ValueError(
            f"--model_name points to a non-existent directory: {model_name}"
        )


def load_real_inputs(dataset_path: str, tokenizer, max_samples: int = 0):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "judge_details" in data:
        details = data["judge_details"]
    elif "details" in data:
        details = data["details"]
    elif isinstance(data, list):
        details = data
    else:
        raise ValueError(f"Unsupported dataset format in {dataset_path}")

    prompts = []
    tokenized = []
    for i, item in enumerate(details):
        if max_samples > 0 and i >= max_samples:
            break
        prompt = item.get("prompt", item.get("question", ""))
        if not isinstance(prompt, str) or prompt == "":
            continue
        # Keep same format as benchmark_dataset_sglang for consistency.
        prompt_fmt = (
            "<role>SYSTEM</role>detailed thinking off<|role_end|>"
            "<role>HUMAN</role>" + prompt + "<|role_end|><role>ASSISTANT</role>"
        )
        input_ids = tokenizer(prompt_fmt)["input_ids"]
        tokenized.append(torch.tensor(input_ids, dtype=torch.long).unsqueeze(0))
        prompts.append(prompt_fmt)

    if not tokenized:
        raise ValueError(f"No valid prompts found in dataset: {dataset_path}")
    return tokenized, prompts


def _get_bucket_length(length: int, bucket_size: int = 32) -> int:
    return bucket_size * (length // bucket_size)


def _calc_padded_gen_lens(input_ids_list: List[torch.Tensor], gen_len: int) -> List[int]:
    padded = []
    for ids in input_ids_list:
        total = ids.shape[1] + gen_len
        padded_len = _get_bucket_length(total)
        padded.append(max(1, padded_len - ids.shape[1]))
    return padded


@torch.no_grad()
def run_real_inference_profile(args, model, config, device, server_args, profiler: "TCExpertProfiler"):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, local_files_only=True)
    all_input_ids, _ = load_real_inputs(args.dataset, tokenizer, args.max_samples)
    padded_gen_lens = _calc_padded_gen_lens(all_input_ids, args.gen_len)

    input_lengths = [inp.size(-1) for inp in all_input_ids]
    max_length = max(input_lengths) + args.gen_len
    aligned_lengths = np.unique(
        [
            max(
                args.block_length,
                min((length // args.block_length) * args.block_length, args.prefilling_limit),
            )
            for length in input_lengths
        ]
    )
    aligned_lengths = [int(x) for x in aligned_lengths]
    supported_batch_sizes = [2**i for i in range(int(np.log2(max(1, args.mini_batch_size))) + 1)]

    runner = ModelRunner(
        model,
        device,
        server_args=server_args,
        max_length=max_length,
        block_length=args.block_length,
        prefill_lengths=aligned_lengths,
        enable_cuda_graph=True,
        supported_batch_sizes=supported_batch_sizes,
        use_cross_block=(args.batch_size == 1),
    )

    # Install hooks only after ModelRunner completes cuda-graph capture.
    profiler.install_hooks(model)

    decoder = ThresholdParallelDecoder(
        temperature=0.0,
        threshold=args.threshold,
        mask_id=156895,
        eos_id=156892,
    )

    if args.cache in ("prefix", "dual"):
        cache_factory = KVCacheFactory(
            args.cache, is_bd_model=True, backend="sglang", max_length=max_length
        )
    else:
        cache_factory = None

    dllm = BlockDiffusionLLM(
        runner,
        decoder,
        BlockIteratorFactory(start_block_align=True, use_block_diffusion=True),
        cache_factory=cache_factory,
        early_stop=True,
        maximum_unroll=1,
        expected_tpf=15,
        backend="sglang",
        mini_batch_size=args.mini_batch_size,
        prefilling_limit=args.prefilling_limit,
        use_naive_batching=args.use_naive_batching,
    )

    # Warmup decode path
    warmup_ids = torch.randint(0, 100000, (args.mini_batch_size, 64), dtype=torch.long, device=device)
    dllm.generate(warmup_ids, gen_length=max(64, args.block_length), block_length=args.block_length)

    sorted_indices = sorted(range(len(input_lengths)), key=lambda i: input_lengths[i])
    sorted_input_ids = [all_input_ids[i] for i in sorted_indices]
    sorted_padded_gen_lens = [padded_gen_lens[i] for i in sorted_indices]

    iterator = range(0, len(sorted_input_ids), args.batch_size)
    step_times_ms = []
    for i in iterator:
        batch = sorted_input_ids[i:i + args.batch_size]
        max_prompt_len = batch[-1].shape[1]
        gen_len = sorted_padded_gen_lens[i + len(batch) - 1]

        batch_input_ids = torch.full(
            (len(batch), max_prompt_len),
            156895,
            dtype=torch.long,
            device=device,
        )
        for j, ids in enumerate(batch):
            batch_input_ids[j, :ids.shape[1]] = ids.to(device)

        torch.cuda.synchronize()
        t0 = time.time()
        _ = dllm.generate(batch_input_ids, gen_length=gen_len, block_length=args.block_length)
        torch.cuda.synchronize()
        t1 = time.time()
        step_times_ms.append((t1 - t0) * 1000.0)

    return step_times_ms


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")
    if args.tp_size != 1 or args.ep_size != 1:
        raise ValueError("This profiler currently supports tp_size=1 and ep_size=1 only.")
    validate_model_name_or_path(args.model_name)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    setup_dist(args.tp_size, args.ep_size, args.port)

    try:
        local_only = os.path.isdir(args.model_name)
        config = AutoConfig.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            local_files_only=local_only,
        )
        config.routing_strategy = "token_choice"
        used_quant = load_quant_config(config, args.model_name)

        if used_quant:
            server_args = ServerArgs(
                model_path=args.model_name,
                quantization="modelopt_fp8",
                modelopt_quant="fp8",
                enable_dp_attention=True,
                trust_remote_code=True,
                tp_size=args.tp_size,
                dp_size=1,
                pp_size=1,
            )
        else:
            server_args = ServerArgs(
                model_path=args.model_name,
                enable_dp_attention=True,
                trust_remote_code=True,
                tp_size=args.tp_size,
                dp_size=1,
                pp_size=1,
            )

        try:
            from sglang.srt.server_args import set_global_server_args_for_scheduler

            set_global_server_args_for_scheduler(server_args)
        except ImportError:
            pass

        initialize_dp_attention(server_args=server_args, model_config=config)
        initialize_moe_config(server_args)

        if used_quant:
            model = LLaDA2SGLangLM(
                config=config,
                quant_config=config.quant_config,
                expert_map_path=".",
            ).eval()
        else:
            model = LLaDA2SGLangLM(config=config, expert_map_path=".").eval()

        torch.set_default_dtype(torch.bfloat16)
        model.load_weights(args.model_name, device=device)
        initialize_moe_config(server_args)
        model = model.to(device).eval()
        model.after_processing()

        profiler = TCExpertProfiler(
            detailed_expert_timing=args.detailed_expert_timing,
            detailed_warmup=args.detailed_warmup,
            detailed_iters=args.detailed_iters,
            max_detailed_experts_per_layer=args.max_detailed_experts_per_layer,
        )

        if args.mode == "real":
            if not args.dataset:
                raise ValueError("--dataset is required when --mode real")
            print(
                f"[Profile] TC real inference mode, batch={args.batch_size}, gen_len={args.gen_len}, "
                f"block_length={args.block_length}, dataset={args.dataset}"
            )
            step_times = run_real_inference_profile(args, model, config, device, server_args, profiler)
        else:
            profiler.install_hooks(model)
            print(
                f"[Profile] TC synthetic mode, batch={args.batch_size}, seq_len={args.seq_len}, "
                f"warmup={args.warmup_steps}, profile_steps={args.profile_steps}"
            )
            vocab_size = int(getattr(config, "vocab_size", 32000))
            step_times = []
            for step in range(args.warmup_steps + args.profile_steps):
                input_ids = torch.randint(
                    low=0,
                    high=max(vocab_size - 1, 1),
                    size=(args.batch_size, args.seq_len),
                    device=device,
                    dtype=torch.long,
                )
                position_ids = torch.arange(args.seq_len, device=device, dtype=torch.long).unsqueeze(0).repeat(args.batch_size, 1)

                torch.cuda.synchronize()
                t0 = time.time()
                _ = model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    use_cache=False,
                )
                torch.cuda.synchronize()
                t1 = time.time()

                if step >= args.warmup_steps:
                    step_times.append((t1 - t0) * 1000.0)

        profiler.uninstall_hooks()
        profiler.dump(args.out_dir)
        profiler.print_top_stragglers(topn=10)
        profiler.print_layer_internal_stragglers(topn_layers=10, min_avg_assignments=5.0)

        avg_step_ms = sum(step_times) / max(len(step_times), 1)
        print(f"\n[Profile] Average end-to-end forward time: {avg_step_ms:.3f} ms")
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
