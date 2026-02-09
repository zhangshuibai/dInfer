# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Callable,
    NamedTuple,
    Optional,
    Protocol,
    TypeGuard,
    runtime_checkable,
)

import torch
import torch.nn.functional as F

from sglang.srt.custom_op import CustomOp
from sglang.srt.eplb import expert_location_dispatch
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location_dispatch import (
    ExpertLocationDispatchInfo,
    topk_ids_logical_to_physical,
)
from sglang.srt.layers.moe import (
    get_moe_runner_backend,
    should_use_flashinfer_trtllm_moe,
)
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    get_compiler_backend,
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
)

if TYPE_CHECKING:
    from sglang.srt.layers.quantization import QuantizationConfig

try:
    from triton_kernels.routing import GatherIndx, RoutingData, ScatterIndx, routing
except ImportError:
    pass
logger = logging.getLogger(__name__)


_is_cuda = is_cuda()
_is_hip = is_hip()
_is_cpu = is_cpu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _is_cuda:
    from sgl_kernel import moe_fused_gate

if _is_cuda or _is_hip:
    from sgl_kernel import topk_softmax
if _use_aiter:
    try:
        from aiter import biased_grouped_topk as aiter_biased_grouped_topk
    except ImportError:
        raise ImportError("aiter is required when SGLANG_USE_AITER is set to True")
if _is_npu:
    import torch_npu

# -------------------------------- TopKConfig ---------------------------------------


@dataclass
class TopKConfig:
    top_k: int
    use_grouped_topk: bool = False
    topk_group: Optional[int] = None
    num_expert_group: Optional[int] = None
    renormalize: bool = True
    num_fused_shared_experts: int = 0
    custom_routing_function: Optional[Callable] = None
    correction_bias: Optional[torch.Tensor] = None
    torch_native: bool = False
    routed_scaling_factor: Optional[float] = None
    apply_routed_scaling_factor_on_output: bool = False
    output_format: Optional[TopKOutputFormat] = None


# -------------------------------- TopKOutput ---------------------------------------


class TopKOutputChecker:

    @staticmethod
    def format_is_standard(topk_output: TopKOutput) -> TypeGuard[StandardTopKOutput]:
        return topk_output.format.is_standard()

    @staticmethod
    def format_is_triton_kernel(
        topk_output: TopKOutput,
    ) -> TypeGuard[TritonKernelTopKOutput]:
        return topk_output.format.is_triton_kernel()

    @staticmethod
    def format_is_bypassed(topk_output: TopKOutput) -> TypeGuard[BypassedTopKOutput]:
        return topk_output.format.is_bypassed()


class TopKOutputFormat(Enum):
    STANDARD = auto()
    TRITON_KERNEL = auto()
    BYPASSED = auto()

    def is_standard(self) -> bool:
        return self == TopKOutputFormat.STANDARD

    def is_triton_kernel(self) -> bool:
        return self == TopKOutputFormat.TRITON_KERNEL

    def is_bypassed(self) -> bool:
        return self == TopKOutputFormat.BYPASSED


@runtime_checkable
class TopKOutput(Protocol):
    """Protocol for top-k outputs in different formats."""

    @property
    def format(self) -> TopKOutputFormat:
        """The format of the output."""
        ...


class StandardTopKOutput(NamedTuple):
    """Standard top-k output format."""

    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor

    @property
    def format(self) -> TopKOutputFormat:
        return TopKOutputFormat.STANDARD


class TritonKernelTopKOutput(NamedTuple):
    """Triton kernel top-k output format."""

    routing_data: RoutingData
    gather_indx: GatherIndx
    scatter_indx: ScatterIndx

    @property
    def format(self) -> TopKOutputFormat:
        return TopKOutputFormat.TRITON_KERNEL


class BypassedTopKOutput(NamedTuple):
    """Bypassed top-k output format."""

    hidden_states: torch.Tensor
    router_logits: torch.Tensor
    topk_config: TopKConfig
    num_token_non_padded: Optional[torch.Tensor] = None
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None

    @property
    def format(self) -> TopKOutputFormat:
        return TopKOutputFormat.BYPASSED


# -------------------------------- TopK ---------------------------------------


class TopK(CustomOp):

    def __init__(
        self,
        top_k: int,
        *,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        renormalize: bool = True,
        num_fused_shared_experts: int = 0,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        correction_bias: Optional[torch.Tensor] = None,
        quant_config: Optional[QuantizationConfig] = None,
        routed_scaling_factor: Optional[float] = None,
        apply_routed_scaling_factor_on_output: Optional[bool] = False,
        output_format: Optional[TopKOutputFormat] = None,
    ):
        # NOTE: scoring_func is not used for now, but we keep it for future use
        # see https://github.com/sgl-project/sglang/pull/4505 for more details
        super().__init__()

        if use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None

        self.topk_config = TopKConfig(
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            num_fused_shared_experts=num_fused_shared_experts,
            custom_routing_function=custom_routing_function,
            correction_bias=correction_bias,
            routed_scaling_factor=routed_scaling_factor,
            apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            output_format=output_format,
        )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        num_token_non_padded: Optional[torch.Tensor] = None,
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    ) -> TopKOutput:
        self.topk_config.torch_native = True
        return select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk_config=self.topk_config,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        num_token_non_padded: Optional[torch.Tensor] = None,
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    ) -> TopKOutput:
        if self.topk_config.output_format is not None:
            output_format = self.topk_config.output_format
        elif get_moe_runner_backend().is_triton_kernel():
            output_format = TopKOutputFormat.TRITON_KERNEL
        elif (
            should_use_flashinfer_trtllm_moe()
            or get_moe_runner_backend().is_flashinfer_mxfp4()
        ):
            output_format = TopKOutputFormat.BYPASSED
        else:
            output_format = TopKOutputFormat.STANDARD

        if output_format == TopKOutputFormat.TRITON_KERNEL:
            # renormalize=True is equivalent to sm_first=False
            routing_data, gather_idx, scatter_idx = routing(
                router_logits,
                self.topk_config.top_k,
                sm_first=not self.topk_config.renormalize,
            )
            return TritonKernelTopKOutput(routing_data, gather_idx, scatter_idx)
        elif output_format == TopKOutputFormat.BYPASSED:
            return BypassedTopKOutput(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_config=self.topk_config,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
            )
        else:
            self.topk_config.torch_native = False
            return select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_config=self.topk_config,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
            )

    def forward_cpu(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        num_token_non_padded: Optional[torch.Tensor] = None,
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    ) -> TopKOutput:
        return select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk_config=self.topk_config,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )

    def forward_npu(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        num_token_non_padded: Optional[torch.Tensor] = None,
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    ) -> TopKOutput:
        global_num_experts = router_logits.shape[-1]

        # NOTE: now npu_moe_gating_top_k can only support `group_count=256` pattern
        if global_num_experts == 256:

            routed_scaling_factor = self.topk_config.routed_scaling_factor or 1
            router_logits = router_logits.to(torch.float32)

            topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
                router_logits,
                k=self.topk_config.top_k,
                bias=self.topk_config.correction_bias.to(torch.float32),
                k_group=self.topk_config.topk_group,
                group_count=self.topk_config.num_expert_group,
                group_select_mode=1,
                renorm=0,
                norm_type=1,
                routed_scaling_factor=routed_scaling_factor,
                eps=float(1e-20),
            )

            if self.topk_config.renormalize:
                topk_weights_sum = (
                    topk_weights.sum(dim=-1, keepdim=True)
                    if self.topk_config.num_fused_shared_experts == 0
                    else topk_weights[:, :-1].sum(dim=-1, keepdim=True)
                )
                topk_weights = topk_weights / topk_weights_sum

            if expert_location_dispatch_info is not None:
                topk_ids = topk_ids_logical_to_physical(
                    topk_ids, expert_location_dispatch_info
                )
            get_global_expert_distribution_recorder().on_select_experts(
                topk_ids=topk_ids
            )

            return StandardTopKOutput(topk_weights, topk_ids, _)
        else:
            self.topk_config.torch_native = True
            return select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_config=self.topk_config,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
            )

    def empty_topk_output(self, device: torch.device) -> TopKOutput:
        topk = self.topk_config.top_k - self.topk_config.num_fused_shared_experts
        topk_weights = torch.empty((0, topk), dtype=torch.float32, device=device)
        topk_idx = torch.full((0, topk), -1, dtype=torch.int32, device=device)
        router_logits = torch.empty((0, topk), dtype=torch.float32, device=device)
        return StandardTopKOutput(topk_weights, topk_idx, router_logits)


# ------------------------------- TopK implementation -------------------------------------


def fused_topk_torch_native(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: torch.Tensor = None,
):
    if correction_bias is not None:
        n_routed_experts = gating_output.shape[-1]
        scores = gating_output.softmax(dim=-1)
        scores_for_choice = scores.view(
            -1, n_routed_experts
        ) + correction_bias.unsqueeze(0)
        topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_ids)
    else:
        assert (
            hidden_states.shape[0] == gating_output.shape[0]
        ), f"Number of tokens mismatch, {hidden_states.shape=} vs {gating_output.shape=}"
        M, _ = hidden_states.shape
        topk_weights = torch.empty(
            M, topk, dtype=torch.float32, device=hidden_states.device
        )
        topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
        topk_weights = F.softmax(gating_output.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(topk_weights, topk, dim=-1)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


def fused_topk_cpu(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    correction_bias: torch.Tensor = None,
):
    topk_weights, topk_ids = torch.ops.sgl_kernel.topk_softmax_cpu(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
    )
    topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
    return topk_weights, topk_ids


def apply_topk_weights_cpu(need_apply, topk_weights, inputs):
    if not need_apply:
        return inputs, topk_weights

    # TODO: fuse below processing in fused_experts_cpu kernel
    inputs = inputs * topk_weights.to(inputs.dtype)
    topk_weights = torch.ones_like(
        topk_weights, dtype=torch.float32
    )  # clear topk_weights as already applied

    return inputs, topk_weights


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    M, _ = hidden_states.shape

    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)

    topk_softmax(
        topk_weights,
        topk_ids,
        gating_output,
        renormalize,
    )

    topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
    return topk_weights, topk_ids


# This is used by the Deepseek V2/V3/R1 series models
@torch.compile(dynamic=True, backend=get_compiler_backend())
def grouped_topk_gpu(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = torch.softmax(gating_output, dim=-1)
    # NPU compiler limitation
    if _is_npu and scores.dtype == torch.bfloat16:
        scores = scores.to(torch.float16)
    num_token = scores.shape[0]
    num_experts = scores.shape[1]
    group_scores = (
        scores.view(num_token, num_expert_group, -1).max(dim=-1).values
    )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
    # TODO: NPU can't support directly evaluating a comparison for now
    topk_weights, topk_ids = torch.topk(
        tmp_scores,
        k=topk,
        dim=-1,
        sorted=(True if num_fused_shared_experts > 0 else False),
    )
    if num_fused_shared_experts:
        topk_ids[:, -1] = torch.randint(
            low=num_experts,
            high=num_experts + num_fused_shared_experts,
            size=(topk_ids.size(0),),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        topk_weights[:, -1] = topk_weights[:, :-1].sum(dim=-1) / routed_scaling_factor

    if renormalize:
        topk_weights_sum = (
            topk_weights.sum(dim=-1, keepdim=True)
            if num_fused_shared_experts == 0
            else topk_weights[:, :-1].sum(dim=-1, keepdim=True)
        )
        topk_weights = topk_weights / topk_weights_sum
        if apply_routed_scaling_factor_on_output:
            topk_weights *= routed_scaling_factor

    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)
    topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
    return topk_weights, topk_ids


def grouped_topk_cpu(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    assert not apply_routed_scaling_factor_on_output
    assert expert_location_dispatch_info is None
    return torch.ops.sgl_kernel.grouped_topk_cpu(
        hidden_states,
        gating_output,
        topk,
        renormalize,
        num_expert_group,
        topk_group,
        num_fused_shared_experts,
        routed_scaling_factor,
        num_token_non_padded,
    )


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def biased_grouped_topk_impl(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = gating_output.sigmoid()
    num_token = scores.shape[0]
    num_experts = scores.shape[1]
    scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)
    group_scores = (
        scores_for_choice.view(num_token, num_expert_group, -1)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores_for_choice.masked_fill(
        ~score_mask.bool(), float("-inf")
    )  # [n, e]
    # TODO: NPU can't support directly evaluating a comparison for now
    _, topk_ids = torch.topk(
        tmp_scores,
        k=topk,
        dim=-1,
        sorted=(True if num_fused_shared_experts > 0 else False),
    )
    topk_weights = scores.gather(1, topk_ids)

    if num_fused_shared_experts:
        topk_ids[:, -1] = torch.randint(
            low=num_experts,
            high=num_experts + num_fused_shared_experts,
            size=(topk_ids.size(0),),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        topk_weights[:, -1] = topk_weights[:, :-1].sum(dim=-1) / routed_scaling_factor

    if renormalize:
        topk_weights_sum = (
            topk_weights.sum(dim=-1, keepdim=True)
            if num_fused_shared_experts == 0
            else topk_weights[:, :-1].sum(dim=-1, keepdim=True)
        )
        topk_weights = topk_weights / topk_weights_sum
        if apply_routed_scaling_factor_on_output:
            topk_weights *= routed_scaling_factor

    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)
    topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
    return topk_weights, topk_ids


def is_power_of_two(n):
    return n > 0 and math.log2(n).is_integer()


def _mask_topk_ids_padded_region(
    topk_ids: torch.Tensor,
    num_token_non_padded: Optional[torch.Tensor] = None,
):
    if num_token_non_padded is None:
        return
    indices = torch.arange(0, topk_ids.shape[0], device=topk_ids.device)
    topk_ids[indices >= num_token_non_padded, :] = -1


@torch.compile(dynamic=True, backend=get_compiler_backend())
def _biased_grouped_topk_postprocess(
    topk_ids, expert_location_dispatch_info, num_token_non_padded
):
    topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
    return topk_ids


def biased_grouped_topk_gpu(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    assert (
        routed_scaling_factor is not None
    ), "routed_scaling_factor is required for biased_grouped_topk"
    # TODO: moe_fused_gate kernel is not supported for num_fused_shared_experts > 0 now.
    if (
        _is_cuda
        and gating_output.shape[1] // num_expert_group
        <= 32  # moe_fused_gate kernel ensure that num_experts/num_expert_group does not exceed MAX_VPT=32 now. And when kernel can handle MAX_VPT > 32, we can remove this assertion.
        and is_power_of_two(correction_bias.shape[0])
    ):
        topk_weights, topk_ids = moe_fused_gate(
            gating_output.to(dtype=torch.float32),
            correction_bias,
            num_expert_group,
            topk_group,
            topk,
            num_fused_shared_experts,
            routed_scaling_factor,
            apply_routed_scaling_factor_on_output,
        )
        # TODO merge into kernel
        if (expert_location_dispatch_info is not None) or (
            num_token_non_padded is not None
        ):
            topk_ids = _biased_grouped_topk_postprocess(
                topk_ids, expert_location_dispatch_info, num_token_non_padded
            )
        return topk_weights, topk_ids
    elif _use_aiter:
        assert not apply_routed_scaling_factor_on_output, "Not implemented"
        token = gating_output.shape[0]
        device = gating_output.device
        assert (
            hidden_states.shape[0] == gating_output.shape[0]
        ), f"Number of tokens mismatch: hidden_states.shape[0] = {hidden_states.shape[0]}, gating_output.shape[0] = {gating_output.shape[0]}"
        topk_weights = torch.empty((token, topk), dtype=torch.float32, device=device)
        topk_ids = torch.empty((token, topk), dtype=torch.int32, device=device)
        aiter_biased_grouped_topk(
            gating_output.to(dtype=torch.float32),
            correction_bias,
            topk_weights,
            topk_ids,
            num_expert_group,
            topk_group,
            renormalize,
            routed_scaling_factor,
        )
        return topk_weights, topk_ids
    else:
        return biased_grouped_topk_impl(
            hidden_states,
            gating_output,
            correction_bias,
            topk,
            renormalize,
            num_expert_group,
            topk_group,
            num_fused_shared_experts=num_fused_shared_experts,
            routed_scaling_factor=routed_scaling_factor,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
            apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
        )


def biased_grouped_topk_cpu(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    compiled: bool = True,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    assert expert_location_dispatch_info is None
    assert not apply_routed_scaling_factor_on_output, "Not implemented"
    return torch.ops.sgl_kernel.biased_grouped_topk_cpu(
        hidden_states,
        gating_output,
        correction_bias,
        topk,
        renormalize,
        num_expert_group,
        topk_group,
        num_fused_shared_experts,
        routed_scaling_factor,
        num_token_non_padded,
    )


if _is_cpu and _is_cpu_amx_available:
    biased_grouped_topk = biased_grouped_topk_cpu
    grouped_topk = grouped_topk_cpu
    fused_topk_native = fused_topk_cpu
    fused_topk = fused_topk_cpu
else:
    biased_grouped_topk = biased_grouped_topk_gpu
    grouped_topk = grouped_topk_gpu
    fused_topk_native = fused_topk_torch_native


# -------------------------------- Expert-Choice Implementation ---------------------------------------


@torch.compile(dynamic=True, backend=get_compiler_backend())
def expert_choice_topk_gpu(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    capacity: int,
    num_experts: int,
    topk: int,
    renormalize: bool,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
):
    """
    True Expert-choice routing: Each expert selects exactly `capacity` tokens.

    This implementation ensures every expert's selection is honored by dynamically
    expanding the output topk to accommodate all selections.

    Key difference from Token Choice:
    - Token Choice: each token selects K experts -> unbalanced expert load
    - Expert Choice: each expert selects C tokens -> balanced expert load

    Args:
        hidden_states: [N, hidden_dim]
        gating_output: [N, E] router logits
        capacity: Tokens per expert (C) - each expert selects exactly C tokens
        num_experts: Total number of experts (E)
        topk: Minimum experts per token (may be expanded to fit all selections)
        renormalize: Whether to renormalize weights

    Returns:
        topk_weights: [N, K'] - weights, K' >= topk to fit all expert selections
        topk_ids: [N, K'] - expert IDs selected for each token
    """
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    num_tokens = gating_output.shape[0]

    # EXPERT CHOICE: Column-wise softmax (experts compete for tokens)
    scores = torch.softmax(gating_output, dim=0)  # [N, E]

    # Transpose to expert-centric view: [E, N]
    scores_transposed = scores.t()  # [E, N]

    # Each expert selects exactly `capacity` tokens
    actual_capacity = min(capacity, num_tokens)

    expert_topk_scores, expert_topk_token_ids = torch.topk(
        scores_transposed,
        k=actual_capacity,
        dim=1,
        sorted=False
    )

    # Build assignment matrix [N, E]
    expert_ids_flat = torch.arange(num_experts, device=gating_output.device).unsqueeze(1).expand(-1, actual_capacity).flatten()
    token_ids_flat = expert_topk_token_ids.flatten()
    scores_flat = expert_topk_scores.flatten()

    token_expert_scores = torch.zeros((num_tokens, num_experts), device=gating_output.device)
    token_expert_scores.index_put_((token_ids_flat, expert_ids_flat), scores_flat)

    # Dynamic topk: use conservative upper bound to avoid .item() which breaks CUDA graphs
    # Upper bound: in the worst case, all experts could select the same token
    # But realistically, use 2x average as a safe estimate
    avg_experts_per_token = (num_experts * actual_capacity + num_tokens - 1) // num_tokens
    # Cap at num_experts (theoretical maximum)
    expanded_topk = max(topk, min(2 * avg_experts_per_token, num_experts))

    # For each token, get all experts that selected it
    topk_weights, topk_ids = torch.topk(token_expert_scores, k=expanded_topk, dim=1, sorted=False)

    # Mark invalid (score=0 means not selected)
    invalid_mask = topk_weights == 0
    topk_weights = torch.where(invalid_mask, torch.zeros_like(topk_weights), topk_weights)
    topk_ids = torch.where(invalid_mask, torch.full_like(topk_ids, -1), topk_ids)

    # Renormalize
    if renormalize:
        valid_mask = topk_ids >= 0
        weights_sum = topk_weights.sum(dim=-1, keepdim=True)
        weights_sum = torch.where(weights_sum > 0, weights_sum, torch.ones_like(weights_sum))
        topk_weights = torch.where(valid_mask, topk_weights / weights_sum, topk_weights)

    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)

    return topk_weights, topk_ids


def expert_choice_topk_cpu(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    capacity: int,
    num_experts: int,
    topk: int,
    renormalize: bool,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
):
    """CPU version of expert-choice routing."""
    # Use same implementation as GPU for now
    return expert_choice_topk_gpu(
        hidden_states,
        gating_output,
        capacity,
        num_experts,
        topk,
        renormalize,
        num_token_non_padded,
        expert_location_dispatch_info,
    )


@torch.compile(dynamic=True, backend=get_compiler_backend())
def grouped_expert_choice_topk_gpu(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    capacity: int,
    num_experts: int,
    topk: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    capacity_per_group: Optional[int] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
):
    """
    True Expert-choice routing: Each expert selects exactly `capacity` tokens.

    This ensures PERFECT expert load balance - every expert processes exactly
    the same number of tokens. The output topk is dynamically expanded to
    accommodate all expert selections without dropping any.

    Key insight: Instead of limiting tokens to topk experts, we expand topk
    to fit all experts that selected each token.

    Args:
        hidden_states: [N, hidden_dim]
        gating_output: [N, E] router logits
        capacity: Tokens per expert (C) - each expert selects exactly C tokens
        num_experts: Total number of experts (E)
        topk: Minimum experts per token (expanded as needed)
        renormalize: Whether to renormalize weights
        num_expert_group: Number of expert groups (unused in pure expert choice)
        topk_group: Number of groups (unused in pure expert choice)

    Returns:
        topk_weights: [N, K'] - weights, K' >= topk to fit all selections
        topk_ids: [N, K'] - expert IDs selected for each token
    """
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    num_tokens = gating_output.shape[0]

    # Expert Choice: softmax over tokens (column-wise) so experts compete for tokens
    scores = torch.softmax(gating_output, dim=0)  # [N, E]

    # Transpose to [E, N] for expert-centric processing
    scores_t = scores.t()  # [E, N]

    # Each expert selects exactly `capacity` tokens
    actual_capacity = min(capacity, num_tokens)

    expert_scores, expert_tokens = torch.topk(scores_t, k=actual_capacity, dim=1, sorted=False)
    # expert_tokens shape: [E, actual_capacity]
    # VERIFY: Every expert selects exactly actual_capacity tokens (perfect balance!)
    # expert_tokens[i] contains the token indices that expert i selected

    # Build assignment matrix [N, E]
    assignment = torch.zeros(num_tokens, num_experts, device=gating_output.device)

    expert_idx = torch.arange(num_experts, device=gating_output.device).unsqueeze(1).expand(-1, actual_capacity)
    expert_idx_flat = expert_idx.flatten()
    token_idx_flat = expert_tokens.flatten()
    scores_flat = expert_scores.flatten()

    assignment.index_put_((token_idx_flat, expert_idx_flat), scores_flat)

    # Dynamic topk: use conservative upper bound to avoid .item() which breaks CUDA graphs
    # Upper bound: 2x average experts per token, capped at num_experts
    avg_experts_per_token = (num_experts * actual_capacity + num_tokens - 1) // num_tokens
    expanded_topk = max(topk, min(2 * avg_experts_per_token, num_experts))

    topk_weights, topk_ids = torch.topk(assignment, k=expanded_topk, dim=1, sorted=False)

    # Mark invalid (score=0 means not selected)
    invalid_mask = topk_weights == 0
    topk_weights = torch.where(invalid_mask, torch.zeros_like(topk_weights), topk_weights)
    topk_ids = torch.where(invalid_mask, torch.full_like(topk_ids, -1), topk_ids)

    # Renormalize
    if renormalize:
        valid_mask = topk_ids >= 0
        weights_sum = topk_weights.sum(dim=-1, keepdim=True)
        weights_sum = torch.where(weights_sum > 0, weights_sum, torch.ones_like(weights_sum))
        topk_weights = torch.where(valid_mask, topk_weights / weights_sum, topk_weights)

    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)

    return topk_weights, topk_ids


if _is_cpu and _is_cpu_amx_available:
    expert_choice_topk = expert_choice_topk_cpu
else:
    expert_choice_topk = expert_choice_topk_gpu


def select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    topk_config: TopKConfig,
    *,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
) -> StandardTopKOutput:

    top_k = topk_config.top_k
    use_grouped_topk = topk_config.use_grouped_topk
    topk_group = topk_config.topk_group
    num_expert_group = topk_config.num_expert_group
    renormalize = topk_config.renormalize
    num_fused_shared_experts = topk_config.num_fused_shared_experts
    custom_routing_function = topk_config.custom_routing_function
    correction_bias = topk_config.correction_bias
    torch_native = topk_config.torch_native
    routed_scaling_factor = topk_config.routed_scaling_factor
    apply_routed_scaling_factor_on_output = (
        topk_config.apply_routed_scaling_factor_on_output
    )

    router_logits, correction_bias = (
        expert_location_dispatch.transform_select_experts_inputs(
            router_logits=router_logits,
            correction_bias=correction_bias,
            info=expert_location_dispatch_info,
        )
    )

    # DeepSeek V2/V3/R1 series models use grouped_top_k
    if use_grouped_topk:
        assert topk_group is not None
        assert num_expert_group is not None
        # Check if custom routing function is provided first
        if custom_routing_function is not None:
            # Use custom routing function (e.g., Expert Choice)
            topk_weights, topk_ids = custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
            )
        elif correction_bias is None:
            topk_weights, topk_ids = grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                num_fused_shared_experts=num_fused_shared_experts,
                routed_scaling_factor=routed_scaling_factor,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
                apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            )
        else:
            topk_weights, topk_ids = biased_grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                correction_bias=correction_bias,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                num_fused_shared_experts=num_fused_shared_experts,
                routed_scaling_factor=routed_scaling_factor,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
                apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            )
    elif torch_native and custom_routing_function is None:
        assert (
            num_token_non_padded is None
        ), "num_token_non_padded is not yet supported in fused_topk_native"
        assert expert_location_dispatch_info is None
        assert not apply_routed_scaling_factor_on_output, "Not implemented"
        topk_weights, topk_ids = fused_topk_native(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
            correction_bias=correction_bias,
        )
    elif custom_routing_function is None:
        assert not apply_routed_scaling_factor_on_output, "Not implemented"
        # Qwen3MOE uses fused_topk
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )
    else:
        assert (
            num_token_non_padded is None
        ), "num_token_non_padded is not yet supported in custom_routing_function"
        assert expert_location_dispatch_info is None
        assert not apply_routed_scaling_factor_on_output, "Not implemented"
        topk_weights, topk_ids = custom_routing_function(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
        )

    get_global_expert_distribution_recorder().on_select_experts(topk_ids=topk_ids)

    return StandardTopKOutput(topk_weights, topk_ids, router_logits)


# -------------------------------- Expert Choice Routing Factory ---------------------------------------


def create_expert_choice_routing_function(
    capacity: int,
    num_experts: int,
    use_grouped_topk: bool = False,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
):
    """
    Factory function to create an expert choice routing function.

    This creates a routing function that uses expanded_topk = num_experts
    to guarantee ZERO information loss - every expert's selection is preserved.

    Args:
        capacity: Tokens per expert (C) - each expert selects exactly C tokens
        num_experts: Total number of experts (E) - also used as expanded_topk
        use_grouped_topk: Whether to use grouped topk
        num_expert_group: Number of expert groups
        topk_group: Number of groups to select

    Returns:
        A routing function compatible with TopK's custom_routing_function

    Example:
        >>> routing_fn = create_expert_choice_routing_function(
        ...     capacity=8, num_experts=256
        ... )
        >>> topk = TopK(
        ...     top_k=256,  # Set to num_experts for expert choice
        ...     custom_routing_function=routing_fn,
        ...     renormalize=True,
        ... )
    """

    @torch.compile(dynamic=True, backend=get_compiler_backend())
    def expert_choice_routing_full(
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        num_token_non_padded: Optional[torch.Tensor] = None,
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    ):
        """
        Expert-choice routing with expanded_topk = num_experts.

        This guarantees zero information loss by using all experts in the topk output.
        """
        num_tokens = gating_output.shape[0]

        # Expert Choice: softmax over tokens (column-wise) so experts compete for tokens
        scores = torch.softmax(gating_output, dim=0)  # [N, E]

        # Transpose to [E, N] for expert-centric processing
        scores_t = scores.t()  # [E, N]

        # Each expert selects exactly `capacity` tokens
        actual_capacity = min(capacity, num_tokens)

        expert_scores, expert_tokens = torch.topk(scores_t, k=actual_capacity, dim=1, sorted=False)

        # Build assignment matrix [N, E]
        assignment = torch.zeros(num_tokens, num_experts, device=gating_output.device)

        expert_idx = torch.arange(num_experts, device=gating_output.device).unsqueeze(1).expand(-1, actual_capacity)
        expert_idx_flat = expert_idx.flatten()
        token_idx_flat = expert_tokens.flatten()
        scores_flat = expert_scores.flatten()

        assignment.index_put_((token_idx_flat, expert_idx_flat), scores_flat)

        # KEY: Use expanded_topk = num_experts to guarantee zero information loss
        # This ensures every expert's selection is preserved in the output
        expanded_topk = num_experts

        topk_weights, topk_ids = torch.topk(assignment, k=expanded_topk, dim=1, sorted=False)

        # Mark invalid (score=0 means not selected)
        invalid_mask = topk_weights == 0
        topk_weights = torch.where(invalid_mask, torch.zeros_like(topk_weights), topk_weights)
        topk_ids = torch.where(invalid_mask, torch.full_like(topk_ids, -1), topk_ids)

        # Renormalize
        if renormalize:
            valid_mask = topk_ids >= 0
            weights_sum = topk_weights.sum(dim=-1, keepdim=True)
            weights_sum = torch.where(weights_sum > 0, weights_sum, torch.ones_like(weights_sum))
            topk_weights = torch.where(valid_mask, topk_weights / weights_sum, topk_weights)

        topk_weights = topk_weights.to(torch.float32)
        topk_ids = topk_ids.to(torch.int32)

        topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
        _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)

        return topk_weights, topk_ids

    return expert_choice_routing_full
