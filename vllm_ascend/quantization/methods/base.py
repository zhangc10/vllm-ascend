#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
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
#
"""Abstract base classes for Ascend quantization schemes."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
from torch.nn.functional import pad
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEMlpComputeInput
from vllm_ascend.quantization.quant_type import QuantType


def get_moe_num_logical_experts(
    layer: torch.nn.Module,
    num_experts: int,
    global_redundant_expert_num: int = 0,
    num_shared_experts: int = 0,
) -> int:
    moe_config = getattr(layer, "moe_config", None)
    num_logical_experts = getattr(moe_config, "num_logical_experts", None)
    if num_logical_experts is not None:
        return int(num_logical_experts)

    return int(num_experts - global_redundant_expert_num - num_shared_experts)


class AscendLinearScheme(ABC):
    """Base class for all linear quantization schemes.

    Subclasses must implement get_weight() and apply() methods.
    Other methods have default implementations that return empty dicts
    or do nothing.
    """

    @abstractmethod
    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        """Return weight tensor specifications.

        Args:
            input_size: Input dimension of the linear layer.
            output_size: Output dimension of the linear layer.
            params_dtype: Data type for parameters.

        Returns:
            Dictionary mapping parameter names to empty tensors with
            the correct shape and dtype.
        """
        ...

    def get_pertensor_param(self, params_dtype: torch.dtype, **kwargs: Any) -> dict[str, Any]:
        """Return per-tensor parameter specifications (e.g., input_scale).

        Args:
            params_dtype: Data type for parameters.
            **kwargs: Additional keyword arguments for subclass extensions

        Returns:
            Dictionary mapping parameter names to empty tensors.
        """
        return {}

    def get_perchannel_param(self, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        """Return per-channel parameter specifications (e.g., weight_scale).

        Args:
            output_size: Output dimension of the linear layer.
            params_dtype: Data type for parameters.

        Returns:
            Dictionary mapping parameter names to empty tensors.
        """
        return {}

    def get_pergroup_param(
        self, input_size: int, output_size: int, params_dtype: torch.dtype, layer_type: str | None = None
    ) -> dict[str, Any]:
        """Return per-group parameter specifications.

        Args:
            input_size: Input dimension of the linear layer.
            output_size: Output dimension of the linear layer.
            params_dtype: Data type for parameters.
            layer_type: Type of layer (e.g., "row" for RowParallelLinear).

        Returns:
            Dictionary mapping parameter names to empty tensors.
        """
        return {}

    @abstractmethod
    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None, tp_rank: int | None = 0
    ) -> torch.Tensor:
        """Forward computation.

        Args:
            layer: The linear layer module.
            x: Input tensor.
            bias: Optional bias tensor.
            tp_rank: Tensor parallel rank.

        Returns:
            Output tensor after quantized linear operation.
        """
        ...

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Post-loading weight processing (transpose, format conversion, etc.).

        Args:
            layer: The linear layer module.
        """
        return


class AscendAttentionScheme(ABC):
    """Base class for all attention quantization schemes.

    Subclasses must implement apply() method.
    Other methods have default implementations.
    """

    def create_weights(self, layer: torch.nn.Module) -> None:
        """Create weights for attention quantization.

        Args:
            layer: The attention layer module.
        """
        return

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Post-loading weight processing for attention layer.

        Args:
            layer: The attention layer module.
        """
        return

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache,
        attn_metadata,
        attn_type,
        scale,
        output,
    ) -> torch.Tensor:
        """Forward computation for attention layer.

        Args:
            layer: The attention layer module.
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            kv_cache: KV cache.
            attn_metadata: Attention metadata.
            attn_type: Attention type.
            scale: Scale factor.
            output: Output tensor.

        Returns:
            Output tensor after attention computation.
        """
        ...


class AscendMoEScheme(ABC):
    """Base class for all MoE quantization schemes.

    Subclasses must implement get_weight(), get_dynamic_quant_param(),
    and apply() methods.

    Attributes:
        quant_type: The quantization type for this scheme. Subclasses should
                   override this class attribute to declare their quant type.
    """

    # Default quant type - subclasses should override this
    quant_type: QuantType = QuantType.NONE

    @abstractmethod
    def get_weight(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        """Return weight tensor specifications for MoE layer.

        Args:
            num_experts: Number of experts.
            intermediate_size_per_partition: Intermediate size per partition.
            hidden_sizes: Hidden dimension size.
            params_dtype: Data type for parameters.

        Returns:
            Dictionary mapping parameter names to empty tensors.
        """
        ...

    @abstractmethod
    def get_dynamic_quant_param(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        """Return dynamic quantization parameters for MoE layer.

        Args:
            num_experts: Number of experts.
            intermediate_size_per_partition: Intermediate size per partition.
            hidden_sizes: Hidden dimension size.
            params_dtype: Data type for parameters.

        Returns:
            Dictionary mapping parameter names to empty tensors.
        """
        ...

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor | None = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: Any | None = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        mc2_mask: torch.Tensor | None = None,
        tid2eid: Any | None = None,
    ) -> torch.Tensor:
        """Forward computation for MoE layer.

        Args:
            layer: The MoE layer module.
            x: Input hidden states.
            router_logits: Router logits for expert selection.
            top_k: Number of experts to select per token.
            renormalize: Whether to renormalize expert weights.
            use_grouped_topk: Whether to use grouped top-k selection.
            num_experts: Number of experts.
            expert_map: Mapping from local to global expert indices.
            topk_group: Group size for grouped top-k.
            num_expert_group: Number of expert groups.
            custom_routing_function: Custom routing function.
            scoring_func: Scoring function name.
            routed_scaling_factor: Scaling factor for routed experts.
            e_score_correction_bias: Expert score correction bias.
            is_prefill: Whether in prefill phase.
            enable_force_load_balance: Whether to force load balancing.
            log2phy: Logical to physical expert mapping.
            global_redundant_expert_num: Number of redundant experts.
            pertoken_scale: Optional per-token activation scale from prepare stage.
            activation: Expert MLP activation type.
            apply_router_weight_on_input: Whether to pre-scale hidden states by router weights.
            mc2_mask: Optional mask used by MC2 dispatch.

        Returns:
            Output tensor after MoE computation.
        """
        ...

    @abstractmethod
    def apply_mlp(
        self, mlp_compute_input: MoEMlpComputeInput
    ) -> tuple[torch.Tensor, torch.npu.Event | None]:
        """Execute MoE MLP compute (GMM1 + activation + requant + GMM2).

        Args:
            mlp_compute_input: Typed runtime payload carrying hidden states,
                group list, quant params, and activation config.

        Returns:
            Tuple of (output hidden states, before_gmm2 event or None).
        """
        ...

    @staticmethod
    def cumsum_group_list(
        group_list: torch.Tensor,
        src_list_type: int,
        dst_list_type: int,
        active_num: int = 0,
        expert_num: int = 0,
    ) -> torch.Tensor:
        if src_list_type not in [0, 1, 2]:
            raise ValueError(f"group_list_type should be in [0, 1, 2], but received {src_list_type}")

        if src_list_type == dst_list_type:
            return group_list
        if src_list_type == 1 and dst_list_type == 0:
            return group_list.cumsum(dim=0)
        if src_list_type == 0 and dst_list_type == 1:
            group_diff = torch.diff(group_list)
            new_group = torch.cat([group_list[0].unsqueeze(0), group_diff], dim=0)
            return new_group
        if src_list_type == 2 and dst_list_type == 0:
            experts = pad(group_list[:, 0], (1, 0))
            tokens = pad(group_list[:, 1].cumsum(dim=0), (1, 0))
            cumsum_group_list = torch.full(
                size=(expert_num,), fill_value=active_num, dtype=group_list.dtype, device=group_list.device
            )

            for i, (start, end) in enumerate(zip(experts[:-1], experts[1:])):
                if end > start:
                    cumsum_group_list[start:end] = tokens[i]

            return cumsum_group_list
        raise NotImplementedError(
            f"Conversion from src_list_type={src_list_type} to dst_list_type={dst_list_type} is not implemented yet. "
            "This feature is under development."
        )

    @staticmethod
    def _can_use_fused_op(activation) -> bool:
        """Determine whether a fused GMM+SwiGLU+Quant op can be used
        for the given activation type.

        Returns False for GELU/GELU_TANH (no fused op supports GELU),
        SWIGLUSTEP (uses separate gmm1+swiglustep+requant), and
        swigluoai_uninterleave (fused ops don't support uninterleaved
        clipped swiglu).
        """
        act_name = getattr(activation, "value", activation)
        return (
            activation not in (MoEActivation.GELU, MoEActivation.GELU_TANH, MoEActivation.SWIGLUSTEP)
            and act_name != "swigluoai_uninterleave"
        )

    @staticmethod
    def _enable_custom_op() -> bool:
        from vllm_ascend.utils import enable_custom_op

        return enable_custom_op()

    @staticmethod
    def _non_fused_act_quant(
        activation,
        hidden_states: torch.Tensor,
        *,
        swiglu_limit: float = 0.0,
        swiglu_alpha: float = 1.0,
        swiglu_beta: float = 0.0,
        use_mxfp_quant: bool = False,
        act_quant_type: torch.dtype = torch.float8_e4m3fn,
        group_list: torch.Tensor | None = None,
        group_list_type: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Common activation + requantization for the non-fusion path.

        Called after GMM1 produces bf16/fp16 output.  Branches by activation
        type:
        - SWIGLUSTEP: swiglustep_forward → npu_dynamic_quant
        - GELU/GELU_TANH: gelu → npu_dynamic_quant
        - swigluoai_uninterleave: npu_clipped_swiglu → npu_dynamic_quant
        - default (silu): swiglu_quant (triton) or npu_swiglu → npu_dynamic_quant

        Returns (hidden_states, swiglu_out_scale).
        """
        import torch_npu
        from vllm_ascend.device.device_op import DeviceOperator
        from vllm_ascend.ops.activation import AscendSwigluStepAndMul

        if activation == MoEActivation.SWIGLUSTEP:
            hidden_states = AscendSwigluStepAndMul.swiglustep_forward(hidden_states, limit=swiglu_limit or 7.0)
            hidden_states, swiglu_out_scale = DeviceOperator.npu_dynamic_quant(
                hidden_states, act_quant_type=act_quant_type, use_mxfp_quant=use_mxfp_quant
            )
        elif activation in (MoEActivation.GELU, MoEActivation.GELU_TANH):
            gate, up = hidden_states.chunk(2, dim=-1)
            approximate = "tanh" if activation == MoEActivation.GELU_TANH else "none"
            hidden_states = torch.nn.functional.gelu(gate, approximate=approximate) * up
            hidden_states, swiglu_out_scale = torch_npu.npu_dynamic_quant(hidden_states)
        elif getattr(activation, "value", activation) == "swigluoai_uninterleave":
            hidden_states = torch_npu.npu_clipped_swiglu(
                hidden_states,
                interleaved=False,
                alpha=swiglu_alpha,
                limit=swiglu_limit,
                bias=swiglu_beta,
            )
            hidden_states, swiglu_out_scale = DeviceOperator.npu_dynamic_quant(
                hidden_states, act_quant_type=act_quant_type, use_mxfp_quant=use_mxfp_quant
            )
        else:
            if HAS_TRITON:
                from vllm_ascend.ops.triton.activation.swiglu_quant import swiglu_quant

                hidden_states, swiglu_out_scale = swiglu_quant(
                    hidden_states, group_list=group_list, group_list_type=group_list_type
                )
            else:
                hidden_states = torch_npu.npu_swiglu(hidden_states)
                hidden_states, swiglu_out_scale = torch_npu.npu_dynamic_quant(hidden_states)
        return hidden_states, swiglu_out_scale

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Post-loading weight processing for MoE layer.

        Args:
            layer: The MoE layer module.
        """
        return
