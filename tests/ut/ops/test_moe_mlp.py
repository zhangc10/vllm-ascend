import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock, patch

import torch
import torch_npu  # noqa: F401  -- registers torch.npu used by the module under test
from torch.nn import functional as F
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEMlpComputeInput,
    MoEQuantParams,
)
from vllm_ascend.ops.fused_moe.moe_stage_params import MoEMxfpParams
from vllm_ascend.quantization.methods.base import AscendMoEScheme
from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod
from vllm_ascend.quantization.methods.w4a16 import AscendW4A16FusedMoEMethod
from vllm_ascend.quantization.quant_type import QuantType

W8A8_DYNAMIC = "vllm_ascend.quantization.methods.w8a8_dynamic"
BASE = "vllm_ascend.quantization.methods.base"


class TestCumsumGroupList(unittest.TestCase):
    glist_dict: ClassVar[dict[int, torch.Tensor]]

    @classmethod
    def setUpClass(cls):
        cls.glist_dict = {
            0: torch.tensor([0, 2, 3, 3]),
            1: torch.tensor([0, 2, 1, 0]),
            2: torch.tensor([[1, 2], [2, 1], [0, 0], [0, 0]]),
        }

    support_combine = [(0, 0), (1, 0), (0, 1)]
    unsupported_combine = [(0, 2), (2, 1), (1, 2)]

    def test_cumsum_group_list_supported_conversion(self):
        for src_list_type, dst_list_type in self.support_combine:
            with self.subTest(src=src_list_type, dst=dst_list_type):
                result = AscendMoEScheme.cumsum_group_list(
                    self.glist_dict[src_list_type], src_list_type, dst_list_type, expert_num=4
                )
                self.assertTrue(torch.equal(result, self.glist_dict[dst_list_type]))

    def test_cumsum_group_list_invalid_type_valueerror(self):
        with self.assertRaises(ValueError) as excinfo:
            AscendMoEScheme.cumsum_group_list(self.glist_dict[0], 4, 0)
        self.assertIn("group_list_type should be in [0, 1, 2], but received", str(excinfo.exception))

    def test_cumsum_group_list_unsupported_conversion_notimplementederror(self):
        for src_list_type, dst_list_type in self.unsupported_combine:
            with self.subTest(src=src_list_type, dst=dst_list_type):
                with self.assertRaises(NotImplementedError) as excinfo:
                    AscendMoEScheme.cumsum_group_list(self.glist_dict[0], src_list_type, dst_list_type)
                self.assertIn("This feature is under development.", str(excinfo.exception))


class TestW4A8RuntimeFlags(unittest.TestCase):
    def test_w4a8_per_channel_gmm_swiglu_flag(self):
        self.assertTrue(
            MoEQuantParams(quant_type=QuantType.W4A8, is_per_channel_weight=True).use_w4a8_per_channel_gmm_swiglu
        )
        self.assertFalse(
            MoEQuantParams(quant_type=QuantType.W4A8, is_per_channel_weight=False).use_w4a8_per_channel_gmm_swiglu
        )
        self.assertFalse(
            MoEQuantParams(quant_type=QuantType.W8A8, is_per_channel_weight=True).use_w4a8_per_channel_gmm_swiglu
        )


class TestCanUseFusedOp(unittest.TestCase):
    def test_gelu_excluded(self):
        self.assertFalse(AscendMoEScheme._can_use_fused_op(MoEActivation.GELU))
        self.assertFalse(AscendMoEScheme._can_use_fused_op(MoEActivation.GELU_TANH))

    def test_swigluoai_uninterleave_excluded(self):
        self.assertFalse(AscendMoEScheme._can_use_fused_op("swigluoai_uninterleave"))

    def test_swiglustep_excluded(self):
        self.assertFalse(AscendMoEScheme._can_use_fused_op(MoEActivation.SWIGLUSTEP))

    def test_silu_swigluoai_allowed(self):
        self.assertTrue(AscendMoEScheme._can_use_fused_op("silu"))
        self.assertTrue(AscendMoEScheme._can_use_fused_op(MoEActivation.SWIGLUOAI))


def _patch_npu_stream():
    """Patch ``torch.npu.current_stream`` so ``record_event()`` returns a tag."""
    evt = MagicMock(name="before_gmm2_evt")
    stream = MagicMock(name="npu_stream")
    stream.record_event.return_value = evt
    return patch("torch.npu.current_stream", return_value=stream), evt


def _make_w8a8_scheme():
    scheme = AscendW8A8DynamicFusedMoEMethod.__new__(AscendW8A8DynamicFusedMoEMethod)
    scheme.dynamic_eplb = False
    return scheme


def _make_w4a16_scheme():
    scheme = AscendW4A16FusedMoEMethod.__new__(AscendW4A16FusedMoEMethod)
    return scheme


def _make_w8a8_layer(*, w1_scale_dtype=torch.float32, w2_scale_dtype=torch.float32, swiglu_limit=0.0):
    layer = MagicMock()
    layer.w13_weight = torch.randn(1, 8, 4)
    layer.w2_weight = torch.randn(1, 4, 1)
    layer.w13_weight_scale_fp32 = torch.randn(1, 8, dtype=w1_scale_dtype)
    layer.w2_weight_scale = torch.randn(1, 4, dtype=w2_scale_dtype)
    layer.swiglu_limit = swiglu_limit
    layer.swiglu_alpha = 1.0
    layer.swiglu_beta = 0.0
    return layer


@contextmanager
def _mock_w8a8_gelu_compute(gate_up, *, gmm2_out=None, capture_quant=False):
    """Mock the W8A8 GELU-path NPU ops: dequant GMM1 (``npu_grouped_matmul``),
    requant (``npu_dynamic_quant``), GMM2 (``npu_grouped_matmul`` second call), plus the
    NPU stream event and ``dispose_tensor``. Yields a namespace with the mocks;
    when ``capture_quant`` is True, ``captured['x']``/``captured['scale']``
    record the requant input and the returned per-token scale."""
    stream_patch, evt = _patch_npu_stream()
    captured = {}

    def _dynamic_quant(x, *args, **kwargs):
        if capture_quant:
            captured["x"] = x.detach().clone()
            scale = torch.ones(1, dtype=torch.float32)
            captured["scale"] = scale
            return x, scale
        return x, torch.ones(1)

    with (
        stream_patch,
        patch("torch_npu.npu_grouped_matmul", return_value=[gate_up], create=True) as mock_gmm,
        patch("torch_npu.npu_dynamic_quant", side_effect=_dynamic_quant, create=True) as mock_dq,
        patch(f"{W8A8_DYNAMIC}.dispose_tensor"),
    ):
        yield SimpleNamespace(gmm=mock_gmm, dq=mock_dq, evt=evt, captured=captured)


def _common_w8a8_mlp_input(
    *,
    activation,
    w1_scale_dtype=torch.float32,
    w2_scale_dtype=torch.float32,
    group_list_type=1,
    group_list=None,
    dynamic_scale=None,
    swiglu_limit=0.0,
    fusion=False,
):
    layer = _make_w8a8_layer(
        w1_scale_dtype=w1_scale_dtype, w2_scale_dtype=w2_scale_dtype, swiglu_limit=swiglu_limit
    )
    return MoEMlpComputeInput(
        hidden_states=torch.randn(1, 4),
        group_list=group_list if group_list is not None else torch.tensor([1], dtype=torch.int64),
        group_list_type=group_list_type,
        dynamic_scale=dynamic_scale if dynamic_scale is not None else torch.randn(1, 1),
        topk_scales=None,
        layer=layer,
        quant=MoEQuantParams(quant_type=QuantType.W8A8),
        fusion=fusion,
        activation=activation,
        need_trans=False,
        dynamic_eplb=False,
        moe_scheme=_make_w8a8_scheme(),
    )


class TestW8A8ApplyMlpGeluPath(unittest.TestCase):
    """GELU path: dispatch, math, and layout coverage for W8A8 scheme."""

    def test_w8a8_gelu_tanh_applies_correct_activation(self):
        """W8A8 + gelu_tanh: GMM1(dequant) -> gelu(tanh)·up -> requant -> GMM2."""
        gate = torch.tensor([[1.0, 2.0, -1.0, 0.5]])
        up = torch.tensor([[0.5, -0.5, 1.0, 2.0]])
        gate_up = torch.cat([gate, up], dim=-1)
        expected = F.gelu(gate, approximate="tanh") * up
        gmm2_out = torch.tensor([[9.0]])
        with _mock_w8a8_gelu_compute(gate_up, gmm2_out=gmm2_out, capture_quant=True) as m:
            scheme = _make_w8a8_scheme()
            out, out_evt = scheme.apply_mlp(_common_w8a8_mlp_input(activation=MoEActivation.GELU_TANH))
        # GELU math applied with tanh approximation before requantization.
        self.assertTrue(torch.allclose(m.captured["x"], expected, atol=1e-6))
        # GMM1 used the dequant form (scale + per_token_scale), not antiquant.
        gmm1_kwargs = m.gmm.call_args.kwargs
        self.assertIn("scale", gmm1_kwargs)
        self.assertIn("per_token_scale", gmm1_kwargs)
        self.assertEqual(gmm1_kwargs["split_item"], 2)
        # Requant invoked.
        m.dq.assert_called_once()
        # Return contract: (hidden_states, before_gmm2_evt).
        self.assertIs(out, gmm2_out)
        self.assertIs(out_evt, m.evt)

    def test_w8a8_gelu_uses_exact_gelu_approximation(self):
        """W8A8 + gelu (not tanh): approximate='none', matching the float path."""
        gate = torch.tensor([[0.5, -0.5, 2.0]])
        up = torch.tensor([[1.0, 1.0, 0.5]])
        gate_up = torch.cat([gate, up], dim=-1)
        expected = F.gelu(gate, approximate="none") * up
        with _mock_w8a8_gelu_compute(gate_up, gmm2_out=torch.zeros(1, 3), capture_quant=True) as m:
            scheme = _make_w8a8_scheme()
            scheme.apply_mlp(_common_w8a8_mlp_input(activation=MoEActivation.GELU))
        # exact GELU (approximate='none') differs from tanh; ensure 'none' used.
        self.assertFalse(torch.allclose(m.captured["x"], F.gelu(gate, approximate="tanh") * up, atol=1e-6))
        self.assertTrue(torch.allclose(m.captured["x"], expected, atol=1e-6))

    def test_w4a16_gelu_uses_antiquat_path(self):
        """W4A16 + gelu: antiquant GMM1 -> gelu·up -> antiquant GMM2, no requant."""
        gate = torch.tensor([[1.0, -1.0]])
        up = torch.tensor([[0.5, 2.0]])
        gate_up = torch.cat([gate, up], dim=-1)
        expected = F.gelu(gate, approximate="tanh") * up
        gmm2_out = torch.tensor([[3.0]])
        stream_patch, evt = _patch_npu_stream()
        layer = MagicMock()
        layer.w13_weight_packed = torch.randn(1, 8, 4)
        layer.w2_weight_packed = torch.randn(1, 4, 1)
        layer.w13_weight_scale = torch.randn(1, 8, 4)
        layer.w2_weight_scale = torch.randn(1, 4, 1)
        layer.w13_weight_offset = torch.randn(1, 8, 4)
        layer.w2_weight_offset = torch.randn(1, 4, 1)
        layer.w2_weight_scale.dtype = torch.float32
        layer.swiglu_limit = 0.0
        layer.swiglu_alpha = 1.0
        layer.swiglu_beta = 0.0
        with (
            stream_patch,
            patch("torch_npu.npu_grouped_matmul", side_effect=[[gate_up], [gmm2_out]], create=True) as mock_gmm,
            patch("torch_npu.npu_dynamic_quant", create=True) as mock_dq,
        ):
            scheme = _make_w4a16_scheme()
            mlp_input = MoEMlpComputeInput(
                hidden_states=torch.randn(1, 4),
                group_list=torch.tensor([1], dtype=torch.int64),
                group_list_type=1,
                dynamic_scale=None,
                topk_scales=None,
                layer=layer,
                quant=MoEQuantParams(quant_type=QuantType.W4A16),
                fusion=False,
                activation=MoEActivation.GELU_TANH,
                moe_scheme=scheme,
            )
            out, out_evt = scheme.apply_mlp(mlp_input)

        self.assertEqual(mock_gmm.call_count, 2)
        # Both GMM calls use antiquant (not scale/per_token_scale).
        for call in mock_gmm.call_args_list:
            self.assertIn("antiquant_scale", call.kwargs)
            self.assertIn("antiquant_offset", call.kwargs)
            self.assertNotIn("scale", call.kwargs)
        # GMM2 (second call) input is the GELU activation output.
        gmm2_input = mock_gmm.call_args_list[1].kwargs["x"][0]
        self.assertTrue(torch.allclose(gmm2_input, expected, atol=1e-6))
        # W4A16 path does NOT requantize.
        mock_dq.assert_not_called()
        self.assertIs(out, gmm2_out)
        self.assertIs(out_evt, evt)

    def test_w8a8_gelu_converts_w1_scale_dtype_to_output_dtype(self):
        """When w1_scale dtype != _output_dtype, it is cast before GMM1."""
        # w1_scale fp32, w2_scale bf16 -> _output_dtype = bfloat16, so the GELU
        # path must cast w1_scale to bfloat16 before GMM1.
        with _mock_w8a8_gelu_compute(torch.zeros(1, 8)) as m:
            scheme = _make_w8a8_scheme()
            scheme.apply_mlp(
                _common_w8a8_mlp_input(
                    activation=MoEActivation.GELU_TANH,
                    w1_scale_dtype=torch.float32,
                    w2_scale_dtype=torch.bfloat16,
                )
            )
        self.assertEqual(m.gmm.call_args.kwargs["scale"][0].dtype, torch.bfloat16)

    def test_gelu_path_does_not_call_swiglu_op(self):
        """GELU path must use torch.gelu, never the SwiGLU NPU op."""
        with _mock_w8a8_gelu_compute(torch.zeros(1, 8)), patch("torch_npu.npu_swiglu", create=True) as mock_swiglu:
            scheme = _make_w8a8_scheme()
            scheme.apply_mlp(_common_w8a8_mlp_input(activation=MoEActivation.GELU_TANH))
        mock_swiglu.assert_not_called()

    def test_fusion_on_gelu_skips_fused_swiglu_quant(self):
        """Guard: with fusion ON (default), GELU must still skip the fused
        SwiGLU+quant op and use the non-fused GELU path."""
        mlp_input = _common_w8a8_mlp_input(activation=MoEActivation.GELU_TANH, fusion=True)
        with (
            _mock_w8a8_gelu_compute(torch.zeros(1, 8)) as m,
            patch("vllm_ascend.device.device_op.DeviceOperator.npu_grouped_matmul_swiglu_quant") as mock_fused,
        ):
            scheme = _make_w8a8_scheme()
            scheme.apply_mlp(mlp_input)
        # Fused SwiGLU+quant op must NOT be called for GELU.
        mock_fused.assert_not_called()
        # Non-fused dequant GMM1 (scale + per_token_scale) IS used.
        self.assertIn("scale", m.gmm.call_args.kwargs)
        self.assertIn("per_token_scale", m.gmm.call_args.kwargs)

    def test_mc2_gelu_skips_mc2_fused_branch(self):
        """Guard: GELU must skip the dequant_swiglu_quant fused branch and
        use the non-fused GELU path."""
        with (
            _mock_w8a8_gelu_compute(torch.zeros(1, 8)) as m,
            patch("torch.ops._C_ascend.npu_dequant_swiglu_quant", create=True) as mock_mc2_fused,
            patch("vllm_ascend.device.device_op.DeviceOperator.npu_grouped_matmul_swiglu_quant") as mock_fused,
        ):
            scheme = _make_w8a8_scheme()
            scheme.apply_mlp(_common_w8a8_mlp_input(activation=MoEActivation.GELU_TANH))
        # MC2 fused SwiGLU op must NOT be called for GELU.
        mock_mc2_fused.assert_not_called()
        mock_fused.assert_not_called()
        # Non-fused dequant GMM1 IS used instead.
        self.assertIn("scale", m.gmm.call_args.kwargs)


class TestW8A8ApplyMlpNoGeluImpact(unittest.TestCase):
    """Non-GELU activations must NOT enter the GELU path (no regression)."""

    def _run_non_gelu(self, activation):
        with (
            _mock_w8a8_gelu_compute(torch.zeros(1, 8)),
            patch(f"{BASE}.HAS_TRITON", False),
            patch("torch_npu.npu_swiglu", return_value=torch.zeros(1, 4), create=True) as mock_swiglu,
            patch("torch.nn.functional.gelu") as mock_gelu,
        ):
            scheme = _make_w8a8_scheme()
            scheme.apply_mlp(_common_w8a8_mlp_input(activation=activation))
        return mock_gelu, mock_swiglu

    def test_silu_activation_skips_gelu_path(self):
        mock_gelu, mock_swiglu = self._run_non_gelu("silu")
        mock_gelu.assert_not_called()
        # SwiGLu op IS used by the existing path -> existing logic intact.
        mock_swiglu.assert_called()

    def test_swiglustep_activation_skips_gelu_path(self):
        mock_gelu, _ = self._run_non_gelu(MoEActivation.SWIGLUSTEP)
        mock_gelu.assert_not_called()

    def test_swigluoai_activation_skips_gelu_path(self):
        mock_gelu, _ = self._run_non_gelu(MoEActivation.SWIGLUOAI)
        mock_gelu.assert_not_called()


class TestUnquantApplyMlp(unittest.TestCase):
    """Test the unquant apply_mlp path."""

    def test_unquant_apply_mlp_wraps_tensor_weights_for_grouped_matmul(self):
        hidden_states = torch.randn(2, 8)
        gate_up_out = torch.randn(2, 16)
        expected = torch.randn(2, 8)
        layer = MagicMock()
        layer.w13_weight = torch.randn(2, 8, 16)
        layer.w2_weight = torch.randn(2, 8, 8)
        layer.w13_bias = None
        layer.w2_bias = None
        layer.swiglu_limit = 0.0
        layer.swiglu_alpha = 1.0
        layer.swiglu_beta = 0.0
        layer.moe = MagicMock()
        layer.moe.has_bias = False
        layer._ascend_moe_lora_context = None

        with (
            patch(
                "vllm_ascend.ops.fused_moe.routed_experts.torch_npu.npu_grouped_matmul",
                side_effect=[[gate_up_out], [expected]],
                create=True,
            ) as mock_grouped_matmul,
            patch(
                "vllm_ascend.ops.fused_moe.routed_experts.torch_npu.npu_swiglu",
                return_value=gate_up_out,
                create=True,
            ),
        ):
            from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod

            scheme = AscendUnquantizedFusedMoEMethod.__new__(AscendUnquantizedFusedMoEMethod)
            mlp_input = MoEMlpComputeInput(
                hidden_states=hidden_states,
                group_list=torch.tensor([1, 1]),
                group_list_type=1,
                dynamic_scale=None,
                topk_scales=None,
                layer=layer,
                quant=MoEQuantParams(quant_type=QuantType.NONE),
                fusion=False,
                activation="silu",
                need_trans=True,
                moe_scheme=scheme,
            )
            output, _ = scheme.apply_mlp(mlp_input)

        self.assertTrue(output is expected)
        first_call, second_call = mock_grouped_matmul.call_args_list
        self.assertEqual(len(first_call.kwargs["weight"]), 1)
        self.assertEqual(len(second_call.kwargs["weight"]), 1)


class TestSchemeApplyMlpDispatch(unittest.TestCase):
    """Test that the correct apply_mlp is dispatched based on quant type."""

    def test_w8a8_scheme_dispatches_to_w8a8_apply_mlp(self):
        """W8A8 quant type dispatches to W8A8's apply_mlp."""
        scheme = _make_w8a8_scheme()
        with patch.object(AscendW8A8DynamicFusedMoEMethod, "apply_mlp", return_value=(torch.randn(2, 8), None)) as mock_apply:
            mlp_input = _common_w8a8_mlp_input(activation="silu")
            scheme.apply_mlp(mlp_input)
        mock_apply.assert_called_once()

    def test_w4a16_scheme_dispatches_to_w4a16_apply_mlp(self):
        """W4A16 quant type dispatches to W4A16's apply_mlp."""
        scheme = _make_w4a16_scheme()
        with patch.object(AscendW4A16FusedMoEMethod, "apply_mlp", return_value=(torch.randn(2, 8), None)) as mock_apply:
            layer = MagicMock()
            layer.w13_weight_packed = torch.randn(1, 8, 4)
            layer.w2_weight_packed = torch.randn(1, 4, 1)
            layer.w13_weight_scale = torch.randn(1, 8, 4)
            layer.w2_weight_scale = torch.randn(1, 4, 1)
            layer.w13_weight_offset = torch.randn(1, 8, 4)
            layer.w2_weight_offset = torch.randn(1, 4, 1)
            layer.w2_weight_scale.dtype = torch.float32
            layer.swiglu_limit = 0.0
            layer.swiglu_alpha = 1.0
            layer.swiglu_beta = 0.0
            mlp_input = MoEMlpComputeInput(
                hidden_states=torch.randn(1, 4),
                group_list=torch.tensor([1], dtype=torch.int64),
                group_list_type=1,
                dynamic_scale=None,
                topk_scales=None,
                layer=layer,
                quant=MoEQuantParams(quant_type=QuantType.W4A16),
                fusion=False,
                activation="silu",
                moe_scheme=scheme,
            )
            scheme.apply_mlp(mlp_input)
        mock_apply.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
