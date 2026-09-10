import torch
import torch.nn as nn

from qmllm.methods.lat_awq.quantize.saliency import get_act_scale
from qmllm.methods.lat_awq.quantize.loss import token_weighted_mse_in_batches
from qmllm.quantization.quant_funcs import pseudo_quantize_tensor
from qmllm.quantization.qlinear import WALinear
from qmllm.utils.device import finite_act_scale_in_batches, normalized_power_scales


def test_token_aware_saliency_and_lambda_zero():
    x = torch.tensor([[[1.0, 10.0], [3.0, 2.0], [5.0, 1.0]]], dtype=torch.float16)
    token_w = torch.tensor([[1.0, 0.0, 0.0]])

    a_global, a_imp, a_final = get_act_scale(x, token_w, mix_lambda=1.0)
    assert a_global.dtype == torch.float32
    assert a_imp.dtype == torch.float32
    assert a_global.shape == a_imp.shape == a_final.shape == (2,)
    assert torch.allclose(a_global, torch.tensor([3.0, 13.0 / 3.0]))
    assert torch.allclose(a_imp, torch.tensor([1.0, 10.0]))
    assert torch.equal(get_act_scale(x, token_w, mix_lambda=0.0)[2], a_global)


def test_saliency_zero_and_nonfinite_weights_fall_back_to_global():
    x = torch.randn(2, 3, 4)
    a_global = get_act_scale(x)[0]
    for token_w in (torch.zeros(2, 3), torch.full((2, 3), float("nan"))):
        _, a_imp, a_final = get_act_scale(x, token_w, mix_lambda=1.0)
        assert torch.equal(a_imp, a_global)
        assert torch.equal(a_final, a_global)


def test_nonfinite_calibration_values_are_sanitized():
    x = torch.tensor(
        [[[1.0, float("nan"), float("inf")], [3.0, 5.0, 7.0]]],
        dtype=torch.float16,
    )
    clean, stats = finite_act_scale_in_batches(x, batch_size=1, device="cpu")
    a_global, _, a_final = get_act_scale(x, mix_lambda=0.0)

    assert torch.isfinite(clean).all()
    assert torch.isfinite(stats).all()
    assert torch.isfinite(a_global).all()
    assert torch.isfinite(a_final).all()


def test_fp16_quantization_and_scale_normalization_stay_finite():
    x = torch.tensor(
        [[float("nan"), float("inf"), -float("inf"), 1.25, -2.5, 3.0, -4.0, 0.0]],
        dtype=torch.float16,
    )
    quantized = pseudo_quantize_tensor(x, n_bits=4)
    scales = normalized_power_scales(
        torch.tensor([1024.0, 60000.0], dtype=torch.float16), ratio=0.95
    )

    assert quantized.dtype == torch.float16
    assert torch.isfinite(quantized).all()
    assert scales.dtype == torch.float32
    assert torch.isfinite(scales).all()


def test_saliency_rejects_silent_broadcasting_and_bad_lambda():
    x = torch.randn(2, 3, 4)
    for bad_weight in (torch.ones(2, 3, 1), torch.ones(1, 3)):
        try:
            get_act_scale(x, bad_weight, mix_lambda=1.0)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid token_w shape was accepted")
    try:
        get_act_scale(x, torch.ones(2, 3), mix_lambda=1.01)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid saliency_mix_lambda was accepted")


class _Identity(nn.Module):
    def forward(self, x):
        return x


class _Zero(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


class _DropToken(nn.Module):
    def forward(self, x):
        return x[:, :1]


class _NaNSecondToken(nn.Module):
    def forward(self, x):
        y = x.clone()
        y[:, 1] = float("nan")
        return y


def test_token_weighted_reconstruction_is_per_token_mse():
    x = torch.tensor([[[1.0, 3.0], [2.0, 4.0]]])
    token_w = torch.tensor([[1.0, 0.0]])
    loss = token_weighted_mse_in_batches(
        _Identity(),
        baseline_x=x,
        test_module=_Zero(),
        token_w=token_w,
        batch_size=1,
        device="cpu",
    )
    assert loss == 5.0  # mean([1^2, 3^2]) for the selected token


def test_token_weighted_reconstruction_rejects_output_broadcasting():
    x = torch.randn(1, 2, 3)
    try:
        token_weighted_mse_in_batches(
            _Identity(), x, _DropToken(), torch.ones(1, 2), device="cpu"
        )
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched output token shapes were accepted")


def test_token_weighted_reconstruction_ignores_nonfinite_unweighted_tokens():
    x = torch.tensor([[[1.0, 3.0], [2.0, 4.0]]])
    loss = token_weighted_mse_in_batches(
        _Identity(),
        baseline_x=x,
        test_module=_NaNSecondToken(),
        token_w=torch.tensor([[1.0, 0.0]]),
        batch_size=1,
        device="cpu",
    )
    assert loss == 0.0


def test_walinear_w4a8_quantizes_weights_and_activations():
    linear = nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        linear.weight.copy_(
            torch.tensor(
                [[-1.1, -0.4, 0.3, 1.2], [0.8, -0.7, 0.2, -0.1], [0.9, 0.5, -0.2, -1.0]]
            )
        )
    quantized = WALinear.from_float(linear, w_bit=4, a_bit=8)
    x = torch.tensor([[[0.13, -0.27, 0.42, 1.31], [1.7, -0.2, 0.01, -0.8]]])

    assert quantized.w_bit == 4
    assert quantized.a_bit == 8
    assert quantized.weight_quant_name == "per_channel"
    assert quantized.act_quant_name == "per_token"
    assert torch.isfinite(quantized(x)).all()
    assert not torch.equal(quantized.weight, linear.weight)
