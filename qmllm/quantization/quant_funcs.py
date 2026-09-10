import torch

@torch.no_grad()
def pseudo_quantize_tensor(tensor, n_bits=8, zero_point=True, q_group_size=-1, per_tensor=False, inplace=False):
    """
    The basic quantization function for weight, activation and KV cache.
    """
    org_tensor_shape = tensor.shape
    org_dtype = tensor.dtype
    if q_group_size > 0:
        assert org_tensor_shape[-1] % q_group_size == 0
        tensor = tensor.reshape(-1, q_group_size)
    if per_tensor:
        tensor = tensor.reshape(1, -1)
    assert tensor.dim() == 2

    # Quantization range estimation in fp16 can overflow (notably max-min for
    # asymmetric quantization), and one non-finite activation otherwise
    # poisons an entire token/channel.  Compute the fake-quantization math in
    # FP32 and map exceptional source values to zero.  The output is cast back
    # to preserve the public behaviour and accelerator memory footprint.
    tensor = torch.nan_to_num(
        tensor.float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    if zero_point:
        max_val = tensor.amax(dim=1, keepdim=True)
        min_val = tensor.amin(dim=1, keepdim=True)
        max_int = 2**n_bits - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:
        max_val = tensor.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bits - 1) - 1
        min_int = -(2 ** (n_bits - 1))
        scales = max_val / max_int
        zeros = 0

    if inplace:
        (
            (tensor.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        tensor = (
            torch.clamp(torch.round(tensor / scales) + zeros, min_int, max_int) - zeros
        ) * scales

    if not torch.isfinite(tensor).all():
        raise FloatingPointError("pseudo quantization produced NaN or Inf")

    tensor = tensor.reshape(org_tensor_shape).to(dtype=org_dtype)

    # return the quantized tonsor, the scaling factor and the zero point value
    # return tensor, scales.view(tensor.shape[0], -1), zeros.view(tensor.shape[0], -1)
    return tensor


@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8, zero_point=False):
    """
    The basic quantization function for weight, activation and KV cache.
    """
    tensor = pseudo_quantize_tensor(w, n_bits=n_bits, zero_point=zero_point, q_group_size=-1, per_tensor=False, inplace=False)
    return tensor
    
@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8, zero_point=False):
    t_shape = t.shape
    t = t.view(-1, t_shape[-1])
    t = pseudo_quantize_tensor(t, n_bits=n_bits, zero_point=zero_point, q_group_size=-1, per_tensor=False, inplace=False)
    return t.reshape(t_shape)
    
@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8, zero_point=False):
    """
    The basic quantization function for weight, activation and KV cache.
    """
    tensor = pseudo_quantize_tensor(w, n_bits=n_bits, zero_point=zero_point, q_group_size=-1, per_tensor=True, inplace=False)
    return tensor
    
@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8, zero_point=False):
    t_shape = t.shape
    t = t.view(-1, t_shape[-1])
    t = pseudo_quantize_tensor(t, n_bits=n_bits, zero_point=zero_point, q_group_size=-1, per_tensor=True, inplace=False)
    return t.reshape(t_shape)
